"""
05_eval_checkpoint.py - Baseline-vs-checkpoint field-level eval (build order §11 step 2).

Runs a loaded HF checkpoint (base model or distilled student) as a JSON extractor
over a JSONL eval corpus and reports the field-level metrics from
``schemaforge.evaluation.harness``.  Run it once against the un-distilled base and
once against a distilled checkpoint to get the comparison table behind the README
status item and the PROJECT_CHARTER baseline number.

HF-``generate``-only (batched), no vLLM: the AMD/ROCm server has no vLLM build,
so this is what actually executes there.  Greedy decoding (``do_sample=False``),
since evaluation is deterministic.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemaforge import get_schema  # noqa: E402
from schemaforge.evaluation.harness import EvalRecord, evaluate, load_records  # noqa: E402
from schemaforge.evaluation.json_utils import parse_json  # noqa: E402


def build_prompt(schema_name: str, document_text: str) -> str:
    """Build the teacher prompt for one corpus record.

    Includes the schema name and the sorted list of ``semantic_fields`` (the
    fields the model is actually asked to extract -- deterministic_fields are
    out of scope per research direction §0).  Deterministic and free of any
    randomness/timestamps.
    """
    spec = get_schema(schema_name)
    fields = ", ".join(sorted(spec.semantic_fields))
    return (
        f"Extract the following fields as JSON from the text below. "
        f"Schema: {schema_name}.\n"
        f"Fields to extract: {fields}.\n"
        f"Text:\n{document_text}\n"
        f"JSON Output:"
    )


def load_model(model_name: str):
    """Load tokenizer + model, resolving device/dtype at runtime like the teacher script."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def batch_generate(model, tokenizer, prompts: list[str], *, max_new_tokens: int, batch_size: int) -> list[str]:
    """Greedily decode one JSON object per prompt, in prompt order."""
    import torch

    n_batches = (len(prompts) + batch_size - 1) // batch_size
    generated_texts: list[str] = []
    for batch_idx, start in enumerate(range(0, len(prompts), batch_size), start=1):
        print(f"[*] generate: batch {batch_idx}/{n_batches}")
        batch = prompts[start:start + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)
        input_seq_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = generated[:, input_seq_len:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        generated_texts.extend(text.strip() for text in decoded)
    return generated_texts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run field-level eval of an HF checkpoint over an eval JSONL.")
    parser.add_argument("--model", required=True, help="HF model id or local checkpoint path")
    parser.add_argument("--label", required=True, help="report label, e.g. 'baseline' or 'distilled'")
    parser.add_argument("--eval-jsonl", required=True, help="path to eval corpus (serialize_records JSONL format)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--out", help="optional path to write the JSON report")
    args = parser.parse_args()

    records = load_records(args.eval_jsonl)
    print(f"[*] Loaded {len(records)} eval records from {args.eval_jsonl}")

    prompts = [build_prompt(record.schema, record.source_text) for record in records]
    model, tokenizer = load_model(args.model)
    raw_outputs = batch_generate(
        model,
        tokenizer,
        prompts,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )

    by_id: dict[int, dict | None] = {}
    for record, raw in zip(records, raw_outputs):
        by_id[id(record)] = parse_json(raw)

    def predict_fn(record: EvalRecord) -> dict | None:
        return by_id[id(record)]

    metrics = evaluate(records, predict_fn)

    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print("=" * 50)
    print(f"[*] label: {args.label}")
    print(f"[*] model: {args.model}")
    print(f"[*] records: {len(records)}")
    print(f"[*] timestamp: {timestamp}")
    print("=" * 50)
    print(json.dumps(metrics, indent=2, sort_keys=True))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        report = {
            "label": args.label,
            "model": args.model,
            "timestamp": timestamp,
            "n_records": len(records),
            "metrics": metrics,
        }
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"[+] Saved report to {args.out}")


if __name__ == "__main__":
    main()
