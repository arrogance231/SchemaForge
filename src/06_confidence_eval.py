"""
06_confidence_eval.py - GPU-side driver for confidence-aware inference (research direction §5).

Runs a loaded HF checkpoint (base model or distilled student) as a JSON
extractor over a JSONL eval corpus while capturing per-token generation
probabilities, derives a raw mean-token-logprob confidence per record,
evaluates correctness with the existing field-level metrics, fits temperature
scaling on one half of the records and evaluates calibration on the other, and
writes a JSON report with the risk-coverage curve and coverage-at-risk
operating points.

HF-``generate``-only (batched), no vLLM, greedy decoding (``do_sample=False``)
matching ``05_eval_checkpoint.py``; ``output_scores=True`` exposes the per-step
logits that the confidence signal is built from.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemaforge import get_schema  # noqa: E402
from schemaforge.calibration import (  # noqa: E402
    RiskCoveragePoint,
    TemperatureScaler,
    coverage_at_risk,
    expected_calibration_error,
    fit_temperature,
    risk_coverage_curve,
)
from schemaforge.evaluation.harness import EvalRecord, load_records  # noqa: E402
from schemaforge.evaluation.json_utils import parse_json  # noqa: E402
from schemaforge.evaluation.metrics import RecordResult, evaluate_record  # noqa: E402


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


def batch_confidences(tokenizer, new_tokens, scores) -> list[float]:
    """Raw per-sample mean-token-logprob confidence, stopping at first EOS/pad.

    ``scores`` is the per-step logits tuple from ``generate(..., output_scores=True)``;
    with left-padding, step ``i`` always predicts position ``input_len + i`` of every
    sample, so the log-prob of the token actually chosen at step ``i`` is gathered from
    ``log_softmax(scores[i])``.  A finished sample's trailing generated tokens are its
    pad token, so per-sample accumulation stops at the first EOS/pad; a sample that
    emits zero real tokens (immediate EOS) gets confidence ``0.0`` instead of a
    divide-by-zero.
    """
    import torch

    n_steps = min(len(scores), new_tokens.shape[1])
    new_tokens = new_tokens[:, :n_steps]
    step_logprobs = torch.stack(
        [torch.log_softmax(logits, dim=-1) for logits in scores[:n_steps]], dim=1
    )
    chosen_logprobs = step_logprobs.gather(-1, new_tokens.unsqueeze(-1)).squeeze(-1)

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    confidences: list[float] = []
    for row_logprobs, tokens in zip(chosen_logprobs, new_tokens):
        real: list[float] = []
        for logprob, token in zip(row_logprobs.tolist(), tokens.tolist()):
            if (eos_id is not None and token == eos_id) or (
                pad_id is not None and token == pad_id
            ):
                break
            real.append(logprob)
        if not real:
            confidences.append(0.0)
            continue
        confidences.append(min(max(math.exp(sum(real) / len(real)), 0.0), 1.0))
    return confidences


def batch_generate_with_scores(
    model, tokenizer, prompts: list[str], *, max_new_tokens: int, batch_size: int
) -> tuple[list[str], list[float]]:
    """Greedily decode one JSON object per prompt, capturing raw confidence per sample."""
    import torch

    n_batches = (len(prompts) + batch_size - 1) // batch_size
    generated_texts: list[str] = []
    confidences: list[float] = []
    for batch_idx, start in enumerate(range(0, len(prompts), batch_size), start=1):
        print(f"[*] generate: batch {batch_idx}/{n_batches}")
        batch = prompts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)
        input_seq_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )
        new_tokens = outputs.sequences[:, input_seq_len:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        generated_texts.extend(text.strip() for text in decoded)
        confidences.extend(batch_confidences(tokenizer, new_tokens, outputs.scores))
    return generated_texts, confidences


def field_f1(result: RecordResult) -> float:
    """Per-record field F1 from the record-level leaf-unit counts."""
    precision = result.n_correct / result.n_predicted if result.n_predicted else 0.0
    recall = result.n_correct / result.n_reference if result.n_reference else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run confidence-aware (§5) eval of an HF checkpoint over an eval JSONL."
    )
    parser.add_argument("--model", required=True, help="HF model id or local checkpoint path")
    parser.add_argument("--label", required=True, help="report label, e.g. 'baseline' or 'distilled'")
    parser.add_argument("--eval-jsonl", required=True, help="path to eval corpus (serialize_records JSONL format)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--out", help="optional path to write the JSON report")
    parser.add_argument("--f1-threshold", type=float, default=0.5)
    parser.add_argument("--max-risk", default="0.1,0.2,0.3", help="comma-separated risk thresholds")
    args = parser.parse_args()

    max_risk_thresholds = [float(value) for value in args.max_risk.split(",")]

    records = load_records(args.eval_jsonl)
    print(f"[*] Loaded {len(records)} eval records from {args.eval_jsonl}")

    prompts = [build_prompt(record.schema, record.source_text) for record in records]
    model, tokenizer = load_model(args.model)
    raw_outputs, confidences = batch_generate_with_scores(
        model,
        tokenizer,
        prompts,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )

    field_f1s: list[float] = []
    correct: list[bool] = []
    for record, raw in zip(records, raw_outputs):
        prediction = parse_json(raw)
        result = evaluate_record(
            prediction,
            record.reference,
            get_schema(record.schema),
            source_text=record.source_text,
        )
        f1 = field_f1(result)
        field_f1s.append(f1)
        correct.append(f1 >= args.f1_threshold)

    fit_confidences = [confidences[i] for i in range(0, len(records), 2)]
    fit_correct = [correct[i] for i in range(0, len(records), 2)]
    holdout_confidences = [confidences[i] for i in range(1, len(records), 2)]
    holdout_correct = [correct[i] for i in range(1, len(records), 2)]

    raw_ece_fit = expected_calibration_error(fit_confidences, fit_correct)
    raw_ece_holdout = expected_calibration_error(holdout_confidences, holdout_correct)
    scaler = fit_temperature(fit_confidences, fit_correct)
    calibrated_ece_holdout = expected_calibration_error(
        [scaler.calibrate(confidence) for confidence in holdout_confidences],
        holdout_correct,
    )
    # Full-set curve (re-uses the fit split): a known looseness at ~70 records, accepted so
    # the operating points describe the deployable confidence signal.
    curve = risk_coverage_curve(confidences, correct)
    coverage_map = {
        str(threshold): coverage_at_risk(curve, threshold)
        for threshold in max_risk_thresholds
    }

    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print("=" * 50)
    print(f"[*] label: {args.label}")
    print(f"[*] model: {args.model}")
    print(f"[*] records: {len(records)}")
    print(f"[*] fitted temperature: {scaler.temperature:.4f}")
    print(f"[*] raw ECE (fit): {raw_ece_fit:.4f}")
    print(f"[*] raw ECE (holdout): {raw_ece_holdout:.4f}")
    print(f"[*] calibrated ECE (holdout): {calibrated_ece_holdout:.4f}")
    print("[*] coverage@risk (full-set curve):")
    for threshold in max_risk_thresholds:
        print(f"    risk <= {threshold:.2f}: coverage {coverage_map[str(threshold)]:.4f}")
    print(f"[*] timestamp: {timestamp}")
    print("=" * 50)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        report = {
            "label": args.label,
            "model": args.model,
            "timestamp": timestamp,
            "n_records": len(records),
            "f1_threshold": args.f1_threshold,
            "fitted_temperature": scaler.temperature,
            "raw_ece_fit": raw_ece_fit,
            "raw_ece_holdout": raw_ece_holdout,
            "calibrated_ece_holdout": calibrated_ece_holdout,
            "risk_coverage_curve": [asdict(point) for point in curve],
            "coverage_at_risk": coverage_map,
            "per_record": [
                {
                    "schema": record.schema,
                    "tags": record.tags,
                    "raw_confidence": confidence,
                    "field_f1": f1,
                    "correct": ok,
                }
                for record, confidence, f1, ok in zip(
                    records, confidences, field_f1s, correct
                )
            ],
        }
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"[+] Saved report to {args.out}")


if __name__ == "__main__":
    main()
