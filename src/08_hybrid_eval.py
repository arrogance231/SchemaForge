"""
08_hybrid_eval.py - GPU-side driver for the §6 hybrid pipeline benchmark and the
§7 routing-threshold sweep.

Runs a loaded HF checkpoint (base model or distilled student) as the residual-field
extractor over an eval JSONL.  For every record the deterministic pre-pass
(``schemaforge.deterministic.prepass.run_prepass``) resolves the fields it owns, the
model is asked ONLY for the residual (``unresolved``) fields, and the two are merged
with ``schemaforge.hybrid.pipeline.merge_prediction``.  THREE systems are scored on
the SAME records: the rules pre-pass alone, the model alone, and the hybrid merge.
Raw per-record mean-token-logprob confidences (captured exactly like
``06_confidence_eval.py``) drive a coverage-at-risk sweep answering, for the hybrid
system, "at what coverage does field F1 stay above target if low-confidence records
are escalated instead of trusted" -- no calibrator is fitted here, just operating
points reported.

HF-``generate``-only (batched), no vLLM, greedy decoding (``do_sample=False``),
matching ``05_eval_checkpoint.py`` / ``06_confidence_eval.py``; ``output_scores=True``
exposes the per-step logits the confidence signal is built from.  This driver is NOT
runnable without a model; the CPU-only pre-pass + merge integration is covered by
``tests/test_hybrid.py``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
from dataclasses import asdict
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemaforge import get_schema  # noqa: E402
from schemaforge.calibration import RiskCoveragePoint, coverage_at_risk, risk_coverage_curve  # noqa: E402
from schemaforge.deterministic.prepass import run_prepass  # noqa: E402
from schemaforge.evaluation.harness import EvalRecord, load_records  # noqa: E402
from schemaforge.evaluation.json_utils import parse_json  # noqa: E402
from schemaforge.evaluation.metrics import RecordResult, aggregate, evaluate_record  # noqa: E402
from schemaforge.hybrid.pipeline import merge_prediction, rules_only_prediction  # noqa: E402


def build_prompt(
    schema_name: str, document_text: str, unresolved_fields: list[str]
) -> str:
    """Build the residual-fields prompt for one corpus record.

    Same template wording as ``01_generate_teacher.py``'s ``build_prompt`` but the
    "Fields to extract:" line lists ``unresolved_fields`` (sorted) instead of every
    semantic field unconditionally: the pre-pass already owns the resolved
    deterministic fields, so the model is only asked for the residual.  Deterministic
    and free of any randomness/timestamps.
    """
    fields = ", ".join(sorted(unresolved_fields))
    return (
        f"Extract the following fields as JSON from the text below. "
        f"Schema: {schema_name}.\n"
        f"Fields to extract: {fields}.\n"
        f"Text:\n{document_text}\n"
        f"JSON Output:"
    )


def load_model(model_name: str) -> tuple[Any, Any]:
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


def _format_metrics_table(
    rules_metrics: dict[str, float],
    model_metrics: dict[str, float],
    hybrid_metrics: dict[str, float],
) -> str:
    """Render the three-system field-metrics comparison as a plain-text table."""
    metric_names = (
        "field_precision",
        "field_recall",
        "field_f1",
        "hallucination_rate",
        "missing_field_rate",
        "schema_validity",
    )
    rows = (
        ("rules", rules_metrics),
        ("model", model_metrics),
        ("hybrid", hybrid_metrics),
    )
    lines = ["".join(f"{name:<24}" for name in (("system",) + metric_names)).rstrip()]
    for system, metrics in rows:
        cells = "".join(
            f"{metrics.get(name, 0.0):<24.4f}" for name in metric_names
        )
        lines.append((f"{system:<24}" + cells).rstrip())
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run §6 hybrid (pre-pass + model) benchmark with a §7 escalation sweep."
    )
    parser.add_argument("--model", required=True, help="HF model id or local checkpoint path")
    parser.add_argument("--label", required=True, help="report label, e.g. 'baseline' or 'distilled'")
    parser.add_argument("--eval-jsonl", required=True, help="path to eval corpus (serialize_records JSONL format)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--f1-threshold", type=float, default=0.5)
    parser.add_argument("--max-risk", default="0.1,0.2,0.3", help="comma-separated risk thresholds")
    parser.add_argument("--out", help="optional path to write the JSON report")
    args = parser.parse_args()

    max_risk_thresholds = [float(value) for value in args.max_risk.split(",")]

    records = load_records(args.eval_jsonl)
    print(f"[*] Loaded {len(records)} eval records from {args.eval_jsonl}")

    prepass_results = [
        run_prepass(record.source_text, get_schema(record.schema)) for record in records
    ]
    prompts = [
        build_prompt(record.schema, record.source_text, result.unresolved)
        for record, result in zip(records, prepass_results)
    ]

    model, tokenizer = load_model(args.model)
    raw_outputs, confidences = batch_generate_with_scores(
        model,
        tokenizer,
        prompts,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )

    rules_results: list[RecordResult] = []
    model_results: list[RecordResult] = []
    hybrid_results: list[RecordResult] = []
    for record, prepass_result, raw in zip(records, prepass_results, raw_outputs):
        spec = get_schema(record.schema)
        model_prediction = parse_json(raw)
        rules_pred = rules_only_prediction(prepass_result.resolved)
        model_pred = model_prediction
        hybrid_pred = merge_prediction(prepass_result.resolved, model_prediction, spec)
        rules_results.append(
            evaluate_record(rules_pred, record.reference, spec, source_text=record.source_text)
        )
        # NOTE: this "model_alone" number is NOT the same experiment as
        # 05_eval_checkpoint.py's model-alone run: THIS driver's prompt asked for
        # the residual fields only, not every semantic field, so the two are not
        # directly comparable.
        model_results.append(
            evaluate_record(model_pred, record.reference, spec, source_text=record.source_text)
        )
        hybrid_results.append(
            evaluate_record(hybrid_pred, record.reference, spec, source_text=record.source_text)
        )

    rules_metrics = aggregate(rules_results)
    model_metrics = aggregate(model_results)
    hybrid_metrics = aggregate(hybrid_results)

    hybrid_correct = [field_f1(result) >= args.f1_threshold for result in hybrid_results]
    curve = risk_coverage_curve(confidences, hybrid_correct)
    coverage_map = {
        str(threshold): coverage_at_risk(curve, threshold)
        for threshold in max_risk_thresholds
    }

    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print("=" * 50)
    print(f"[*] label: {args.label}")
    print(f"[*] model: {args.model}")
    print(f"[*] records: {len(records)}")
    print("[*] field metrics (micro-averaged):")
    print(_format_metrics_table(rules_metrics, model_metrics, hybrid_metrics))
    print("[*] coverage@risk (hybrid system, escalation sweep):")
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
            "rules_metrics": rules_metrics,
            "model_metrics": model_metrics,
            "hybrid_metrics": hybrid_metrics,
            "risk_coverage_curve": [asdict(point) for point in curve],
            "coverage_at_risk": coverage_map,
        }
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"[+] Saved report to {args.out}")


if __name__ == "__main__":
    main()
