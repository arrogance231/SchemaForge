"""
09_loop.py - Research direction §8: Continual Distillation Loop driver.

Orchestrates the iterative generate -> teacher-query -> gate -> retrain ->
benchmark -> failure-analysis cycle by chaining the existing standalone numbered
scripts in ``src/`` as subprocesses (this repo's convention: ``src/*.py`` scripts
are standalone and never cross-imported, so nothing internal is imported here --
only stdlib).  Every iteration's dataset version, gate rejection rate, checkpoint
identity, benchmark table and failure histogram are appended as one JSON line to
``--state-file`` so the loop is resumable and no result (including a failed
iteration) is silently overwritten.

This driver defaults to a single deliberate, reviewable iteration per invocation
(``--max-iterations 1``), not an unattended infinite loop: each real §8 step in
this project so far has been run, logged and reviewed manually, and each call of
this script is one such iteration.  It also never auto-tunes its own next-iteration
parameters: the ``top_failure_category`` signal is recorded and printed for a
human/agent to react to in the next invocation's ``--schemas``/``--operators``
choice, per the manual per-iteration practice documented in ``logs/V2_TRAINING_FAILURES.md``.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = "models/distilled_minicpm5_1b_v2_amd"
CHECKPOINT_FILE = os.path.join(MODEL_DIR, "model.safetensors")
GENERATED_CORPUS = "data/hard_examples_train.jsonl"
GATE_LINE_RE = re.compile(r"\[\+\] Gate: (\d+)/(\d+) admitted \(([\d.]+)%")
WRITTEN_RECORDS_RE = re.compile(r"wrote (\d+) records")
EXCLUDED_FAILURE_CATEGORY = "ambiguous_input"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args; the loop's own params, not forwarded to any subprocess."""
    parser = argparse.ArgumentParser(
        description="Run one or more §8 continual-distillation loop iterations."
    )
    parser.add_argument("--eval-jsonl", required=True, help="eval corpus for the hybrid + failure benchmarks")
    parser.add_argument("--schemas", nargs="+", default=None, help="schema names for hard-example generation")
    parser.add_argument("--n-per-schema", type=int, default=30, help="examples per schema per severity")
    parser.add_argument("--severities", default="0.0,0.3,0.6,0.9", help="comma-separated severity levels")
    parser.add_argument("--operators", nargs="+", default=None, help="corruption operator names for generation")
    parser.add_argument("--seed", type=int, default=42, help="deterministic corpus seed")
    parser.add_argument("--state-file", default="logs/loop_state.jsonl", help="append-only JSON-lines bookkeeping file")
    parser.add_argument("--max-iterations", type=int, default=1, help="loop iterations per invocation (default: one deliberate iteration)")
    parser.add_argument("--dry-run", action="store_true", help="print each subprocess command without running anything")
    return parser.parse_args(argv)


def _run(cmd: list[str], dry_run: bool) -> tuple[str, str]:
    """Run ``cmd`` from the repo root, returning ``(stdout, stderr)``.

    In ``dry_run`` mode nothing executes: print the command and return empty
    strings.  Non-zero exits raise :class:`subprocess.CalledProcessError` so the
    caller can abort the loop.
    """
    if dry_run:
        print(f"[dry-run] would run: {cmd}")
        return "", ""
    proc = subprocess.run(
        cmd, capture_output=True, text=True, check=True, cwd=REPO_ROOT
    )
    return proc.stdout, proc.stderr


def append_record(record: dict, state_file: str) -> None:
    """Append one JSON object per line to ``state_file``, never overwriting."""
    Path(state_file).parent.mkdir(parents=True, exist_ok=True)
    with open(state_file, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def dataset_version(args: argparse.Namespace) -> dict:
    """The exact generate.py CLI args used, for corpus reproducibility."""
    return {
        "split": "train",
        "n": args.n_per_schema,
        "severities": args.severities,
        "seed": args.seed,
        "out": GENERATED_CORPUS,
        "schemas": args.schemas or [],
        "operators": args.operators or [],
    }


def generate_command(args: argparse.Namespace) -> list[str]:
    """Assemble the ``schemaforge.hardexamples.generate`` invocation."""
    cmd = [
        sys.executable,
        "-m",
        "schemaforge.hardexamples.generate",
        "--split",
        "train",
        "--n",
        str(args.n_per_schema),
        "--severities",
        args.severities,
        "--seed",
        str(args.seed),
        "--out",
        GENERATED_CORPUS,
    ]
    if args.schemas:
        cmd += ["--schemas"] + args.schemas
    if args.operators:
        cmd += ["--operators"] + args.operators
    return cmd


def parse_written_records(stdout: str, stderr: str) -> int | None:
    """Extract ``wrote N records`` from generate.py's output (it logs to stderr)."""
    match = WRITTEN_RECORDS_RE.search(stdout + "\n" + stderr)
    return int(match.group(1)) if match else None


def parse_gate(stdout: str) -> tuple[int | None, int | None, float | None]:
    """Extract admitted/total/rejection-rate from 01's ``[+] Gate:`` line."""
    match = GATE_LINE_RE.search(stdout)
    if not match:
        return None, None, None
    return int(match.group(1)), int(match.group(2)), float(match.group(3))


def checkpoint_sha256(dry_run: bool) -> str | None:
    """Content hash of the trained checkpoint for provenance, or ``None`` if absent."""
    if dry_run:
        return None
    path = os.path.join(REPO_ROOT, CHECKPOINT_FILE)
    if not os.path.exists(path):
        return None
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_report_field(report_path: str, field: str) -> dict | None:
    """Read one field back from a subprocess-written JSON report, or ``None``."""
    try:
        with open(os.path.join(REPO_ROOT, report_path), "r", encoding="utf-8") as handle:
            report = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    value = report.get(field)
    return value if isinstance(value, dict) else None


def top_failure_category(by_category: dict | None) -> tuple[str | None, int | None]:
    """Highest-count failure category excluding ``ambiguous_input`` (scored separately per §3/§6)."""
    if not by_category:
        return None, None
    candidates = {
        name: count
        for name, count in by_category.items()
        if name != EXCLUDED_FAILURE_CATEGORY and count > 0
    }
    if not candidates:
        return None, None
    return max(candidates.items(), key=lambda item: item[1])


def run_iteration(args: argparse.Namespace, idx: int, dry_run: bool) -> dict:
    """Run one full loop iteration and return its state-file record."""
    record: dict = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "iteration_index": idx,
        "dataset_version": dataset_version(args),
        "n_hard_examples": None,
        "teacher_n_admitted": None,
        "teacher_n_total": None,
        "teacher_rejection_rate_pct": None,
        "checkpoint_sha256": None,
        "hybrid_metrics": None,
        "by_category": None,
        "top_failure_category": None,
        "top_failure_category_count": None,
        "dry_run": dry_run,
    }
    try:
        stdout, stderr = _run(generate_command(args), dry_run)
        record["n_hard_examples"] = parse_written_records(stdout, stderr)

        stdout, _ = _run([sys.executable, "src/01_generate_teacher.py"], dry_run)
        admitted, total, rejection = parse_gate(stdout)
        record["teacher_n_admitted"] = admitted
        record["teacher_n_total"] = total
        record["teacher_rejection_rate_pct"] = rejection

        _run([sys.executable, "src/02_train_distill.py"], dry_run)
        record["checkpoint_sha256"] = checkpoint_sha256(dry_run)

        hybrid_out = f"logs/loop_iter_{idx}_hybrid.json"
        _run(
            [
                sys.executable,
                "src/08_hybrid_eval.py",
                "--model",
                os.path.join(".", MODEL_DIR),
                "--label",
                f"loop_iter_{idx}",
                "--eval-jsonl",
                args.eval_jsonl,
                "--out",
                hybrid_out,
            ],
            dry_run,
        )
        record["hybrid_metrics"] = load_report_field(hybrid_out, "hybrid_metrics")

        failures_out = f"logs/loop_iter_{idx}_failures.json"
        _run(
            [
                sys.executable,
                "src/07_failure_eval.py",
                "--model",
                os.path.join(".", MODEL_DIR),
                "--label",
                f"loop_iter_{idx}",
                "--eval-jsonl",
                args.eval_jsonl,
                "--out",
                failures_out,
            ],
            dry_run,
        )
        record["by_category"] = load_report_field(failures_out, "by_category")

        category, count = top_failure_category(record["by_category"])
        record["top_failure_category"] = category
        record["top_failure_category_count"] = count
        if category is not None:
            print(
                f"[*] top failure category this iteration: {category} ({count} instances) "
                "-- consider a targeted corpus next"
            )
    except subprocess.CalledProcessError as exc:
        if not dry_run:
            append_record(record, args.state_file)
        print(f"[-] subprocess failed: {' '.join(exc.cmd)}", file=sys.stderr)
        if exc.stderr:
            print(exc.stderr, file=sys.stderr)
        sys.exit(1)

    if not dry_run:
        append_record(record, args.state_file)
    return record


def print_summary(args: argparse.Namespace, records: list[dict]) -> None:
    """Report how many iterations ran, the last hybrid field F1, and the state file."""
    print(f"[*] loop finished: {len(records)} iteration(s) run")
    if records:
        hybrid = records[-1].get("hybrid_metrics")
        if isinstance(hybrid, dict) and hybrid.get("field_f1") is not None:
            print(f"[*] final hybrid field F1: {hybrid['field_f1']:.4f}")
    print(f"[*] state file: {args.state_file}")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: run ``--max-iterations`` loop iterations, logging each."""
    args = parse_args(argv)
    records: list[dict] = []
    for idx in range(args.max_iterations):
        records.append(run_iteration(args, idx, args.dry_run))
    print_summary(args, records)


if __name__ == "__main__":
    main()
