#!/usr/bin/env python3
"""
generate_graphs.py
=================
Data-driven evidence graphs for SchemaForge V2/V3. Reads REAL experiment
metrics from the V3 iteration manifests under experiments/v3-iter{N}-*/ and
plots the documented V2 iteration table + V2-FINAL release comparison.

Output: 300-dpi PNGs written to docs/graphs/ (created if missing).

Usage:
    python3 src/generate_graphs.py

Notes:
    * Requires matplotlib + seaborn in the ACTIVE python3. The project .venv
      does not install them -- use the system interpreter
      (/usr/bin/python3, with --user packages) instead.
    * V2 iteration numbers are documented constants (V2 manifests predate
      this script); V3 numbers are read live from the manifests, falling
      back to documented constants only when a manifest genuinely lacks a
      field. Sources for the constants are cited inline.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: no display required
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = ROOT / "experiments"
OUTPUT_DIR = ROOT / "docs" / "graphs"

sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.sans-serif": "DejaVu Sans", "font.size": 11})


# ---------------------------------------------------------------------------
# Documented constants (V2 iteration table / V2-FINAL release comparison)
# Sources: docs/WHITEPAPER.md (V2 iteration table, lines 148-153) and
# logs/V2_TRAINING_FAILURES.md (V2-FINAL model-alone F1, line 350).
# ---------------------------------------------------------------------------
V2_ITERATION_F1 = {
    "iter 3": 0.3873,
    "iter 5/10": 0.6858,   # all-time best hybrid F1 on the 72-record eval
    "iter 12": 0.6432,
    "iter 13": 0.6830,
    "iter 14": 0.6583,
    "iter 15": 0.6827,     # V2-FINAL release checkpoint
}
V2_FINAL_F1 = 0.6827        # V2-FINAL release checkpoint, 72-record eval
V2_BEST_F1 = 0.6858         # iteration 5/10 all-time best
V2_FINAL_RULES_F1 = 0.2911  # rules alone (WHITEPAPER.md: 0.9185 prec / 0.1729 rec)
V2_FINAL_MODEL_F1 = 0.4846  # model alone, residual prompt (logs/V2_TRAINING_FAILURES.md)

# V3 documented fallbacks (docs/FINDINGS_V3_ITERATIONS_1_4.md / WHITEPAPER_V3.md)
V3_FALLBACK_F1 = {1: 0.6581, 2: 0.6745, 3: 0.6597, 4: 0.6671}
V3_FALLBACK_MFS = {1: 58.8, 2: 52.3, 3: 57.0, 4: 56.4}
V3_FALLBACK_F1_288 = {2: 0.6742, 3: 0.6524, 4: 0.6650}
V3_FALLBACK_EPOCHS = {1: 3, 2: 2, 3: 1, 4: 2}  # epochs used per iteration


# ---------------------------------------------------------------------------
# Manifest loader -- defensive against both metric key placements:
#   * iter1/iter3 (and iter2): training_performance.evaluation_metrics_72record_eval
#   * iter4:                    top-level evaluation_metrics_72record_eval
# ---------------------------------------------------------------------------
def _find_eval(manifest: dict, key: str) -> dict:
    """Return the eval dict for `key` from either top-level or nested layout."""
    if isinstance(manifest.get(key), dict):
        return manifest[key]
    tp = manifest.get("training_performance")
    if isinstance(tp, dict) and isinstance(tp.get(key), dict):
        return tp[key]
    return {}


def _hybrid_f1(manifest: dict, eval_key: str):
    """Extract hybrid field F1 from an eval section, or None."""
    ev = _find_eval(manifest, eval_key)
    hybrid = ev.get("hybrid") if isinstance(ev, dict) else None
    if isinstance(hybrid, dict):
        v = hybrid.get("field_f1")
        if isinstance(v, (int, float)):
            return float(v)
    return None


def load_v3_metrics():
    """Read real metrics from every experiments/v3-iter{N}-*/manifest.json.
    Falls back to documented constants only where a manifest lacks a field.
    Returns {1: {...}, 2: {...}, 3: {...}, 4: {...}}.
    """
    out = {}
    for it in range(1, 5):
        entry = {"manifest": None, "f1_72": None, "mfs": None, "f1_288": None, "epochs": None, "warnings": []}
        matches = sorted(EXPERIMENTS_DIR.glob(f"v3-iter{it}-*/manifest.json"))
        if not matches:
            entry["warnings"].append(f"no manifest found for v3-iter{it}")
        else:
            try:
                manifest = json.loads(matches[0].read_text())
            except (OSError, json.JSONDecodeError) as exc:
                entry["warnings"].append(f"unparseable manifest {matches[0].name}: {exc}")
            else:
                entry["manifest"] = str(matches[0])
                entry["f1_72"] = _hybrid_f1(manifest, "evaluation_metrics_72record_eval")
                entry["f1_288"] = _hybrid_f1(manifest, "evaluation_metrics_288record_eval")
                ev72 = _find_eval(manifest, "evaluation_metrics_72record_eval")
                mfs = ev72.get("missing_field_share_of_failures") if isinstance(ev72, dict) else None
                if isinstance(mfs, (int, float)):
                    entry["mfs"] = float(mfs) * 100.0  # fraction -> percent
                cfg = manifest.get("training_configuration") or {}
                txt = str(cfg.get("changes_from_iteration_15", "")) + " " + str(manifest.get("status_note", ""))
                # First NUM_EPOCHS= token wins: iter3's config mentions both
                # "NUM_EPOCHS=1 ... Restored to NUM_EPOCHS=2" -- the first is
                # the setting this run actually trained with.
                import re
                m = re.search(r"NUM_EPOCHS=(\d)", txt)
                if m:
                    entry["epochs"] = int(m.group(1))
        # Fallbacks (only where the manifest genuinely lacks the field)
        if entry["f1_72"] is None:
            entry["warnings"].append(f"v3-iter{it}: 72-rec F1 fallback to documented {V3_FALLBACK_F1[it]}")
            entry["f1_72"] = V3_FALLBACK_F1[it]
        if entry["mfs"] is None:
            entry["warnings"].append(f"v3-iter{it}: missing_field share fallback to documented {V3_FALLBACK_MFS[it]}%")
            entry["mfs"] = V3_FALLBACK_MFS[it]
        if entry["f1_288"] is None:
            if it in V3_FALLBACK_F1_288:
                entry["warnings"].append(f"v3-iter{it}: 288-rec F1 fallback to documented {V3_FALLBACK_F1_288[it]}")
                entry["f1_288"] = V3_FALLBACK_F1_288[it]
        if entry["epochs"] is None:
            entry["epochs"] = V3_FALLBACK_EPOCHS[it]
            entry["warnings"].append(f"v3-iter{it}: epochs inferred as {entry['epochs']} (no NUM_EPOCHS token)")
        out[it] = entry
    return out


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------
def _save(fig, filename: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] wrote {path} ({path.stat().st_size:,} bytes)")
    return path


def _annotate_bar(ax, bars, fmt="{:.4f}", dy=0.004, color="#1a1a1a", bold=True):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=9, color=color, fontweight="bold" if bold else "normal")


# ---------------------------------------------------------------------------
# Chart (a): V2 iteration hybrid F1 trajectory (documented constants)
# ---------------------------------------------------------------------------
def chart_v2_iteration_f1():
    labels = list(V2_ITERATION_F1.keys())
    values = list(V2_ITERATION_F1.values())
    x = list(range(len(labels)))
    best_idx = labels.index("iter 5/10")
    rel_idx = labels.index("iter 15")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(x, values, marker="o", linewidth=2.5, color="#2b5c8f", markersize=8,
            label="V2 hybrid field F1 (72-rec eval)")
    ax.fill_between(x, values, 0.30, alpha=0.08, color="#2b5c8f")
    ax.axhline(V2_FINAL_F1, color="#d95f02", linestyle="--", linewidth=1.6,
               label=f"V2-FINAL release checkpoint ({V2_FINAL_F1:.4f})")
    ax.annotate(f"All-time best {V2_BEST_F1:.4f}", xy=(best_idx, V2_BEST_F1),
                xytext=(best_idx - 1.1, V2_BEST_F1 + 0.035), fontsize=10, fontweight="bold",
                color="#1b9e77", arrowprops=dict(arrowstyle="->", color="#1b9e77", lw=1.4))
    ax.annotate(f"V2-FINAL {V2_FINAL_F1:.4f}", xy=(rel_idx, V2_FINAL_F1),
                xytext=(rel_idx - 1.05, V2_FINAL_F1 - 0.065), fontsize=10, fontweight="bold",
                color="#d95f02", arrowprops=dict(arrowstyle="->", color="#d95f02", lw=1.4))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.35, 0.78)
    ax.set_xlabel("V2 iteration")
    ax.set_ylabel("Hybrid field F1 (72-record eval)")
    ax.set_title("V2 Iteration Trajectory: Hybrid Field F1 (72-record eval)", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    return _save(fig, "v2_iteration_f1.png")


# ---------------------------------------------------------------------------
# Chart (b): V3 hybrid F1 across iterations 1-4 (read from manifests)
# ---------------------------------------------------------------------------
def chart_v3_iteration_f1(v3: dict):
    labels = [f"iter {i}" for i in range(1, 5)]
    values = [round(v3[i]["f1_72"], 4) for i in range(1, 5)]
    colors = ["#e0a5a0"] * 4
    colors[1] = "#1b9e77"  # iter 2 = confirmed optimum (2 epochs)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    bars = ax.bar(labels, values, color=colors, width=0.6)
    _annotate_bar(ax, bars)
    ax.axhline(V2_FINAL_F1, color="#d95f02", linestyle="--", linewidth=1.6,
               label=f"V2-FINAL release ({V2_FINAL_F1:.4f})")
    ax.text(1, values[1] + 0.02, "optimum\n(2 epochs)", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#1b9e77")
    ax.set_ylim(0.62, 0.72)
    ax.set_xlabel("V3 iteration")
    ax.set_ylabel("Hybrid field F1 (72-record eval)")
    ax.set_title("V3 Iterations 1-4: Hybrid Field F1 (from manifests)", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    return _save(fig, "v3_iteration_f1.png")


# ---------------------------------------------------------------------------
# Chart (c): epoch sweep (1/2/3 epochs -> F1)
# ---------------------------------------------------------------------------
def chart_v3_epoch_sweep(v3: dict):
    # map: 1 epoch (iter3), 2 epochs (iter2), 3 epochs (iter1)
    by_epoch = {}
    for i in range(1, 4):  # iter4 is the LR test at 2 epochs, not part of the sweep
        by_epoch[v3[i]["epochs"]] = round(v3[i]["f1_72"], 4)
    labels = ["1 epoch", "2 epochs", "3 epochs"]
    try:
        values: list = [by_epoch[1], by_epoch[2], by_epoch[3]]
    except KeyError as exc:
        raise SystemExit(f"epoch sweep incomplete; need 1/2/3-epoch entries, got {by_epoch}") from exc
    colors = ["#a6bddb", "#1b9e77", "#a6bddb"]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(labels, values, color=colors, width=0.55)
    _annotate_bar(ax, bars)
    ax.text(1, values[1] + 0.012, "sweep optimum", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#1b9e77")
    ax.axhline(V2_FINAL_F1, color="#d95f02", linestyle="--", linewidth=1.4,
               label=f"V2-FINAL release ({V2_FINAL_F1:.4f})")
    ax.set_ylim(0.64, 0.71)
    ax.set_xlabel("Training epochs (fuzzy-gate corpus, 2691 admitted)")
    ax.set_ylabel("Hybrid field F1 (72-record eval)")
    ax.set_title("V3 Epoch Sweep: 1 vs 2 vs 3 Training Epochs", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    return _save(fig, "v3_epoch_sweep.png")


# ---------------------------------------------------------------------------
# Chart (d): missing_field share of failures across V3 (from manifests)
# ---------------------------------------------------------------------------
def chart_v3_missing_field_share(v3: dict):
    labels = [f"iter {i}" for i in range(1, 5)]
    values = [round(v3[i]["mfs"], 1) for i in range(1, 5)]
    colors = ["#2b5c8f"] * 4
    lowest = min(values)
    low_idx = values.index(lowest)
    colors[low_idx] = "#1b9e77"
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    bars = ax.bar(labels, values, color=colors, width=0.6)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.8, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.text(low_idx, values[low_idx] + 4.5, "lowest\n(2 epochs)", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#1b9e77")
    ax.set_ylim(0, 70)
    ax.set_xlabel("V3 iteration")
    ax.set_ylabel("missing_field share of all failures (%)")
    ax.set_title("V3 Iterations 1-4: missing_field Share of Failures (from manifests)", fontweight="bold")
    return _save(fig, "v3_missing_field_share.png")


# ---------------------------------------------------------------------------
# Chart (e): V2-FINAL system comparison (rules / model / hybrid)
# ---------------------------------------------------------------------------
def chart_hybrid_vs_rules_vs_model():
    labels = ["rules alone", "model alone\n(residual prompt)", "hybrid\n(rules \u2192 model)"]
    values = [V2_FINAL_RULES_F1, V2_FINAL_MODEL_F1, V2_FINAL_F1]
    colors = ["#7570b3", "#e7298a", "#1b9e77"]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(labels, values, color=colors, width=0.55)
    _annotate_bar(ax, bars, dy=0.006)
    ax.set_ylim(0, 0.80)
    ax.set_ylabel("Field F1 (72-record eval)")
    ax.set_title("V2-FINAL Release Checkpoint: Rules vs Model vs Hybrid", fontweight="bold")
    ax.text(2, values[2] + 0.03, "release\ncheckpoint", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#1b9e77")
    return _save(fig, "hybrid_vs_rules_vs_model_v2final.png")


# ---------------------------------------------------------------------------
# Chart (f): 288-record eval hybrid F1 across the V3 iterations that ran it
# ---------------------------------------------------------------------------
def chart_v3_288record_evals(v3: dict):
    labels = [f"iter {i}" for i in range(1, 5) if v3[i]["f1_288"] is not None]
    values = [round(v3[i]["f1_288"], 4) for i in range(1, 5) if v3[i]["f1_288"] is not None]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(labels, values, color="#2b5c8f", width=0.55)
    _annotate_bar(ax, bars)
    ax.set_ylim(0.62, 0.71)
    ax.set_xlabel("V3 iteration (iter 1 did not run the 288-record eval)")
    ax.set_ylabel("Hybrid field F1 (288-record eval)")
    ax.set_title("V3 Iterations: Hybrid Field F1 on the 288-Record Eval", fontweight="bold")
    return _save(fig, "v3_288record_evals.png")


# ---------------------------------------------------------------------------
def main():
    print(f"\n=== SchemaForge evidence graph generator ===")
    print(f"root: {ROOT}")
    print(f"output: {OUTPUT_DIR}")

    v3 = load_v3_metrics()
    print("\n--- V3 metrics loaded from manifests ---")
    for i in range(1, 5):
        e = v3[i]
        print(f"  iter {i}: f1_72={e['f1_72']:.4f}  mfs={e['mfs']:.1f}%  f1_288={e['f1_288'] if e['f1_288'] is None else f'{e["f1_288"]:.4f}'}  epochs={e['epochs']}")
        for w in e["warnings"]:
            print(f"    ! {w}")

    print("\n--- generating charts ---")
    chart_v2_iteration_f1()
    chart_v3_iteration_f1(v3)
    chart_v3_epoch_sweep(v3)
    chart_v3_missing_field_share(v3)
    chart_hybrid_vs_rules_vs_model()
    chart_v3_288record_evals(v3)

    print(f"\n[+] all graphs written to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
