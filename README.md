# SchemaForge V2

> **This repository is a curated export of an actively developed research project.** It was
> split out from a larger internal monorepo so the public-facing code, evaluation results, and
> experiment provenance for this specific distillation project stand on their own, without
> exposing unrelated internal content. See `experiments/` for machine- and human-readable
> provenance of each training run, and `logs/V2_TRAINING_FAILURES.md` for the full,
> never-overwritten iteration-by-iteration research log (including negative results).

A lightweight semantic extraction engine that complements deterministic parsing. See
`docs/PROJECT_CHARTER.md` (v2.0.0) for scope and `docs/SCHEMAFORGE_V2_RESEARCH_DIRECTION.md`
for the implementation contract. `docs/WHITEPAPER.md` is a **draft, not final** writeup of
progress so far — see it for the current results table and an explicit list of what is not yet
built (full benchmark suite against external systems, further corpus scaling).

## Status

**V2 phase — methodologically closed at iteration 15.** The published Hugging Face checkpoint
remains **iteration 5/10**; the **V2-FINAL iteration-15 release checkpoint** (sha256
`c13f7f6c…`, hybrid field F1 **0.6827**) is prepared for publication at
`models/schemaforge-v2-distilled-minicpm5-1b/`. **V3 phase — open:** iterations 1–4 recorded in
`docs/WHITEPAPER_V3.md` and `logs/V3_TRAINING_FAILURES.md` — the recipe sweep confirms
**2 epochs / LR=2e-5** as the optimum; best V3 hybrid field F1 **0.6745**; `missing_field` share
of failures reduced to **52.3%**.

Build order per research direction §11:

| Step | Component | Status |
|---|---|---|
| 1 | Schema registry, 12 domains (`schemaforge/registry.py`, `schemaforge/schemas/`) | done |
| 2 | Field-level evaluation harness (`schemaforge/evaluation/`) | done |
| 3 | Deterministic pre-pass / primary baseline (`schemaforge/deterministic/`) | done |
| 4 | Hard-example generator (`schemaforge/hardexamples/`) | done |
| 5 | Teacher generation + validation gate, retrain (`schemaforge/validation/`) | done — full §4.2 gate wired, iterated from 166 → 636 gated training examples |
| 6 | Confidence + calibration (`schemaforge/calibration/`) | done — ECE, reliability diagram, risk-coverage curve, temperature scaling |
| 7 | Hybrid pipeline + routing threshold sweep (`schemaforge/hybrid/`) | done — hybrid (rules → model) beats both systems alone on every metric |
| 7 | Failure analysis (`schemaforge/failure_analysis/`) | done — 8-category classifier + honest catch-all, real bug found and fixed |
| 8 | Continual distillation loop (`src/09_loop.py`) | done — V2 loop ran through iteration 15 (V2-FINAL); V3 iterations 1–4 in `docs/WHITEPAPER_V3.md` |

**Headline result so far (V2-FINAL):** on the 72-record eval set, the hybrid system
(deterministic pre-pass + distilled model, routed by field ownership) reaches field F1
**0.6827**, vs. **0.291** for the deterministic pass alone and **~0.484** for the model alone —
beating both individual systems on every metric simultaneously. Full numbers and provenance:
`docs/WHITEPAPER.md` (V2, closed at iteration 15) and `docs/WHITEPAPER_V3.md` (V3, iterations
1–4).

**This is still a research-in-progress result, not a finished benchmark.** Several negative and
mixed results are documented rather than hidden (see iterations 4, 6, 8, 13 in the V2 training
log) — reading the full logs, not just this summary table, is the accurate picture.

## Roadmap

**V2 phase — closed.** Pipeline reproducible end-to-end; hybrid claim validated (V2-FINAL
checkpoint, sha256 `c13f7f6c…`); iteration-15 release checkpoint prepared for publication.

**V3 phase — in progress.** Next levers:
1. **Delabel/implicit stacking test** over the full operator mix — both corruption families
   composited on the same document.
2. **Held-out validation-based early stopping** — replace the fixed epoch grid with a
   held-out-val stop criterion.
3. **Larger eval set** — more held-out schemas and documents to tighten the 72-record CIs.

**Pending external actions.**
- **Hugging Face upload** of the V2-FINAL release checkpoint
  (`models/schemaforge-v2-distilled-minicpm5-1b/`, sha256 `c13f7f6c…`).
- **Optional GitHub release/tag** for this standalone repository (`v2.0.0`).

## Hardware

Training runs on an **AMD Instinct MI300X (192GB)** via SSH — this replaces the V1 target of
an NVIDIA RTX 6000 Ada/Pro (Blackwell). GPU access for this project is provided by the
**AMD AI Developer Program**, whose credits made this hardware available; thank you to the
program for the compute.

Stack: ROCm 7.2.4, PyTorch `2.10.0.dev+rocm6.4`, device/dtype resolved at runtime (no
hardcoded CUDA assumptions — see `docs/SCHEMAFORGE_V2_RESEARCH_DIRECTION.md` §9). No ROCm
vLLM build is installed; teacher generation uses the batched HuggingFace `generate` fallback
in `src/01_generate_teacher.py`.

First end-to-end pipeline validation (teacher generation → sequence-level KD training) ran
successfully on 2026-08-08 on the original 15-document V1 sample set (iterations 1–2 in
`logs/V2_TRAINING_FAILURES.md`; a real bug was found and fixed in teacher JSON extraction
during this run).

Iteration 3 replaced that sample set with a real 288-record hard-example corpus
(`data/hard_examples_train.jsonl`, `schemaforge/hardexamples/generate.py`) and wired the full
§4.2 teacher-validation gate (`schemaforge/validation/gate.py`); 166/288 records were admitted
(42.4% rejected, reasons logged to `data/teacher_dataset_rejections.json`), and the student was
retrained on the admitted set. That checkpoint lives at `models/distilled_minicpm5_1b_v2_amd`.

Iteration 4 ran the first baseline comparison (`src/05_eval_checkpoint.py`) — un-distilled base
`openbmb/MiniCPM5-1B` vs. this checkpoint, same 72-record eval set. **Result is a mixed/negative
one:** the distilled checkpoint gains schema validity (+0.077) and recall (+0.042) but loses
field F1 (-0.039), precision (-0.179), and hallucination rate worsens (+0.035) — likely
overfitting from only ~18 gated examples/schema over 3 epochs. Full numbers and analysis in
`logs/V2_TRAINING_FAILURES.md` iteration 4; this is **not yet a charter-compliant improvement**,
and is reported as a negative result rather than hidden, per research direction §8.

The legacy `src/*.py` scripts are V1 and are **superseded**; they carry the defects listed in
charter §7 and are retained only for provenance.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# .venv/bin/python -m pip install -r requirements.txt         # Linux/AMD server
```

Steps 1–4 require **no torch, no GPU, no network**.

## Test

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```

97 tests, all passing.

## Generate a hard-example dataset

```bash
.venv/Scripts/python.exe -m schemaforge.hardexamples.generate \
    --schemas invoice medical_note support_ticket \
    --n 50 --severities 0.0 0.3 0.6 1.0 \
    --operators ocr_noise typo delabel implicit abbreviate \
    --seed 11 --out data/hard_v1.jsonl
```

Output is deterministic: the same `--seed` produces byte-identical JSONL.

## Reproduce the deterministic baseline curve

The deterministic pre-pass is both stage 1 of the pipeline and the primary baseline
SchemaForge is measured against. Its degradation under corruption is the x-axis of the
project's headline crossover plot. Measured over the 9 training schemas, 4 docs each
(n=36 per row), operators `ocr_noise,typo,delabel`, seed 11:

| severity | field F1 | precision | recall | missing-field rate |
|---|---|---|---|---|
| 0.0 | 0.393 | 0.917 | 0.250 | 0.727 |
| 0.3 | 0.230 | 0.727 | 0.136 | 0.812 |
| 0.6 | 0.138 | 0.528 | 0.080 | 0.849 |
| 1.0 | 0.092 | 0.474 | 0.051 | 0.892 |

Reading these numbers correctly:

- **Precision starts at 0.917 and recall at 0.250.** That shape is the whole thesis in one
  row: rules are nearly always right about the fields they own, and structurally silent on
  everything else. Recall is computed over *all* schema leaves, including the semantic fields
  the pre-pass never attempts, so the 0.25 is not a failure — it is the headroom SchemaForge
  exists to fill.
- **Precision decays from 0.917 to 0.474 as corruption rises.** This is the result that
  matters: under OCR noise, typos and missing labels, rules do not merely go quiet, they
  start being *wrong*. That decay curve is what the model has to beat.
- Reproduce with `split='train'`, passing `run_prepass(...).resolved` straight into
  `evaluate_record` (it is already in flattened dotted-path form).

## Layout

```
schemaforge/
├── registry.py            SchemaSpec + leaf_paths; enforces the deterministic/semantic
│                          field-ownership partition at construction time
├── schemas/               12 domains; conversation, insurance_claim, kg_triple are held out
├── deterministic/         extractors.py (dates, emails, urls, phones, money, ids, percents)
│                          prepass.py (nearest_label binding; never fills a semantic field)
├── evaluation/            json_utils.py (balanced-brace parser), metrics.py (micro-averaged
│                          field P/R/F1, hallucination, missing), harness.py (slicing)
└── hardexamples/          seeds.py (clean docs + gold), operators.py (10 corruptions),
                           generate.py (CLI, deterministic JSONL)
```

## Invariants worth knowing

- **Field ownership.** `deterministic_fields | semantic_fields == leaf_paths(model)`, and the
  two are disjoint. Enforced in `SchemaSpec.__post_init__`, so a bad split raises at import.
- **The pre-pass never fills a semantic field.** Asserted in `run_prepass` before returning.
  A deterministic field the extractors miss falls through to `unresolved` rather than
  vanishing.
- **Labels are never inferred from corrupted text.** Corruptions are applied to a document
  whose gold is already known; only `nest` changes the gold, and its result still validates.
- **`ambiguate` items are excluded from accuracy.** `harness.evaluate` drops them from
  `overall` and `by_schema` by default (`exclude_tags=("ambiguate",)`) and reports them under
  an `excluded` key instead, so contestable gold never penalises a correct system. Pass
  `exclude_tags=()` to pool everything.
- **Held-out schemas cannot leak into training data.** `generate_dataset(..., split="train")`
  raises if a held-out schema is requested; `split="eval"` defaults to exactly the three.
- **Metric units.** `missing_field_rate` and `hallucination_rate` count leaf *units* in both
  numerator and denominator — a list of 3 elements contributes 3, not 1.
