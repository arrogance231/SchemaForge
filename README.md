# SchemaForge

A hybrid deterministic + knowledge-distilled-LLM pipeline for structured JSON extraction across 12 document schemas, with a full research trail (whitepapers, failure logs, evidence graphs) documenting every positive and negative result.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](#license)
[![Version](https://img.shields.io/badge/version-2.0.0-informational)](VERSION)
[![Tests](https://img.shields.io/badge/tests-97%20passing-brightgreen)](tests/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-arrochi112-yellow)](https://huggingface.co/arrochi112)

> **Status.** V2 is methodologically closed at iteration 15 (release checkpoint prepared for publication). V3 is open, iterations 1-4 complete. See [`docs/WHITEPAPER.md`](docs/WHITEPAPER.md) and [`docs/WHITEPAPER_V3.md`](docs/WHITEPAPER_V3.md) for the full, honest record, including negative results.

## Overview

Pure LLM extraction of structured fields from noisy documents (OCR errors, typos, missing labels, implicit values) is unreliable, and pure deterministic/regex extraction is structurally blind to semantic fields it was never written to look for. SchemaForge tackles this with a **hybrid pipeline**: a deterministic pre-pass owns the fields it can extract reliably (dates, emails, IDs, money, phones), and a small distilled language model (MiniCPM5-1B, ~1.04B params) owns the semantic fields the pre-pass cannot touch, routed by explicit field ownership, not a learned gate. The model itself is trained via sequence-level knowledge distillation from a much larger teacher (gemma-4-31B), using a validation-gated hard-example corpus built specifically to stress corruption operators (OCR noise, typos, delabeling, implicit values, abbreviation, etc.) across 12 document schemas (invoice, receipt, resume, contract, support ticket, medical note, insurance claim, CRM record, email, conversation, form, KG triple).

## Results

| Model | Dataset | Method | Field F1 |
|---|---|---|---|
| Deterministic pre-pass alone | 72-record eval | Regex/rule extraction | 0.291 |
| MiniCPM5-1B (distilled) alone | 72-record eval | Sequence-level KD, iteration 15 (V2-FINAL) | ~0.484 |
| **Hybrid (rules to model, by field ownership)** | 72-record eval | V2-FINAL, iteration 15 | **0.6827** |
| Hybrid, V3 best | 72-record eval | 2-epoch recipe, fuzzy-gated n=100 corpus (iteration 2) | 0.6745 |

The hybrid system beats both individual systems on every metric simultaneously, not just F1, but precision, recall, hallucination rate, and schema validity. Full provenance and per-iteration numbers: [`docs/WHITEPAPER.md`](docs/WHITEPAPER.md) (V2) and [`docs/WHITEPAPER_V3.md`](docs/WHITEPAPER_V3.md) (V3).

## Architecture

```
                     +-----------------------+
   raw document --->  | Deterministic pre-pass | ---> deterministic_fields (dates, emails,
                     |  (regex/nearest-label) |     urls, phones, money, ids, percents)
                     +-----------+-----------+
                                 | unresolved / semantic fields only
                                 v
                     +-----------------------+
                     |  Distilled MiniCPM5-1B | ---> semantic_fields (never overwrites
                     |  (KD from gemma-4-31B) |     a deterministic field)
                     +-----------+-----------+
                                 v
                     +-----------------------+
                     |   Hybrid merge (by     | ---> final structured JSON,
                     |   field-ownership map) |     validated against schema
                     +-----------------------+
```

deterministic_fields | semantic_fields == leaf_paths(schema), and the two sets are disjoint, enforced at schema construction time (schemaforge/registry.py), so a bad split raises at import rather than silently mis-routing fields at inference.

## Quick Start

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# .venv/bin/python -m pip install -r requirements.txt         # Linux/AMD server
```

Steps 1-4 below require no torch, no GPU, no network. Smallest runnable example, generate a hard-example dataset and run the deterministic pre-pass:

```bash
.venv/Scripts/python.exe -m schemaforge.hardexamples.generate \
    --schemas invoice medical_note support_ticket \
    --n 50 --severities 0.0 0.3 0.6 1.0 \
    --operators ocr_noise typo delabel implicit abbreviate \
    --seed 11 --out data/hard_v1.jsonl
```

Output is deterministic: the same --seed produces byte-identical JSONL.

## Dataset

**Format.** Each record pairs a corrupted document with gold-labeled structured JSON for one of 12 schemas (schemaforge/schemas/). Corruptions are applied via 10 operators (ocr_noise, typo, delabel, implicit, abbreviate, nest, ambiguate, etc.) at configurable severity, generated deterministically from a seed (schemaforge/hardexamples/generate.py, operators.py, seeds.py).

**Preprocessing / validation.** Teacher outputs pass a 4-step gate before admission to the training set (schemaforge/validation/gate.py): JSON parse, then Pydantic schema validation, then source-support/ontology-derivation check, then no-over-assertion check. Rejection reasons are logged (e.g. data/teacher_dataset_rejections.json).

**Splits.** 9 of the 12 schemas are used for training; insurance_claim, conversation, and kg_triple are **held out entirely** and evaluated only for generalization. generate_dataset(..., split="train") raises if a held-out schema is requested, this is enforced in code, not just convention.

## Training

Sequence-level cross-entropy distillation (no cross-tokenizer logit KL, found invalid and dropped, see docs/PROJECT_CHARTER.md section 7.1). Confirmed-optimum recipe: **2 epochs, LR=2e-5**, weight decay 0.01.

### Single GPU

```bash
.venv/Scripts/python.exe src/02_train_distill.py
```

### Multi-GPU (DeepSpeed ZeRO-2)

```bash
deepspeed src/02_train_distill.py --deepspeed configs/ds_config.json
```

### Distributed Training

This project trains on a single **AMD Instinct MI300X (192GB)** node via SSH; the DeepSpeed config (configs/ds_config.json, ZeRO stage 2, bf16, gradient accumulation 4) generalizes to multi-node without code changes but has not been run multi-node in this project.

## Configuration

| Hyperparameter | Value | Notes |
|---|---|---|
| NUM_EPOCHS | 2 | Swept 1/2/3 in V3 iterations 2-3; 2 is the confirmed optimum (0.6597 / 0.6745 / 0.6581 field F1) |
| Learning rate | 2e-5 | Swept vs 1e-5 in V3 iteration 4; 2e-5 wins (lower LR under-trains on this corpus size) |
| Weight decay | 0.01 | |
| Distillation objective | Sequence-level CE only | No logit KL, teacher and student use different tokenizers |
| Precision | bf16 | fp16 disabled in configs/ds_config.json |
| ZeRO stage | 2 | Gradient accumulation steps: 4 |
| Corpus size (V3 best) | n=100 docs, fuzzy-gated to 2691/3600 admitted | See docs/WHITEPAPER_V3.md iteration 2 |
| Teacher decoding | Greedy (temperature=0.0) | Fixed for determinism, see below |

## Evaluation

```bash
.venv/Scripts/python.exe src/05_eval_checkpoint.py     # base vs. distilled checkpoint
.venv/Scripts/python.exe src/08_hybrid_eval.py          # hybrid vs. rules-only vs. model-only
.venv/Scripts/python.exe -m pytest tests/ -q            # 97 tests, all passing
```

Metrics (schemaforge/evaluation/metrics.py): micro-averaged field precision/recall/F1, hallucination rate, missing-field rate, schema validity, all computed over flattened dotted-path leaves, with list-valued fields counted per-element (not per-path) in both numerator and denominator to keep units consistent. ambiguate-tagged items are excluded from accuracy by default (exclude_tags=("ambiguate",)) since their gold is contestable; pass exclude_tags=() to pool everything.

Reproduce the deterministic-baseline degradation curve (the pipeline's headline crossover plot) with split='train', feeding run_prepass(...).resolved directly into evaluate_record, it is already in flattened dotted-path form.

## Checkpoints

Published checkpoints live under [arrochi112 on Hugging Face](https://huggingface.co/arrochi112):

- [arrochi112/schemaforge-v2-distilled-minicpm5-1b](https://huggingface.co/arrochi112/schemaforge-v2-distilled-minicpm5-1b), V2-FINAL release checkpoint (iteration 15), sha256 c13f7f6c...
- [arrochi112/schemaforge-v3-distilled-minicpm5-1b](https://huggingface.co/arrochi112/schemaforge-v3-distilled-minicpm5-1b), V3 best research checkpoint (iteration 2), sha256 c1b51015...7dfb

Local copies live at models/schemaforge-v2-distilled-minicpm5-1b/ and models/schemaforge-v3-distilled-minicpm5-1b/. Every training run backs up the pre-run checkpoint automatically before overwriting (a since-fixed bug once lost a checkpoint without this, see iteration 12 postmortem in logs/V2_TRAINING_FAILURES.md); resume training from the latest ..._pre_<run> backup if a run needs to be re-launched.

## Experiments

| Iteration | Phase | Change tested | Hybrid field F1 | Verdict |
|---|---|---|---|---|
| 5/10 | V2 | (pipeline pre-determinism-fixes) | 0.6858 | Highest raw score, but checkpoint later lost (no backup existed yet) |
| 15 (V2-FINAL) | V2 | Full bug-fixed pipeline: fixed seed, checkpoint backup, greedy teacher decoding | **0.6827** | Release checkpoint, reproducible (+/-0.0003 on controlled re-run) |
| v3-iter1 | V3 | Corpus scale 75 to 100 docs, strict gate | 0.6581 | Negative, corpus size alone does not close the omission gap |
| v3-iter2 | V3 | Epochs 3 to 2, fuzzy-gated n=100 corpus | **0.6745** | Positive, best V3 result; confirms total-training-steps hypothesis |
| v3-iter3 | V3 | Epochs 2 to 1 | 0.6597 | Negative, completes epoch sweep, confirms 2 is optimal |
| v3-iter4 | V3 | LR 2e-5 to 1e-5 | 0.6671 | Negative, lower LR under-trains, schema validity regresses -0.0833 |

Full manifests: experiments/loop-iter*-*/ (V2) and experiments/v3-iter*-*/ (V3), each with a manifest.json. Failure-category breakdown, calibration (ECE, reliability diagrams, risk-coverage curves), and the full negative-result history live in logs/V2_TRAINING_FAILURES.md and logs/V3_TRAINING_FAILURES.md.

## Hardware

Training runs on a single **AMD Instinct MI300X (192GB)** via SSH, provided by the **AMD AI Developer Program**. Stack: ROCm 7.2.4, PyTorch 2.10.0.dev+rocm6.4; device/dtype resolved at runtime (no hardcoded CUDA assumptions, see docs/SCHEMAFORGE_V2_RESEARCH_DIRECTION.md section 9). No ROCm vLLM build is installed; teacher generation uses the batched HuggingFace generate fallback (src/01_generate_teacher.py).

## Project Structure

```
schemaforge/
|-- registry.py            SchemaSpec + leaf_paths; enforces the deterministic/semantic
|                           field-ownership partition at construction time
|-- schemas/                12 domains; insurance_claim, conversation, kg_triple held out
|-- deterministic/          extractors.py, prepass.py (nearest-label binding)
|-- evaluation/             json_utils.py, metrics.py, harness.py
|-- hardexamples/           seeds.py, operators.py (10 corruptions), generate.py
|-- validation/             gate.py, 4-step teacher-output validation
|-- calibration/            ECE, reliability diagrams, risk-coverage, temperature scaling
|-- failure_analysis/       8-category failure classifier
|-- hybrid/                 routing threshold sweep, hybrid merge
src/                        01-09 numbered pipeline scripts (teacher gen to loop)
docs/                       WHITEPAPER.md (V2), WHITEPAPER_V3.md (V3), PROJECT_CHARTER.md,
                             SCHEMAFORGE_V2_RESEARCH_DIRECTION.md, graphs/
experiments/                per-iteration manifests, provenance
logs/                       V2_TRAINING_FAILURES.md, V3_TRAINING_FAILURES.md (append-only)
models/                     local checkpoint copies + model cards
tests/                      97 tests
```

## Results & Analysis

Evidence graphs (docs/graphs/): iteration-over-iteration field F1 for V2 and V3, the epoch sweep (1/2/3 epochs), missing_field share of failures over time, and the hybrid-vs-rules-vs-model comparison at V2-FINAL.

**Deterministic-pass degradation curve** (9 training schemas x 4 docs, n=36/row, operators ocr_noise,typo,delabel, seed 11):

| severity | field F1 | precision | recall | missing-field rate |
|---|---|---|---|---|
| 0.0 | 0.393 | 0.917 | 0.250 | 0.727 |
| 0.3 | 0.230 | 0.727 | 0.136 | 0.812 |
| 0.6 | 0.138 | 0.528 | 0.080 | 0.849 |
| 1.0 | 0.092 | 0.474 | 0.051 | 0.892 |

Precision starts high (0.917) because rules are almost always right about the fields they own, but decays to 0.474 under heavy corruption, rules do not just go quiet under noise, they start being wrong. That decay curve is the target the distilled model has to beat, and recall starts at only 0.250 because it is computed over all schema leaves, including semantic fields the pre-pass never attempts, that headroom is what the hybrid model exists to fill.

**Dominant unsolved failure mode.** missing_field (omission) accounted for 55-62% of every failure breakdown across all 15 V2 iterations and remains 52-59% through V3 iterations 1-4, no corpus-scale, epoch, or LR change tested so far has closed it. This is V3's primary open target (see docs/WHITEPAPER_V3.md open questions).

## Reproducibility

- **Seeds.** All corpus generation is seeded (--seed) and produces byte-identical JSONL. Training uses a fixed random seed; a controlled re-run (V2 iteration 15) confirmed field F1 moves by only 0.0003 across a byte-identical corpus re-run.
- **Teacher decoding.** Greedy (temperature=0.0), required for reproducible teacher labels; this was not always true (see logs/V2_TRAINING_FAILURES.md early iterations).
- **Versions.** ROCm 7.2.4, PyTorch 2.10.0.dev+rocm6.4. See requirements.txt for full Python dependency pins.
- **Configs.** configs/ds_config.json (DeepSpeed ZeRO-2), per-run hyperparameters recorded in each experiments/*/manifest.json.
- **Checkpoints.** Every run auto-backs-up the pre-run checkpoint before overwriting, closing a bug that once lost the highest-scoring V2 checkpoint (iteration 5/10) without a backup.

## Citation

```bibtex
@software{schemaforge2026,
  author  = {arrogance231},
  title   = {SchemaForge: Hybrid Deterministic + Distilled-LLM Structured Extraction},
  year    = {2026},
  url     = {https://github.com/arrogance231/SchemaForge},
  note    = {V2 release checkpoint: arrochi112/schemaforge-v2-distilled-minicpm5-1b (Hugging Face)}
}
```

## License

Apache 2.0 (as declared in the published model cards). GPU access for training provided by the **AMD AI Developer Program**.