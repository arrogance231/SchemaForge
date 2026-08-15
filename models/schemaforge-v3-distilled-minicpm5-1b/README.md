---
license: apache-2.0
base_model: openbmb/MiniCPM5-1B
tags:
  - json-extraction
  - structured-extraction
  - knowledge-distillation
  - text-generation
language:
  - en
pipeline_tag: text-generation
---

# SchemaForge V3 - Distilled MiniCPM5-1B (V3 best checkpoint, iteration 2)

Sequence-level knowledge-distilled MiniCPM5-1B, the best of SchemaForge's four V3 research iterations targeting the missing_field failure mode.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](#license)
[![Base model](https://img.shields.io/badge/base-openbmb%2FMiniCPM5--1B-informational)](https://huggingface.co/openbmb/MiniCPM5-1B)
[![Status](https://img.shields.io/badge/status-research%20checkpoint-orange)](#results)

> **This is a public research/development checkpoint from the SchemaForge V3 phase. It is NOT the project's release checkpoint, V2-FINAL (`arrochi112/schemaforge-v2-distilled-minicpm5-1b`) is the recommended deployment choice**, and its hybrid field F1 (0.6745) sits -0.0082 below V2-FINAL (0.6827) on the 72-record eval. It is published because it is the best of V3's four iterations and demonstrates the project's primary improvement: missing_field's share of failures dropped to 52.3%, the lowest measured across V2 and V3. All training and evaluation compute was contributed by the AMD AI Developer Program on an AMD Instinct MI300X (192GB, ROCm). See the [SchemaForge V3 whitepaper](../../docs/WHITEPAPER_V3.md), the [V3 findings note (iterations 1-4)](../../docs/FINDINGS_V3_ITERATIONS_1_4.md), and `logs/V3_TRAINING_FAILURES.md` for the full iteration history.

## Overview

V2's own research closed out with one dominant unsolved problem: `missing_field` (omission) accounted for 55-62% of every measured failure breakdown, across all 15 V2 iterations, unmoved by any intervention tried. V3 exists to attack that problem directly. This checkpoint is a student (`openbmb/MiniCPM5-1B`, ~1.04B params) sequence-level knowledge-distilled from a `google/gemma-4-31B` teacher, same architecture and pipeline as the V2 release checkpoint, across the same 12 document schemas. It is the best of V3's four iterations: the first change to help since V2 iterations 13/15, it cut `missing_field`'s share of failures to 52.3%, the lowest measured in V3, by stopping training one epoch earlier under a larger, fuzzy-gate-admitted corpus.

## Results

72-record held-out eval set spanning all 12 schemas (`data/eval_holdout.jsonl`):

| Model | Dataset | Method | Field F1 |
|------|---------|--------|-------|
| Deterministic rules alone | 72-record eval | Regex/rule extraction | 0.2911 |
| This checkpoint alone | 72-record eval | Residual-field prompt, greedy decode | 0.4803 |
| **Hybrid (rules -> this model)** | 72-record eval | Field-ownership routing, V3 iteration 2 | **0.6745** |
| Hybrid, V2-FINAL (for comparison) | 72-record eval | V2 release checkpoint, iteration 15 | 0.6827 |

Full metric breakdown:

| system | field precision | field recall | field F1 | hallucination rate | schema validity | missing_field rate |
|---|---|---|---|---|---|---|
| deterministic rules alone | 0.9185 | 0.1729 | 0.2911 | 0.0000 | 1.0000 | 0.8117 |
| this model alone (residual-field prompt) | 0.4927 | 0.4686 | 0.4803 | 0.1056 | 0.8611 | 0.3026 |
| **hybrid (rules -> this model)** | **0.7110** | **0.6416** | **0.6745** | **0.0201** | **0.8889** | **0.1144** |

Corroboration on the 288-record eval (`data/eval_holdout_v2.jsonl`): hybrid field F1 0.6742 (precision 0.7169, recall 0.6363, hallucination rate 0.0324, schema validity 0.8715, missing_field_rate 0.1207), consistent with the 72-record readout. This checkpoint does not beat the V2-FINAL release checkpoint on the comparable eval (-0.0082); that is the explicit reason V2-FINAL, not this checkpoint, is the recommended deployment checkpoint.

## Architecture

```
   raw document ---> deterministic pre-pass ---> deterministic_fields (dates, emails, ids, ...)
                                |
                                | unresolved / semantic fields only
                                v
                    this checkpoint (residual-field prompt) ---> semantic_fields
                                |
                                v
                    hybrid merge (by field-ownership map) ---> final structured JSON
```

Identical routing architecture to the V2 release checkpoint. V3 does not re-test the hybrid architectural claim, it optimizes this model's contribution to it.

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("<this-repo>")
model = AutoModelForCausalLM.from_pretrained("<this-repo>", trust_remote_code=True)

prompt = (
    "Extract the following fields as JSON from the text below. Schema: invoice.\n"
    "Fields to extract: line_items[].description, vendor_name.\n"
    "Text:\nINVOICE #INV-1001. Vendor: Acme Supply Co. ...\n"
    "JSON Output:"
)
inputs = tokenizer(prompt, return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

For production use, run the **release V2-FINAL checkpoint** behind the deterministic pre-pass (`schemaforge/deterministic/`), not this checkpoint (it underperforms V2-FINAL). If you do use this checkpoint, the prompt format (schema name + explicit residual field list from the schema's `semantic_fields` minus whatever the deterministic pass already resolved) must match training; see `src/08_hybrid_eval.py`'s `build_prompt` in the source repository for the exact template.

## Dataset

- 3600-record corpus generated at n=100/schema/severity (`data/corpus_v3_iter1.jsonl`), same generation command, operator mix, and seed as V2's corpus, scaled up from n=75.
- **Preprocessing.** Iteration 2's attempted fresh teacher generation wedged (deadlock at batch 208/900, busy-spin >1h, killed). Because teacher generation is deterministic (greedy, temperature=0.0), the saved iteration-1 raw outputs are byte-identical to a fresh run, so the admission gate was re-run on iteration 1's saved rejections (`data/teacher_dataset_rejections_v3iter1_bak.json`, 1443 records) with `fuzzy_support=True` (threshold 0.85) by `src/09_recover_fuzzy.py` on CPU, recovering 534/1443 (37%), giving 2691/3600 admitted (25.2% rejected vs 40.1% under the strict gate used in iteration 1). Training dataset: `data/teacher_dataset_v3iter2.json`.
- **Splits.** Same 9 training / 3 held-out schema split as V2; evaluated on both the 72-record and 288-record held-out sets.
- Greedy teacher decoding (`temperature=0.0`) throughout, for reproducibility.

## Training

Sequence-level cross-entropy distillation only (no cross-tokenizer logit KL, invalid in an earlier project phase, dropped). 2 epochs, the confirmed optimum of V3's epoch sweep.

### Single GPU

```bash
.venv/Scripts/python.exe src/02_train_distill.py
```

### Multi-GPU (DeepSpeed ZeRO-2)

```bash
deepspeed src/02_train_distill.py --deepspeed configs/ds_config.json
```

### Distributed Training

Trained on a single AMD Instinct MI300X (192GB) node via SSH; the DeepSpeed config (`configs/ds_config.json`, ZeRO stage 2, bf16, gradient accumulation 4) generalizes to multi-node without code changes but has not been run multi-node in this project.

## Configuration

| Hyperparameter | Value | Notes |
|---|---|---|
| Epochs | 2 | Confirmed optimum: 1ep 0.6597 / 2ep 0.6745 / 3ep 0.6581 field F1 |
| Learning rate | 2e-5 | Confirmed optimum vs 1e-5 (iteration 4, F1 0.6671, schema validity regressed -0.0833) |
| Batch size | 2 | AdamW, cosine schedule |
| Distillation objective | Sequence-level CE only | No logit KL, teacher and student use different tokenizers |
| Precision | bf16 | |
| Teacher decoding | Greedy (temperature=0.0) | Fuzzy-gate recovery used `fuzzy_support=True`, threshold 0.85, on CPU only for gate re-scoring |
| Training corpus | 3600 records, 2691 admitted (74.8%) | Fuzzy gate, recovered from iteration 1's rejections |
| Seed | 42 | |

## Evaluation

```bash
.venv/Scripts/python.exe src/05_eval_checkpoint.py     # base vs. distilled checkpoint
.venv/Scripts/python.exe src/08_hybrid_eval.py          # hybrid vs. rules-only vs. model-only
```

Metrics (`schemaforge/evaluation/metrics.py`): micro-averaged field precision/recall/F1, hallucination rate, schema validity, missing_field rate, computed over flattened dotted-path leaves. Reproduce the headline numbers above against `data/eval_holdout.jsonl` (72-record) or `data/eval_holdout_v2.jsonl` (288-record, corroboration set).

## Checkpoints

This checkpoint's weights live in this repository (`model.safetensors`). Locally in the source repository it is mirrored at `models/schemaforge-v3-distilled-minicpm5-1b/`. Every training run backs up the pre-run checkpoint automatically before overwriting. To resume or retrain, start from the run manifest at `experiments/v3-iter2-20260811T0228Z/manifest.json` in the source repository.

## Experiments

V3 ran four iterations (full history in `logs/V3_TRAINING_FAILURES.md`; summary in `docs/FINDINGS_V3_ITERATIONS_1_4.md`):

| Iteration | Change tested | Hybrid field F1 | Verdict |
|---|---|---|---|
| v3-iter1 | Corpus scale 75->100 docs, strict gate, 2157/3600 admitted, 3 epochs | 0.6581 | Negative, corpus size alone does not close the omission gap |
| **v3-iter2 (this checkpoint)** | Epochs 3->2, fuzzy gate, 2691/3600 admitted | **0.6745** | Positive, best V3 result; confirms total-training-steps hypothesis |
| v3-iter3 | Epochs 2->1, same corpus | 0.6597 | Negative, underfits; completes the epoch sweep, confirms 2 is optimal |
| v3-iter4 | LR 2e-5->1e-5, 2 epochs, same corpus | 0.6671 | Negative, under-trains; brackets LR=2e-5 as the optimum |

The recipe grid is now bracketed: 2 epochs at LR=2e-5 is the confirmed optimum, V2/V3's default recipe was already near-optimal for this corpus. This checkpoint sits at that optimum, produced by the fully deterministic pipeline (greedy teacher labels, fixed seed, verified checkpoint backup), so its score is a stable, reproducible measurement.

## Hardware

**Compute acknowledgment: all training and evaluation for this checkpoint was contributed by the AMD AI Developer Program**, on an AMD Instinct MI300X (192GB VRAM) with ROCm 7.0.2 / PyTorch 2.11.0.dev+rocm7.0. This checkpoint: 2 epochs x 1346 steps = 2692 total gradient updates; epoch-mean losses 0.0486 / 0.0162 (near-zero-loss plateau by epoch 2, as in every run).

## Project Structure

```
schemaforge/
|-- registry.py            SchemaSpec + leaf_paths; field-ownership partition
|-- schemas/                12 domains; 3 held out from training
|-- deterministic/          extractors.py, prepass.py
|-- hybrid/                 pipeline.py (merge_prediction), routing threshold sweep
|-- validation/             gate.py, fuzzy_support recovery path
src/02_train_distill.py     training entrypoint (this checkpoint's recipe, NUM_EPOCHS=2)
src/09_recover_fuzzy.py     fuzzy-gate rejection recovery (this checkpoint's corpus)
src/08_hybrid_eval.py       hybrid vs. rules-only vs. model-only, build_prompt template
docs/WHITEPAPER_V3.md       full V3 research writeup
docs/FINDINGS_V3_ITERATIONS_1_4.md  iterations 1-4 summary
logs/V3_TRAINING_FAILURES.md  append-only, all V3 iterations including negative results
experiments/v3-iter2-20260811T0228Z/  this checkpoint's run manifest
```

## Results & Analysis

`missing_field` (omission) remains the dominant failure category, 52.3% share here, V3's confirmed best, down from V2's 55-62% band. The total-training-steps mechanism (fewer epochs over a larger, fuzzy-gated corpus) improved it but did not close it.

Model-alone field F1 (0.4803) remains near the un-distilled base model's zero-shot level: the distillation's value is realized through the hybrid architecture, not as a standalone extractor, same caveat as the V2 card.

Known limitations (V2 limitations carry forward):

- **Below the release checkpoint (V2-FINAL)** on the comparable 72-record eval (-0.0082): archived for reproducibility of V3's results, not for deployment.
- Confidence output needs post-hoc temperature scaling (raw mean-token confidence measures badly overconfident); no hybrid escalation policy to a frontier LLM is implemented.
- `incorrect_normalization` has never fired in any measured eval set across V2 or V3, genuinely unresolved whether the category is reachable.

## Reproducibility

- **Seeds.** Fixed training random seed 42; teacher generation uses greedy decoding (`temperature=0.0`) for byte-identical labels across re-runs; fuzzy-gate recovery re-scores existing byte-identical outputs rather than regenerating them.
- **Versions.** ROCm 7.0.2, PyTorch 2.11.0.dev+rocm7.0.
- **Configs.** `configs/ds_config.json` (DeepSpeed ZeRO-2); this run's manifest at `experiments/v3-iter2-20260811T0228Z/manifest.json` in the source repository.
- **Checkpoint identity.** `model.safetensors` sha256 `c1b51015b12091fd8b71bbae2c9fc16e2091a408e32d15f2c752242bf2227dfb`; training run `v3-iter2-20260811T0228Z`; git commits `a848a7c` (2-epoch retrain), `a8be53d` (dataset recovery). Published 2026-08-11 as a public research checkpoint; V2-FINAL remains the recommended release/deployment checkpoint.

## Citation

```bibtex
@software{schemaforge_v3_iter2_2026,
  author  = {arrogance231},
  title   = {SchemaForge V3 - Distilled MiniCPM5-1B (best checkpoint, iteration 2)},
  year    = {2026},
  url     = {https://huggingface.co/arrochi112/schemaforge-v3-distilled-minicpm5-1b},
  note    = {Research checkpoint; recommended deployment checkpoint is V2-FINAL. Source: https://github.com/arrogance231/SchemaForge}
}
```

## License

Apache 2.0. See the [SchemaForge GitHub repository](https://github.com/arrogance231/SchemaForge) for the full pipeline (schema registry, deterministic pre-pass, hard-example generator, teacher-validation gate, evaluation harness, calibration module, failure-category classifier, hybrid routing), the V2/V3 whitepapers, all experiment manifests, and evidence graphs.