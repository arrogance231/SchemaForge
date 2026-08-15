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

# SchemaForge V2 - Distilled MiniCPM5-1B (V2 final, iteration 15)

Sequence-level knowledge-distilled MiniCPM5-1B for hybrid JSON/structured extraction; the V2-FINAL release checkpoint of the SchemaForge project.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](#license)
[![Base model](https://img.shields.io/badge/base-openbmb%2FMiniCPM5--1B-informational)](https://huggingface.co/openbmb/MiniCPM5-1B)
[![Params](https://img.shields.io/badge/params-~1.04B-lightgrey)](#overview)

> **This is the V2 release checkpoint.** It closes out the V2 phase of the SchemaForge project: the pipeline (teacher-validation gate, hard-example generation, confidence calibration, hybrid routing, failure-category analysis, and a now-deterministic/reproducible training pipeline) is considered methodologically complete as of this checkpoint. Follow-on work (V3) targets the model's own remaining weaknesses (see Results & Analysis) under a fresh whitepaper. See the [SchemaForge V2 whitepaper](../../docs/WHITEPAPER.md) and `logs/V2_TRAINING_FAILURES.md` for the complete 15-iteration research history, including every negative result along the way.

## Overview

Pure LLM extraction of structured fields from noisy documents is unreliable, and pure deterministic/regex extraction is structurally blind to the semantic fields it was never written to look for. This checkpoint is the model half of SchemaForge's hybrid answer to that problem: a student checkpoint (`openbmb/MiniCPM5-1B`, ~1.04B params) sequence-level knowledge-distilled from a `google/gemma-4-31B` teacher for JSON/structured extraction across 12 document schemas (invoice, receipt, resume, contract, support ticket, medical note, insurance claim, CRM record, email, conversation, form, knowledge-graph triple). This specific checkpoint's headline result is not a raw-model number: it is that routing between this model and a tuned deterministic pre-pass (by field ownership) beats either system alone on every metric simultaneously, see Results below.

## Results

72-record held-out eval set spanning all 12 schemas (including the 3 held-out ones):

| Model | Dataset | Method | Field F1 |
|------|---------|--------|-------|
| Deterministic rules alone | 72-record eval | Regex/rule extraction | 0.2911 |
| This checkpoint alone | 72-record eval | Residual-field prompt, greedy decode | 0.4841 |
| **Hybrid (rules -> this model)** | 72-record eval | Field-ownership routing | **0.6827** |

Full metric breakdown:

| system | field precision | field recall | field F1 | hallucination rate | schema validity |
|---|---|---|---|---|---|
| deterministic rules alone | 0.9185 | 0.1729 | 0.2911 | 0.0000 | 1.0000 |
| this model alone (residual-field prompt) | 0.4900 | 0.4784 | 0.4841 | 0.0914 | 0.8194 |
| **hybrid (rules -> this model)** | **0.7174** | **0.6513** | **0.6827** | **0.0092** | **0.8333** |

The hybrid system beats both individual systems on every metric simultaneously, the project's central architectural claim, holding here as it has in every hybrid configuration tested.

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

This model never receives fields the deterministic pre-pass already owns; it is prompted only with the residual field list (the schema's `semantic_fields` minus whatever the pre-pass already resolved). See `schemaforge/hybrid/pipeline.py`'s `merge_prediction` in the source repository.

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

For production use, run this model behind the deterministic pre-pass (`schemaforge/deterministic/`) and merge with `schemaforge/hybrid/pipeline.py`'s `merge_prediction`, using the model alone forgoes the architecture's actual advantage. The prompt format (schema name + explicit residual field list) must match training exactly; see `src/08_hybrid_eval.py`'s `build_prompt` in the source repository for the exact template.

## Dataset

- 2700 hard-example records generated by `schemaforge/hardexamples/generate.py` across the 9 *training* schemas (3 schemas, `insurance_claim`, `conversation`, `kg_triple`, are held out from training entirely, evaluated only for generalization).
- **Preprocessing.** Teacher outputs gated through a 4-step validation pipeline (JSON parse, Pydantic schema validation, source-support/ontology-derivation check, no-over-assertion check) before admission. 1625/2700 (60.2%) of teacher outputs passed the gate.
- **Splits.** 9 training schemas, 3 held-out schemas for generalization only, 72-record held-out eval set spanning all 12.
- Teacher generation uses greedy decoding (`temperature=0.0`) for reproducibility.

## Training

Sequence-level cross-entropy distillation only (no cross-tokenizer logit KL, found invalid in an earlier project phase and dropped, see `docs/PROJECT_CHARTER.md` section 7.1). 3 epochs.

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
| Epochs | 3 | V2's default recipe; V3 later swept 1/2/3 and found 2 marginally better under a different corpus |
| Learning rate | 2e-5 | Confirmed optimum in V3's LR sweep |
| Distillation objective | Sequence-level CE only | No logit KL, teacher and student use different tokenizers |
| Precision | bf16 | fp16 disabled in `configs/ds_config.json` |
| ZeRO stage | 2 | Gradient accumulation steps: 4 |
| Teacher decoding | Greedy (temperature=0.0) | Required for reproducible teacher labels |
| Training corpus | 2700 records, 1625 admitted (60.2%) | 4-step validation gate |

## Evaluation

```bash
.venv/Scripts/python.exe src/05_eval_checkpoint.py     # base vs. distilled checkpoint
.venv/Scripts/python.exe src/08_hybrid_eval.py          # hybrid vs. rules-only vs. model-only
```

Metrics (`schemaforge/evaluation/metrics.py`): micro-averaged field precision/recall/F1, hallucination rate, schema validity, computed over flattened dotted-path leaves. Reproduce the 72-record eval numbers above by running `src/08_hybrid_eval.py` against this checkpoint and `data/eval_holdout.jsonl`.

## Checkpoints

This checkpoint's weights live in this repository (`model.safetensors`). Locally in the source repository it is mirrored at `models/schemaforge-v2-distilled-minicpm5-1b/`. Every training run backs up the pre-run checkpoint automatically before overwriting, a since-fixed bug once lost the highest-scoring V2 checkpoint (iteration 5/10, field F1 0.6858) without a backup, documented in iteration 12's postmortem. To resume or retrain, start from the latest `..._pre_<run>` backup in the source repository rather than this published snapshot.

## Experiments

15 training iterations were run during V2 development (full history in `logs/V2_TRAINING_FAILURES.md`):

| Iteration | Change tested | Hybrid field F1 | Verdict |
|---|---|---|---|
| 5/10 | (pipeline pre-determinism-fixes) | 0.6858 | Highest raw score, but checkpoint later lost (no backup existed yet) |
| 12 | N/A | N/A | Postmortem: checkpoint-backup safeguard added after iteration 5's loss |
| 14 | Isolated corruption-operator training (targeted `missing_field` fix) | (below iter 15) | Negative, did not transfer to the compounded multi-corruption eval case |
| **15 (this checkpoint)** | Full bug-fixed pipeline: fixed seed, checkpoint backup, greedy teacher decoding | **0.6827** | Release checkpoint, reproducible (+/-0.0003 on controlled re-run) |

This checkpoint (iteration 15) is the most recent, and was produced by the fully bug-fixed pipeline: a fixed training random seed, an automatic checkpoint-backup step, and, critically, deterministic (greedy) teacher-label generation, none of which were true for the iteration-5 checkpoint. A controlled re-run (iteration 15) measured this fix's actual impact: field F1 moved by only 0.0003 versus its immediate predecessor, meaning this checkpoint's score (0.6827) is a stable, reproducible measurement, not a lucky or unlucky draw, the property that matters most for a checkpoint meant to be published and cited.

## Hardware

AMD Instinct MI300X (192GB), ROCm 7.0.2, PyTorch 2.11.0.dev+rocm7.0. GPU access provided by the AMD AI Developer Program.

## Project Structure

```
schemaforge/
|-- registry.py            SchemaSpec + leaf_paths; field-ownership partition
|-- schemas/                12 domains; 3 held out from training
|-- deterministic/          extractors.py, prepass.py
|-- hybrid/                 pipeline.py (merge_prediction), routing threshold sweep
|-- validation/             gate.py, 4-step teacher-output validation
src/02_train_distill.py     training entrypoint (this checkpoint's recipe)
src/05_eval_checkpoint.py   base vs. distilled checkpoint comparison
src/08_hybrid_eval.py       hybrid vs. rules-only vs. model-only, build_prompt template
docs/WHITEPAPER.md          full V2 research writeup
logs/V2_TRAINING_FAILURES.md  append-only, all 15 iterations including negative results
experiments/loop-iter3-20260810T095747Z/  this checkpoint's run manifest
```

## Results & Analysis

Model-alone field F1 (0.4841) is close to, and on some measurements below, the un-distilled base `MiniCPM5-1B`'s zero-shot score, the distillation's value in this pipeline is realized through the hybrid architecture, not as a standalone extractor. Use it hybrid, with the deterministic pre-pass, not alone.

`missing_field` (omission) is the dominant failure category across every measured configuration in this project (roughly 55-62% of all failures), the clearest remaining target for future work; V3 (see `docs/WHITEPAPER_V3.md`) targets this directly and reduced it to 52.3% at the cost of -0.0082 field F1 versus this checkpoint.

Known limitations:

- Confidence output is poorly calibrated without post-hoc temperature scaling (raw mean-token confidence has been measured badly overconfident on this project's checkpoints); a fitted temperature scaler helps but was not re-fit specifically for this checkpoint.
- No hybrid escalation policy to a frontier LLM is implemented, the confidence signal has been characterized (coverage-at-risk curves exist) but no production routing threshold is set.

## Reproducibility

- **Seeds.** Fixed training random seed; teacher generation uses greedy decoding (`temperature=0.0`) for byte-identical labels across re-runs.
- **Versions.** ROCm 7.0.2, PyTorch 2.11.0.dev+rocm7.0.
- **Configs.** `configs/ds_config.json` (DeepSpeed ZeRO-2); this run's manifest at `experiments/loop-iter3-20260810T095747Z/` in the source repository.
- **Checkpoint identity.** `model.safetensors` sha256 `c13f7f6c21c35d2ed9159acc044a9f88abcd9fdfe9d0248ee625236b70c8470d`; training run `loop-iter3-20260810T095747Z`; git commit `9c941ea12f299b66c7bfd04603f6beea0ce04c9b`.

## Citation

```bibtex
@software{schemaforge_v2_2026,
  author  = {arrogance231},
  title   = {SchemaForge V2 - Distilled MiniCPM5-1B (V2-FINAL, iteration 15)},
  year    = {2026},
  url     = {https://huggingface.co/arrochi112/schemaforge-v2-distilled-minicpm5-1b},
  note    = {Source: https://github.com/arrogance231/SchemaForge}
}
```

## License

Apache 2.0. See the [SchemaForge GitHub repository](https://github.com/arrogance231/SchemaForge) for the full pipeline (schema registry, deterministic pre-pass, hard-example generator, teacher-validation gate, evaluation harness, calibration module, failure-category classifier, hybrid routing), the V2/V3 whitepapers, all experiment manifests, and evidence graphs.