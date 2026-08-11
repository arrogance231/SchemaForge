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

# SchemaForge V3 — Distilled MiniCPM5-1B (V3 best checkpoint, iteration 2)

> **This is a public research/development checkpoint from the SchemaForge V3 phase. It is NOT
> the project's release checkpoint — V2-FINAL (models/schemaforge-v2-distilled-minicpm5-1b/)
> is the recommended deployment choice** — and its hybrid field F1 (0.6745) sits **-0.0082
> below V2-FINAL (0.6827)** on the 72-record eval. It is published because it is the best of
> V3's four iterations and demonstrates the project's primary improvement: missing_field's
> share of failures dropped to 52.3%, the lowest measured across V2 and V3. All training and
> evaluation compute was contributed by the AMD AI Developer Program on an AMD Instinct MI300X
> (192GB, ROCm). See
> **[SchemaForge V3 whitepaper](../../docs/WHITEPAPER_V3.md)**, the
> **[V3 findings note (iterations 1-4)](../../docs/FINDINGS_V3_ITERATIONS_1_4.md)**, and
> `logs/V3_TRAINING_FAILURES.md` for the full iteration history.

## What this is

A student checkpoint (`openbmb/MiniCPM5-1B`, ~1.04B params) sequence-level knowledge-distilled
from a `google/gemma-4-31B` teacher for JSON/structured extraction across 12 document schemas
(invoice, receipt, resume, contract, support ticket, medical note, insurance claim, CRM record,
email, conversation, form, knowledge-graph triple) — the same architecture and pipeline as the
SchemaForge V2 release checkpoint (`models/schemaforge-v2-distilled-minicpm5-1b/`, prepared for publication). V3's goal
is to close `missing_field` (omission), the dominant failure category across every V2
configuration measured (55-62% of all failures). **This checkpoint is the best of V3's four
iterations**: the first change to help since V2 iterations 13/15, it cut `missing_field`'s share
of failures to 52.3% — the lowest measured in V3 — by stopping training one epoch earlier under
a larger, fuzzy-gate-admitted corpus.

## Why this checkpoint, specifically

V3 ran four iterations (full history in `logs/V3_TRAINING_FAILURES.md`; summary in
`docs/FINDINGS_V3_ITERATIONS_1_4.md`):

- **V3 iter 1** (3 epochs, strict gate, 2157/3600 admitted): corpus scale-up n=75→100 *hurt*
  (hybrid F1 0.6827→0.6581), isolating "total training steps, not corpus size" as the mechanism.
- **V3 iter 2 (this checkpoint)** (2 epochs, fuzzy gate, 2691/3600 admitted): F1 recovered to
  **0.6745**, `missing_field` share down to **52.3%** — best-in-V3 on every headline metric.
- **V3 iter 3** (1 epoch, same corpus): 0.6597 — underfits; completes the epoch sweep
  1/2/3 → 0.6597/0.6745/0.6581, confirming **2 epochs** as the optimum.
- **V3 iter 4** (LR=1e-5, 2 epochs, same corpus): 0.6671 — under-trains; brackets
  **LR=2e-5** as the optimum.

The recipe grid is now bracketed: **2 epochs at LR=2e-5 is the confirmed optimum** — V2/V3's
default recipe was already near-optimal for this corpus. This checkpoint sits at that optimum,
produced by the fully deterministic pipeline (greedy teacher labels, fixed seed, verified
checkpoint backup), so its score is a stable, reproducible measurement. It does not, however,
beat the release V2-FINAL (prepared for publication) baseline on the comparable eval — which is exactly why it is
published as a research checkpoint, not the project's release checkpoint.

## Training data

- 3600-record corpus generated at n=100/schema/severity (`data/corpus_v3_iter1.jsonl`), same
  generation command, operator mix, and seed as V2's corpus, scaled up from n=75.
- Iteration 2's attempted fresh teacher generation **wedged** (deadlock at batch 208/900,
  busy-spin >1h, killed). Because teacher generation is deterministic (greedy,
  temperature=0.0), the saved iteration-1 raw outputs are **byte-identical to a fresh run** — so
  the admission gate was re-run on iter-1's saved rejections
  (`data/teacher_dataset_rejections_v3iter1_bak.json`, 1443 records) with `fuzzy_support=True`
  (threshold 0.85) by `src/09_recover_fuzzy.py` on CPU, recovering **534/1443 (37%)** →
  **2691/3600 admitted (25.2% rejected vs 40.1% under the strict gate)**. Training dataset:
  `data/teacher_dataset_v3iter2.json`.
- 2 epochs, sequence-level cross-entropy distillation (no cross-tokenizer logit KL — invalid in
  an earlier project phase, dropped). Greedy teacher decoding (temperature=0.0) throughout.

## Results

On the 72-record held-out eval set spanning all 12 schemas (including the 3 held-out ones),
`data/eval_holdout.jsonl`:

| system | field precision | field recall | field F1 | hallucination rate | schema validity | missing_field rate |
|---|---|---|---|---|---|---|
| deterministic rules alone | 0.9185 | 0.1729 | 0.2911 | 0.0000 | 1.0000 | 0.8117 |
| this model alone (residual-field prompt) | 0.4927 | 0.4686 | 0.4803 | 0.1056 | 0.8611 | 0.3026 |
| **hybrid (rules → this model)** | **0.7110** | **0.6416** | **0.6745** | **0.0201** | **0.8889** | **0.1144** |

The hybrid system again beats both individual systems on every metric simultaneously — V3 does
not re-test this architectural claim, it optimizes the model's contribution to it.
`missing_field` share of failures: **52.3% (196/375)**, the lowest measured in V3 and the first
V3 configuration to move it below V2's 55-62% band — the intended V3 target, though not yet
enough to beat V2-FINAL overall.

Corroboration on the 288-record eval (`data/eval_holdout_v2.jsonl`): hybrid field F1 **0.6742**
(precision 0.7169, recall 0.6363, hallucination rate 0.0324, schema validity 0.8715,
missing_field_rate 0.1207) — consistent with the 72-record readout.

**Comparison to the V2 release checkpoint:** 72-rec hybrid field F1 **0.6745 vs 0.6827**
(V2-FINAL, iteration 15) — **-0.0082**. This checkpoint does not beat the release
checkpoint on the comparable eval. That is the explicit reason V2-FINAL, not this checkpoint, is the recommended deployment
checkpoint (`checkpoint_used_for_huggingface_upload` was recorded as false at eval time, prior to this
publication decision).

## Known limitations

- **Below the release checkpoint (V2-FINAL, prepared for publication)** on the comparable 72-record eval (-0.0082):
  archived for reproducibility of V3's results, not for deployment.
- `missing_field` (omission) remains the dominant failure category — 52.3% share here, V3's
  confirmed best. The total-training-steps mechanism improved it but did not close it.
- Model-alone field F1 (0.4803) remains near the un-distilled base model's zero-shot level: the
  distillation's value is realized through the hybrid architecture, not as a standalone
  extractor — same caveat as the V2 card.
- V2 limitations carry forward: confidence output needs post-hoc temperature scaling (raw
  mean-token confidence measures badly overconfident); no hybrid escalation policy to a frontier
  LLM is implemented; `incorrect_normalization` has never fired in any measured eval set.

## Usage

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

For production use, run the **release V2-FINAL (prepared for publication) checkpoint** behind the deterministic pre-pass
(`schemaforge/deterministic/`) and merge with `schemaforge/hybrid/pipeline.py`'s
`merge_prediction` — do not use this checkpoint in production (it underperforms V2-FINAL). The
prompt format (schema name + explicit residual field list from the schema's `semantic_fields`
minus whatever the deterministic pass already resolved) must match training; see
`src/08_hybrid_eval.py`'s `build_prompt` in the source repository for the exact template.

## Training/compute

**Compute acknowledgment: all training and evaluation for this checkpoint was contributed by
the AMD AI Developer Program, on an AMD Instinct MI300X (192GB VRAM) with ROCm 7.0.2 /
PyTorch 2.11.0.dev+rocm7.0.** This checkpoint: 2 epochs × 1346 steps = 2692 total gradient
updates; epoch-mean losses 0.0486 / 0.0162 (near-zero-loss plateau by epoch 2, as in every
run); LR 2e-5, batch_size 2, AdamW, cosine schedule, bf16, seed 42.

## Checkpoint identity

- `model.safetensors` sha256: `c1b51015b12091fd8b71bbae2c9fc16e2091a408e32d15f2c752242bf2227dfb`
- Training run: `v3-iter2-20260811T0228Z` (see `experiments/v3-iter2-20260811T0228Z/manifest.json`
  for full machine-readable provenance)
- Git commits: `a848a7c` (2-epoch retrain; dataset recovery landed at `a8be53d`)
- Published 2026-08-11 as a public research checkpoint (huggingface.co/arrochi112/schemaforge-v3-distilled-minicpm5-1b); V2-FINAL remains the recommended release/deployment checkpoint (checkpoint_used_for_huggingface_upload was recorded as false at eval time, prior to this publication decision)

## Source repository

Full pipeline (schema registry, deterministic pre-pass, hard-example generator, teacher
validation gate incl. fuzzy-support recovery, evaluation harness, calibration module,
failure-category classifier, hybrid routing): see the project repository, not included in this
checkpoint upload.
