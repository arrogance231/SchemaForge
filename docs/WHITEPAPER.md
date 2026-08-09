---
title: "SchemaForge V2 — Whitepaper (DRAFT)"
status: "DRAFT — NOT FINAL"
last_updated: "2026-08-09"
---

# SchemaForge V2: Hybrid Deterministic + Distilled Semantic Structured Extraction

> **STATUS: DRAFT, NOT THE FINAL VERSION.** This document is assembled from the project's
> iteration logs (`logs/V2_TRAINING_FAILURES.md`) as work is ongoing. Several sections marked
> below are placeholders for work that has not been run yet. Do not cite numbers here as final
> results — cross-check against `logs/V2_TRAINING_FAILURES.md`, which is the authoritative,
> append-only record, before quoting anything from this document externally.

## Abstract (draft)

SchemaForge V2 is a hybrid extraction system: a tuned deterministic pre-pass handles fields that
regex/rule-based parsing already solves well (dates, IDs, amounts, emails, phones), and a
distilled ~1B-parameter language model (student: `openbmb/MiniCPM5-1B`, teacher:
`google/gemma-4-31B`) handles the semantic residual — fields that require language understanding.
The project's contribution is not a checkpoint alone but a full pipeline: a schema registry with
held-out-schema evaluation discipline, a hard-example generation framework with per-corruption-
operator attribution, a mandatory teacher-output validation gate that bounds label noise, a
field-level evaluation harness, and (in progress) confidence calibration and hybrid routing.
**The hybrid pipeline, failure-category analysis, and continual-distillation loop are not yet
built — this whitepaper describes the distillation and evaluation work completed so far, not a
finished system.**

## 1. Motivation and scope

See `docs/PROJECT_CHARTER.md` (v2.0.0) for full scope and `docs/SCHEMAFORGE_V2_RESEARCH_DIRECTION.md`
for the module-by-module implementation contract this project follows. In short: don't compete
with regex where regex already wins; measure the model on what's left after a tuned deterministic
pass has taken everything it can.

## 2. Method (as implemented so far)

### 2.1 Schema registry
12 domains (invoice, receipt, resume, contract, support_ticket, medical_note, insurance_claim,
crm_record, email, conversation, form, kg_triple), each with a Pydantic v2 model and an explicit
deterministic/semantic field-ownership split, enforced at import time. Three schemas
(`insurance_claim`, `conversation`, `kg_triple`) are held out from training entirely and used
only to evaluate generalization to unseen schemas.

### 2.2 Hard-example generation
`schemaforge/hardexamples/generate.py` applies ten corruption operators (OCR noise, delabeling,
reordering, abbreviation, synonym substitution, typos, code-switching, nesting, implicit
inference, genuine ambiguation) at parameterized severity to clean seed documents, deterministic
given a seed. Training corpus used in the current best checkpoint: 1080 records across the 9
training schemas at severities 0.0/0.3/0.6/0.9.

### 2.3 Teacher-output validation gate
Every teacher (`google/gemma-4-31B`) output must pass four checks before entering the training
set: (1) parses as JSON, (2) validates against the schema's Pydantic model, (3) every semantic
string value is either a literal substring of the source or a registered ontology derivation,
(4) no field asserted beyond what the schema licenses. Rejection rate is reported, not hidden —
41.1% on the current corpus (`schemaforge/validation/gate.py`; see `logs/V2_TRAINING_FAILURES.md`
iterations 3 and 5 for the full rejection breakdown per step and per schema).

### 2.4 Distillation
Sequence-level knowledge distillation (cross-entropy on validated teacher outputs), not the
cross-tokenizer logit KL used in the superseded V1 approach (invalid due to mismatched
tokenizers — see `docs/PROJECT_CHARTER.md` §7.1 appendix). 3 epochs on the 636 gated-admitted
examples from the 1080-record corpus.

### 2.5 Evaluation harness
`schemaforge/evaluation/harness.py`: micro-averaged field precision/recall/F1, exact match,
schema validity, hallucination rate, missing-field rate, sliced per schema and per corruption tag
(research direction §6: "slicing is the point" — an aggregate mean would hide the result this
project exists to produce).

### 2.6 Confidence calibration (in progress)
`schemaforge/calibration/`: expected calibration error, reliability diagrams, risk-coverage
curves, and grid-search temperature scaling. Raw confidence signal (mean token log-probability)
is currently badly overconfident (mean 0.978 vs. 47% actual correctness); temperature scaling
partially corrects this (holdout ECE 0.59 → 0.16 after widening the calibration search grid).
**Self-consistency sampling and a trained calibration head, both named in the research direction
as stronger alternatives, have not been tried.**

## 3. Results (draft — see `logs/V2_TRAINING_FAILURES.md` for full precision)

| checkpoint | training examples | field F1 | field precision | field recall | hallucination rate | schema validity |
|---|---|---|---|---|---|---|
| base `MiniCPM5-1B` (zero-shot) | 0 | 0.4263 | 0.5711 | 0.3401 | 0.1132 | 0.8615 |
| iteration 3 (166 gated examples) | 166 | 0.3873 | 0.3923 | 0.3824 | 0.1479 | 0.9385 |
| **iteration 5 (636 gated examples, current best)** | 636 | 0.4216 | 0.4403 | 0.4044 | **0.0717** | **0.9538** |

Measured on a 72-record eval set spanning all 12 schemas (including the 3 held-out ones), 0.0/0.5
corruption severities. **Not yet a clean win on field F1** — iteration 5 is within 0.005 of the
un-distilled base model, having traded some precision for large gains in schema validity, recall,
and (notably) a hallucination rate now below the base model's. The iteration 3→5 trend (more
gated training data closing the gap on every axis simultaneously) suggests further corpus scaling
is the productive next lever, not a change to the KD objective — this has not been tested at
larger scale yet.

**Negative results, reported per the project's stated practice of not overwriting failures:**
iteration 4 found iteration 3's smaller-corpus checkpoint (166 examples) was worse than the base
model on field F1, consistent with overfitting (near-zero training loss by epoch 2 of 3);
iteration 6 found the raw confidence signal badly overconfident and the initial calibration
search under-corrected (grid ceiling artifact, fixed in iteration 7).

## 4. What is NOT in this whitepaper yet

- **Hybrid pipeline** (§6/§7 of the research direction): routing between the deterministic
  pre-pass and the model based on a calibrated confidence threshold, with an empirically-set
  escalation threshold and a cost-per-document number. **Not built.**
- **Failure-category analysis** (§7): automatic classification of every eval error into missing
  field / incorrect normalization / wrong entity boundary / wrong inferred value / hallucinated
  field / schema violation / incorrect nesting / ambiguous input, with per-schema and
  per-corruption-operator breakdowns and worst-N examples. **Not built.**
- **Continual distillation loop** (§8): the generate → query teacher → validate → admit → retrain
  → benchmark → failure-analyze → targeted-regenerate cycle. **Not built**; the current checkpoint
  is the result of two manual scale-up iterations (3→5), not an automated loop.
- **Benchmark suite** (§6, full form): comparison against a traditional parser, ≥2 open-source
  extraction models, and a commercial API; latency p50/p95, tokens, peak memory, docs/s, cost per
  document. **Not run.**
- The deterministic pre-pass's own baseline curve exists (see `README.md`, "Reproduce the
  deterministic baseline curve") but has not yet been placed side-by-side with the distilled
  model's curve on the same corruption sweep — that crossover plot is described in the research
  direction as the headline figure of the finished paper and does not exist yet.

## 5. Reproducibility

Every iteration in `logs/V2_TRAINING_FAILURES.md` records: dataset version/size, teacher
rejection rate, checkpoint identity, and a benchmark table where applicable, per the project's
stated whitepaper-readiness practice. That log, not this document, is the source of truth while
the project is in progress.

## 6. Acknowledgments

GPU access for this project's AMD Instinct MI300X training was provided by the **AMD AI Developer
Program**.
