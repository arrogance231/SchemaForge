---
id: "garc://iod/dir-quant/team-distill/project/gemma_minicpm_json_distillation@v2.0.0"
title: "SchemaForge V2 — A Lightweight Semantic Structured Extraction Engine"
version: "2.0.0"
status: "PROPOSED"
created_date: "2026-08-03"
last_updated: "2026-08-07"
owner:
  tier: "TeamLead"
  entity_id: "garc://iod/dir-quant/team-distill"
tags:
  - "llm"
  - "knowledge-distillation"
  - "semantic-extraction"
  - "hybrid-architecture"
  - "gemma"
  - "minicpm"
  - "confidence-estimation"
  - "amd-rocm"
---

# Project Charter: SchemaForge V2

> Supersedes v1.0.0 ("Gemma-to-MiniCPM-1B Knowledge Distillation"). The V1 charter is
> preserved in §8 for provenance. The scope change is recorded in
> `consortium/shared_knowledge_network/decision_log/RDR-0001_schemaforge_v2_semantic_extraction_pivot.md`.

## 1. Research Question

V1 established that a distilled 1B student can emit schema-shaped JSON. That is no longer
the objective. V2 is governed by a single question:

> **When can a distilled 1B language model replace deterministic extractors, and when
> should the two be combined into a hybrid system?**

Every design decision in this project must be traceable to that question. Work that only
improves "JSON validity on invoices" does not qualify.

## 2. Positioning

SchemaForge V2 is **not** marketed, benchmarked, or written up as "a better JSON extractor."
It is:

> A lightweight semantic extraction engine that complements deterministic parsing by
> handling the ambiguity, variability, and linguistic complexity that rule-based systems
> cannot robustly address, while operating at a fraction of the computational cost of
> frontier LLMs.

The deliverable is a map of three operating regions, with empirical boundaries:

| Region | Owner | Best at |
|---|---|---|
| Explicit structured patterns | Deterministic rules | Dates, emails, phones, currency, URLs, IDs — near-100% precision, ~0 cost |
| Semantic extraction under a constrained schema | **SchemaForge** | Implicit entities, normalization, paraphrase, OCR/typo noise, inferred categoricals |
| Open-ended reasoning | Frontier LLMs | Multi-hop inference, unconstrained schemas, novel domains |

We do not attempt to beat regex where regex is already correct. We measure where regex
breaks and own that territory.

## 3. Scope Change vs. V1

| Dimension | V1 | V2 |
|---|---|---|
| Target fields | Explicit, labelled, single-domain (invoices) | Implicit, normalized, paraphrased, multi-domain |
| Architecture | Model-only | Hybrid: deterministic pre-pass → model → schema validation |
| Output | Bare JSON | JSON + per-field confidence + `uncertain_fields` |
| Schemas | 1 implicit shape | ≥12 registered schemas, held-out schema generalization test |
| Dataset | 15 hand-written invoices | Programmatic hard-example generation, continually expanded |
| Metrics | JSON syntax validity, tok/s | EM, schema validity, field P/R/F1, hallucination rate, missing-field rate, latency, cost/doc |
| Baselines | None | Regex, traditional parsers, teacher LLM, open-source extractors, commercial APIs |
| Contribution | A checkpoint | Pipeline + methodology + benchmark suite + deployment reference |

## 4. Primary Objectives

The model is trained to extract what pattern matching cannot reach:

- implicit entities — `"He recently celebrated his forty-second birthday."` → `age: 42`
- ontology normalization — `HTN` → `Hypertension`
- intent from conversational language — `"I'd like my money back."` → `intent: Refund`
- inferred categoricals — `"Works intermittently after charging."` → `issue: Battery`
- abbreviations, paraphrase, reordered information, multilingual expressions
- OCR corruption, spelling errors, missing labels
- nested relationships and entity boundaries

## 5. Hybrid-First Architecture

```
Input document
      │
      ▼
Deterministic extractor pass        (regex / dateparser / libphonenumber / currency / URL / ID)
      │
      ├──► resolved fields ─────────────────────────────┐
      ▼                                                 │
Remaining unresolved fields  ──►  SchemaForge (1B)      │
                                        │               │
                                        ▼               │
                              prediction + confidence    │
                                        │               │
                                        ▼               ▼
                                  Schema validation (Pydantic/JSON Schema)
                                        │
                                        ├──► valid + confident ──► Final JSON
                                        └──► low confidence / violation ──► escalation
                                                                            (human review or frontier LLM)
```

The model is only ever asked for fields the deterministic pass could not resolve. Routing
decisions, escalation thresholds, and the cost curve they produce are themselves an
experimental result, not a fixed configuration.

## 6. Deliverables

1. **Schema registry** (`schemas/`) — ≥12 domains: invoices, receipts, resumes, contracts,
   customer support, medical notes, insurance, CRM records, emails, conversations, forms,
   KG triples. Each with a Pydantic model and a normalization ontology where applicable.
2. **Hard-example generator** (`src/10_generate_hard_examples.py`) — programmatic corruption
   and paraphrase: OCR noise, label removal, field reordering, abbreviation substitution,
   synonym swap, typo injection, code-switching, nesting, deliberate ambiguity.
3. **Teacher generation suite** (`src/01_generate_teacher.py`) — guided-decoding teacher
   sampling with automatic validation of teacher output before admission to the dataset.
4. **Distillation engine** (`src/02_train_distill.py`) — schema-aware multi-schema training
   with a mathematically valid teacher-student objective (see §7.1).
5. **Confidence head + calibration** (`src/05_confidence.py`) — per-field confidence with a
   reported calibration curve (ECE, risk–coverage), not an uncalibrated softmax score.
6. **Hybrid inference pipeline** (`src/06_pipeline.py`) — deterministic pass, routing,
   model call, validation, escalation.
7. **Evaluation suite** (`src/07_benchmark.py`) — all metrics in §9 against all baselines,
   reproducible from a single command, on a frozen held-out set.
8. **Failure analysis** (`src/08_failure_analysis.py`) — automatic classification into the
   taxonomy in §10, with per-category statistics and plots.
9. **Continual distillation loop** (`src/09_loop.py`) — the cycle in §11.
10. **Whitepaper** (`docs/WHITEPAPER_GUIDE.md`) — positioned per §2.
11. **Deployment reference** (`docs/AMD_ROCM_DEPLOYMENT.md`) — production serving example.

## 7. Known V1 Defects That V2 Must Fix

These are recorded because they invalidate parts of the V1 result, and silently inheriting
them would invalidate V2 as well.

### 7.1 Cross-tokenizer logit distillation is unsound
`src/02_train_distill.py` computes KL between the student's logits and the teacher's logits
truncated to the student's vocabulary size
(`t_logits = t_out.logits[:, :, :s_out.logits.size(-1)]`). Gemma and MiniCPM have different
tokenizers, different vocabularies, and different token-to-position alignment for the same
string. Truncating one vocabulary to the length of another does not align them; the KL term
in V1 is comparing unrelated distributions. It must be replaced with one of:
- **(a)** sequence-level KD (train on teacher-generated text with cross-entropy only) — the
  safe default;
- **(b)** a documented cross-tokenizer alignment (e.g. token-level alignment via minimum
  edit distance over decoded strings, or ULD/optimal-transport logit matching);
- **(c)** a same-tokenizer teacher.
The chosen option must be stated in the whitepaper with an ablation.

### 7.2 Evaluation measures syntax, not correctness
`03_eval.py` / `04_eval_public_dataset.py` report only "does `json.loads` succeed" and
tokens/sec, on a prompt whose answer is never checked. `{}` scores as a success. V2 metrics
are field-level and reference-compared (§9).

### 7.3 No held-out set
15 training examples, evaluated on the same distribution, with no split. V2 requires a
frozen held-out set including **held-out schemas** the model never trained on, to support
the "learns structured extraction rather than memorizing one schema" claim.

### 7.4 Brace-matching JSON extraction
Both eval scripts cut at the first `}`, which truncates any nested object. Replace with a
real balanced-brace/streaming parser.

### 7.5 Loss masking silently drops CE
`DistillationLoss` returns the KL term alone when CE is NaN. NaN CE means all labels are
`-100` (prompt longer than `max_len`) — that must raise, not be swallowed.

## 8. Infrastructure

**Training target: AMD GPU server via SSH.** Specific accelerator TBC by the project owner;
the stack must therefore be written vendor-neutral and must not hardcode CUDA assumptions.

Concretely, V1's stack requires substitution:

| V1 (NVIDIA/Nebius) | V2 (AMD/ROCm) |
|---|---|
| `pip install torch --index-url .../whl/cu121` | ROCm PyTorch wheel matching the server's ROCm version |
| `flash-attn` | ROCm FlashAttention (`flash-attn` ROCm fork) or PyTorch SDPA fallback |
| vLLM (CUDA build) | vLLM ROCm build, or HF `generate` fallback for teacher sampling |
| DeepSpeed ZeRO-2 | DeepSpeed ROCm build, or FSDP |
| `bitsandbytes` | ROCm build, or skip quantized training |

`docs/NEBIUS_DEPLOYMENT.md` is retained as V1 history and is **not** the V2 deployment path.
`docs/AMD_ROCM_DEPLOYMENT.md` is authoritative once the GPU model is confirmed. All scripts
must select device/dtype/attention implementation at runtime rather than assuming `cuda`.

## 9. Evaluation Suite

**Baselines:** regex/rule extractor, a traditional parser (e.g. invoice2data / spaCy-based),
the teacher LLM, ≥2 open-source extraction models, and a commercial extraction API where
licensing permits.

**Metrics:** exact match · schema validity · field precision · field recall · field F1 ·
latency (p50/p95) · tokens in/out · peak memory · throughput (docs/s) · cost per document ·
hallucination rate (fields asserted but unsupported by the source) · missing-field rate.

**Slices:** every metric is reported per schema, and per hard-example corruption type, so the
regex-vs-SchemaForge crossover point is visible rather than averaged away.

## 10. Failure Taxonomy

Every error is auto-classified as: missing field · incorrect normalization · wrong entity
boundary · wrong inferred value · hallucinated field · schema violation · incorrect nesting ·
ambiguous input (reference itself contestable). Per-category counts and visualizations are
produced each iteration and drive the next round of hard-example generation.

## 11. Continual Distillation Loop

1. Generate difficult documents → 2. query teacher → 3. **validate teacher output**
(schema + support check; reject silently wrong targets) → 4. admit high-quality examples →
5. retrain → 6. benchmark → 7. classify failures → 8. generate targeted hard examples for the
dominant failure categories → 9. repeat.

The dataset is a versioned artifact that evolves with the model; each iteration records its
dataset version, checkpoint, and benchmark table.

## 12. Success Criteria

V2 is complete when the project can state, with reproducible evidence:
- the empirical boundary at which SchemaForge beats a tuned deterministic extractor, per
  corruption type;
- the boundary at which escalation to a frontier LLM becomes necessary, and the cost per
  document of the hybrid at that operating point;
- field-F1 on **held-out schemas**, demonstrating schema generalization rather than
  memorization;
- a calibration curve showing that reported confidence is usable as a routing signal.

A checkpoint alone does not satisfy this charter.

## 13. Directory Structure (V2)

```text
projects/gemma_minicpm_json_distillation/
├── configs/
│   ├── ds_config.json
│   └── train_v2.yaml
├── schemas/                            # per-domain Pydantic models + ontologies
├── docs/
│   ├── PROJECT_CHARTER.md              # this document (v2.0.0)
│   ├── SCHEMAFORGE_V2_RESEARCH_DIRECTION.md
│   ├── WHITEPAPER_GUIDE.md
│   ├── AMD_ROCM_DEPLOYMENT.md          # authoritative V2 deployment path
│   └── NEBIUS_DEPLOYMENT.md            # V1 history, superseded
├── src/
│   ├── 01_generate_teacher.py          ├── 06_pipeline.py
│   ├── 02_train_distill.py             ├── 07_benchmark.py
│   ├── 03_eval.py                      ├── 08_failure_analysis.py
│   ├── 04_eval_public_dataset.py       ├── 09_loop.py
│   ├── 05_confidence.py                └── 10_generate_hard_examples.py
├── data/                               # versioned dataset iterations + frozen held-out set
├── logs/
└── models/
```

---

## Appendix: Superseded V1 Charter (v1.0.0)

**Objective.** End-to-end KD pipeline compressing the structured reasoning, instruction
following, and schema compliance of a Gemma teacher into a MiniCPM-1B student, for
high-throughput enterprise JSON and schema-constrained extraction at sub-second latency.

**Infrastructure.** NVIDIA RTX 6000 Ada/PRO (48GB) on Nebius AI Cloud; remote SSH execution;
PyTorch DDP / DeepSpeed ZeRO-2, vLLM continuous batching, FlashAttention-2, BF16.

**Deliverables.** `01_generate_teacher.py` (vLLM guided decoding), `02_train_distill.py`
(temperature-scaled KL + CE), `03_eval.py` (JSER, SCR, F1, tokens/sec),
`WHITEPAPER_GUIDE.md`.

**Status.** Delivered through Iteration 3 (`models/distilled_minicpm5_1b_iter2` on disk).
Superseded by v2.0.0; see §7 for defects that must not be carried forward.
