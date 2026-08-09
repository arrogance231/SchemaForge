---
id: "garc://iod/dir-quant/team-distill/spec/schemaforge_v2_research_direction@v1.0.0"
title: "SchemaForge V2 — Research Direction and Implementation Guide"
version: "1.0.0"
status: "PROPOSED"
created_date: "2026-08-07"
last_updated: "2026-08-07"
owner:
  tier: "TeamLead"
  entity_id: "garc://iod/dir-quant/team-distill"
tags:
  - "semantic-extraction"
  - "hybrid-architecture"
  - "hard-example-generation"
  - "confidence-calibration"
  - "continual-distillation"
---

# SchemaForge V2 — Research Direction and Implementation Guide

Companion to `PROJECT_CHARTER.md` v2.0.0. The charter states *what* and *why*; this document
states *how*, module by module, with the concrete contracts each script must satisfy.

---

## 0. Standing Rule

> Do not attempt to compete with regex on problems regex already solves perfectly.

Any experiment whose headline result is "the model also extracts `invoice_number` from
`INVOICE #INV-1001`" is out of scope. That field belongs to the deterministic pass. The model
is evaluated on what is left after the deterministic pass has taken everything it can.

---

## 1. Schema Registry (`schemas/`)

One module per domain, each exporting:

```python
class SchemaSpec:
    name: str                      # "medical_note"
    model: type[BaseModel]         # Pydantic v2 model, the ground-truth schema
    deterministic_fields: set[str] # fields the regex pass owns; model never asked for these
    semantic_fields: set[str]      # fields requiring language understanding
    ontology: dict[str, str] | None  # surface form -> canonical value, e.g. {"HTN": "Hypertension"}
    hard_example_hooks: list[Callable]  # domain-specific corruptions
```

Minimum registry: `invoice`, `receipt`, `resume`, `contract`, `support_ticket`,
`medical_note`, `insurance_claim`, `crm_record`, `email`, `conversation`, `form`,
`kg_triple`.

**Split discipline.** Three of the twelve are designated **held-out schemas**: never trained
on, only evaluated. They carry the "learned structured extraction, not one schema" claim. If
that claim is dropped, the split can be dropped — not otherwise.

**Prompt contract.** The schema is passed to the model at inference time (JSON Schema in the
prompt or a schema-name token plus registry lookup). A model that only works when the schema
is baked into its weights cannot generalize to held-out schemas.

---

## 2. Deterministic Pre-Pass (`src/06_pipeline.py`, stage 1)

Extractors, each returning `(value, span, confidence=1.0)` or `None`:

| Extractor | Implementation |
|---|---|
| dates | `dateparser`, plus explicit format regexes; must emit ISO-8601 |
| emails / URLs | RFC-shaped regex |
| phone numbers | `phonenumbers` (libphonenumber) |
| currency / amounts | regex + locale-aware decimal normalization |
| IDs / reference numbers | schema-supplied patterns |

Rules:
- The pre-pass is **tuned**, not a strawman. It is also the primary baseline in §6, so
  weakening it to flatter the model corrupts the central result.
- A field resolved deterministically is removed from the model's request. The prompt asks for
  the residual field set only.
- Every extractor records its span, so hallucination checking (§5) can distinguish
  model-asserted from rule-asserted values.

---

## 3. Hard-Example Generation (`src/10_generate_hard_examples.py`)

Corruption operators, each parameterized by severity and independently toggleable so failures
can be attributed per operator:

| Operator | Example transformation |
|---|---|
| `ocr_noise` | `rn`→`m`, `0`↔`O`, `1`↔`l`, dropped/duplicated chars, broken line joins |
| `delabel` | remove `"Total:"` — value must be recovered from context alone |
| `reorder` | shuffle field order, split one field across distant sentences |
| `abbreviate` | `Hypertension`→`HTN`, `Incorporated`→`Inc.`, domain glossary driven |
| `synonym` | paraphrase field labels and values |
| `typo` | keyboard-adjacency and phonetic errors |
| `code_switch` | inject a second language for a field or the whole document |
| `nest` | promote flat fields into nested entities / repeated line items |
| `implicit` | replace a literal with an inference — `Age: 42` → "celebrated his forty-second birthday" |
| `ambiguate` | introduce a genuine second reading (labelled as such; see below) |

**Label provenance.** The generator applies corruptions to a document whose gold JSON is
already known, so the label survives the corruption by construction. Labels are never
inferred back from corrupted text.

**`ambiguate` is special.** Deliberately ambiguous items are tagged and scored separately.
They belong in the *confidence* evaluation (the model should report low confidence), not in
the accuracy numerator, where the reference itself is contestable.

**Severity curriculum.** Generation runs at increasing severity across iterations, driven by
§8's failure statistics — not at a fixed random mix.

---

## 4. Distillation (`src/01`, `src/02`)

### 4.1 Objective
Charter §7.1 stands: V1's cross-tokenizer logit KL is invalid and must not be inherited.
Default to **sequence-level KD** — train on validated teacher outputs with cross-entropy —
and treat any logit-level scheme as a separate, ablated experiment that must first
demonstrate correct token alignment on a unit test.

### 4.2 Teacher output validation (mandatory gate)
Before an example enters the dataset:
1. parses as JSON;
2. validates against the schema's Pydantic model;
3. every extracted value is **supported** — string values appear in the source, or are a
   registered ontology mapping of something that appears, or are marked `inferred` with the
   supporting span;
4. no field asserted that the source does not license.

Rejected teacher outputs are logged with reasons, not silently dropped — the rejection rate
is itself reported (it bounds label noise).

### 4.3 Training
- Multi-schema batches; schema passed in-context.
- Loss masked to completion tokens; a fully-masked example **raises** (V1 swallowed it).
- Held-out schemas excluded from every training split, including any validation split used
  for early stopping.

---

## 5. Confidence-Aware Inference (`src/05_confidence.py`)

Output contract:

```json
{
  "prediction": { "...": "..." },
  "confidence": 0.96,
  "field_confidence": { "diagnosis": 0.41, "patient_name": 0.99 },
  "uncertain_fields": ["diagnosis"],
  "provenance": { "patient_name": "deterministic", "diagnosis": "model" }
}
```

Signal sources, in order of preference: mean token log-probability over the field's value
span; agreement across k sampled generations (self-consistency); a trained calibration head.
Whichever is used, the raw signal is **calibrated** (temperature scaling or isotonic
regression fit on a validation split) — an uncalibrated softmax score is not a confidence.

Required reporting: reliability diagram, expected calibration error (ECE), and a
**risk–coverage curve** — the operational question is "at what coverage does field-F1 exceed
threshold X", since that is what sets the escalation rate.

**Hallucination check** runs alongside: any model-asserted string value with no source
support and no ontology derivation is flagged, regardless of its confidence.

---

## 6. Benchmarking (`src/07_benchmark.py`)

Single command, frozen held-out set, results written as a versioned table plus plots.

**Systems compared:** tuned regex/rule pipeline · a traditional parser · SchemaForge alone ·
**hybrid (rules → SchemaForge)** · teacher LLM · ≥2 open-source extraction models · a
commercial API where licensing permits.

**Metrics:** exact match · schema validity · field precision / recall / F1 · latency p50/p95 ·
input+output tokens · peak memory · docs/s · cost per document · hallucination rate ·
missing-field rate.

**Slicing is the point.** Every metric is reported per schema *and* per corruption operator at
each severity. The headline figure of the paper is the crossover plot: deterministic accuracy
falling as corruption severity rises, SchemaForge holding, and the severity at which the two
curves cross. An aggregate mean hides exactly the result this project exists to produce.

Cost per document is computed for the hybrid at its chosen escalation threshold, including
the frontier-LLM calls that escalation triggers. That number is the production argument.

---

## 7. Failure Analysis (`src/08_failure_analysis.py`)

Automatic classifier over every eval error:

| Category | Detection |
|---|---|
| missing field | key absent or null, reference non-null |
| incorrect normalization | value maps to the same ontology entry as the reference but differs in surface form, or maps to the wrong entry |
| wrong entity boundary | predicted span overlaps the reference span but does not equal it |
| wrong inferred value | no source span supports either value; both are inferences, they disagree |
| hallucinated field | key or value with no source support and no ontology derivation |
| schema violation | Pydantic validation failure |
| incorrect nesting | correct leaf values at the wrong path |
| ambiguous input | item carries the `ambiguate` tag |

Output: per-category counts, per-schema and per-operator breakdowns, confusion between
predicted and reference ontology entries, and the worst-N examples per category dumped for
inspection. This file's output is the direct input to the next round of §3.

---

## 8. Continual Distillation Loop (`src/09_loop.py`)

```
generate hard docs ──► query teacher ──► validate teacher (§4.2) ──► admit examples
        ▲                                                                  │
        │                                                                  ▼
targeted generation ◄── failure analysis (§7) ◄── benchmark (§6) ◄──── retrain (§4.3)
   for top failure
    categories
```

Each iteration records: dataset version and size, corruption mix, teacher rejection rate,
checkpoint hash, full benchmark table, failure histogram. An iteration that does not improve
the target failure category is reported as a negative result, not overwritten.

---

## 9. Infrastructure Notes (AMD)

Training runs on the owner's AMD server over SSH; the specific accelerator is TBC. Until it
is confirmed:

- No script hardcodes `cuda`. Device, dtype, and attention implementation are resolved at
  runtime (`torch.cuda.is_available()` is true on ROCm builds, but attention backend and
  library availability are not — probe explicitly and fall back to PyTorch SDPA).
- Teacher sampling must work without a CUDA-only vLLM build: vLLM ROCm if available,
  otherwise batched HF `generate`. `01_generate_teacher.py` currently imports `vllm`
  unconditionally at module level — that import must become optional.
- FlashAttention-2, DeepSpeed, and bitsandbytes each need ROCm builds or a documented
  fallback (SDPA, FSDP, no quantized training respectively).
- Confirm the ROCm version on the server first and pin the matching PyTorch wheel; ROCm
  wheel/driver mismatches are the dominant failure mode.
- BF16 support depends on the specific AMD part. Verify before assuming; fall back to FP16
  with loss scaling or FP32 for the small student if needed.

The V1 model sizes should be re-checked against the confirmed VRAM before scheduling: a 31B
teacher in BF16 needs ~62GB of weights alone, which V1's stated 48GB target could not hold —
V1 relied on `device_map="auto"` offload. If the AMD server cannot host the teacher, teacher
generation is decoupled from training and run as a separate offline pass (which the §8 loop
already assumes).

---

## 10. Research Contributions

The project ships, beyond a checkpoint:

1. an automatic distillation pipeline with teacher-output validation;
2. a schema-aware, multi-schema training methodology with held-out-schema evaluation;
3. a hard-example generation framework with per-operator attribution;
4. a hybrid extraction architecture with an empirically-set routing threshold;
5. confidence-aware inference with a published calibration curve;
6. comprehensive benchmarking against deterministic systems, not only against LLMs;
7. a reproducible evaluation suite;
8. a production deployment example.

---

## 11. Build Order

1. Schema registry + Pydantic models (§1) — everything else depends on it.
2. Field-level evaluation harness (§6, metrics only) — before any retraining, so V1's
   checkpoint gets a real number and there is a baseline to beat.
3. Deterministic pre-pass (§2) — the primary baseline.
4. Hard-example generator (§3).
5. Teacher generation + validation gate (§4.2), retrain (§4.3).
6. Confidence + calibration (§5).
7. Hybrid pipeline + routing threshold sweep (§2, §5).
8. Failure analysis (§7), then close the loop (§8).

Steps 1–4 are CPU-only and can be built before the AMD server is available.
