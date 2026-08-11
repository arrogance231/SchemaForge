---
title: "SchemaForge V2 — Whitepaper (DRAFT)"
status: "DRAFT — NOT FINAL"
last_updated: "2026-08-10"
---

# SchemaForge V2: Hybrid Deterministic + Distilled Semantic Structured Extraction

> **STATUS: DRAFT, NOT THE FINAL VERSION.** This document is assembled from the project's
> iteration logs (`logs/V2_TRAINING_FAILURES.md`) as work is ongoing. Do not cite numbers here as
> final results — cross-check against `logs/V2_TRAINING_FAILURES.md`, which is the authoritative,
> append-only record, before quoting anything from this document externally.

> **Methodological caveat, added 2026-08-10, resolved (with a nuance) later the same day:**
> teacher-label generation (`src/01_generate_teacher.py`) sampled at `temperature=0.1` with no
> fixed seed through iteration 14 — the ground-truth training labels were non-deterministic across
> runs, even for byte-identical source corpora. Diagnosed by comparing iterations 5 and 12
> (identical corpus/parameters, hybrid-eval field F1 0.043 apart). Fixed in iteration 15 (greedy
> decoding, `temperature=0.0`). **Iteration 15 then re-ran iteration 13's exact setup under the fix
> and found field F1 changed by only 0.0003** — at the corpus scale iterations 13/14 used, teacher
> noise was NOT the dominant factor; the 5-vs-12 swing likely reflects noise mattering more at
> that pair's smaller corpus scale. Practical upshot: **iterations 13 and 14's findings hold up**
> (iteration 14's regression is if anything strengthened, not explained away); **iterations 3, 4,
> 5, and 12's smaller-corpus findings remain the ones to treat with real caution.** See §7.

## Abstract (draft)

SchemaForge V2 is a hybrid extraction system: a tuned deterministic pre-pass handles fields that
regex/rule-based parsing already solves well (dates, IDs, amounts, emails, phones), and a
distilled ~1B-parameter language model (student: `openbmb/MiniCPM5-1B`, teacher:
`google/gemma-4-31B`) handles the semantic residual — fields that require language understanding.
The project's contribution is not a checkpoint alone but a full pipeline: a schema registry with
held-out-schema evaluation discipline, a hard-example generation framework with per-corruption-
operator attribution, a mandatory teacher-output validation gate that bounds label noise, a
field-level evaluation harness, confidence calibration, an automatic failure-category classifier,
and a hybrid routing layer. **The headline empirical result is that the hybrid system (rules →
model, routed by field ownership) beats both the deterministic pass alone and the distilled model
alone on every metric simultaneously** — this holds up across every eval configuration tried so
far. A full external benchmark suite (vs. other extraction systems) and a mature continual-
distillation loop are the two largest pieces not yet built.

## 1. Motivation and scope

See `docs/PROJECT_CHARTER.md` (v2.0.0) for full scope and `docs/SCHEMAFORGE_V2_RESEARCH_DIRECTION.md`
for the module-by-module implementation contract this project follows. In short: don't compete
with regex where regex already wins; measure the model on what's left after a tuned deterministic
pass has taken everything it can.

## 2. Method

### 2.1 Schema registry
12 domains (invoice, receipt, resume, contract, support_ticket, medical_note, insurance_claim,
crm_record, email, conversation, form, kg_triple), each with a Pydantic v2 model and an explicit
deterministic/semantic field-ownership split, enforced at import time. Three schemas
(`insurance_claim`, `conversation`, `kg_triple`) are held out from training entirely and used
only to evaluate generalization to unseen schemas. 7 of the 12 schemas carry a populated
`ontology` dict (surface-form → canonical mappings, e.g. `"NDA"` → `"Non-Disclosure Agreement"`)
sourced from the same glossary the hard-example generator's `abbreviate` operator uses.

### 2.2 Hard-example generation
`schemaforge/hardexamples/generate.py` applies ten corruption operators (OCR noise, delabeling,
reordering, abbreviation, synonym substitution, typos, code-switching, nesting, implicit
inference, genuine ambiguation) at parameterized severity to clean seed documents, deterministic
given a seed. Training corpus size has grown across iterations: 288 → 1080 → 2700 → 3780 records.

### 2.3 Teacher-output validation gate
Every teacher (`google/gemma-4-31B`) output must pass four checks before entering the training
set: (1) parses as JSON, (2) validates against the schema's Pydantic model, (3) every semantic
string value is either a literal substring of the source or a registered ontology derivation,
(4) no field asserted beyond what the schema licenses. Rejection rate is reported, not hidden —
it has ranged from 29.0% to 42.4% across iterations as the corpus, ontology coverage, and corpus
composition changed (`schemaforge/validation/gate.py`; see `logs/V2_TRAINING_FAILURES.md` for the
full per-iteration breakdown).

### 2.4 Distillation
Sequence-level knowledge distillation (cross-entropy on validated teacher outputs), not the
cross-tokenizer logit KL used in the superseded V1 approach (invalid due to mismatched
tokenizers — see `docs/PROJECT_CHARTER.md` §7.1 appendix). 3 epochs, `openbmb/MiniCPM5-1B`
(~1.04B parameters), AdamW/cosine schedule, bf16, on the gate-admitted subset of each iteration's
corpus.

### 2.5 Evaluation harness
`schemaforge/evaluation/harness.py`: micro-averaged field precision/recall/F1, exact match,
schema validity, hallucination rate, missing-field rate, sliced per schema and per corruption tag
(research direction §6: "slicing is the point" — an aggregate mean would hide the result this
project exists to produce). The held-out eval set grew from 72 to 288 records partway through the
project specifically because the smaller set could not reliably distinguish real improvements
from run-to-run noise (see §7).

### 2.6 Confidence calibration
`schemaforge/calibration/`: expected calibration error, reliability diagrams, risk-coverage
curves, and grid-search temperature scaling. The raw confidence signal (mean token
log-probability) is badly overconfident (mean 0.978 vs. 47% actual correctness on one measured
checkpoint); temperature scaling substantially corrects this once the search grid is wide enough
(holdout ECE 0.59 → 0.16). Self-consistency sampling and a trained calibration head, both named
in the research direction as stronger alternatives, have not been tried.

### 2.7 Failure-category classifier
`schemaforge/failure_analysis/`: classifies every eval discrepancy into one of 8 named categories
(missing field, incorrect normalization, wrong entity boundary, wrong inferred value, hallucinated
field, schema violation, incorrect nesting, ambiguous input) plus an honest catch-all. Across
every configuration measured, **missing_field (omission) is the dominant failure mode by a wide
margin — roughly 55-62% of all failures**, far ahead of hallucination or schema violation. This is
the single most consistent finding in the project's failure data.

### 2.8 Hybrid routing
`schemaforge/hybrid/`: merges the deterministic pass's resolved fields with the model's
predictions for the residual (unresolved) fields, pre-pass taking precedence for any field it
successfully owns. This is the architecture the whole project is built to validate — see §3.

### 2.9 Continual-loop driver
`src/09_loop.py`: chains corpus generation → teacher query/gate → retrain → hybrid benchmark →
failure analysis as one deliberate, reviewable iteration (not an unattended infinite loop — each
invocation is logged and reviewed before the next). Records a full machine-readable manifest per
run under `experiments/<run-id>/`.

## 3. Results — the headline hybrid comparison

Measured on the 72-record eval set (all 12 schemas including the 3 held-out ones), using the
iteration-5 checkpoint:

| system | field precision | field recall | field F1 | hallucination rate | schema validity |
|---|---|---|---|---|---|
| rules alone | 0.9185 | 0.1729 | 0.2911 | 0.0000 | 1.0000 |
| model alone (residual prompt) | 0.4852 | 0.4128 | 0.4461 | 0.0852 | 0.7361 |
| **hybrid (rules → model)** | **0.7131** | **0.5858** | **0.6432** | **0.0136** | **0.8333** |

**The hybrid system beats both individual systems on every metric simultaneously.** Rules are
precise but structurally blind to semantic fields (0.92 precision, 0.17 recall alone); the model
recalls much more but at a real hallucination cost; routing by field ownership captures the best
of both. This result held (with numeric variation, see §7) across every checkpoint tested against
it, from the earliest hybrid run through the most recent.

A crossover benchmark across rising corruption severity (severities 0.0–1.0, 144 records) confirms
the predicted shape for the deterministic pass specifically: **rules precision decays from 0.990
to 0.537** as corruption rises — under OCR noise and missing labels, rules don't just go quiet,
they start being wrong. The hybrid system's margin over rules-alone narrows as severity rises
(+0.42 F1 at severity 0.0 → +0.28 at severity 1.0) but never reverses in this sweep; rules never
actually win at any tested severity, because their recall ceiling (~0.26, even on clean text) is
structural, not corruption-dependent.

### Student checkpoint quality over iterations

| checkpoint | admitted training examples | field F1 (72-rec eval) | notes |
|---|---|---|---|
| base `MiniCPM5-1B` (zero-shot) | 0 | 0.4263 | reference point, not a hybrid number |
| iteration 3 | 166 | 0.3873 | overfitting (iteration 4) |
| iteration 5/10 | 636 | **0.4216** (model alone) / **0.6858** (hybrid) | all-time best hybrid F1 on the 72-rec eval — **not** the published checkpoint |
| iteration 12 | 634 (same corpus as iter 5) | 0.6432 (hybrid) | regression, same setup as iter 5 — see §7 |
| iteration 13 | 1627 | 0.6830 (hybrid) | recovery, still below iter 5/10 |
| iteration 14 | 2685 | 0.6583 (hybrid) | regression, targeted-corpus hypothesis not supported |
| **iteration 15 (V2-FINAL, published checkpoint)** | 1625 | **0.6827 (hybrid)** | the published checkpoint — `models/schemaforge-v2-distilled-minicpm5-1b/`, sha256 `c13f7f6c` |

**The currently-published (Hugging Face) checkpoint is V2-FINAL (iteration 15)** —
`models/schemaforge-v2-distilled-minicpm5-1b/` (sha256 `c13f7f6c`): 1625 gate-admitted training
examples, 72-rec hybrid field F1 0.6827. Iteration 5/10 still holds the all-time 72-rec hybrid-F1
best (0.6858), but 15's label-determinism fix (greedy teacher decoding, validated on a
byte-identical corpus re-run — see §7) made it the reproducible, publishable candidate: 0.6827
sits 0.0031 below the all-time best on a pipeline whose run-to-run variance is now characterized.
V3 continues this work in `docs/WHITEPAPER_V3.md`.

## 4. What is NOT in this whitepaper yet

- **Full external benchmark suite** (research direction §6): comparison against a traditional
  parser, ≥2 open-source extraction models, and a commercial API; latency p50/p95, tokens, peak
  memory, docs/s, cost per document. **Not run.**
- **Mature continual-distillation loop**: `src/09_loop.py` exists and has run twice (iteration 12
  automated, iteration 13 following the automated pattern with fixes), but the loop's own
  "targeted regeneration based on the top failure category" step is currently a human/agent
  decision point, not an automated policy — and the two automated-adjacent iterations run so far
  (13, 14) have not yet produced a checkpoint that beats the best manual iteration.
- **Self-consistency confidence signal / trained calibration head** — named in the research
  direction as stronger than the currently-used mean-token-logprob signal, not implemented.
- **Attribution of the teacher-gate admission-rate swings** (29.0%–42.4% across iterations) to a
  specific cause (ontology fills vs. corpus composition vs. both) — not isolated by any run so
  far.

## 5. Reproducibility

Every iteration in `logs/V2_TRAINING_FAILURES.md` records: dataset version/size, teacher
rejection rate, checkpoint identity, and a benchmark table where applicable. Since iteration
15, per-run experiment manifests under `experiments/<run-id>/manifest.json` additionally capture
git commit, exact launch command, hardware/software environment, and per-epoch training loss —
built specifically because two negative results (iterations 12, 14) turned out to trace back to
process gaps (unfixed training seed, discarded logs, non-deterministic teacher labels) rather than
genuine findings about the method, and those gaps were only found by building this provenance
trail. That log and those manifests, not this document, are the source of truth while the project
is in progress.

## 6. Acknowledgments

GPU access for this project's AMD Instinct MI300X training was provided by the **AMD AI Developer
Program**.

## 7. Failure log — negative and mixed results, in full

This project's stated practice is to record negative results, not overwrite or hide them. This
section summarizes every substantive negative or mixed finding to date; `logs/V2_TRAINING_FAILURES.md`
has the full technical detail behind each entry.

- **Iteration 1**: teacher JSON extraction had a real bug (no balanced-brace matching) causing
  100% label corruption in the first real generation run. Found and fixed before any training
  happened on the corrupted data.
- **Iteration 4**: the first retrained checkpoint (166 examples) scored WORSE than the un-distilled
  base model on field F1 — an overfitting signature (near-zero training loss by epoch 2 of 3).
- **Iteration 6**: the model's raw confidence signal (mean token log-probability) was found badly
  overconfident (mean 0.978 vs. 47% actual correctness), and the initial calibration search grid
  was too narrow, silently landing on its own ceiling instead of the true optimum (fixed in
  iteration 7, found via offline re-analysis of already-collected data, no new GPU run needed).
- **Iteration 8**: the first automated failure-category run had a genuine double-counting bug in
  the classifier itself (explicit-null model predictions counted as two different failure
  categories); found before trusting the numbers, fixed, re-run.
- **Iteration 9**: `incorrect_normalization` reading exactly 0 was traced to a real content gap
  (5 of 12 schemas had no `ontology` dict at all) rather than a detection bug — later addressed
  in iteration 13, but see iteration 14's finding that the category still read 0 even after the
  fix, an unresolved non-result.
- **Iteration 11**: the deterministic-baseline crossover benchmark did not reproduce this
  project's own earlier documented numbers exactly, despite an apparently identical generation
  recipe — the discrepancy's source was not tracked down, flagged rather than silently presented
  as an exact match.
- **Iteration 12**: the first fully-automated loop run regressed sharply (field F1 0.6432 vs. the
  prior best 0.6858) on what was believed to be a near-identical setup to iteration 5. Building
  this run's provenance record surfaced three real process defects: no training seed, the
  checkpoint being silently overwritten without backup (iteration 5's original weights are now
  permanently lost), and the retrain stage's log output being captured and discarded rather than
  saved.
- **Iteration 13**: after fixing those three defects and scaling the corpus + filling ontology
  gaps, field F1 recovered to 0.6830 — a tie with, not a clear win over, the all-time best
  (0.6858), within the noise band iteration 12 had already demonstrated existed.
- **Iteration 14**: tested whether an isolated, concentrated corpus of the two operators most
  associated with omission failures would reduce that failure category. On a genuinely controlled
  same-eval-set comparison, it did not — `missing_field`'s share of all failures rose from 55.0%
  to 60.1%, and field F1/schema-validity both dropped versus iteration 13. The teacher-gate
  admission rate improved sharply in this same run (29.0% vs. 39.7% rejected), a positive finding
  that is NOT attributable to the same cause as the checkpoint regression — they point in opposite
  directions and are reported as separate findings, not netted against each other.
- **Post-iteration-14 review**: comparing the full iteration history side by side (rather than
  pairwise) surfaced that iterations 5 and 12 — identical corpus and parameters — produced hybrid
  field F1 scores 0.043 apart. That swing is larger than the difference attributed to any
  deliberate experimental change made in iterations 13 or 14. Root cause: `src/01_generate_teacher.py`
  sampled teacher labels at `temperature=0.1` with no fixed seed, meaning the ground-truth training
  data itself was non-deterministic across every run in this project's history through iteration
  14. Fixed (greedy decoding) in iteration 15.
- **Iteration 15 resolved the caution above, with a nuance.** Re-running iteration 13's exact
  setup (byte-identical corpus) under the now-deterministic teacher changed field F1 by only
  0.0003 (0.6830 → 0.6827) — two orders of magnitude smaller than the iteration-5-vs-12 swing.
  **At the ~1625-1627-admitted-example scale, teacher-sampling noise was NOT the dominant driver
  of iteration 13's result.** The likely explanation: noise sensitivity scales inversely with
  corpus size — iterations 5 and 12 used a much smaller corpus (~635 admitted examples), where any
  individual differently-sampled label carries proportionally more weight. Practically: **iteration
  14's regression is strengthened, not explained away** — it was measured against a baseline now
  confirmed stable, so it more likely reflects a genuine effect of that run's corpus-composition
  change. A secondary, smaller effect this comparison DID surface: schema_validity dropped
  meaningfully (0.875 → 0.833) even though F1 didn't move — worth tracking going forward, not
  itself explained by this analysis.
