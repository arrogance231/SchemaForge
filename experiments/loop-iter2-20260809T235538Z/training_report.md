---
run_id: loop-iter2-20260809T235538Z
status: COMPLETED — 2026-08-10T08:50:31Z
last_updated: 2026-08-10 (run completion)
---

# Training Report — loop-iter2-20260809T235538Z

## What this document is

Provenance for the targeted retraining run proposed in chat after iteration 13's "tie within
noise" result. Hypothesis: give the model concentrated, isolated exposure to the two corruption
operators most associated with the dominant failure category (`missing_field`), and fix the eval
set's small size so the next comparison is decisive rather than another coin-flip. Orchestrated
manually (not via `src/09_loop.py`, which can't concatenate two separately-generated corpora).

## What changed for this run

1. **Targeted supplementary corpus.** `data/corpus_targeted_missing.jsonl` (1080 records):
   `delabel`/`implicit` operators ONLY (no other corruption stacked on top, unlike the baseline
   corpus's all-ten-operators-per-example default), n=40/schema, severities 0.3/0.6/0.9, seed=99.
   Concatenated with `data/corpus_baseline.jsonl` (2700 records, identical params to iteration
   13's corpus) into `data/hard_examples_train.jsonl` (3780 total).
2. **4x larger eval set.** `data/eval_holdout_v2.jsonl` (288 records vs. the 72-record set used
   since iteration 4), same generation recipe scaled up (n=12 instead of n=3). The old eval set
   is kept and this checkpoint was ALSO run against it, specifically to get one number directly
   comparable to the project's history.
3. **Training configuration unchanged** from iteration 13 (same seed, LR, epochs, batch size) —
   only the training data changed, per the plan's single-variable-at-a-time intent.

## Results

**Teacher gate — the clearest positive finding this run:** 2685/3780 admitted (**29.0% rejected**),
down sharply from iteration 13's 39.7% and iteration 5's 41.1% — the largest single-iteration
improvement in admission rate seen in this project. Not isolated to one cause (ontology fills
carried over unchanged from iteration 13, or the isolated-operator corpus structure, or both).

**Field metrics — mixed to negative on the eval set comparable to project history (72 records):**

| metric | iteration 13 | this run | delta |
|---|---|---|---|
| field F1 | 0.6830 | 0.6583 | **-0.0247** |
| field precision | 0.7302 | 0.7536 | +0.0234 |
| field recall | 0.6416 | 0.5844 | **-0.0572** |
| exact match | 0.2222 | 0.2917 | +0.0695 |
| schema validity | 0.8750 | 0.8056 | **-0.0694** |
| hallucination rate | 0.0095 | 0.0162 | +0.0067 (worse) |

On the new 288-record eval set (not comparable to prior iterations, only to future ones): field
F1 0.6409, `missing_field` at 59.6% of all failures.

**Controlled comparison (same 72-record eval set, both iterations) — the key negative finding for
this run's specific hypothesis:** `missing_field` was **60.1% (221/368) of all failures this run,
up from iteration 13's 55.0% (202/367)**. `schema_violation` also worsened on this identical
comparison (1 → 4 instances), matching the schema_validity metric drop. **The targeted
delabel/implicit corpus did not reduce the dominance of the failure category it was built to
address — it went the other direction, and this is not a noisy/different-eval-set artifact since
both numbers are measured on the exact same 72 records.**

## Interpreting this run honestly

**This is a negative-to-mixed result, not a win, despite the striking gate-admission-rate
improvement.** Three things are true simultaneously and none should be smoothed over:

1. The pipeline got measurably better at admitting valid teacher labels (29.0% vs 39.7%
   rejected) — a real, useful finding independent of what it did to the final checkpoint.
2. The checkpoint itself, evaluated on the apples-to-apples 72-record set, scored WORSE on field
   F1, recall, and schema validity than iteration 13's, and worse than the all-time-best
   iteration 5/10 checkpoint.
3. The specific hypothesis this run was designed to test (isolated delabel/implicit training
   examples reduce omission failures) is not supported by the data collected — if anything,
   pointing the other direction.

A plausible (not confirmed) explanation: the isolated-operator supplementary corpus, having no
other corruptions stacked on top, may be an easier/less representative training signal than the
model needs — teaching it to handle delabel/implicit in isolation doesn't necessarily transfer to
handling them combined with OCR noise, typos, and reordering simultaneously, which is what the
eval set (and most of the baseline training corpus) actually contains. This is a hypothesis for
the next iteration to test, not a conclusion this run establishes.

## Checkpoint decision

**Not recommended for Hugging Face upload.** Scores below both the currently-published
iteration-5 checkpoint and iteration 13's (un-uploaded) checkpoint on the directly-comparable eval
set. `models/distilled_minicpm5_1b_v2_amd_pre_iter14` (iteration 13's checkpoint) remains
available on the training server as backup, per the Tier-0 fix confirmed working again this run.

## Timing

- Start: 2026-08-09 23:55:38 UTC (teacher-generation stage launch).
- End: 2026-08-10 08:50:31 UTC.
- Total: 8h 54m 53s — the corpus is 1.4x iteration 13's size, but the duration also reflects a
  mixed GPU-contention profile (a second agent's session was heavily active for roughly the first
  5 hours of teacher generation, then fully exited — throughput measurably increased afterward,
  confirmed by direct batch-count sampling, though not precisely quantified end-to-end). Not a
  clean throughput comparison to iteration 13's 6h2m in either direction.

## What this run suggests for the next iteration

1. **Don't isolate corruption operators in the supplementary corpus next time** — stack
   delabel/implicit ON TOP OF the other operators (matching the eval set's actual composition)
   rather than training on them in isolation, to test whether that changes the missing_field
   result.
2. **Attribute the gate-admission-rate improvement.** A run holding the corpus composition fixed
   but toggling the ontology fills on/off would isolate whether that's the real driver, separate
   from the corpus-structure change.
3. **The 288-record eval set is now the standard going forward** for any run that wants a result
   comparable to this one; the 72-record set remains useful only for continuity with iterations
   4-13's specific numbers, not as the primary decision metric anymore.
4. **schema_validity dropped notably** (0.875 → 0.806 on the comparable eval set) — worth a
   failure-analysis pass specifically on `schema_violation` instances (only 1 in iteration 13, but
   worth checking this run's count on the same eval set) to see if the isolated-operator corpus
   introduced any schema-shape confusion.
