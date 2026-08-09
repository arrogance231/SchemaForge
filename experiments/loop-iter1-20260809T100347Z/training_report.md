---
run_id: loop-iter1-20260809T100347Z
status: COMPLETED — 2026-08-09T16:06:27Z
last_updated: 2026-08-09 (run completion)
---

# Training Report — loop-iter1-20260809T100347Z

## What this document is

Provenance record for the retraining run executed per the approved plan
(`/home/arro/.claude/plans/buzzing-tickling-sphinx.md`), which itself was written in response to
iteration 12's negative result (`experiments/loop-iter0-20260809T073238Z/`, hybrid F1 dropped to
0.6432 from the prior best 0.6858). See that plan file and `logs/V2_TRAINING_FAILURES.md` for full
background. This report follows the same measured-not-guessed convention as loop-iter0's.

## What changed for this run

Two categories of change, applied together (per the plan's stated tradeoff: comparable
attribution in one run vs. splitting across two slower, separately-isolated runs):

**Process fixes (Tier 0):**
1. `torch.manual_seed(42)` added to `src/02_train_distill.py` — the first run in this project with
   a fixed training seed.
2. `src/09_loop.py` now backs up the existing checkpoint (renamed to
   `models/distilled_minicpm5_1b_v2_amd_pre_iter0`) before retraining overwrites it — **confirmed
   working**: loop-iter0's checkpoint is preserved, unlike iteration-5's which was lost.
3. `src/09_loop.py` now redirects the retrain stage's stdout to a log file instead of discarding
   it — **confirmed working**: this is the first automated-loop run with a real per-epoch loss
   record (`logs/loop_iter1_train.log`).

**Data/config changes (Tier 1):**
4. Corpus scaled from `n=30` to `n=75` per schema/severity (2700 records vs. 1080).
5. `ontology` dicts populated for 5 previously-empty schemas (`contract`, `crm_record`, `invoice`,
   `receipt`, `resume`), sourced from the existing `_ABBREVIATIONS` glossary.

## Results

| system | field precision | field recall | field F1 | hallucination rate | schema validity |
|---|---|---|---|---|---|
| rules alone | 0.9185 | 0.1729 | 0.2911 | 0.0000 | 1.0000 |
| model alone (residual prompt) | 0.4828 | 0.4686 | 0.4756 | 0.1121 | 0.8611 |
| **hybrid** | **0.7302** | **0.6416** | **0.6830** | **0.0095** | **0.8750** |

Teacher gate: **1627/2700 admitted (60.3%)**, 39.7% rejected — a modest improvement over
loop-iter0's 41.3% and iteration-5's 41.1%, directionally consistent with the ontology fills
reducing false rejections, though not isolated as the sole cause (corpus is also larger and
covers different specific documents).

Per-epoch training loss (recovered for the first time in an automated run): epoch 1 avg 0.0382 →
epoch 2 avg 0.0089 → epoch 3 avg 0.0052. Shape matches every prior manually-launched run.

Failure-category breakdown: `missing_field` 202 (still dominant), `unclassified_mismatch` 89,
`hallucinated_field` 45, `incorrect_nesting` 15, `ambiguous_input` 7, `wrong_entity_boundary` 6,
`schema_violation` 1, `incorrect_normalization` 0, `wrong_inferred_value` 0.

## Interpreting the headline number

**Hybrid field F1 (0.6830) recovered strongly from loop-iter0's regression (0.6432, +0.0398) but
did not clearly beat the all-time-best iteration 5/10 checkpoint (0.6858, -0.0028).** Given
loop-iter0 already demonstrated run-to-run variance of a similar magnitude on an almost-identical
setup, this result is best read as a **tie within this project's observed noise band**, not a
clean win or a clean loss. This run's real value is methodological, not a new headline number:

- It confirms the Tier 0 process fixes work (checkpoint preserved, training log recoverable,
  seed fixed for future reproducibility).
- It confirms the corpus-scale + ontology-fill changes together produce a result consistent with
  (not contradicting) the project's prior evidence that more gated data helps.
- **It does NOT isolate which of the five simultaneous changes (seed, corpus scale, 5x ontology
  fill, or interaction effects between them) drove the recovery from loop-iter0.** That was a
  known, accepted tradeoff of this plan, not an oversight — a future iteration wanting to
  attribute effect sizes precisely would need to vary one factor at a time.

**`incorrect_normalization` staying at exactly 0 despite the ontology fills is a real, unresolved
non-result**, not swept under the rug: either the small 72-record eval set didn't happen to
produce a normalization-mismatch case this time (plausible given how rare the category already
was), or the ontology fix's effect is concentrated in training-corpus admission rather than
eval-time model behavior. Not resolved here.

## Checkpoint decision

**This checkpoint is NOT recommended as the next Hugging Face upload.** It's not an unambiguous
improvement over the currently-uploaded iteration-5 checkpoint — a tie within noise isn't grounds
to replace a documented, already-published artifact. The prior checkpoint (this run's
`_pre_iter0` backup) and the already-uploaded HF model remain the reference points until a future
run shows a clearer margin.

## Timing

- Start: 2026-08-09 10:03:47 UTC
- End: 2026-08-09 16:06:27 UTC
- Total: 6h 2m 40s — teacher generation over the 2700-record corpus was overwhelmingly the
  dominant cost (consistent with loop-iter0's finding that retrain+eval together take only
  minutes regardless of corpus scale). Measured on a shared GPU; not a clean single-tenant
  benchmark.

## What this run suggests for the next iteration

1. **Attribute the recovery.** A controlled follow-up (same seed/corpus-scale, ontology fills
   toggled off) would isolate whether the ontology fix alone helped, versus corpus scale alone,
   versus the fixed seed alone.
2. **`missing_field` is still the dominant failure category** (202 instances, was 214/219 in prior
   runs) — barely moved by this run's changes. None of Tier 0/Tier 1 directly targeted it; the
   plan's Tier 2 (deferred) items — gate source-support loosening, more epochs, self-consistency
   confidence — remain the more direct levers for this specific category.
3. **`model_alone` hallucination rate rose** (0.085 in loop-iter0 → 0.112 here) alongside a
   near-zero final training loss — worth a closer look in a future failure-analysis pass to rule
   out early overfitting at this corpus scale, even though the hybrid system's hallucination rate
   stayed low (rules never hallucinate, cushioning the effect).
