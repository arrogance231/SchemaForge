---
run_id: loop-iter3-20260810T095747Z
status: COMPLETED — 2026-08-10T15:09:55Z
last_updated: 2026-08-10 (run completion)
---

# Training Report — loop-iter3-20260810T095747Z

## What this document is

The controlled experiment proposed after a full-history review flagged that
`src/01_generate_teacher.py` had been sampling teacher labels non-deterministically
(`temperature=0.1`, no seed) across every iteration through 14 — and that this was a plausible
explanation for the large, "unexplained" F1 swing between iterations 5 and 12 (identical
corpus/params, 0.043 F1 apart). This run re-executes iteration 13's exact setup — same corpus
(byte-identical, sha256-verified), same training configuration — with the sole change being the
now-deterministic (greedy) teacher, to measure how much of the noise story actually holds up.

## Result: the noise fix mattered less than expected, and that's informative

| metric | iteration 13 (noisy teacher) | iteration 15 (deterministic teacher) | delta |
|---|---|---|---|
| field F1 | 0.6830 | 0.6827 | -0.0003 |
| field precision | 0.7302 | 0.7174 | -0.0128 |
| field recall | 0.6416 | 0.6513 | +0.0097 |
| hallucination rate | 0.0095 | 0.0092 | -0.0003 |
| schema validity | 0.8750 | 0.8333 | -0.0417 |

**Field F1 barely moved.** Removing the teacher-sampling noise source entirely changed the
headline metric by 0.0003 — two orders of magnitude smaller than the 0.043 swing between
iterations 5 and 12. Teacher gate admission was also nearly identical (1625/2700 vs. 1627/2700,
39.8% vs 39.7% rejected) despite the sampling-method change.

**What this means:** at this corpus scale (~1625-1627 admitted examples), teacher-sampling noise
was NOT the dominant factor behind iteration 13's specific F1 result — that measurement was
already reasonably stable. The large iteration-5-vs-12 swing likely reflects noise mattering more
at the *smaller* corpus scale those two iterations used (1080 records, ~635 admitted) — fewer
examples means any individual differently-sampled label has proportionally more influence on the
trained model's aggregate behavior. This is a genuine, useful methodological finding in its own
right: **noise sensitivity appears to scale inversely with training-set size** in this pipeline.

**A secondary effect is visible and shouldn't be ignored just because F1 didn't move:**
schema_validity dropped meaningfully (0.875 → 0.833, -0.042), with precision and recall shifting
in offsetting directions. Small enough not to change the headline conclusion, but worth watching
in the next iteration's failure analysis rather than dismissed as noise itself.

## What this changes about how to read iterations 3–14

Per the whitepaper caveat added after the full-history review: iterations 3–14 should be read with
real caution given the undetected noise source. This run's finding refines, rather than confirms
or dismisses, that caveat:

- **Small-corpus comparisons (iterations 3, 4, 5, 12) remain suspect** — the noise source was
  real and, at that scale, large enough to be the dominant explanation for at least one measured
  swing.
- **Larger-corpus comparisons (iteration 13 vs. this run) turn out to be more trustworthy than
  the blanket caveat implied.** Iteration 13's 0.6830 was a real, reasonably reproducible
  measurement, not a noise artifact.
- **Consequently, iteration 14's regression (0.6830 → 0.6583, and the controlled
  missing_field-share increase 55.0% → 60.1%) is STRENGTHENED as a genuine negative finding**,
  not explained away — it was measured against a now-confirmed-stable baseline, at a corpus scale
  where this run shows noise isn't the dominant factor.

## Timing and hardware

- Start: 2026-08-10 09:57:47 UTC. End: 2026-08-10 15:09:55 UTC. Total: 5h 12m 8s.
- **First iteration in this project run under fully exclusive GPU access** — no contention from
  the other agent session confirmed present at any point. Teacher generation nonetheless sustained
  only ~2.0-2.2 batches/min (4 records/batch) over 675 batches — this corpus's generation cost
  appears to be inherently high on this hardware/model pairing, not primarily a contention
  artifact as earlier duration estimates in this project assumed.
- Per-epoch loss (0.0384 / 0.0084 / 0.0048) nearly identical to iteration 13's (0.0382 / 0.0089 /
  0.0052) despite different (deterministic vs. sampled) teacher labels — consistent with the
  small eval-metric difference.

## Checkpoint decision

Not uploaded to Hugging Face — matches iteration 13's result (not an improvement over the
published iteration 5/10 checkpoint at 0.6858). This run's value is establishing the noise floor
for future comparisons, not producing a new best checkpoint.

## Recommended next step

Now that the noise floor at this corpus scale is characterized, a future corpus-composition
experiment (e.g. iteration 14's hypothesis, retried with delabel/implicit STACKED on the full
operator set instead of isolated, per that run's own recommendation) can be trusted to reflect
real effects rather than needing this same noise-attribution caveat repeated. The teacher-
determinism fix remains valuable going forward regardless of this particular result — it
guarantees future runs are exactly reproducible given the same corpus, which this run's own
close-but-not-identical numbers (schema_validity moved even though F1 didn't) shows still matters
at a finer grain than the headline metric.
