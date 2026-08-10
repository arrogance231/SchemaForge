# V3 Training Failure Log

Per SCHEMAFORGE_V2_RESEARCH_DIRECTION.md §8 (carried forward as V3 practice): negative results
and failures are recorded, not overwritten. This log starts fresh for V3; V2's full 15-iteration
history remains at `logs/V2_TRAINING_FAILURES.md`, unmodified.

V3 starts from the V2-final checkpoint (`models/schemaforge-v2-distilled-minicpm5-1b/`, iteration
15, field F1 0.6827 hybrid on the 72-record eval set) and targets `missing_field`, the dominant
failure category across every V2 configuration measured (55-62% of all failures). See
`docs/WHITEPAPER_V3.md` for the starting hypotheses.

## V3 Iteration 1 — 2026-08-10 — Pure corpus scale-up (n=75→100): negative result, points to total training steps as the real lever

**What ran.** Corpus scaled from iteration 13/15's n=75/schema/severity to n=100 (3600 records,
same severities/seed/full-operator-mix), under the now-deterministic teacher pipeline
(commit `9c941ea`, before the fuzzy-gate fix was ready — this run used the strict gate).
Training config unchanged (3 epochs, same LR/batch size). Full provenance to be finalized in
`experiments/`.

**Result — negative, and directly comparable to iteration 15 (same clean pipeline, only corpus
size differs):**

| metric | iteration 15 (n=75, 1625 admitted) | V3 iter 1 (n=100, 2157 admitted) | delta |
|---|---|---|---|
| field F1 | 0.6827 | 0.6581 | **-0.0246** |
| field precision | 0.7174 | 0.7229 | +0.0055 |
| field recall | 0.6513 | 0.6039 | **-0.0474** |
| schema validity | 0.8333 | 0.8472 | +0.0139 |
| missing_field share of failures | ~55% (iter 13's measurement) | 58.8% (211/359) | worse |

**More admitted training examples made the model worse on this eval set, not better** — the
opposite of iteration 3→5's earlier finding. Reconciling this: iteration 5→13 (636→1627 admitted)
was already a "tie, not an improvement" (0.6858→0.6830); this run (1625→2157) shows an actual
decline. The trend across the whole project now looks like **diminishing returns past ~1600
admitted examples, turning negative beyond that**, not a monotonic "more data always helps"
relationship as the early 3→5 result suggested in isolation.

**A specific, testable mechanism for why:** total training steps scale with corpus size under
this project's FIXED 3-epoch schedule. Iteration 15 trained 2439 total steps (813/epoch × 3);
this run trained 3237 total steps (1079/epoch × 3) — 33% more gradient updates — and BOTH runs
converged to essentially the same final training loss (0.0052 both), meaning the larger corpus
didn't produce a meaningfully different training signal by the end, just more steps to reach the
same near-zero-loss plateau. Combined with every V2/V3 run so far showing this same near-zero-loss-
by-epoch-2-3 pattern, the working hypothesis is: **the training recipe over-converges regardless
of corpus size, and scaling corpus under a fixed epoch count just means more steps spent in that
over-converged regime, hurting generalization rather than helping.**

**Next test (V3 iteration 2):** hold corpus size at n=100 (reuse the same source corpus,
`data/corpus_v3_iter1.jsonl`) but reduce epochs (3→2) to test whether stopping earlier salvages
the larger corpus's potential benefit — this isolates "total steps" from "corpus size" for the
first time in this project. Also enabling the newly-validated fuzzy-support gate
(`fuzzy_support=True`, recovers ~45% of step-3 rejections with no false positives found in
spot-checking) for this run, since it's independently justified and ready.
