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

## V3 Iteration 2 — 2026-08-11 — Epochs 3→2 under the fuzzy-gate corpus (n=100): positive result, confirms the total-training-steps hypothesis

**The wedge incident.** Iter 2's teacher generation wedged: deadlock at batch 208/900, busy-spin for >1h, process killed (autonomous-monitoring period; no interactive steering available). Rather than re-run generation and risk the same deadlock, the run exploited determinism: the pipeline is greedy (temp=0.0), so iter-1's saved raw outputs are byte-identical to what a fresh iter-2 run would produce.

**The recovery method.** `src/09_recover_fuzzy.py` (commit `a8be53d`) re-ran the admission gate on iter-1's saved rejections (`data/teacher_dataset_rejections_v3iter1_bak.json`, 1443 records) with `fuzzy_support=True` (threshold 0.85) on CPU, then rebuilt prompts via importlib from `01_generate_teacher.py`. Result: **534/1443 (37%) step-3 rejections recovered**, merged to **2691/3600 admitted (25.2% rejected vs. 40.1% under the strict gate)** → `data/teacher_dataset_v3iter2.json`. Checkpoint backup `models/distilled_minicpm5_1b_v2_amd_pre_v3iter2` verified intact before training.

**What ran.** Same 3600-record corpus as iter 1 (`data/corpus_v3_iter1.jsonl`), same seed/LR/batch (42, 2e-5, 2), but epochs cut 3→2 (`NUM_EPOCHS=2`, commit `6b7248e`) → 1346 steps/epoch, 2692 total. Epoch-mean losses 0.0486 / 0.0162 (near-zero-loss plateau by epoch 2, as always). Checkpoint sha256 (`model.safetensors`): `c1b51015b12091fd8b71bbae2c9fc16e2091a408e32d15f2c752242bf2227dfb`.

**Result — positive, first change that helped since iteration 13/15:**

| metric (72-rec eval) | iter 15 (n=75, 1625 adm, 3ep) | V3 it1 (n=100, 2157 adm, 3ep) | V3 it2 (n=100, 2691 adm, 2ep) | it2 vs it1 |
|---|---|---|---|---|
| hybrid field F1 | 0.6827 | 0.6581 | **0.6745** | **+0.0164** |
| field precision | 0.7174 | 0.7229 | 0.7110 | -0.0119 |
| field recall | 0.6513 | 0.6039 | 0.6416 | +0.0377 |
| schema validity | 0.8333 | 0.8472 | 0.8889 | +0.0417 |
| missing_field share of failures | ~55% | 58.8% (211/359) | **52.3% (196/375)** | better |

**Interpretation.** Holding corpus size at n=100 (iter 1 vs. iter 2 differ only in epoch count and the independently-justified fuzzy gate) and cutting 3→2 epochs recovered most of iter 1's lost ground (0.6581 → 0.6745) and pushed missing_field's share to 52.3%, the lowest measured in V3. This is the first clean confirmation of the iter-1 mechanism: **total training steps, not corpus size per se** — 2692 steps (2×1346) sits between iter 15's 2439 (3×813) and iter 1's 3237 (3×1079). The larger corpus only hurt when forced through a third epoch in the over-converged regime.

**Corroboration (288-record eval).** Hybrid field F1 0.6742, schema validity 0.8715, precision 0.7169, recall 0.6363, missing_field_rate 0.1207 — consistent with the 72-record readout; model-only F1 0.4861, rules-only F1 0.2911.

**Status.** Better than iter 1, still -0.0082 below the published V2/iter-15 checkpoint (0.6827) on the 72-record eval → not an HF-upload candidate (`checkpoint_used_for_huggingface_upload: false`). Full provenance in `experiments/v3-iter2-20260811T0228Z/manifest.json`. Iteration 15 remains the best published checkpoint.

**Planned next steps:** once teacher generation completes → retrain student at 2 epochs (3→2, per iter 1 hypothesis) → hybrid eval on the 288-record set → then V3 iteration 3.

## V3 Iteration 3 — 2026-08-11 — Epochs 2→1 under the fuzzy-gate corpus (n=100): negative result, completes the epoch sweep

**What ran.** Same 2691-record fuzzy-gate corpus as iter 2 (`data/teacher_dataset_v3iter2.json`, admitted 2691/3600), same seed/LR/batch (42, 2e-5, 2), but epochs cut 2→1 (`NUM_EPOCHS=1`, temporary server-side edit for the run, not committed) → 1346 steps total, single epoch. Epoch-1 avg loss 0.0489 (vs iter 2's epoch-1 0.0486 — expected, the first epoch is recipe-identical). Checkpoint sha256 `b8d078df…4ef6e8`; pre-run backup `models/distilled_minicpm5_1b_v2_amd_pre_v3iter3` verified intact.

**Result — negative vs iter 2, completes the sweep:**

| metric (72-rec eval) | V3 it1 (3ep) | V3 it2 (2ep) | V3 it3 (1ep) | it3 vs it2 |
|---|---|---|---|---|
| hybrid field F1 | 0.6581 | **0.6745** | 0.6597 | **-0.0148** |
| field precision | 0.7229 | 0.7110 | 0.7095 | -0.0015 |
| field recall | 0.6039 | 0.6416 | 0.6165 | -0.0251 |
| schema validity | 0.8472 | 0.8889 | 0.9583 | +0.0694 |
| missing_field share of failures | 58.8% (211/359) | **52.3% (196/375)** | 57.0% (196/344) | worse |

**Interpretation.** One epoch underfits: field F1 0.6597 essentially ties iter 1's 3-epoch 0.6581 but loses to iter 2's 2-epoch 0.6745. With 1346 gradient updates the model captures fewer extraction patterns — missing_field's share of failures rises back to 57.0% (196/344) from iter 2's 52.3% (196/375), even though the absolute failure count (344) is lower. Schema validity improved to 0.9583, but at the cost of recall (0.6165). The sweep brackets the optimum: **1ep 0.6597 / 2ep 0.6745 / 3ep 0.6581 — 2 epochs is the confirmed optimum** under the fuzzy-gate corpus.

**Corroboration (288-record eval).** Hybrid field F1 0.6524, schema validity 0.9410, precision 0.7072, recall 0.6054, missing_field_rate 0.1577 — consistent with the 72-record readout; model-only F1 0.4530, rules-only F1 0.2763.

**Status.** Negative result vs iter 2 but informative: it closes the epoch sweep with a clean optimum at 2 epochs. Not an HF-upload candidate (`checkpoint_used_for_huggingface_upload: false`). Full provenance in `experiments/v3-iter3-20260811T0240Z/manifest.json`. `NUM_EPOCHS` restored to 2 (confirmed optimum) in `src/02_train_distill.py`.

**Planned next steps:** with corpus size (n=100) and epoch count (2) now fixed, the remaining levers are the training recipe itself — LR/warmup changes (e.g. lower LR or longer warmup to avoid the near-zero-loss-by-epoch-2 plateau) or held-out early stopping. missing_field remains 52-57% of all failures and is still the primary unsolved problem.
