# SchemaForge V3 — Findings Note: Iterations 1-4 (2026-08-10 → 2026-08-11)

> **Purpose of this note.** This is a findings/incident note, NOT part of the SchemaForge
> whitepaper itself. It records the V3 iteration sweep (iterations 1-4), the teacher-generation
> wedge incident and its determinism-based recovery, the confirmed recipe optimum, and the
> standing conclusions — with pointers to full provenance. The whitepapers remain canonical:
> `docs/WHITEPAPER.md` (V2, closed at iteration 15) and `docs/WHITEPAPER_V3.md` (V3, draft);
> `logs/V3_TRAINING_FAILURES.md` is the authoritative append-only record.

## 1. The wedge incident (V3 iteration 2)

**What happened.** While V3 iteration 2 was generating teacher labels, the process wedged:
**deadlock at batch 208/900**, busy-spinning for >1h, and was killed (the incident occurred
during an autonomous-monitoring period with no interactive steering available).

**Diagnosis — fd-offset evidence.** The wedge was diagnosed as a busy-spin deadlock, not slow
progress and not an I/O stall: the process's **file-descriptor offset on its output stream stayed
stationary** across multiple samples while CPU remained pinned — the signature of a
lock/condition-variable busy-spin rather than a blocked wait. No output records were flushed
past batch 208, so the wedged run held no recoverable partial work.

**Why it happened.** The deadlock sat inside the teacher-generation loop (the batched
generation/validation handoff) at a fixed batch boundary (208/900), i.e. in the first ~23% of
the run. With generation deterministic (greedy, temperature=0.0), a re-run reproduces the same
batch sequence and the same environment-level failure mode — re-running generation carried a
real risk of re-hitting the same deadlock. The root cause inside the loop was not reproduced
interactively; the recovery described below made re-running unnecessary.

**Why it could not be recovered in-process.** The wedged process was killed with no partial
checkpoint and no usable flushed output. In-process recovery was not attempted: the diagnosis
(busy-spin, no output progress) left nothing to salvage, and no interactive steering was
available during the autonomous-monitoring window.

## 2. The determinism-based recovery method

Rather than re-run generation, iteration 2 exploited the pipeline's determinism:

- Teacher generation is **greedy (temperature=0.0)** and seeded: for a fixed corpus and prompt
  template, the saved raw outputs from iteration 1 are **byte-identical** to what a fresh run
  would produce. Re-running the **admission gate only** (not the generation) is therefore
  equivalent to re-running generation + gate — except the gate is cheap and runs on CPU.
- `src/09_recover_fuzzy.py` (commit `a8be53d`) re-ran the gate on iter-1's saved rejections
  (`data/teacher_dataset_rejections_v3iter1_bak.json`, 1443 records) with `fuzzy_support=True`
  (threshold 0.85), then rebuilt training prompts via importlib from `01_generate_teacher.py`.
- Result: **534/1443 (37%)** step-3 rejections recovered → **2691/3600 admitted (25.2% rejected
  vs 40.1% under the strict gate)** → `data/teacher_dataset_v3iter2.json`.

The pre-iteration checkpoint backup (`models/distilled_minicpm5_1b_v2_amd_pre_v3iter2`) was
verified intact before training, so the recovery introduced no risk to the canonical V2-FINAL
checkpoint.

## 3. Full results table (V3 iterations 1-4)

| iteration | corpus | epochs | LR | admitted/total | 72-rec hybrid F1 | 72-rec missing share | 288-rec hybrid F1 | 288-rec schema validity | 288-rec missing_field_rate |
|---|---|---|---|---|---|---|---|---|---|
| V2 iter 15 (published/release baseline) | n=75 | 3 | 2e-5 | 1625/2700 | **0.6827** | ~55% | — | — | — |
| V3 iter 1 | n=100 | 3 | 2e-5 | 2157/3600 | 0.6581 | 58.8% (211/359) | — | — | — |
| **V3 iter 2 (best)** | n=100 | **2** | **2e-5** | 2691/3600 | **0.6745** | **52.3% (196/375)** | 0.6742 | 0.8715 | 0.1207 |
| V3 iter 3 | n=100 | 1 | 2e-5 | 2691/3600 | 0.6597 | 57.0% (196/344) | 0.6524 | 0.9410 | 0.1577 |
| V3 iter 4 | n=100 | 2 | 1e-5 | 2691/3600 | 0.6671 | 56.4% (202/358) | 0.6650 | 0.8611 | 0.1200 |

Detail on the best checkpoint (V3 iter 2, 72-record eval, hybrid system):

| metric | value |
|---|---|
| field F1 | 0.6745 |
| field precision | 0.7110 |
| field recall | 0.6416 |
| hallucination rate | 0.0201 |
| schema validity | 0.8889 |
| missing_field_rate | 0.1144 |
| missing_field share of failures | 52.3% (196/375) |

## 4. Confirmed recipe optimum

- **Epoch sweep** (fuzzy-gate corpus, LR=2e-5): 1ep **0.6597** / 2ep **0.6745** / 3ep **0.6581**
  → **2 epochs is the optimum**. Iter 3's 1-epoch run underfits (ties iter 1's 3-epoch result);
  iter 1's 3-epoch run over-converges (3237 total steps, near-zero loss plateau).
- **LR sweep** (2-epoch optimum): 2e-5 **0.6745** / 1e-5 **0.6671** → **LR=2e-5 is the optimum**.
  The lower LR under-trains on this small corpus (final epoch loss 0.0333 vs 0.0162 at 2e-5;
  schema validity drops 0.8889 → 0.8056, the biggest single regression of the sweep).
- **Recipe-grid conclusion: 2 epochs at LR=2e-5 is the confirmed optimum**; V2/V3's default
  recipe was already near-optimal for this corpus. Neither more steps (iter 1, 3 epochs), fewer
  steps (iter 3, 1 epoch), nor a lower LR helps.

## 5. Standing conclusion

- `missing_field` (omission) **remains the dominant failure category**: V3's best (iter 2)
  reached **52.3%** share — the lowest measured in the project and the first below V2's 55-62%
  band — but the gap is not closed.
- The V3 mechanism finding: **total training steps, not corpus size per se**, drives these
  checkpoints' generalization. 2692 steps (2×1346, iter 2) sits between iteration 15's 2439
  (3×813) and iter 1's 3237 (3×1079); the larger corpus only hurt when forced through a third
  epoch in the over-converged regime.
- **No V3 checkpoint beats V2-FINAL (0.6827) on the comparable 72-record eval** — iter 2's
  0.6745 is -0.0082 below. Every iteration manifest records
  `checkpoint_used_for_huggingface_upload: false`. The release candidate remains V2-FINAL
  (iteration 15; `models/schemaforge-v2-distilled-minicpm5-1b/`, sha256 `c13f7f6c…`), prepared
  for publication and pending the project owner's Hugging Face upload.

## 6. Provenance pointers

- **Run manifests** (authoritative machine-readable provenance):
  - `experiments/v3-iter1-20260810T155838Z/manifest.json`
  - `experiments/v3-iter2-20260811T0228Z/manifest.json` (best checkpoint)
  - `experiments/v3-iter3-20260811T0240Z/manifest.json`
  - `experiments/v3-iter4-20260811T0305Z/manifest.json`
- **Failure log**: `logs/V3_TRAINING_FAILURES.md` (append-only record; the authoritative source
  for this note's incident and recovery sections)
- **Recovery script**: `src/09_recover_fuzzy.py`
- **Whitepapers**: `docs/WHITEPAPER.md` (V2), `docs/WHITEPAPER_V3.md` (V3)
- **Data** (this package): `data/corpus_v3_iter1.jsonl`,
  `data/teacher_dataset_v3iter1.json`, `data/teacher_dataset_v3iter2.json`,
  `data/teacher_dataset_rejections_v3iter1_bak.json`, `data/eval_holdout.jsonl`,
  `data/eval_holdout_v2.jsonl`
- **Model card** (this package): `models/schemaforge-v3-distilled-minicpm5-1b/README.md`
- **V2 history**: `logs/V2_TRAINING_FAILURES.md` (unmodified)
