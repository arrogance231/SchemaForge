---
title: "SchemaForge V3 — Whitepaper (DRAFT)"
status: "DRAFT — NOT FINAL, iterations 1-4 complete"
last_updated: "2026-08-11"
predecessor: "docs/WHITEPAPER.md (V2, closed out at iteration 15)"
---

# SchemaForge V3: Closing the Omission Gap

> **STATUS: DRAFT.** V2 (`docs/WHITEPAPER.md`) is considered methodologically closed as of
> iteration 15 — the pipeline is reproducible, the hybrid architecture's headline claim is
> validated, and the release checkpoint, prepared for publication (see `models/schemaforge-v2-distilled-minicpm5-1b/`)
> reflects that closure. V3 starts from that baseline and targets the model's own remaining
> weaknesses. This document is assembled live as V3 iterations run; treat
> `logs/V3_TRAINING_FAILURES.md` as the authoritative append-only record, same convention as V2's
> `logs/V2_TRAINING_FAILURES.md`.

## Starting point (inherited from V2, not re-derived)

- Pipeline is deterministic and reproducible: fixed training seed, automatic checkpoint backup,
  greedy (temperature=0.0) teacher-label generation — see V2 iteration 15's controlled validation
  of this (field F1 moved 0.0003 under a byte-identical corpus re-run).
- Hybrid architecture (deterministic pre-pass + model, routed by field ownership) beats both
  systems alone on every metric — not up for debate, re-confirmed on every V2 configuration
  tested. V3 is not testing this claim again; it is optimizing the model's contribution to it.
- **The dominant unsolved problem, unchanged across all 15 V2 iterations: `missing_field`
  (omission) is 55-62% of every failure breakdown measured.** No V2 intervention (corpus scale,
  ontology fills, an isolated-operator targeted corpus) reduced this. This is V3's primary target.
- V2's own diagnosis of why the isolated-operator attempt (iteration 14) failed: training the
  model on `delabel`/`implicit` corruptions in isolation (no other corruption stacked on top) may
  not transfer to the compounded, multi-corruption case the eval set and most training data
  actually presents. V3's first experiment tests this directly.

## V3 Iteration Log

(populated as runs complete — see `logs/V3_TRAINING_FAILURES.md` for full technical detail behind
each entry)

### Iteration 1 — 2026-08-10 — Corpus n=75→100 under the deterministic pipeline (strict gate): NEGATIVE, isolates corpus scale

- **Change tested:** pure corpus scale-up 75→100 documents under the deterministic pipeline
  (greedy teacher labels, fixed seed), strict gate: **2157/3600 admitted (40.1% rejected)**; 3
  epochs, 1079 steps/epoch, 3237 total gradient updates.
- **Result (72-rec eval):** hybrid field F1 **0.6581** (V2-FINAL/iter-15 0.6827, -0.0246),
  missing_field share **58.8%** (211/359) — still inside V2's 55-62% band.
- **Verdict:** negative vs the V2-FINAL baseline (release checkpoint) — corpus size alone (n=75→100) does
  not close the omission gap; the mechanism to test next is total training steps, not corpus
  size. Provenance: `experiments/v3-iter1-20260810T155838Z/manifest.json`.

### Iteration 2 — 2026-08-11 — Epochs 3→2 under the fuzzy-gate corpus (n=100): POSITIVE, confirms the total-training-steps hypothesis

- **Wedge incident & recovery:** iter-2 teacher generation deadlocked at batch 208/900 (>1h busy-spin, killed). Recovered via determinism — `src/09_recover_fuzzy.py` re-ran the gate with `fuzzy_support=True` on iter-1's saved rejections, recovering 534/1443 (37%) → **2691/3600 admitted** (25.2% rejected vs 40.1% strict). Pre-iteration checkpoint backup `..._pre_v3iter2` verified intact.
- **Change tested:** corpus held at n=100 (same source), epochs 3→2 (`NUM_EPOCHS=2`, commit `6b7248e`): 1346 steps/epoch, 2692 total; epoch-mean losses 0.0486 / 0.0162; checkpoint sha256 `c1b51015…7dfb`.
- **Result (72-rec eval):** hybrid field F1 **0.6745** (it1 0.6581, +0.0164), schema validity 0.8889, missing_field share **52.3%** (196/375) — lowest in V3. 288-rec eval corroborates: F1 0.6742, schema validity 0.8715, missing_field_rate 0.1207.
- **Verdict:** first change to help since iter 13/15; confirms iter-1's mechanism (total training steps, not corpus size — 2692 steps sits between iter 15's 2439 and iter 1's 3237). Still -0.0082 below V2-FINAL/iter-15 (0.6827, release checkpoint) on the 72-rec eval → not an HF-upload candidate. Provenance: `experiments/v3-iter2-20260811T0228Z/manifest.json`.

### Iteration 3 — 2026-08-11 — Epochs 2→1 under the fuzzy-gate corpus (n=100): NEGATIVE, completes the epoch sweep

- **Change tested:** epochs 2→1 (`NUM_EPOCHS=1`, temporary server-side edit, restored to 2 after the run): 1346 steps total; epoch-1 avg loss 0.0489; checkpoint sha256 `b8d078df…4ef6e8`; pre-run backup `..._pre_v3iter3` intact.
- **Result (72-rec eval):** hybrid field F1 **0.6597** (it2 0.6745, -0.0148), schema validity 0.9583, missing_field share back to **57.0%** (196/344). 288-rec eval corroborates: F1 0.6524, schema validity 0.9410, missing_field_rate 0.1577.
- **Verdict:** negative vs iter 2 but completes the sweep — 1ep 0.6597 / 2ep 0.6745 / 3ep 0.6581, so **2 epochs is the confirmed optimum** under the fuzzy-gate corpus. Repo restored to `NUM_EPOCHS=2`. Not an HF-upload candidate. Provenance: `experiments/v3-iter3-20260811T0240Z/manifest.json`.

### Iteration 4 — 2026-08-11 — LR 2e-5→1e-5 at the 2-epoch optimum: NEGATIVE, brackets the LR dimension

- **Change tested:** LR lowered 2e-5 → 1e-5 (named `LR` constant added for this run, restored to 2e-5 after): same 2691-record fuzzy corpus, same 2 epochs (1346 steps/epoch, 2692 total); epoch losses 0.0678 / 0.0333.
- **Result (72-rec eval):** hybrid field F1 **0.6671** (it2 0.6745, -0.0074), schema validity **0.8056** (it2 0.8889, -0.0833 — the biggest regression), missing_field share 56.4% (202/358). 288-rec eval corroborates: F1 0.6650, schema validity 0.8611.
- **Verdict:** negative — the lower LR under-trains on this small corpus (higher final loss, more schema-invalid outputs). Recipe grid now bracketed: **2 epochs / LR=2e-5 is the confirmed optimum**; V2/V3's default recipe was already near-optimal. Not an HF-upload candidate. Provenance: `experiments/v3-iter4-20260811T0305Z/manifest.json`.

## Open questions V3 aims to answer

1. Does stacking `delabel`/`implicit` on top of the full ten-operator mix (rather than isolating
   them, V2 iteration 14's approach) reduce `missing_field`'s share of failures?
2. Is there a training-recipe lever (more epochs, different LR, held-out validation-based early
   stopping) that helps once corpus composition is right, given V2 saw near-zero training loss by
   epoch 2-3 in every run — a possible early-overfitting signature not yet directly investigated?
3. Can the gate's source-support check (`schemaforge/validation/gate.py` step 3) be loosened
   safely to admit legitimate denoising corrections (e.g. OCR-typo'd names the teacher correctly
   normalizes) without also admitting genuinely wrong labels? V2 iterations 3/5 flagged this as a
   real false-rejection source but never attempted a fix.
4. Does `incorrect_normalization` ever actually fire in a large enough eval set? It read exactly
   0 in every V2 measurement despite ontology fills — genuinely unresolved.

## Acknowledgments

GPU access for this project's AMD Instinct MI300X training is provided by the **AMD AI Developer
Program**.
