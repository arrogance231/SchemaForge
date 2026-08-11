# Changelog

All notable changes to SchemaForge V2 are documented in this file.

Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [2.0.0] — 2026-08-11

**V2 phase complete** — pipeline reproducible end-to-end, hybrid claim validated, V2-FINAL
release checkpoint prepared for publication.

### Added

- `VERSION` file (`2.0.0`); annotated git tag `v2.0.0` on the release commit.
- README Status section refreshed to the true current state (V2 closed at iteration 15, V3
  iterations 1–4 recorded) plus a new `## Roadmap` section.

### V2 release state

- V2 methodologically closed at **iteration 15**; the published Hugging Face checkpoint remains
  **iteration 5/10**.
- **V2-FINAL** iteration-15 release checkpoint at `models/schemaforge-v2-distilled-minicpm5-1b/`
  (sha256 `c13f7f6c…`, hybrid field F1 **0.6827** on the 72-record eval set) — prepared for
  publication.
- Headline result (72-record eval): hybrid field F1 **0.6827** vs **0.291** deterministic pass
  alone vs **~0.484** model alone.

### V3 (in progress)

- V3 opened with `docs/WHITEPAPER_V3.md`; iterations 1–4 recorded in `docs/WHITEPAPER_V3.md`
  and `logs/V3_TRAINING_FAILURES.md`.
- Recipe sweep: **2 epochs / LR=2e-5** confirmed optimum (1ep 0.6597 / 2ep 0.6745 / 3ep 0.6581);
  best V3 hybrid field F1 **0.6745**; `missing_field` share of failures reduced to **52.3%**.
