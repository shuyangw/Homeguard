# WS-3d Gate 1 (Detector Diagnostic Rerun) -- BLOCKED -- 2026-05-24

## Summary

WS-3d Gate 1 BLOCKED. v1 LightGBM detector trained successfully (best CV mean
logloss = 0.347 on 48-combination purged sweep), but on the apples-to-apples
G4-event-basis H5 measurement, v1 median lag = 21d vs v0 baseline 14d
(50% INCREASE, not the >=30% REDUCTION required by Pre-commitment 5). v1
also missed the 2025_dec_drawdown entirely within a 50-day event window.
Per spec, Gates 2-6 are NOT run; spec amendment is required before
continuing.

## Changes Made

- **src/strategies/advanced/market_regime_detector_v1.py** (new): LightGBM
  detector schema-compatible with v0. Loads joblib artifact, exposes
  `classify_regime(spy_data, vix_data, timestamp)` with 5 regime keys plus
  `last_indicators`, `last_regime_scores`, `last_classification_timestamp`,
  and (new) `last_bear_probability`. BEAR score is `predict_proba(X)[0, 1]`;
  4 non-BEAR scores derived from v0-style rules over the 4 backwards-compat
  features (above_20, above_50, above_200, vix_percentile).
- **scripts/diagnostics/train_detector_v1.py** (new): Purged 6-fold CV with
  5d purge + 5d embargo, 48-combination bounded hyperparameter sweep
  (4 n_estimators x 4 max_depth x 3 lr). Persists model + metadata.json
  under `H:/Stock_Data/alt_data/models/v20_detector/<git_sha>/`. Best HP:
  n_estimators=100, max_depth=5, learning_rate=0.01, num_leaves=31.
- **scripts/diagnostics/regime_detector_v1_replay.py** (new): Day-by-day
  replay across 2017-2026 (2360 rows). Output schema mirrors v0
  `labels.parquet` plus `bear_proba` column.
- **scripts/diagnostics/regime_detector_v1_diagnostic.py** (new): H1-H5
  rerun on v1 vs v0 baseline. Computes H5 lag on TWO bases:
  - G4-event basis (decides Gate 1 verdict, matches 20260523 baseline of record)
  - G1_BEAR-onset basis (spec methodology)
- **tests/strategies/advanced/test_market_regime_detector_v1.py** (new):
  11 unit tests covering schema compat, argmax-flip-on-threshold, missing
  artifact, insufficient coverage, idempotency, v0-style helper.
- **docs/reports/ramp/20260601_ws3d_regime_diagnostic_rerun.md** (new):
  Full H1-H5 results + Diagnosis and recommendation section.
- **diagnostics/regime/v1/labels.parquet** (new): 2360 day-rows partitioned
  by year.

## Commits

- `e192f5e` feat(detector): MarketRegimeDetectorV1 -- LightGBM on leading
  indicators (WS-3d)
- `f3c11c9` feat(diagnostics): WS-3d detector replay + H1-H5 diagnostic
  rerun

## Known Issues / Remaining Work

**Gate 1 is BLOCKED.** Gates 2-6 NOT run.

H5 per-event detail (G4 basis):

| Event | v0 lag | v1 lag | Delta |
|---|---|---|---|
| Q4_2018_selloff | 14d | 26d | WORSE 12d |
| COVID_crash | 14d | 8d | BETTER 6d |
| 2022_bear_market | 14d | 20d | WORSE 6d |
| 2025_tariff_drawdown | 5d | 22d | WORSE 17d |
| 2025_dec_drawdown | 36d | DID NOT FIRE | WORST |

Recommended next steps for spec revision (documented in the report's
Diagnosis section):

1. Lower BEAR_PROB_THRESHOLD or evaluate Gate 1 on a Schmitt-trigger label
   rather than argmax-at-0.5. Raw P(BEAR) reaches 0.25-0.30 days before
   the argmax flip.
2. Train on a LEADING target (G2_BEAR forward-window, or G1_BEAR.shift(-k))
   rather than the coincident G1_BEAR.
3. Consider fallback architectures in the spec Appendix (HMM, threshold
   ensemble) -- but none of these address the underlying issue that a
   confirmation-label supervised model cannot be predicted ahead of itself
   at a 0.5 decision threshold.
4. Escalate to halt-or-redirect per parent WS-3 spec Appendix.

H2 recall is the silver lining: v1 = 96.5% vs v0 = 46.1%. On confirmed
G1_BEAR days, v1 is dominant. The failure mode is purely lag (when the
argmax flips), not coverage (whether it flips at all).

## Validation

- 11 new detector unit tests PASS
- Existing test suite PASS (177 tests across ramp_phase4, diagnostics,
  strategies.advanced, data.leading_indicators)
- Training: 48-combination sweep ran cleanly; no degenerate folds reported.
- Replay: 2360 day-rows written to `diagnostics/regime/v1/labels.parquet`
  with 381 BEAR labels (16.1% of replay days, close to G1_BEAR's 17.6%
  positive rate).
- Diagnostic: `docs/reports/ramp/20260601_ws3d_regime_diagnostic_rerun.md`
  (9177 chars) contains H1-H5 comparison tables, G4 per-event detail, G1
  per-onset detail, and a Diagnosis section enumerating spec revision
  options.
- Gate 1 verdict (G4 basis): FAIL.
- No code changes to v0 detector, no V11-V14 variants modified. Schema
  compat preserved per Pre-commitment 4.

## Trial-chain bookkeeping

Per Pre-commitment 1, WS-3d's `n_trials_project` counter remains at the
spec-time value of 1 (the V20-rd-bear-cash primary, not yet implemented
because Gate 1 BLOCKED before Gate 3). The hyperparameter sweep counts
as 1 trial, not 48, per the bounded-sweep discipline.
