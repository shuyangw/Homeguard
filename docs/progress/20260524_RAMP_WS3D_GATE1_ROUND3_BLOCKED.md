# WS-3d Gate 1 Round 3 -- Leading Target Retrain Verdict - 2026-05-24

## Summary

Third attempt at Gate 1 for the WS-3d LightGBM regime detector. After round 1
(argmax-at-0.5, BLOCKED at 21d vs 14d) and round 2 (Schmitt-tau-0.30 per
Amendment 6, BLOCKED at 19.5d vs 2.0d), this round retrained the model on a
LEADING target `G1_BEAR.shift(-10)` so the model predicts whether G1_BEAR will
fire 10 days from t. Result: median Schmitt-tau-0.30 lag = 11.0d (5/5 fired
criterion missed at 4/5). Both pre-registered round-3 success criteria
(median <= 5d AND 5/5 events fired) FAIL. WS-3d under the leading-target
hypothesis is unambiguously FALSIFIED.

## Pre-registered Round 3 framing

Original "30% reduction from v0 2.0d baseline" was unbeatable (30% off 2.0d
is 1.4d -- detector cannot fire faster than day 0-1 of a hand-curated event
window). Round-3 criteria pre-registered BEFORE observing the new model's
lag:

- PASS iff: v1 median H5 lag at Schmitt-tau-0.30 (G4-event basis) <= 5d
  AND v1 fires on >= 5/5 G4 drawdown events.
- FAIL otherwise.

Observed: 11.0d median, 4/5 fired. Both fail.

## Changes Made

- **scripts/diagnostics/train_detector_v1.py**: added `--target` CLI flag
  accepting `{g1_bear, g1_shift10, g2_bear}` (default g1_bear for backward
  compat). New `TARGET_CONFIG` dict encodes per-target purge/embargo:
  g1_bear keeps the original 5/5 (point-in-time label), g1_shift10 widens
  to 15/15 (10d shift + 5d buffer each side), g2_bear to 35/35 (30d forward
  window). Model artifacts now suffixed by target name
  (`model_g1_shift10.pkl`); the original `model.pkl` for g1_bear is
  unchanged. Metadata records target name + shift + purge/embargo. Last 10
  rows of the shifted target are masked False to avoid lookahead.
- **scripts/diagnostics/regime_detector_v1_replay.py**: added
  `--model-suffix` (default `g1_bear`) and `--output-dir`. Round-1 outputs
  to `diagnostics/regime/v1/`, round-3 leading target outputs to
  `diagnostics/regime/v1_g1_shift10/`. Model auto-discovery filters by
  suffix.
- **scripts/diagnostics/regime_detector_v1_diagnostic.py**: added
  `--labels-dir` and `--report-path` CLI flags. Diagnostic can now be run
  against any v1 replay output (round 1 / round 3 / future rounds) and
  write to a separate report file. Default behavior unchanged.
- **docs/reports/ramp/20260602_ws3d_diagnostic_g1_shift10.md**: new
  round-3 verdict report. Pre-registered criteria + result tables +
  per-event G4 lag detail at Schmitt-tau-0.30 + best CV hyperparameters +
  feature importance. The shared diagnostic-script body follows the
  preamble; its hardcoded round-1 diagnosis text is preserved as
  informational context.

## Round 3 Results

- v0 baseline (cross-check): Schmitt-tau-0.30 median lag 2.0d, 5/5 fired
- v1 round 1 (G1_BEAR target): 19.5d, 4/5 fired
- v1 round 3 (G1_BEAR.shift(-10) target): 11.0d, 4/5 fired

Per-event G4 detail at Schmitt-tau-0.30 (round 3 v1):
- Q4_2018_selloff (2018-10-03):   v0=2,  v1=8       -> WORSE by 6d
- COVID_crash (2020-02-19):       v0=2,  v1=5       -> WORSE by 3d
- 2022_bear_market (2022-01-04):  v0=14, v1=14      -> TIE
- 2025_tariff_drawdown (2025-02-19): v0=2, v1=19    -> WORSE by 17d
- 2025_dec_drawdown (2025-12-15):    v0=36, v1=DID NOT FIRE -> WORSE

The shifted target shrunk the median lag from round 1's 19.5d to 11.0d but
still doesn't meet the 5d criterion. v1 missing the 2025_dec_drawdown
entirely (the most recent event) is the dominant failure -- 4/5 fails the
5/5 criterion regardless of any median.

## Best CV Hyperparameters (round 3)

From `H:/Stock_Data/alt_data/models/v20_detector/146e5d2/model_g1_shift10.metadata.json`:
- n_estimators=50, max_depth=3, learning_rate=0.05, num_leaves=7
- Best CV mean logloss = 0.38289 (std 0.26953) over 6 purged folds at
  purge=15 / embargo=15
- Feature importance (gain): skew_close=69, hy_proxy_ratio=66,
  vix_percentile=50, above_200=39, vix_term_ratio=32, breadth_pct=24,
  above_50=17, above_20=3

## Commits

- `8c476f0` feat(detector): WS-3d Gate 1 round 3 -- leading target G1_BEAR.shift(-10) retrain
- `e3c030d` diag(ws3d): Gate 1 round 3 verdict at G1_BEAR.shift(-10) -- BLOCKED at Schmitt-tau-0.30

## Known Issues / Remaining Work

- WS-3d under the leading-target hypothesis is FALSIFIED. The user must
  decide:
  (i)   Halt WS-3d. Three independent measurements of structural detector
        lag (V12 gap_days=-3.42, v0 H5=14d, E8 exit-to-low=-8d) +
        three failed Gate 1 attempts suggest the regime-aware approach is
        at its useful limit for RAMP regardless of detector iteration.
        **Recommended.**
  (ii)  One more alternative target: retrain on `G2_BEAR`
        (forward 30d return < -5% AND forward vol > 25%). The training
        script already supports `--target g2_bear` with purge=35/embargo=35.
        Note: G2 is fully forward-looking, has tighter class balance, and
        is harder to learn than G1.shift(-10); it is plausible to be even
        less informative.
  (iii) Try the fallback architectures (HMM, threshold ensemble) from the
        WS-3d spec Appendix. These do not address the underlying issue
        that v0's `score_BEAR` is a near-coincident rule on price decline
        that fires on day 0-2 of every drawdown -- almost no supervised
        ML model on lagging features can outpace it without leakage.

- Gates 2-6 are still blocked. Task #126 marked completed in the round-3
  sense. Tasks #127-#131 remain pending pending user decision.

## Validation

- Training: `scripts/diagnostics/train_detector_v1.py --target g1_shift10`
  succeeded; best CV logloss 0.38289 < round-1's not-recorded baseline.
- Replay: `scripts/diagnostics/regime_detector_v1_replay.py
  --model-suffix g1_shift10` wrote 2360 rows to
  `diagnostics/regime/v1_g1_shift10/labels.parquet`. Regime distribution:
  WEAK_BULL 745, STRONG_BULL 484, BEAR 416, UNPREDICTABLE 347,
  SIDEWAYS 319, SAFE_MODE 49.
- Diagnostic: `scripts/diagnostics/regime_detector_v1_diagnostic.py
  --labels-dir diagnostics/regime/v1_g1_shift10 --report-path
  docs/reports/ramp/20260602_ws3d_diagnostic_g1_shift10.md` produced the
  binding-metric headline: v0 2.0d (5/5), v1 11.0d (4/5), reduction
  -450.0%, v1 meets <= 10d absolute floor: False. GATE 1 (binding,
  Amendment 6): FAIL.
- v0 baseline cross-check matches Round 2 baseline (2.0d, 5/5). Good.
- Round-1 outputs (`H:/Stock_Data/alt_data/models/v20_detector/80d2a8d/model.pkl`,
  `diagnostics/regime/v1/labels.parquet`) untouched.
