# WS-3d Gate 1 (Detector Diagnostic Rerun) -- BLOCKED -- 2026-05-24

## Summary

WS-3d Gate 1 STILL BLOCKED after Amendment 6 reframing.

**Round 1 (argmax-at-0.5, pre-Amendment 6, commit f3c11c9):** v1 LightGBM
detector trained successfully (best CV mean logloss = 0.347 on 48-combination
purged sweep), but on the apples-to-apples G4-event-basis H5 measurement,
v1 median lag = 21d vs v0 baseline 14d (50% INCREASE, not the >=30%
REDUCTION required by Pre-commitment 5). v1 also missed the
2025_dec_drawdown entirely.

**Round 2 (Schmitt-tau-0.30 first-crossing, Amendment 6, commits 1b24b7b +
<this-commit>):** Hypothesis from Round 1 -- that v1's `P(BEAR)` crosses
0.25-0.30 before the argmax-at-0.5 flip -- was tested by reframing Gate 1's
binding metric to Schmitt-tau-0.30 first-crossing (matching the
V20-rd-bear-cash consumer-layer pattern). Result: v1 median lag at tau=0.30
= 19.5d, v0 median lag at tau=0.30 = 2.0d (v0's rule-based `score_BEAR` is a
fraction-of-criteria-met that flips immediately on the first
above_20/above_50/above_200/vix_percentile criterion). Reduction = -875%.
v1 is still SLOWER than v0 at the binding metric. The Schmitt-trigger
reformulation does NOT rescue Gate 1.

Per Amendment 6's last paragraph, the spec falls back to: (b) retrain on a
leading target, (c) try fallback architectures, or (d) halt WS-3d. Decision
is the user's, not automatic. Gates 2-6 remain NOT run.

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
- `bb97e56` docs(progress): WS-3d Gate 1 session log -- BLOCKED at
  diagnostic gate (Round 1)
- `1b24b7b` spec(ws3d): Amendment 6 -- Gate 1 H5 metric changes to
  Schmitt-tau-0.30 first-crossing
- `<this-commit>` diag(ws3d): Gate 1 H5 rerun under Amendment 6 -- STILL
  BLOCKED at Schmitt-tau-0.30

## Known Issues / Remaining Work

**Gate 1 is STILL BLOCKED under Amendment 6.** Gates 2-6 NOT run.

H5 per-event detail under BOTH metrics:

| Event | v0 argmax | v1 argmax | v0 tau=0.30 | v1 tau=0.30 |
|---|---|---|---|---|
| Q4_2018_selloff | 14d | 26d | 2d | 26d |
| COVID_crash | 14d | 8d | 2d | 7d |
| 2022_bear_market | 14d | 20d | 14d | 17d |
| 2025_tariff_drawdown | 5d | 22d | 2d | 22d |
| 2025_dec_drawdown | 36d | DID NOT FIRE | 36d | DID NOT FIRE |

Median lag summary:

| Metric | v0 | v1 | Reduction |
|---|---|---|---|
| argmax-at-0.5 (legacy, informational) | 14.0d | 21.0d | -50.0% |
| **Schmitt-tau-0.30 (Amendment 6 binding)** | **2.0d** | **19.5d** | **-875.0%** |

Why Amendment 6 did not rescue Gate 1: v0's `score_BEAR` is a rule-based
fraction-of-criteria-met (above_20 / above_50 / above_200 / vix_percentile),
each independent and binary. The first time SPY closes below its 20-day MA
on a drawdown, v0's score jumps to 0.20-0.40 immediately. So v0 at tau=0.30
fires on day 1-2 of essentially every drawdown. v1's `bear_proba` is a
LightGBM joint posterior P(G1_BEAR | indicators) -- a smoothly-rising
function that takes weeks to cross 0.30 because G1_BEAR itself is a 10%
drawdown confirmation and the supervised model learned to predict that
confirmation, not its onset. The leading-indicator hypothesis (that
VIX_term + HY proxy + breadth + SKEW would let LightGBM cross the
consumer-layer threshold before v0's rule-based score) is FALSIFIED on
G4 events.

Recommended escalation (Amendment 6's last paragraph):

(b) **Retrain on a LEADING target** -- use label = G1_BEAR.shift(-k) for
    k in {5, 10, 15} (picking k that maximizes G4 lead-time), or use
    G2_BEAR (forward 30-day return < -5% AND forward vol > 25%). G2 has
    more class imbalance and is harder to learn but is fundamentally
    leading.
(c) **Try a fallback architecture** -- HMM (5-state) or threshold-ensemble.
    Risk: HMM EM is unstable across rolling fits (mode collapse). None of
    these address the supervised-on-confirmation-label issue, but they may
    produce different lag characteristics.
(d) **HALT WS-3d.** Three independent measurements of structural detector
    lag (V12 gap_days=-3.42, v0 H5=14d argmax / 2d Schmitt, E8
    exit-to-low=-8d) led to this spec. If a fresh architecture AND a
    fresh leading-indicator input set AND a Schmitt-trigger evaluation
    cannot reduce H5 lag, the regime-aware approach may be at its useful
    limit for RAMP regardless of detector iteration. v0's rule-based
    score at tau=0.30 (2d median lag) is the ceiling for any consumer
    that uses the WS-3d input set and consumes via tau=0.30 -- v1 trails
    that ceiling, so swapping the detector for V20-rd-bear-cash is a
    REGRESSION at the consumer layer.

The decision is the user's, not automatic.

H2 recall is the silver lining: v1 = 96.5% vs v0 = 46.1% on confirmed
G1_BEAR days. The failure mode is purely lag (when the score crosses the
gate), not coverage.

## Validation

- 11 detector unit tests PASS (re-run after Amendment 6 changes; the
  diagnostic-script extension is independent of the detector module so
  detector tests are unaffected).
- Diagnostic: `docs/reports/ramp/20260601_ws3d_regime_diagnostic_rerun.md`
  rewritten with three H5 bases (Schmitt-tau-0.30 binding, argmax-at-0.5
  legacy, G1_BEAR-onset informational). Now ~13k chars including
  Amendment 6 section, per-event tables for both metrics, verdict block,
  and updated diagnosis.
- Gate 1 verdict (Amendment 6 binding, Schmitt-tau-0.30): FAIL.
- Gate 1 verdict (legacy argmax-at-0.5): FAIL.
- No code changes to v0 detector, v1 detector, or any V11-V14 variant.
  Only `scripts/diagnostics/regime_detector_v1_diagnostic.py` and the
  spec / report / session log were modified.
- Schema compat preserved per Pre-commitment 4.

## Amendment 6 self-review checklist

- v0 baseline at Schmitt-tau-0.30 computed honestly from
  `diagnostics/regime/v0_scores/labels.parquet` `score_BEAR` column (NOT
  inherited from the 14d argmax-at-0.5 baseline). Result: 2.0d.
- tau_eval=0.30 pre-registered from E3's soft-score finding, cited in spec
  Amendment 6 and the report's Amendment 6 section. Not chosen post-hoc.
- ASCII-only throughout (report renders via `encoding='ascii',
  errors='replace'`).
- BLOCKED escalation path documented: (b) retrain on leading target,
  (c) alternative architecture, (d) halt WS-3d.

## Trial-chain bookkeeping

Per Pre-commitment 1, WS-3d's `n_trials_project` counter remains at the
spec-time value of 1 (the V20-rd-bear-cash primary, not yet implemented
because Gate 1 BLOCKED before Gate 3). The hyperparameter sweep counts
as 1 trial, not 48, per the bounded-sweep discipline.
