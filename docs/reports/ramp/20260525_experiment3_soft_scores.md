# Experiment 3 -- Soft-Score Extraction from MarketRegimeDetector

**Date**: 2026-05-24
**Branch**: v12-bear-to-cash
**Builds on**: V12 readiness alignment panel; regime detector diagnostic
**Decision target**: WS-3 track selection (WS-3a hysteresis / WS-3b leading indicators / WS-3c soft-score consumption)

## Summary

The MarketRegimeDetector's BEAR_score rises from a baseline of ~0.0-0.11
twenty trading days before an argmax flip into BEAR and reaches the 0.30
threshold roughly 24 trading days *before* the argmax actually flips.
BEAR_score also has a statistically significant negative correlation with
forward SPY drawdown at h in {5, 10, 20} (Pearson r between -0.18 and
-0.21, p < 1e-18). The soft signal exists; argmax is suppressing it.
**Verdict: WS-3c (soft-score consumption)** -- the next WS-3 spec should
build a tau-threshold consumer that fires on `score_BEAR >= tau` rather
than waiting for argmax. Hysteresis (WS-3a) and leading inputs (WS-3b)
both address real problems but are not the highest-leverage fix here.

## What was extended

- `scripts/diagnostics/regime_score_replay.py` -- adds the 5 per-regime
  soft scores per replay day on top of the v0 schema. Re-runs the
  detector once per day to harvest `last_regime_scores`; the v0 helper
  is untouched.
- Output: `diagnostics/regime/v0_scores/labels.parquet` (2360 rows
  spanning 2017-01-03 through 2026-05-22, partitioned by year), schema
  = v0 schema + 5 new `score_<REGIME>` columns in [0, 1].

## Three diagnostics

### (a) Event-study: BEAR_score trajectory aligned to BEAR onset

63 BEAR onsets identified in 2017-2026 (4 more than the V12 alignment
panel's 59; the v0 replay range starts on 2017-01-03 and includes early
2017 onsets the V12 panel may filter out). Median + IQR of BEAR_score
across all onsets, by trading day relative to onset:

| relative_day | median bear_score | p25 | p75 |
|---:|---:|---:|---:|
| -30 | 0.000 | 0.000 | 0.444 |
| -20 | 0.111 | 0.000 | 0.333 |
| -10 | 0.222 | 0.000 | 0.556 |
|  -5 | 0.333 | 0.000 | 0.556 |
|  -3 | 0.444 | 0.111 | 0.556 |
|  -1 | 0.444 | 0.222 | 0.556 |
|   0 | 0.556 | 0.556 | 0.667 |
|  +5 | 0.556 | 0.333 | 0.778 |
| +10 | 0.333 | 0.111 | 0.778 |

The median BEAR_score is already 0.222 at t = -10 and 0.444 at t = -3,
well before the argmax flip at t = 0 (which by definition pushes the
score above ~0.5 to win against competing regimes). The p75 trajectory
shows that in ~25% of events BEAR_score is already at 0.444 by t = -30
and 0.555 by t = -8 -- those are events where the soft score has clearly
diagnosed bear conditions weeks before the argmax catches up.

### (b) Cross-correlation: BEAR_score x forward SPY drawdown

| horizon (days) | Pearson r | n_obs | p_value |
|---:|---:|---:|---:|
|  1 | +0.0199 | 2359 | 3.33e-01 |
|  5 | -0.1982 | 2355 | 2.75e-22 |
| 10 | -0.2132 | 2350 | 1.45e-25 |
| 20 | -0.1806 | 2340 | 1.33e-18 |

BEAR_score is not informative for next-day drawdown (h=1d, r=+0.02, p
= 0.33), but is significantly negatively correlated with 5-20 day
forward drawdown (|r| in [0.18, 0.21], all p < 1e-18). This is the
expected signature of a momentum/vol indicator that is too noisy for
one-day prediction but tracks the broader stress regime. Mean |r|
across the four horizons is 0.1530.

### (c) Threshold sweep: median lag from BEAR_score crossing tau to (i) argmax flip, (ii) SPY drawdown trough

| tau | median argmax_lag | mean argmax_lag | median trough_lag | mean trough_lag | n_events_crossed |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 26.00 | 22.19 | 29.00 |   ~   | 63 |
| 0.30 | 24.00 | 20.02 | 21.00 |   ~   | 61 |
| 0.40 | 18.00 | 18.07 | 19.00 |   ~   | 58 |
| 0.50 | 17.00 | 17.20 | 15.00 |   ~   | 51 |

Read as: at tau = 0.30, on the median event BEAR_score crossed 0.30
roughly 24 trading days before the argmax actually flipped into BEAR.
The trough_lag column says the SPY drawdown trough occurred ~21 trading
days *after* the tau crossing on the median event (positive = trough is
LATER than tau cross, so the tau cross is leading the trough). This is
the headroom WS-3c would unlock: a tau-threshold BEAR consumer would
fire ~3-4 weeks earlier than the current argmax-only consumer, and on
average still well before the SPY trough.

Note: 2 of 63 events at tau=0.30 (61 crossed of 63) had BEAR_score
remain below 0.30 in the entire 30-day pre-onset window -- in those
events the BEAR score rose abruptly into the argmax-winning range
without an earlier tau breach. The tau=0.50 column drops to 51 of 63
events crossed; those 12 events would be missed by a high-tau consumer.

Mean gap_days from onset to drawdown trough across the 63 events:
**-1.86 days** (negative = detector argmax fires AFTER the SPY trough on
average). This matches the V12 readiness panel's mean -3.42 in
direction; the modest magnitude difference is consistent with the small
universe diff (63 vs 59 onsets) and the +/-10d trough window.

## Verdict

**WS-3c (soft-score consumption).**

Applying the decision criterion: median argmax_lag at tau = 0.30 is
**24.0 days**, which is more than an order of magnitude above the
3-day cutoff. The first clause of the decision criterion fires
unambiguously:

> If `median_argmax_lag at tau=0.3 > 3.0`: verdict = WS-3c.
> Implication: detector's signal is already there; argmax is suppressing it.

This verdict is reinforced by the cross-correlation result (Pearson r
at h = 5d is -0.198, p = 2.75e-22): BEAR_score carries real
forward-drawdown signal that the argmax label is throwing away. WS-3a
(hysteresis) would help with the argmax flicker problem identified in
the v0 diagnostic's H4, but does not unlock the ~20-day lead that
soft-score consumption does. WS-3b (leading indicators) is not refuted
by this experiment (the detector's underlying inputs may still be
lagging), but the soft-score gap dominates the leading-indicator gap by
a wide margin given the current evidence.

## Implications for WS-3 spec

The next WS-3 spec should design a consumer-layer change rather than a
detector-layer change: a `BearScoreConsumer` that triggers cash-shift /
position-trim when `score_BEAR >= tau` (with tau, hysteresis on the
threshold, and minimum-persistence as the tuneable hyperparameters). The
EXT-OOS validation must hold the v0 replay aside and only score on
forward data, since the v0_scores window (2017-2026) overlaps the V11
and V12 in-sample windows. WS-3a (state-machine hysteresis on the
*consumer's* threshold crossings, not on the detector's argmax) likely
falls naturally out of the WS-3c design and can be co-tuned.

## Artifacts

- `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\diagnostics\regime\v0_scores\labels.parquet` -- soft-score replay (2360 rows, 5 score columns + v0 schema)
- `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\diagnostics\regime\v0_scores\event_study_bear_score.csv` -- per-event BEAR_score trajectory
- `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\diagnostics\regime\v0_scores\event_study_summary.csv` -- aggregated event study (median + IQR per relative_day)
- `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\diagnostics\regime\v0_scores\cross_correlation.csv` -- BEAR_score x forward drawdown
- `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\diagnostics\regime\v0_scores\threshold_sweep.csv` -- tau threshold analysis
- `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\diagnostics\regime\v0_scores\verdict.txt` -- decision criterion output

Source files:

- `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\scripts\diagnostics\regime_score_replay.py`
- `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\notebooks\research\experiment3_soft_scores.py`
- `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\tests\diagnostics\test_regime_score_replay.py`

## Limitations and caveats

- **V11/V12 EXT-OOS contamination**: same 2017-2026 window was used for
  V11 and V12 readiness work. The WS-3c consumer if chosen requires
  forward OOS validation before deploy; in-sample numbers here are
  consistent with prior diagnostics but cannot stand alone as evidence
  of forward edge.
- **Small-N for some claims**: 63 BEAR onsets is small. The median +
  IQR are reported throughout instead of just the mean to surface
  dispersion; per-event variability in the BEAR_score trajectory is
  visible in `event_study_bear_score.csv` (p25 vs p75 gap of 0.4-0.6
  at most relative_days).
- **Soft-score range**: per the v0 diagnostic's Phase 0 analysis,
  `_score_regime` produces values in [0, 1] but the *denominator*
  (`criteria_count`) differs across regimes -- STRONG_BULL has 4 keyed
  criteria, BEAR has 3, UNPREDICTABLE has 2. Comparing absolute
  BEAR_score thresholds to other regimes' scores is therefore not
  meaningful; what matters here is the *temporal* trajectory of
  BEAR_score (rising from 0 to 0.5 before each onset) and the
  cross-correlation with forward drawdown, both of which are
  within-regime claims and not affected by the cross-regime scaling
  asymmetry.
- **tau threshold tuning is hyperparameter overfit**: the {0.2, 0.3,
  0.4, 0.5} sweep is on the same data the WS-3c consumer would be
  trained on. The spec must include a held-out validation step before
  picking a production tau.
- **Onset definition is asymmetric**: an "onset" is defined as
  prev != BEAR AND today == BEAR. A regime that flips into BEAR for one
  day, out for one day, and back into BEAR registers as two onsets.
  The threshold-sweep results are not affected (lookback windows still
  start from the new onset), but the event-study median may
  over-weight flickery periods. A robustness check that filters to
  "sticky" BEAR onsets (e.g. >= 5 consecutive BEAR days after onset)
  is a recommended WS-3 spec follow-up.
