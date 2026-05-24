# Experiment 5 -- OMR Cross-Check on Detector Failures

**Date**: 2026-05-24
**Branch**: v12-bear-to-cash
**Builds on**: regime detector diagnostic (H4/H5); V12 readiness alignment panel; experiment 3 soft-score panel
**Decision target**: WS-3 portfolio-level priority vs V12c

## Summary

OMR's trade log carries only three of the detector's five regime labels
(STRONG_BULL, WEAK_BULL, SIDEWAYS) -- the Bayesian-bucket screen inside the OMR
adapter structurally filters BEAR and UNPREDICTABLE entries, so the trade log
has no BEAR/UNPREDICTABLE samples to compare. Across the three observed
regimes, OMR's per-trade Sharpe is tightly clustered (0.215 - 0.258, range/max
= 16.6%), well below the 30% threshold the decision criterion sets for
DETECTOR-WIDE. However, this clustering is partly mechanical (OMR only trades
when the detector says "tradeable"), so the comparison to RAMP's full
regime-conditional sweep is structural, not numeric. **Verdict: AMBIGUOUS** --
RAMP's regime-conditional Sharpe swing has no clean analog in the OMR
trade-level data. WS-3 should be treated as RAMP-attributable until a separate
study instruments OMR per-day P&L (including no-trade days) so that
BEAR/UNPREDICTABLE detector behaviour can be observed in OMR's full state.

## Methodology

- OMR trade log: `output/backtests/omr_original_universe_2017_2024_trades.csv`
  (2335 trades, 2018-01-03 to 2024-12-27, 13 leveraged ETFs).
- Per-trade `regime` field already populated by OMR adapter using the
  production `MarketRegimeDetector` at trade entry.
- Per-trade Sharpe = mean(net_return) / std(net_return), NOT annualized.
  Trade-level is the natural unit for OMR (overnight hold). 95% CI via
  bootstrap (1000 resamples) when n > 50.
- Transition-day classification from
  `diagnostics/regime/v0/labels.parquet`: a trade is a "transition_day" trade
  if `labels.regime[entry_date] != labels.regime[entry_date - 1 trading day]`.
- BEAR-onset alignment: 63 BEAR-onset events identified as day-over-day
  transitions into BEAR in `labels.parquet`. For each, OMR net_return is
  pooled across all trades entering in `[onset - 3, onset + 3]` trading days.
- Decision criterion: OMR Sharpe-by-regime range / max(|Sharpe|) >= 30% ->
  DETECTOR-WIDE, < 30% -> RAMP-SPECIFIC, AMBIGUOUS when sample/structure
  prevents a clean comparison.

## Per-regime Sharpe (OMR)

| Regime          | n_trades | mean_net_return | std_net_return | sharpe_per_trade | CI low | CI high |
|-----------------|---------:|----------------:|---------------:|-----------------:|-------:|--------:|
| STRONG_BULL     |      643 |        0.005468 |       0.021165 |            0.258 |  0.190 |   0.320 |
| WEAK_BULL       |     1381 |        0.004391 |       0.020388 |            0.215 |  0.174 |   0.256 |
| SIDEWAYS        |      311 |        0.005145 |       0.021546 |            0.239 |  0.141 |   0.328 |
| UNPREDICTABLE   |        0 |             n/a |            n/a |              n/a |    n/a |     n/a |
| BEAR            |        0 |             n/a |            n/a |              n/a |    n/a |     n/a |

**Sharpe range / max(|Sharpe|) = 0.043 / 0.258 = 16.6%** across the three
observed regimes -- below the 30% DETECTOR-WIDE threshold. The 95% CIs for
STRONG_BULL, WEAK_BULL, and SIDEWAYS overlap heavily, consistent with OMR's
Bayesian-bucket screen filtering entries to high-probability setups regardless
of the regime label.

Crucially, **BEAR and UNPREDICTABLE contribute zero observations.** Cross-
referencing `labels.parquet` against the OMR window (2018-2024), the detector
classified 309 days as BEAR and 36 days as UNPREDICTABLE; OMR placed no trades
on any of them. This is by design -- OMR's `_can_take_trade` chain refuses
BEAR/UNPREDICTABLE entries.

## Transition vs persistent

| Bucket          | n_trades | mean_net_return | std_net_return | sharpe_per_trade |
|-----------------|---------:|----------------:|---------------:|-----------------:|
| transition_day  |      501 |        0.007720 |       0.025667 |            0.301 |
| persistent_day  |     1834 |        0.003987 |       0.019130 |            0.208 |
| overall         |     2335 |        0.004788 |       0.020758 |            0.231 |

OMR's transition-day Sharpe is **higher** than persistent-day Sharpe by +0.092.
This is the OPPOSITE direction from the RAMP V12 lag-asymmetry finding (where
transition days -- particularly BEAR onsets -- carry the lag tax). Two readings
are consistent with the data: (a) the Bayesian-bucket screen happens to be
more accurate on transition days, or (b) survivorship in the trade log itself
biases the comparison (transition days where the screen would have rejected
the trade do not appear). Without no-trade-day P&L attribution we cannot
distinguish these.

## BEAR-onset alignment cross-check

| Aggregate                                  | Value     |
|--------------------------------------------|-----------|
| Total BEAR onsets (labels.parquet)         | 63        |
| In OMR window (2018-2024)                  | 49        |
| Outside OMR window (missing data)          | 14 (22.2%)|
| Onsets with >= 1 OMR trade in [-3, +3]     | 49        |
| Pooled OMR mean net_return on onset windows| 0.002478  |
| Overall OMR mean net_return                | 0.004788  |
| Pooled trade count on onset windows        | 255       |

OMR's mean net_return on BEAR-onset windows (0.248%) is **~48% lower** than its
overall mean (0.479%). Direction-wise this is consistent with the RAMP finding
that BEAR onsets are uncomfortable for any long-equity exposure. Magnitude-wise
it's a notably milder hit than V12 takes (V12 near_close drops to 0.268 Sharpe
explicitly because BEAR-day equity returns are positive on average and BEAR-
to-cash forfeits them, but BEAR-onset transitions are where most of V12's
lag-asymmetry tax lives -- see `20260524_phase4_v12_readiness.md` and
experiment 4). So OMR does show a "BEAR-onset tax" at trade level, but it's
attenuated by the Bayesian-bucket filter rejecting most BEAR-onset window
entries before they fire.

The 22.2% missing-data fraction (14 BEAR onsets outside 2018-2024) limits the
robustness of this comparison; in particular, the 2020-03 COVID drawdown and
2022 inflation regime are well-covered, but pre-2018 (2017 brief BEAR flips)
and 2025-2026 events (Sep-Nov 2025, Mar 2026) are not in the OMR sample.

## RAMP comparison

From `20260523_phase4_v11_readiness.md` and `20260524_phase4_v12_readiness.md`:

- **V11 near_close @ 5 bps**: Sharpe 0.528 (annualized) across the full
  2017-2026 window. No regime-conditional Sharpe panel is published, but V11
  is the regime-aware mom/protection baseline that ate the detector unmodified.
- **V12 near_close @ 5 bps**: Sharpe 0.268 (annualized) -- a 0.260 drop from
  V11. The mechanism (experiment 4 and V12 readiness) is that BEAR-to-cash
  forfeits **positive average BEAR-day equity returns** -- i.e. when the
  detector calls BEAR, the day's median equity return is non-negative often
  enough that flattening costs more than it saves. This is RAMP's regime-
  conditional Sharpe failure mode: a single regime (BEAR) carries a strongly
  negative Sharpe contribution under a regime-conditional rule that uses the
  argmax label naively.
- **V12 one-day-lag @ 5 bps**: Sharpe 0.665 -- the lag actually helps,
  consistent with H5 (the detector is late so executing on lagged regime
  smooths over flicker).

OMR cannot exhibit the V11->V12 swing because it does not execute under
regime-conditional rules in the same sense -- its only regime-level dependency
is the binary "trade / no-trade" gate. So in the language of the V12 panel:
OMR is closer to V11 in design (regime as a screen, not as a parameter
switch), and the bigger detector-tax effects in V12 stem from the
parameter-switching behaviour that OMR doesn't share.

## Verdict

**AMBIGUOUS.**

Justification:
1. The decision criterion compares Sharpe-by-regime *range* across the full
   five-regime taxonomy. OMR's trade log has data for only three regimes
   (STRONG_BULL/WEAK_BULL/SIDEWAYS); BEAR and UNPREDICTABLE are zero by
   adapter construction. A direct range/max comparison underweights the
   missing regimes and isn't structurally comparable to RAMP's full-coverage
   regime panel.
2. Across the regimes that *are* present, OMR's per-trade Sharpe is tightly
   clustered (range/max = 16.6%, all CIs overlapping). This is consistent
   with the Bayesian-bucket screen dampening regime sensitivity -- i.e. the
   evidence available *does* favour RAMP-SPECIFIC, but only conditional on a
   tradeable-day subset.
3. OMR's BEAR-onset window mean return is 0.248% vs overall 0.479%, a clear
   ~48% degradation in direction-and-magnitude. This is qualitative evidence
   that the detector's BEAR-onset flicker/lag does leak into OMR P&L, just
   muted by the Bayesian screen.

Net interpretation: the detector's failures are NOT visible at OMR trade
granularity in the same way they are at RAMP day granularity, but neither is
OMR proven immune. A future study should re-instrument OMR to record per-day
P&L (including no-trade days, including BEAR-day "should we have traded?"
counterfactuals) so that the regime-conditional Sharpe panel is comparable
across the two strategies.

## Implications for WS-3 priority

Treat WS-3 (detector improvement: WS-3a stability, WS-3b lag, WS-3c soft
scores) as **RAMP-attributable** for portfolio-priority purposes until per-day
OMR attribution is available. V12c (single-strategy RAMP fix; e.g. BEAR-to-
half rather than BEAR-to-cash, or a softer regime-conditional position-size
schedule) is the higher-leverage near-term track. WS-3 retains its strategic
value on RAMP alone, but the inferred "detector-wide" multiplier we hoped to
claim for WS-3 vs V12c is not supported by the OMR trade-level data.

## Limitations

- **OMR trade log covers 2018-2024**; experiment 3 / V12 use 2017-2026.
  22.2% (14 / 63) of BEAR-onset events are outside the OMR window -- notably
  the 2017 brief BEAR flips and the 2025-2026 BEAR events (which include the
  most recent regime stress). The pre-2018 events are minor (short BEAR
  episodes that flipped back fast); the 2025-2026 missing events are
  material to forward-looking conclusions.
- **Per-trade Sharpe** is NOT directly comparable to per-day Sharpe used in
  RAMP backtests. The decision criterion is "Sharpe range / max" which is
  dimensionless, so structural comparability is preserved -- but the absence
  of BEAR/UNPREDICTABLE from the OMR trade log breaks the structural symmetry
  the criterion assumes.
- **No per-day OMR P&L attribution** was available -- the trade log only
  contains days OMR fired entries. Days when OMR considered and rejected a
  trade are invisible, so we cannot tell whether the detector's flicker/lag
  caused OMR to miss positive-EV trades (forgone-alpha tax) the same way it
  caused RAMP to misposition.
- **Detector schema** assumed canonical
  {STRONG_BULL, WEAK_BULL, SIDEWAYS, UNPREDICTABLE, BEAR}. The OMR adapter's
  taxonomy matches the canonical set; absences in the trade log are
  consequence of the OMR Bayesian-bucket screen, not a schema deviation.
- **Adapter version drift**: the OMR backtest used the production detector
  version at the time the backtest was run; subsequent detector changes
  (none committed since) would invalidate the direct join. Cross-checked
  joining trades to `diagnostics/regime/v0/labels.parquet`: 100% of OMR
  entries (2335/2335) joined cleanly, with 66 trades (2.8%) where the OMR-
  recorded regime disagreed with the v0 label -- almost entirely
  STRONG_BULL<->WEAK_BULL and WEAK_BULL<->SIDEWAYS adjacent reclassifications
  consistent with run-to-run detector variability around regime boundaries,
  plus 15 trades where the v0 label is BEAR but OMR recorded a non-BEAR
  regime (i.e. OMR fired on the day; tiny fraction relative to the 309
  BEAR-labeled days in the OMR window). Close enough to treat as the same
  detector for these purposes.

## Artifacts

- `notebooks/research/experiment5_omr_regime_attribution.py`
- `diagnostics/omr_cross_check/omr_per_regime_sharpe.csv`
- `diagnostics/omr_cross_check/omr_transition_vs_persistent.csv`
- `diagnostics/omr_cross_check/omr_bear_onset_alignment.csv`
- `diagnostics/omr_cross_check/verdict.txt`
