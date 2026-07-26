# FX OHLC / Range-Based Wave: Results

**Date:** 2026-07-25 | **Status:** CLOSED -- all 4 pre-registered trials FAIL
**Pre-registration:** `docs/strategies/research/20260725_fx_ohlc_wave_preregistration.md` (LOCKED 2026-07-25)
**Working copy:** `docs/reports/fx/ohlc_wave_gate.md`

## 1. Summary

All four pre-registered range-based FX specs FAIL the pre-committed combined gate.
The registered prediction (pre-reg Section 6: "all four FAIL") is confirmed.

These were the first FX signals in the campaign that required the intraday RANGE
(high/low), and were literally unexpressible before the `wants_ohlc` engine change
of 2026-07-25 -- the engine previously discarded open/high/low before calling the
strategy. Specs 1-3 (Keltner reversion, bandwidth squeeze, vol-spike fade) are
GENUINELY NEW mechanisms. Spec 4 (ADX-gated trend) is explicitly an ENHANCEMENT of
the already-failed #3/FxTrend and is reported as such.

Three of four produce NEGATIVE OOS Sharpe that widens further negative at 1.5x cost
-- a genuine directional failure, not a marginal edge nudged under by friction. The
single positive spec (OHLC-VOLSPIKE, +0.0873) is economically trivial and fails
decisively on DSR = 0.0000 against a deflated bar of SR_zero = 1.129.

No parameter was swept, no threshold tuned, no pair or window dropped, and N was
never reduced. This wave IS the entire search for this family.

## 2. Per-trial results

Walk-forward: train 36m / test 12m / step 12m, 13 non-overlapping OOS windows,
2014-01-01 .. 2026-04-30 OOS span, 3,217 stitched OOS days. Universe: G10-22
(22 pairs), identical for all four specs. `execution_lag=1`, taker costs, weekly
rebalance, vol target 0.03/instrument, IDM on. PSR/DSR with `periods_per_year=252`.

| # | Trial | Catalog | Sharpe 1x | Sharpe 1.5x | PSR | DSR | PBO | N used | SR_zero | S&P corr | IR vs S&P | Windows | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | OHLC-KELTNER | #12 | **-0.4762** | -0.6621 | 0.0459 | 0.0000 | 0.6806 | 137 | 1.129 | +0.141 | -0.877 | 13 | **FAIL** |
| 2 | OHLC-SQUEEZE | #27 | **-0.4783** | -0.6573 | 0.0437 | 0.0000 | 0.6352 | 138 | 1.130 | -0.105 | -0.783 | 13 | **FAIL** |
| 3 | OHLC-VOLSPIKE | #29 | **+0.0873** | +0.0114 | 0.6235 | 0.0000 | 0.2448 | 139 | 1.131 | +0.116 | -0.684 | 13 | **FAIL** |
| 4 | OHLC-ADX-TREND | #6 | **-0.3089** | -0.4234 | 0.1357 | 0.0000 | 0.2727 | 140 | 1.132 | -0.118 | -0.605 | 13 | **FAIL** |

Gate (pre-committed, methodology Section 2.5): OOS Sharpe > 0 AND positive at 1.5x
cost AND PSR >= 0.95 AND DSR >= 0.95 AND PBO < 0.5, plus the S&P benchmark /
marginal-contribution check.

S&P 500 Sharpe over the same OOS dates = **0.6842** (3,080 aligned days). No spec
comes close to it, and every spec has a strongly negative information ratio against
it, so marginal book contribution is negative for all four. None proceeds to
book-level evaluation. The marginal deflated contribution proxy `DSR x (1 - corr^2)`
is 0.0000 for all four, since DSR is 0.0000 in every case.

### Failure reasons, per trial

1. **OHLC-KELTNER (#12) -- FAIL.** OOS Sharpe -0.4762, widening to -0.6621 at 1.5x
   cost. PSR 0.0459, DSR 0.0000, PBO 0.6806 (the worst of the wave; the window
   ranking is unstable under resampling). Negative in 10 of 13 windows. Fading
   deviations from an EMA(20) center with 2.0 x ATR(10) bands is a directional loser
   on daily G10 spot before deflation even matters.
2. **OHLC-SQUEEZE (#27) -- FAIL.** OOS Sharpe -0.4783, -0.6573 at 1.5x. PSR 0.0437,
   DSR 0.0000, PBO 0.6352. Negative in 9 of 13 windows. Volatility compression does
   precede expansion, but the direction of the release is not predicted by the sign
   of the trailing 20d close change -- the breakout leg is a coin flip paid for with
   spread.
3. **OHLC-VOLSPIKE (#29) -- FAIL (the only positive spec).** OOS Sharpe +0.0873,
   surviving 1.5x cost at +0.0114 -- i.e. cost stress nearly erases it but does not
   flip the sign. It fails on PSR 0.6235 (< 0.95) and decisively on DSR 0.0000: the
   deflated bar at N=139 is SR_zero = 1.131 and the realized +0.087 is more than an
   order of magnitude below it. PBO 0.2448 is the only comfortably acceptable PBO in
   the wave, but PBO alone is not a pass. This spec is deliberately sparse -- it
   gates on Parkinson-RV z > 2 and therefore fires on only ~13% of days, producing
   just 594 OOS fills against 8,723 for the squeeze and 12,171 for the Keltner. The
   per-window Sharpe dispersion is correspondingly enormous (-1.90 to +1.93, with
   window 7 undefined because the gate never fired in it). A +0.087 aggregate built
   from 13 windows with that spread is indistinguishable from noise, which is exactly
   what PSR 0.62 says. **This is a trivial edge, not a cost-destroyed one -- and it
   is not a lead.**
4. **OHLC-ADX-TREND (#6) -- FAIL, and no rescue of the baseline.** See Section 3.

## 3. Did the ADX enhancement improve the FxTrend baseline it filters?

**No -- it degraded it.** This must not be read as a fresh mechanism; the pre-reg
labels it an ENHANCEMENT of #3/FxTrend, which already failed.

| | FxTrend baseline (#3) | ADX-gated (#6) |
|---|---|---|
| OOS Sharpe (1x) | -0.02 | **-0.3089** |
| OOS Sharpe (1.5x) | n/a (baseline row) | -0.4234 |
| DSR | 0.20 | **0.0000** |
| PBO | 0.85 | 0.2727 |
| Windows | 13 | 13 |

Gating the Carver EWMAC forecast to zero when ADX(14) < 25 made the strategy
substantially WORSE (-0.31 vs -0.02) and drove DSR from an already-failing 0.20 to
0.0000. The one metric that improved is PBO (0.85 -> 0.27), which is not a
performance improvement: suppressing the forecast on most days makes the surviving
window ranking more stable while removing the exposure that could have produced a
return.

Per the pre-committed rule for baseline comparisons, the FxTrend numbers are CITED
from the existing catalog record (`FX_60_CATALOG_TRACKER.md` row #3: OOS -0.02,
DSR 0.20, PBO 0.85, IDM on, 13 windows) rather than re-run, so the baseline does not
consume a trial. **Caveat, stated plainly:** the baseline was measured under an
earlier apparatus vintage (it pre-dates the 2026-07-25 PSR/DSR unit fix and the
associated hardening) and its universe/date span is not identical to this wave's, so
the -0.02 -> -0.31 delta is NOT a controlled A/B and the exact magnitude should not
be quoted as one. The qualitative conclusion is nonetheless robust: both are
decisively negative, both fail DSR, and the ADX filter does not rescue the trend
mechanism. Per the pre-reg, we do NOT now try a different ADX threshold.

## 4. Trial count / deflation accounting

Cumulative project-wide N: **137 -> 141** (4 trials, one per spec). Each run queried
the registry before appending its own row, so the N in force per trial was 137, 138,
139, 140 respectively, and the registry ended at 141. Verified directly against
`output/experiments.duckdb` via
`src.backtesting.walkforward_common.get_campaign_trial_distribution()`.

Deflated bar, using the realized cross-trial spread v = 0.42778 measured in the Tier
B wave:

| N | SR_zero (annualized) |
|---|---|
| 137 | 1.1291 |
| 138 | 1.1302 |
| 139 | 1.1312 |
| 140 | 1.1322 |
| 141 | 1.1333 |

The pre-registered prediction of a ~1.13 bar is confirmed to 3 decimal places. The
wave's best result (+0.0873) is 7.5% of the bar. N was never reduced to help a spec
pass; the deflation was applied at the full, growing project-wide count.

The S&P benchmark leg was computed read-only (`register=False`, no FillSink) on a
recomputation of the same stitched OOS series, so it appended nothing to the
registry and N remains 141 -- a diagnostic is not a trial.

## 5. Fills verification (methodology Section 12)

Every run persisted its fills. Verified non-empty `trades_oos.csv.gz` (the OOS
concatenation matching the gated return series) plus a 53-artifact `manifest.csv`
under each run-scoped sink, BEFORE any verdict was accepted:

| Trial | Run ID | OOS fills | Manifest artifacts |
|---|---|---|---|
| KELTNER | `output/backtests/FxOhlcKeltner/runs/20260726T030804Z_62bcb1` | 12,171 | 53 |
| SQUEEZE | `output/backtests/FxOhlcSqueeze/runs/20260726T030831Z_8ee250` | 8,723 | 53 |
| VOLSPIKE | `output/backtests/FxOhlcVolSpike/runs/20260726T030850Z_ed1de3` | 594 | 53 |
| ADX-TREND | `output/backtests/FxOhlcAdxTrend/runs/20260726T030915Z_3e40de` | 6,999 | 53 |

## 6. Apparatus notes

Reported, not redone (verified by the orchestrator during the build):
indicators check out on real data (ATR(10) = 0.00712, ~71 pips EURUSD; ADX(14) =
27.8; Parkinson RV = 5.5% annualized); no-lookahead confirmed end-to-end by
perturbing every bar after index 3000 and observing all earlier forecasts unchanged;
indicator-level causality is unit-tested. These runs used the hardened apparatus
(`execution_lag=1`, un-truncated PBO, publication-lagged FRED rates, silent-skip
fixed, unit-correct PSR/DSR at `periods_per_year=252`). Nothing was disabled.

Two observations filed, neither affecting these verdicts (nothing passed):

1. **`run_fx_walkforward.py` does not compute the S&P benchmark leg** that
   `run_fx_wave2_gate.py` does -- no `correlation_sp500`, `information_ratio_sp500`,
   or `marginal_deflated_contribution_proxy` in its result dict or JSON output. The
   pre-registered gate REQUIRES the S&P benchmark / marginal-contribution check, so
   the walk-forward runner alone cannot adjudicate the full gate. It was computed
   separately here (read-only). The two runners should be reconciled so the benchmark
   leg is not dependent on the adjudicator remembering to add it. Per the integrity
   rule, this was FILED rather than patched mid-adjudication.
2. **`run_fx_walkforward.py` defaults `--report` to the shared baseline path**
   `docs/reports/fx/FX_WALK_FORWARD.md`, so back-to-back gate runs silently clobber
   each other's report and the FxTrend baseline report. Worked around by passing an
   explicit per-trial `--report` path; the default should be made run-scoped.

Minor, expected-by-design: OHLC-VOLSPIKE window 7 has an undefined (nan) window
Sharpe because its z > 2 gate never fired in that window, and the Keltner PBO
computation dropped 1 stub window (12 columns x 260 rows). Neither is a defect.

## 7. Scoped conclusion and stopping rule

Per the pre-registration's stopping rule (Section 7), all four failing means:

**The range-based (ATR / ADX / Parkinson) signal family is unproductive FOR THIS
DAILY-SPOT-TAKER CONSTRUCTION.** The family STOPS here: no parameter sweep, no
alternative ATR multiple, ADX threshold, or z-window, no ML variant.

**Scope of this negative, stated exactly** (CLAUDE.md North Star, and matching the
SCOPE banner in `FX_60_CATALOG_TRACKER.md`): what has been shown is that these four
specific constructions -- retail-accessible, DAILY frequency, SPOT, spread-TAKER,
weekly-rebalanced, on the G10-22 universe -- do not clear the gate. This is **NOT**
a claim that range-based signals have no edge, that ATR/ADX/Parkinson are
uninformative, or that FX has no edge. Range and volatility structure is real and
measurable (the indicator pre-checks above confirm the quantities are correctly
computed on real data); what fails is monetizing it at this frequency, in this
execution style, at these costs. The most plausible untested mis-specifications
remain the ones named in the campaign's scope banner: the cost side (taker-only --
these mechanisms are all spread-paying and were never tested as a liquidity
PROVIDER), and frequency (range/volatility structure is most exploitable
intraday, and daily sampling is structurally blind to it).

Notably, the range-based family is the one where the frequency objection bites
hardest: a "range" measured once per day discards essentially all of the intraday
path that the mechanism is nominally about.

**Remaining catalog blockers are now INTRADAY (21 strategies) and ML (6
strategies), both substantial builds.** No further daily-spot FX specs are unblocked
by this wave's engine work.

No spec passed, so no book-level marginal-contribution evaluation is triggered.

## 8. Integrity self-audit

- Specs, params, universe, gate and trial count were fixed in a LOCKED pre-reg
  BEFORE any backtest; none was edited afterwards.
- Search size = exactly 4 specs. No spec was run more than once for a verdict; the
  only re-execution was a read-only, non-registering benchmark recomputation.
- No parameter swept, no threshold tuned, no pair or window dropped, no date range
  adjusted, no cost model softened, no gate threshold moved, N never reduced.
- One universe (G10-22) for all four specs, chosen a priori, precisely so that no
  per-spec universe selection could occur.
- The outcome matches the registered prediction (all four FAIL), so there is no
  post-hoc rationalization to audit -- and the one positive point estimate
  (VOLSPIKE +0.087) is reported as a FAIL rather than promoted to a "promising
  lead", which is what it would take three degrees of freedom to become.
- The ADX spec is reported as an ENHANCEMENT of a failed mechanism, not as a new
  one, and its modest PBO improvement is explicitly NOT presented as progress.
