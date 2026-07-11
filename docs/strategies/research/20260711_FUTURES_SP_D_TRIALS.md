# Futures SP-D + VRP Finalization: Trial Ledger - 2026-07-11

Sub-project D of the Futures Strategy Testability Campaign, plus a shared-gate
deflation fix. Two questions: (1) does the VIX #26 roll-down verdict survive HONEST
multiple-testing deflation, and (2) is the #28 realized-vs-implied VRP a distinct
premium or a re-expression of #26? All numbers are real gate output, recorded verbatim.

**Gate (methodology Section 2.5):** PASS requires PSR >= 0.95 AND DSR >= 0.95 AND
PBO < 0.25. **Benchmark:** carry_idm OOS Sharpe 0.765 (the incumbent).

## Headline

**Once the DSR is honestly deflated, NOTHING in the futures book clears the gate --
including the incumbent carry.** The VRP family is dead: #26 fails deflation with a
catastrophic tail, and #28 is a degenerate re-expression of #26 with no distinct edge.

## P0 -- the gate was not actually deflating (the core fix)

Two defects made the "deflated" Sharpe ratio inert; both are now fixed
(`src/backtesting/walkforward_common.py`):

1. **Trial count was 1.** `TRIAL_COUNT_PARAMETER_FREE = 1` -> no multiple-testing
   deflation at all. Replaced with `CAMPAIGN_CUMULATIVE_TRIALS = 40` (counted from the
   SP-A/E/B/C ledgers + verdict sweep).
2. **Trial-Sharpe distribution was a single element.** The gate called
   `dsr(sharpe, [sharpe], ...)`, and `expected_max_sharpe` returns 0 for a <2-element
   list -> the deflation benchmark SR_zero was 0, so DSR == PSR regardless of N.
   Fixed by threading `CAMPAIGN_TRIAL_SHARPES` (29 real OOS Sharpes from the ledgers,
   variance 0.112) into the gate. This yields **SR_zero = 0.733** (the expected-max
   Sharpe of 40 trials under the null): a strategy must clear ~0.73 with 95%
   confidence to pass DSR.
3. **PBO NaN.** `_compute_pbo` NaN-ed whenever the shortest OOS window had < 2*s (32)
   rows. Raised the drop threshold to 2*s so a short trailing window is excluded, not
   NaN-ing the whole statistic.

This is a SHARED-gate change (all sleeves now deflate) but only lowers DSR, so no
prior PASS flips.

## Verdicts (deflated)

| Sleeve | OOS Sharpe | PSR | DSR (deflated) | PBO | Verdict |
|--------|-----------|-----|----------------|-----|---------|
| #26 VIX roll-down | 0.564 | 1.00 | 8.88e-06 | 0.613 | FAIL |
| #28 VRP (short-VX1) | 0.055 | 0.972 | 1.5e-124 | 0.363 | FAIL |
| carry (incumbent, indicative*) | 0.588 | 1.00 | 5.4e-14 | 0.093 | FAIL (DSR) / PASS (PBO) |

Deflation benchmark SR_zero = 0.733 for all three.

### #26 -- VIX futures term-structure roll-down: deflated FAIL
The +0.564 OOS Sharpe (real positive VRP, confirmed in SP-E) does NOT survive honest
deflation: its edge (0.564) sits BELOW the 40-trial benchmark (0.733), so DSR
collapses to 8.88e-06. PBO is now a real 0.613 (was spuriously NaN) -- also a fail.
Construction frozen from SP-E (roll-masked, causal, backwardation kill-switch); no
sign or window changes.

**Tail audit (the independent disqualifier):** skew -2.22, kurtosis 15.6, worst day
-47.9%, **max drawdown -81.1%**, 2018-02 (Volmageddon) realized -12.6%, 2020-03
(COVID) flat (the kill-switch went flat during the inversion). A short-vol sleeve
with an 81% drawdown is not a carry-book addition regardless of Sharpe.

### #28 -- realized-vs-implied VRP on ES: FAIL and a re-expression of #26
- **IV extraction validated:** ES ATM-IV (Black-76 on the most-active near-ATM ES
  option prints, underlying = RAW front-future close, rate = FRED DFF) correlates
  0.828 with the VIX front (median ratio 0.71) -- passes the pre-registered validation
  gate. HAR realized-vol forecast (Corsi, causal, within-session RV) median ES
  annualized RV-vol 0.103.
- **Gate:** VRP = IV_ATM - E[RV]_HAR, percentile-sized short-VX1 stream. OOS Sharpe
  0.055 (essentially zero), DSR ~0, PBO 0.363 -> FAIL.
- **Re-expression check (mandatory):** correlation to the #26 stream 0.479; marginal
  Sharpe of #28's residual after regressing out #26 = **0.015**. #28 adds no distinct
  edge -- it is a weaker re-expression of #26 (which itself already failed). The
  VRP-gap signal does not beat #26's curve-slope signal.

### carry -- the incumbent, under honest deflation
Gating the FuturesCarry 2010-2026 equity through the deflated gate: OOS Sharpe 0.588,
DSR 5.4e-14 (FAIL), PBO 0.093 (PASS -- not overfit), SR_zero 0.733. carry's edge sits
at/below the multiple-testing-honest benchmark, so it does NOT clear DSR >= 0.95.
*Indicative: this equity run's Sharpe (0.588) differs from the carry_idm walk-forward
figure (0.765); both are at/below the 0.733 knife-edge. carry remains the best
DEPLOYABLE book (real cash-and-carry mechanism, passes the non-overfitting PBO check),
but the campaign's 40-trial search cannot statistically distinguish its edge from the
best-of-40-under-the-null with 95% confidence.

## What this means
Honest multiple-testing deflation is a high bar (SR_zero 0.733 after 40 trials). The
futures campaign has surfaced real mechanisms (carry, VRP) but no sleeve whose
out-of-sample edge clears that bar with confidence. This is the completed objective:
report the honest degradation, do not engineer around it. The VRP family (#26, #28) is
closed as a negative result -- deflated FAILs, catastrophic short-vol tail, and #28
non-distinct from #26.

## Reproducibility
- Gate constants: `CAMPAIGN_CUMULATIVE_TRIALS = 40`, `CAMPAIGN_TRIAL_SHARPES` (29
  documented values, var 0.112), SR_zero 0.733, in `src/backtesting/walkforward_common.py`.
- #26 stream + gate.json: `output/backtests/futures/sp_d_vix26/` (from the smoke).
- #28 stream + gate.json: `output/backtests/futures/sp_d_vrp/ES/`.
- New module: `src/backtesting/vol/` (option_symbol, atm_iv, har_rv, vrp_strategy).
