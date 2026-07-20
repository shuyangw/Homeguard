# FX Catalog Campaign -- Wave 2 Resolution

**Date:** 2026-07-19
**Pre-registration:** `docs/superpowers/specs/2026-07-19-fx-wave2-selection-design.md`
Section 6 (stopping rule, locked before any Wave 2 result).

## The 6 Wave 2 verdicts

| # | Strategy | Mechanism | OOS Sharpe (1x/1.5x) | PSR | DSR | PBO | N | S&P corr | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 33 | Turn-of-Month USD | Seasonal | -0.28 / -0.36 | 0.00 | 0.00 | 0.84 | 104 | 0.03 | REJECT |
| 39 | PCA Dollar-Factor Residual | Statistical residual | -0.12 / -0.22 | 0.00 | 0.00 | 0.38 | 105 | 0.02 | REJECT |
| 42 | RORO Regime Spread | Macro regime | 0.06 / -0.03 | 1.00 | 0.00 | 0.17 | 106 | 0.00 | WEAK (fails 1.5x cost + DSR) |
| 35 | AUD/NZD pairs | Cointegration RV | -0.24 / -0.30 | 0.00 | 0.00 | 0.82 | 109 | 0.04 | REJECT |
| 37 | Cointegration scanner | Cointegration RV | -0.24 / -0.31 | 0.00 | 0.00 | 0.45 | 110 | -0.01 | REJECT |
| 30 | Relative-vol pair (XAU/XAG) | Vol-differential RV | -0.48 / -0.54 | 0.00 | 0.00 | 0.43 | 111 | 0.14 | REJECT |

Track A (#33, #39, #42): `docs/strategies/research/20260719_fx_wave2_trackA_results.md`.
Track B (#35, #37, #30): `docs/strategies/research/20260719_fx_wave2_trackB_results.md`.

**None of the 6 Wave 2 strategies clears the combined statistical gate
(methodology Section 2.5).** Five are decisively negative OOS Sharpe (DSR
0.0000, no edge to deflate). The sixth (#42 RORO Regime Spread) has a small
positive 1x-cost Sharpe (+0.06) with a naive PSR of 0.9993, but DSR is exactly
0.0000 once deflated for the honest, growing 106-trial project-wide search, and
the edge does not survive a 1.5x cost stress (-0.03) -- it fails the mandatory
cost-sensitivity gate outright.

## Does any strategy meet the pre-registered "genuinely close" bar?

**No.** The stopping rule (Section 6) defines "genuinely close" as a positive
DEFLATED Sharpe (DSR meaningfully positive, short of but approaching 0.95) with
low S&P correlation -- i.e. a real, cost-surviving, statistically-supported
edge that just falls short of the full gate. Every one of the 6 strategies has
DSR = 0.0000 exactly. #42's 1x-cost positive Sharpe is the only candidate that
could be mistaken for "close," and it was evaluated directly against this bar
in the Track A results doc: it fails outright, not marginally -- the 1x
positive Sharpe is cost-fragile (flips negative at 1.5x) and contributes zero
deflated statistical evidence of skill. None of the 6 strategies has a
positive, cost-surviving, DSR-supported edge at any distance from the gate.

## Book-level context (informational, not gate-relevant given no strategy passes)

All 6 strategies show near-zero-to-modest S&P correlation (range -0.01 to
0.14), consistent with their design as either market-neutral relative-value
spreads (#30/#35/#37/#39/#42, beta-weighted or statistically residualized by
construction) or a USD-seasonal basket largely orthogonal to the S&P
(#33). This confirms the mechanisms are structurally distinct from
directional equity risk, as intended -- but market-neutrality alone does not
create book value without a real underlying edge, and none of the 6 produced
one.

## WAVE 2 RESOLUTION: the pre-registered stopping rule resolves to STOP

Per `docs/superpowers/specs/2026-07-19-fx-wave2-selection-design.md` Section 6:

> If all 6 Wave 2 strategies FAIL the combined gate: across two waves the
> campaign will have tested 8+ distinct mechanisms (trend, cross-sectional
> momentum, carry, filtered carry, session breakout, spread-RV, statistical
> residual, macro-regime, seasonal, metals) spanning the full frequency and
> style spectrum, all failing after realistic costs. That is decisive evidence
> the retail G10 FX catalog is exhausted. The campaign DECLARES the finding and
> STOPS: no Wave 3, and specifically no ML meta-labeling harness build
> (#48-53).

**All 6 Wave 2 strategies fail.** Combined with Wave 1's 6/6 fail
(`docs/strategies/research/20260719_fx_catalog_campaign_synthesis.md` and the
2026-07-19 cost-sensitivity regate confirming robustness to the cost
assumption, `docs/strategies/research/20260719_fx_cost_sensitivity_regate.md`),
the campaign has now tested, gated, and rejected:

1. Time-series momentum / trend (Wave 1, #3 TSMOM)
2. Cross-sectional momentum (Wave 1, #4 XSectMom)
3. Plain carry (Wave 1, #15)
4. Filtered/seatbelt carry (Wave 1, #16/#19)
5. Session breakout (Wave 1, #20 London Breakout)
6. Metals ratio-reversion (Wave 1, #43 Gold/Silver)
7. Statistical-residual reversion (Wave 2, #39 PCA dollar-factor)
8. Macro-regime spread (Wave 2, #42 RORO)
9. Seasonal (Wave 2, #33 Turn-of-Month)
10. Cointegration/relative-value spread-RV (Wave 2, #30/#35/#37 -- three
    distinct implementations of the same family: pairwise regression residual,
    systematic scanning, and vol-differential)

Ten distinct mechanisms (or 8+ counting the #30/#35/#37 spread-RV family as
one), spanning trend-following, mean-reversion, carry, seasonality, statistical
arbitrage, and macro-regime timing, at both directional and market-neutral
risk profiles, all fail net of realistic transaction costs -- with the failure
robust to a cost-model sensitivity check (Wave 1 re-gate) and to honest,
growing multiple-testing deflation (DSR computed against a monotonically
increasing project-wide trial count throughout, now at N=111).

**Recommendation: STOP.** Per the pre-registered rule, this is declared as a
structural verdict about the retail G10 FX asset class under this cost regime,
not a coverage gap. Do not proceed to a Wave 3. Do not build the ML
meta-labeling harness (#48-53) -- it was explicitly gated behind a Wave 2
survivor, and there is none. The disciplined action per the North Star
("surfacing a failure is success... never a problem to engineer around") is to
record this finding and redirect research effort to a different asset class or
sleeve rather than continuing to search the same exhausted style space.

## What remains untested (explicitly out of scope, for the record)

- #36 Scandi triangle (needs Brent oil data on top of the now-built spread
  engine) and #40 correlation-breakdown (partial spread dependency) were
  deferred from Wave 2 by design and were not tested.
- EM/data-blocked strategies (#18, #55) and the remaining INTRADAY strategies
  (#21-25) were not in Wave 2's scope.
- These remain technically open, but the stopping rule's judgment is that 8+
  already-tested mechanisms constitute decisive, generalizable evidence; the
  marginal value of testing further neighbors within the same exhausted style
  space is judged low. If revisited, that would be a deliberate, separately
  justified decision, not a default continuation of this campaign.
