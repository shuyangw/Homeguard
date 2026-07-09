# Futures SP-E Trials -- Pre-Registration Ledger (2026-07-07)

Parameter-free, pre-registered trials (DSR trial count +1 each). No post-hoc sign flips.

| # | Strategy | Feed | Universe | Expected sign | Config |
|---|---|---|---|---|---|
| 49 | FuturesFundingCarry | Binance funding | BTC/ETH | long_short | funding_carry.yaml |
| 37 | FuturesCoTTilt | CFTC Legacy CoT | broad | long_short | cot_tilt.yaml |
| 26/27 | VIX roll-down (return stream) | Cboe VX curve | VX1 | short (contango) | vix_rolldown.yaml |

Required checks: #49 correlated-re-expression vs the deployed CME-basis satellite (#48);
#37 contested [C]; #26 backwardation kill-switch is part of the construction (not risk garnish).
Data-only: EIA calendar (consumer #41 needs SP-B).

## Results (2026-07-09)

Gate = PSR>=0.95 AND DSR>=0.95 AND PBO<0.25 AND 1.5x cost. Benchmark carry_idm = 0.765 / PASS.

| # | Strategy | OOS 1x | PBO | PSR | DSR | verdict | notes |
|---|---|---|---|---|---|---|---|
| 37 | CoT tilt | -0.124 | 0.141 | 0.00 | 0.00 | REJECT | negative OOS; contested [C] confirmed. Real CoT data (post whitespace-null fix). |
| 26/27 | VIX roll-down (return stream) | +0.564 | NaN | -- | -- | (needs work) | POSITIVE VRP after the roll-jump fix (was -0.854 contaminated). PBO NaN on a single-config return stream (inherited _compute_pbo limitation) -> needs a proper deflation/robustness pass; real crash tail (skew -2.50, kurt 20.4); pre-cost. The one genuinely promising NEW result. |
| 49 | funding carry | -- | -- | -- | -- | NO DATA | Binance geo-blocked (HTTP 451); unit-tested only. To gate: fetch funding elsewhere, avg-abs-10 calibrate _FUNDING_SCALAR, + #48 re-expression check. |

Bottom line: #37 REJECTS (as the catalog's [C] rating predicted). #26 VIX roll-down is the one
promising new result (+0.564) but its gate is incomplete (PBO NaN) -- the honest next step for VIX is a
proper return-stream deflation + cost model + best-of-N, NOT a claim of a gate pass yet. #49 blocked on data.
