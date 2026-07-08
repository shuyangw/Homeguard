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

Results (append after each run: OOS Sharpe, PBO, PSR, DSR, verdict).
