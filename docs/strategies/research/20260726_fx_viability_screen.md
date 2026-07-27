# Statistical-Viability Screen, and the Slate Recomputed on Measured Costs - 2026-07-26

**Status:** apparatus + analysis. No trial consumed, no verdict produced.

## Why the screen exists

The deflated bar sat at roughly 1.05-1.14 annualized for most of the FX
campaign, while the realistic literature ceiling for a single daily G10 factor
is about 0.3-0.6 net Sharpe. **Most of the 141 trials were arithmetically
incapable of passing even if their thesis was entirely correct**, and each one
still raised the bar for everything tested after it.

So a spec must now publish, before consuming a trial:

```
if_true_sharpe = sqrt(trades_per_year) * (gross_edge_bps - cost_bps) / per_trade_vol_bps
```

and clear the CURRENT SR_zero. Adopted as a FORMAL gate (user decision,
2026-07-26). `src/backtesting/validation/viability.py`, 10 tests.

`gross_edge_bps` and `per_trade_vol_bps` are the author's stated estimates and
belong in the pre-registration. `cost_bps` is **not** an estimate: it is computed
from the measured hour-of-week surface for the pairs and hours the spec actually
trades. The same signal can be viable on EURUSD in the London hours and hopeless
on USDNOK at the rollover, and a spec-level average hides exactly that.

## The recomputation, and it goes the other way

The post-mortem's Section 4.3 viability numbers were computed against the cost
model deleted on 2026-07-25 (1.0 pip/side majors, approx 1.8 bps round trip).
Re-running its own formulas, its own stated parameters, against measured costs:

| spec | doc cost | measured | doc if-true | true if-true | verdict |
|---|---:|---:|---:|---:|---|
| #1 EVT-JUMP | 3.50 | **0.97** | 0.85 | **2.28** | viable |
| #2 FIX-REV | 1.80 | **1.09** | 1.51 | **1.84** | viable |
| #3 CB-DRIFT | 2.50 | **1.04** | 0.94 | **1.33** | viable |
| #5 XLEAD | 2.00 | **1.40** | 1.33 | **2.14** | viable |

Bar: SR_zero = **1.1807** (N = 141 + 50, per the ~50-slate decision).

The doc over-charged every spec, by 1.4x to 3.6x. Two consequences matter:

**Its screen would have killed a live candidate.** #3 CB-DRIFT scored 0.94
against a 1.14 bar on the doc's numbers, i.e. screened out. On measured costs it
is 1.33 and clears. A viability screen fed stale costs does harm, not good.

**The screen-kills need revisiting.** The doc killed time-zone session
decomposition on "gross 1-3 bp/day against >=1.8 bp/RT taker cost". Measured
major cost in liquid hours is 0.6-1.1 bps round trip, so that comparison no
longer holds and the family is a live candidate again rather than a cost-screen
casualty.

## Robustness

| spec | measured cost | break-even cost | headroom | survives 1.5x |
|---|---:|---:|---:|---|
| #1 EVT-JUMP | 0.97 | 2.91 | 3.0x | yes |
| #2 FIX-REV | 1.09 | 2.50 | 2.3x | yes |
| #3 CB-DRIFT | 1.04 | 1.61 | 1.5x | yes |
| #5 XLEAD | 1.40 | 2.11 | 1.5x | yes |

Break-even is the cost at which the if-true Sharpe falls to the bar. All four
clear the mandatory 1.5x cost-stress leg on their stated parameters.

## What this does NOT say

These are **if-true** numbers. They say a spec is worth a trial, never that it
will pass one. Every input except cost is the author's estimate, and an
optimistic `gross_edge_bps` produces an optimistic screen. The screen is a
floor-check on arithmetic feasibility, not evidence about the market.

Two of the post-mortem's specs could not be screened at all: **#6 RANGE-INTRA**
and **#7 GAP-SUN** quote if-true ranges (0.7-1.2 and 0.6-1.0) without stating
the edge and per-trade vol they came from. They must state those inputs before
they can consume a trial. That gap is itself a finding: an unstated-input range
is not a screen result.

## Reproduce

```
PYTHONPATH=$(pwd) python -c "
from src.backtesting.validation.viability import screen_spec
from src.backtesting.walkforward_common import get_campaign_trial_distribution
from src.backtesting.statistics.dsr import expected_max_sharpe
n, s = get_campaign_trial_distribution()
wd = lambda hrs: [d*24+h for d in range(5) for h in hrs]
print(screen_spec(name='#1 EVT-JUMP', trades_per_year=200, gross_edge_bps=5.0,
                  per_trade_vol_bps=25.0,
                  pairs=['EURUSD','USDJPY','GBPUSD','AUDUSD'],
                  hours_of_week=wd([12,13]),
                  sr_zero=expected_max_sharpe(s, n+50)).summary())"
```
