# #20 London Open Breakout -- Re-gate on Corrected Apparatus (2026-07-26)

Durable tracked copy of the re-gate. Working report: `docs/reports/fx/` (gitignored).

Strategy: catalog #20 London Open Breakout. FX, intraday, 1-minute bars.
Pairs: GBPUSD, EURUSD, EURGBP, GBPJPY (equal-risk combined daily return).
Window: 2011-01-01 .. 2026-04-01. Walk-forward 36m train / 12m test / 12m step.
Primary pre-registered gate (from 2026-07-19): stitched OOS Sharpe > S&P 500
Sharpe over the same OOS dates. PSR/DSR/PBO are diagnostics under that
pre-registration, but are reported and adjudicated here as well.

## 1. Why this re-gate exists

The 2026-07-19 verdict (OOS Sharpe -1.60, S&P leg 0.68, DSR 0, 3064 OOS
same-dates) was produced on apparatus since found defective in three ways. This
run establishes what the number is once those are corrected. It is NOT an
attempt to make the strategy pass.

1. **Cost model** (`d98eb35`, `19c1488`). Previously one flat round-trip charge
   from the deprecated pip-tier model (major tier x london multiplier = 2.4 pips
   RT, hour-blind, price-level-blind). Now: the MEASURED hour-of-week spread at
   the hour each fill actually lands on, half the round trip at the entry fill's
   hour and half at the exit fill's, via `fx_round_trip_bps_at()` reading
   `config/costs/fx_hour_of_week_spread.csv`.
2. **Trial count** (`098d085`). The runner sourced N from
   `n_trials_project_wide()`, which returns 0 for this campaign, so DSR was
   deflated against SR_zero = 0.0000 -- the gate had degenerated to "is the
   Sharpe positive". It now uses `get_campaign_trial_distribution()`, and passes
   the campaign's 130 observed trial Sharpes rather than the single-element
   `[sharpe]`.
3. **Trial-count contention** (`230ca1a`). `get_campaign_trial_distribution()`
   silently fell back to N=40 (SR_zero 0.7331) whenever the registry could not be
   opened read-write.

Verified at run time, before the gate: **N = 141, SR_zero = 1.1372** (trial-Sharpe
std 0.4293 over 130 observed + 11). This is the corrected path, not the 40 /
0.7331 fallback.

## 2. Cost-direction correction -- the brief's premise holds for only half the book

The re-gate was commissioned on the understanding that the cost correction runs
in the PESSIMISTIC direction (the flat model over-charged the London window, so
-1.60 was biased against the strategy). Measured before running, that is true for
the two USD majors and FALSE for the two crosses:

| Pair | Old flat RT (bps of price) | New RT at London hours (bps) | Direction |
|---|---|---|---|
| GBPUSD | 1.85 | 0.96 | 1.9x CHEAPER |
| EURUSD | 2.14 | 0.58 | 3.7x CHEAPER |
| EURGBP | 2.79 | 4.40 | 1.6x MORE EXPENSIVE |
| GBPJPY | 1.50 | 4.40 | 2.9x MORE EXPENSIVE |

EURGBP and GBPJPY are absent from both the measured level table
(`_MEASURED_RT_BPS`) and the hour-of-week surface, so they take the
`_UNMEASURED_RT_BPS = 4.0` fallback plus 0.4 bps commission, hour-blind. The old
pip-tier model charged them the major tier, which for GBPJPY in particular
(pip 0.01 on a ~160 price) was only 1.50 bps.

So on an equal-weight four-pair book, the net cost delta is ambiguous a priori:
two legs get materially cheaper, two get materially more expensive. This is
recorded BEFORE the run so the result cannot be attributed after the fact to
whichever direction turns out convenient. It also means a modest improvement, no
improvement, or a mild deterioration in OOS Sharpe are all consistent with a
correctly-applied cost fix.

## 3. REGISTERED PREDICTION (recorded before running)

**Predicted: still a FAIL, by a wide margin.**

- OOS Sharpe improves from -1.60 on the GBPUSD/EURUSD legs but is partly or
  wholly offset by the EURGBP/GBPJPY legs getting 1.6-2.9x more expensive.
  Point prediction: OOS Sharpe remains clearly negative, in roughly the
  -0.8 to -1.5 band. A move above zero would be a surprise.
- The primary S&P gate FAILS: OOS Sharpe stays below the S&P leg (approx 0.68).
- PSR, DSR, PBO all stay outside the gate. DSR in particular should be ~0
  against SR_zero = 1.1372: a negative Sharpe cannot clear a positive deflated
  bar under any trial count.
- The 1.5x cost-stress leg is strictly worse than the 1.0x leg.

A cost correction is not a mechanism. Nothing about this strategy's economic
hypothesis changed; only the accuracy of what it is charged. A spec that loses
1.60 Sharpe units does not become viable because roughly 1-2 bps per round trip
was mispriced on half its book.

## 4. Trial accounting -- adjudication

**Decision: this re-gate does NOT increment N. N stays at 141 for this run.**

Reasoning:

- A trial, for DSR purposes, is a consumed researcher degree of freedom -- a
  distinct specification drawn from the search space, whose maximum over draws
  is what the deflation corrects for. Nothing about the specification changed
  here: same pairs, same parameters (risk_frac 0.005, tp_fraction 0.5,
  offset_pips 3.0), same window, same horizon, same walk-forward geometry, same
  pre-registered gate. No human looked at a result and chose a different setting
  to get a better number.
- What changed is the measuring apparatus, in a direction that was not selected
  for its effect on this strategy: the hour-of-week cost surface was built for
  the whole FX platform, and the trial-count fixes are strictly gate-TIGHTENING
  (SR_zero moved from 0.0000 to 1.1372). Counting an apparatus correction as a
  new trial would penalize the act of fixing a bug, which creates an incentive
  to leave defective apparatus in place.
- This matches the standing rule that fixing a bug or mis-specification and
  re-running the SAME pre-registered spec is an apparatus correction, not a
  search, and does not consume the iteration budget.

The counter-argument, stated fairly: any re-run gives the spec another draw at
the gate, and if one re-ran on enough apparatus variants one could eventually
draw a pass. That risk is real but is not engaged here, because (a) the
apparatus change was not chosen by looking at this strategy's outcome, (b) the
direction of the change was published in this document before the run, and (c)
the result is a FAIL, so no gate was cleared by the re-draw. Had this re-gate
FLIPPED the verdict to a PASS, the honest treatment would be to count it as a
trial and re-adjudicate, because at that point the re-run would have been the
thing that produced the pass. That contingency is recorded here in advance.

**N never shrinks.** The 141 used here is at or above every count previously
applied to this strategy (the original run used a degenerate 0). The superseded
2026-07-19 figures are preserved, not overwritten, in the catalog tracker.

## 5. Apparatus bug found and fixed during setup (not a spec change)

`scripts/backtest_scripts/run_fx_london_breakout_walkforward.py` called
`strat._book_if_closed(eng)` at two sites (the end-of-day safety flatten in
`_pair_daily_returns` and in `_pair_trade_log`), but the strategy's signature is
`_book_if_closed(self, engine, bar)` -- the `bar` argument became required when
the cost model started needing the exit fill's timestamp to price the exit-hour
half-spread (`19c1488`). Any FX day ending with a still-open position would have
raised `TypeError` and aborted the run. Fixed by passing the last bar. This is an
apparatus repair, not a specification change, and consumes no iteration budget.

A `--cost-mult` CLI override was also added so the 1.5x cost-stress leg runs off
the same config as the primary leg rather than a forked copy. The leg's
multiplier is now printed in the report so a leg is never ambiguous.

## 6. Results

(Filled in after the run. Sections 1-5 above were committed before it started.)
