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

Sections 1-5 above were committed in `8cbd84a` BEFORE the run started.

Run date 2026-07-26. Window 2011-01-01 .. 2026-04-01 (full configured range, no
subset). OOS span 2014-01-03 .. 2026-04-01, 13 walk-forward windows, 3064 OOS
days restricted to dates the S&P also trades, 2228 of them active (>=1 trade).
Data frequency: 1-minute. Full daily series 4267 days. All four pairs had 1m data
across the range; EURGBP and GBPJPY are absent from the `fx_daily` cache, so
their daily ATR(14) was aggregated from their own 1m bars per FX trading day, as
designed.

### 6.1 Headline

| Metric | 1.0x costs (PRIMARY) | 1.5x costs (STRESS) | 0.0x costs (BOUND) |
|---|---|---|---|
| OOS Sharpe (net) | **-1.8657** | **-2.7102** | **-0.1381** |
| S&P Sharpe, same OOS dates | 0.6767 | 0.6767 | 0.6767 |
| Beats S&P (primary gate) | **NO** | **NO** | **NO** |
| IS Sharpe (mean per-window) | -1.4842 | -2.2832 | +0.1447 |
| IS/OOS Sharpe ratio | n/a (OOS <= 0) | n/a (OOS <= 0) | n/a (OOS <= 0) |
| PSR | 0.0000 | 0.0000 | 0.3150 |
| DSR (N=142) | 0.0000 | 0.0000 | 0.0000 |
| PBO | 0.4318 | 0.3901 | 0.4992 |
| Correlation to S&P | -0.0278 | -0.0265 | -- |
| Information ratio vs S&P | -0.9089 | -1.0203 | -- |
| CAGR | -4.130% | -5.974% | -0.333% |
| Total return (OOS) | -40.33% | -52.95% | -4.00% |
| Max DD | -40.77% | -53.24% | -9.96% |
| Max DD duration | 4452 days | 4452 days | 2988 days |
| Calmar | -0.1013 | -0.1122 | -0.0334 |
| Monthly win rate | 31.8% (47/148) | 27.7% (41/148) | 51.4% (76/148) |
| Profit factor | 0.7128 | 0.6093 | 0.9754 |
| Annualized vol | 2.26% | 2.28% | 2.25% |
| Skew / Kurtosis | -0.3866 / 3.9350 | -0.4920 / 3.9755 | -- |

Trade log: 6162 entries, 8676 exit fills, average hold 250.0 one-minute bars
(4.17 hours), consistent with an 08:00-09:30 London entry window and a 16:00
London flat.

### 6.2 VERDICT: FAIL, and the FAIL is cost-robust

The primary pre-registered gate (OOS Sharpe > S&P over the same OOS dates) fails
at every cost level including zero. PSR, DSR and PBO all fail as diagnostics.
The 1.5x cost-stress leg is materially worse, as required to be checked and as
predicted. This confirms and hardens the 2026-07-19 FAIL.

### 6.3 The prediction was RIGHT on the verdict and WRONG on the direction

The registered prediction in Section 3 called a FAIL, which is correct. It also
predicted OOS Sharpe would IMPROVE from -1.60 into roughly the -0.8 to -1.5 band.
It did not. It got WORSE, to -1.8657, outside the predicted band on the bad side.
Recording this plainly rather than quietly re-describing the prediction as
"directionally uncertain": Section 2 correctly identified that the net direction
was ambiguous, and then Section 3 still committed to an improving point estimate.
That point estimate was wrong.

The cause is measurable on the realized fills, not a matter of opinion:

| Pair | Entries | Old flat RT (bps) | New measured RT (bps) | Delta | Ratio |
|---|---|---|---|---|---|
| GBPJPY | 1888 | 1.545 | 4.400 | +2.855 | 2.85x |
| EURGBP | 1135 | 2.851 | 4.400 | +1.549 | 1.54x |
| EURUSD | 1585 | 2.056 | 0.586 | -1.470 | 0.29x |
| GBPUSD | 1554 | 1.744 | 0.966 | -0.778 | 0.55x |
| **Book mean** | **6162** | **1.967** | **2.553** | **+0.586** | **1.30x** |

The book-weighted round-trip charge went UP 30 percent, not down. The premise
that the cost correction runs in the pessimistic direction is true only for the
pairs that are IN the measured tables. GBPJPY -- this strategy's single largest
leg by entry count -- and EURGBP are in neither `_MEASURED_RT_BPS` nor the
hour-of-week surface, so they take the conservative `_UNMEASURED_RT_BPS = 4.0`
fallback plus 0.4 bps commission, hour-blind. The old pip-tier model charged
GBPJPY only 1.545 bps because a 0.01 pip on a ~160 price is a small fraction of
notional. Half this book is unmeasured crosses, so the correction made it more
expensive on net.

### 6.4 Why no cost refinement can rescue this, and why we did not try one

The 4.4 bps charged to EURGBP and GBPJPY is probably 2-3x too wide for two
genuinely liquid crosses. The tempting move is to measure those two pairs and
re-run. We did NOT do that, because it would be a researcher degree of freedom
spent in the direction of a better number, and because a strictly stronger
argument was available for the same compute: run the strategy at ZERO cost.

At zero transaction cost -- not reduced, ELIMINATED, an upper bound no real
execution can beat -- OOS Sharpe is **-0.1381**, profit factor **0.9754**, and
the monthly win rate is **51.4%**, i.e. a coin flip with slightly negative
expectancy. The gross mechanism has no edge. It does not clear the S&P bar of
0.6767, and it does not clear the deflated bar SR_zero of 1.1372; it does not
clear zero. Therefore the FAIL is invariant to the entire cost model, and the
EURGBP/GBPJPY spread question is moot for this verdict. This bound is a
diagnostic, not a tradeable specification, and is reported as such.

The trade-level decomposition says the same thing from a second direction:
across all 6162 trades over 12+ years, GROSS pre-cost P&L is **+43.9R** -- an
economically negligible raw edge, roughly 0.007R per trade -- against a total
cost drag of 507.2R, giving net -463.3R. Trade-level expectancy is -0.0752R
(avg winner +0.6619R, avg loser -0.8500R, win rate 51.25%). Per pair, net R is
EURUSD +31.8, GBPUSD -51.0, GBPJPY -219.0, EURGBP -225.0: only one of four
pairs is even marginally positive. A strategy whose gross edge is +43.9R over
6162 trades has no margin to pay any spread at all.

Note the two profit-factor figures in this document are on different bases and
both are reported rather than the flattering one: 0.7128 on the combined daily
RETURN series (1.0x costs) and 0.8186 on per-trade R. They agree in direction.

Supporting evidence that there is nothing to salvage: PBO at the zero-cost bound
is 0.4992, a literal coin flip -- in-sample-selected configurations carry no
out-of-sample persistence. IS Sharpe is +0.1447 against OOS -0.1381, so even the
in-sample edge is negligible before it fails to generalize. At 1.0x costs both IS
(-1.4842) and OOS (-1.8657) are deeply negative, which is the signature of a
genuinely negative-expectancy specification rather than an overfit one.

### 6.5 Scope of this negative (do not over-generalize)

This bounds the SPECIFICATION tested: a filtered Asian-range breakout
(0.25-0.80x daily ATR width gate, tier-1 EUR/GBP event skip, 3-pip offset OCO,
partial target at 1x range, 1x ATR(15m) trail, 16:00 London flat), on four
G10 pairs, at 1-minute resolution, as a pure spread TAKER, 2011-2026. It does
NOT establish that London-session breakout structure is dead generally, and it
does not say anything about the same idea executed as a liquidity PROVIDER,
at a different resolution, or with order-flow rather than price-range triggers.
Those remain untested hypotheses, not refuted ones.

### 6.6 Trial accounting -- what actually happened at run time

The adjudication in Section 4 was that this apparatus re-run should NOT
increment N, leaving it at 141. The runner nonetheless hardcodes
`trial_count = base_trials + 1` and gated against **N = 142**. This is disclosed
rather than smoothed over. It is harmless here in both directions: 142 > 141, so
the bar applied was STRICTER than the adjudication required (N never shrinks),
and DSR is 0.0000 at any N because a negative Sharpe cannot clear a positive
deflated bar.

The registry read succeeded and the fallback did NOT fire. Proof: the reports
show `trials=142`, which is the live count 141 + 1. The contention fallback
would have produced 40 + 1 = 41 and SR_zero 0.7331. Confirmed independently
before the run: `get_campaign_trial_distribution()` returned N = 141 with 130
observed trial Sharpes, std 0.4293, **SR_zero = 1.1372**.

This run was deliberately NOT appended to `output/experiments.duckdb`,
consistent with the Section 4 decision that it is an apparatus correction rather
than a new trial. That is a decision, not an omission.

### 6.7 Section 11.9 exit-schema verification (first real run of this code)

The fills log at
`output/backtests/fx/LondonBreakout/2011-01-01_to_2026-04-01/trades.csv`
(2.3 MB, 14838 rows) was verified on disk, not assumed. All seven exit-schema
columns are **100.0% populated** on all 8676 exit rows (`reason`, `trade_id`,
`entry_ts`, `entry_price`, `mae`, `mfe`, `bars_held` -- zero nulls, zero
sentinels). Entry rows correctly leave them blank / NaN / -1 rather than zero,
which is the intended design (zero would be a false claim about an excursion).

Exit reason distribution: target 2514, trail 2490, stop 1982, flat_1600 1689,
eod_safety 1. MAE min/median/max -1.86 / -0.00245 / 0; MFE 0 / 0.00328 / 2.719;
bars_held 1 / 247 / 479.

Internal consistency check passed: exits (8676) minus entries (6162) equals
exactly the `target` count (2514). Every partial take-profit at
`tp_fraction = 0.5` produces one extra exit fill, so this identity is what a
correct implementation must produce.

**DEFECT FOUND -- `trade_id` is populated but degenerate.** `trade_id` is
constant at the single value `4` across ALL 8676 exit rows (`nunique() == 1`,
min == max == 4), spanning all four pairs and the full 2011-2026 window. It is
100% non-sentinel and therefore PASSES a naive null/sentinel check -- which is
exactly the check that was run first, and which gave a false pass. Cardinality,
not nullity, is what catches this. Recording the miss as well as the defect.

Root cause: `src/backtesting/engine/intraday_order_engine.py::_open_position`
assigns `trade_id` from the engine's `_next_id` counter, but the runner
instantiates a FRESH `OrderEngine()` per FX trading day, so `_next_id` resets to
0 every day. Since this strategy places exactly one deterministic OCO bracket
sequence per day, the counter lands on the same value every day. The column
carries zero discriminating information.

Impact on this verdict: NONE. `trade_id` is a diagnostic join key only; the
gated return series never touches it, and the substantive Section 11.9 fields
(`reason`, `mae`, `mfe`, `bars_held`, `entry_ts`, `entry_price`) are all valid.
Round trips remain fully reconstructable: the composite key
`(date, pair, entry_ts, entry_price)` yields exactly 6162 unique values on the
exit rows, matching the entry count one-for-one.

Follow-up required before any downstream consumer uses `trade_id` as a join key
or uniqueness check: give the intraday engine a globally-incrementing counter
shared across the per-day engines rather than a fresh `OrderEngine()` per day.
Logged here rather than fixed in-flight, because changing the engine mid-verdict
would invalidate the run that was just gated.

### 6.8 The apparatus bug was load-bearing, not cosmetic

The `_book_if_closed(eng)` arity bug fixed in Section 5 fired for real: the
`eod_safety` exit reason appears exactly once in the log. Under the unfixed code
that single day would have raised `TypeError: _book_if_closed() missing 1
required positional argument: 'bar'` and aborted the entire run. The bug was
introduced when the cost model started needing the exit fill's timestamp
(`19c1488`) and had never been exercised, because the previous trade-log run
predates that change. Without this fix the re-gate could not have run at all.

### 6.9 Artifacts

- `docs/reports/fx/FX_LONDON_BREAKOUT_REGATE_20260726_cost1.0x.md` (primary)
- `docs/reports/fx/FX_LONDON_BREAKOUT_REGATE_20260726_cost1.5x.md` (stress)
- `docs/reports/fx/FX_LONDON_BREAKOUT_REGATE_20260726_cost0.0x_BOUND.md` (bound)
- `output/backtests/fx/LondonBreakout/2011-01-01_to_2026-04-01/trades.csv` (fills)

The `docs/reports/` tree is gitignored; this document is the durable tracked
copy. The 2026-07-19 figures are preserved alongside the new ones in
`docs/strategies/FX_60_CATALOG_TRACKER.md` row 20 and are not overwritten.
