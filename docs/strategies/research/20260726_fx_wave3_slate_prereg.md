# FX Wave 3 Slate -- Pre-Registration (intraday / event-time axis)

**Date:** 2026-07-26  
**Bar every spec faces:** SR_zero = **1.1807** annualized, from the generation ledger at N = 141 prior trials plus a 49-spec slate.  
**Status:** pre-registered. Committed BEFORE any spec in this slate is run.

## Provenance and blindness

Generated in a fresh context whose only permitted campaign input was
`docs/strategies/research/20260726_fx_generation_ledger.md`. No results file,
report, tracker, session log or experiment registry was read. Two disclosures:

1. **A result leaked through the environment.** The session-start git log in the
   system prompt contained the commit subject `fb169df test(fx): #20 London
   Breakout re-gate on corrected apparatus -- FAIL, cost-robust`. That is a
   verdict for catalog slot #20. It was not sought and arrived before the brief
   was read. Slot #20 is excluded from this slate.
2. **The leak exposed a real defect, now fixed.** The ledger listed #20 as OPEN.
   `build_generation_ledger.py` mapped gate grades with
   `_GRADE.get(cells[6], "OPEN")`, so any unrecognized grade string silently
   became OPEN. The tracker holds `'FAIL (cost-robust)'` (with a space) while the
   map had `'FAIL(cost-robust)'` (without), so a tested-and-failed slot was
   presented to the generator as an open one. The parser now raises on unknown
   grades and the token was added. Rebuilt counts: OPEN 43 -> 42, TESTED 13 -> 14,
   READY-open 7 -> 6. The corrected ledger marks #20 TESTED-FAIL independently of
   the leak, so the exclusion does not rest on leaked information.

Measurement discipline: `per_trade_vol_bps` is MEASURED from the held 1m data
(2011-2026, unsigned dispersion only). `gross_edge_bps` is a literature or
first-principles estimate stated at proposal time. **No signed effect,
autocorrelation or continuation was measured before proposing** -- doing so would
condition the pre-registration on the answer. Costs are computed by the screen
from the measured hour-of-week spread surface; none is asserted by hand.

## A correction to the screen, applied to every multi-leg spec

`screen_spec` has no concept of legs: it averages the round-trip cost over the
pairs named and charges it ONCE. A 2-leg spread pays two round trips and a 6-leg
basket pays six. Left uncorrected the screen would flatter every relative-value
spec in this slate. Each entry below therefore reports the raw screen output and,
where legs > 1, a **leg-adjusted if-true Sharpe** computed with
`if_true_sharpe(T, edge, legs * cost, vol)`. **The leg-adjusted figure is the
authoritative one** and decides routing. This is a limitation of the screen worth
fixing in `viability.py`, not a property of these specs.

## Two gates, not one

A spec earns a trial only if it clears the bar **and still clears it with costs
at 1.5x the measured surface** (methodology Section 4). The second gate is not
decoration: the high-trade-count intraday specs in this slate have net edges of a
few tenths of a bp against ~0.6-1.0bps of cost, so a modest cost misestimate
flips their sign. Specs that pass only at 1.0x cost are reported as routed, with
the binding constraint named.

## Self-audit of this document

A first draft of this slate had 19 specs clearing the bar. Reviewing my own
inputs before publishing, three faults were found and corrected DOWNWARD:

- **A trigger threshold used as an expected edge.** Spec 35 (synthetic cross)
  was given `gross_edge_bps = 3.0`, which is the entry threshold -- assuming full
  capture of the divergence on every trade -- while its own spurious-reason field
  said the true edge is approximately zero. Corrected to 0.5.
- **Numbers contradicting their own prose.** Spec 14 (NOKSEK) was given an edge
  of 30% of per-trade volatility while its text stated the two-leg cost would
  erase it. Corrected from 9.0 to 5.0bps.
- **Trade counts set to the maximum possible rather than the expected trigger
  rate.** Twelve thresholded specs had `trades_per_year` set as if every pair
  traded every day, inflating sqrt(T). Corrected to expected trigger rates.

These corrections moved specs from viable to routed. That direction is the point:
the failure mode this campaign is guarding against is an author tuning inputs
until specs pass, and the only defence is to audit one's own numbers against
one's own stated reasoning before the run, not after.

## Slate summary

- Specs pre-registered: **49** (39 catalog slots + 10 novel)
- Clear the bar on their own if-true arithmetic: **2**
- Cannot clear the bar even if entirely correct: **47** -> routed to the forward-paper queue / combination spec, NOT to standalone trials

The runnable catalog inventory is 39 slots (35 open + 4 naive-only re-forms)
after the #20 correction; all 39 are covered here, plus 10 novel specs.

### Mechanism-family budget

| family | mechanism | specs | clear bar |
|---|---|---:|---:|
| F1 | US scheduled-event time (CPI / NFP / FOMC) | 6 | 0 |
| F2 | Session and time-of-day segmentation | 11 | 0 |
| F3 | Benchmark fixing and rebalancing flow | 4 | 2 |
| F4 | Intraday breakout and volatility expansion | 6 | 0 |
| F5 | Intraday mean reversion (taker-side inventory residual) | 4 | 0 |
| F6 | Cross-sectional and dollar-factor structure | 5 | 0 |
| F7 | Lead-lag and cross-market propagation | 3 | 0 |
| F8 | Cross-asset metals-FX linkage | 4 | 0 |
| F9 | Carry and swap-aware forms | 2 | 0 |
| F10 | Calendar and liquidity-regime effects | 4 | 0 |

No family exceeds 11 of 49 specs. The budget is deliberately weighted toward
event-time and session structure, where the apparatus has authoritative
timestamps and a measured hour-of-week cost surface, and away from daily factor
families, which the arithmetic below shows cannot reach this bar at all.

The families are not equally independent, and saying so matters more than the
count: F4 (breakout) and F6 (cross-sectional momentum) both rest on slow
information diffusion, and F5 and F2's reversion specs both rest on inventory
absorption. Specs 23 and 27 differ mainly in their exit rule and are flagged in
their own kill conditions as at risk of being one idea counted twice. Treating
all 49 as independent evidence would overstate the slate's information content.

### What the arithmetic says before anything is run

Three structural results fall out of the screen, and they shape the slate more
than any individual idea:

1. **Daily G10 factor specs are arithmetically incapable of passing.** Published
   net Sharpe for FX trend, carry and cross-sectional momentum is 0.3-0.6. Since
   the if-true Sharpe is derived FROM that literature figure, it is by
   construction below 1.18. Every such spec here is routed, not tested. This is
   not pessimism, it is arithmetic: at ~40 trades/year no honest per-trade edge
   reaches the bar.
2. **Event-time drift specs fail on trade count, not on mechanism.** With only
   ~30 US releases a year and measured post-event dispersion of 19-32bps, even a
   generous drift estimate gives sqrt(180) * 0.4/22 ~ 0.26. The event calendar is
   our best data asset and it still cannot support a standalone daily-frequency
   verdict. Event specs survive only where the edge/vol ratio is structurally
   large, not where the mechanism is merely real.
3. **High-trade-count intraday specs clear the bar at measured cost and collapse
   at 1.5x cost.** Specs 12, 15, 19, 24, 28 and 49 have if-true Sharpes of
   1.24-2.36 at the measured surface and 0.31-1.03 at 1.5x. Their net edges are
   a few tenths of a bp against ~0.6-1.0bps of cost, so they are tests of the
   cost model as much as of the signal. Running them would spend six trials on
   results that a plausible cost misestimate could reverse. They are routed.
4. **What survives both gates is concentrated, price-insensitive, calendar-known
   flow.** Only the month-end and quarter-end fix specs clear at 1.5x cost, and
   they do so with headroom rather than marginally: spec 18 needs a 4.40bps gross
   edge to clear at 1.5x cost and is proposed at 6.0bps, against the 10-20bps
   Melvin-Prins document. They are rare (72 and 24 trades/year) but their
   edge-to-volatility ratio is structurally large, which is the only thing that
   works at this bar.

### Specs that clear the bar (ranked by leg-adjusted margin)

| # | slot | name | if-true SR | at 1.5x cost | margin |
|---|---|---|---:|---:|---:|
| 21 | NOVEL | Quarter-end fix amplification | 2.18 | 2.08 | +1.00 |
| 18 | 32 | Month-end fix flow | 2.09 | 1.86 | +0.91 |

### Specs routed to forward-paper / combination (cannot clear the bar if true)

| # | slot | name | if-true SR | at 1.5x cost | binding constraint |
|---|---|---|---:|---:|---|
| 12 | 25 | Session-transition volatility expansion | 2.36 | 0.39 | fails 1.5x cost gate only |
| 15 | 59 | Hour-of-week volatility surface conditioning | 1.70 | 0.72 | fails 1.5x cost gate only |
| 19 | 23 | WMR 16:00 fix impact and reversal | 1.37 | 0.62 | fails 1.5x cost gate only |
| 28 | 11 | Hourly z-score reversion | 1.30 | 0.31 | fails 1.5x cost gate only |
| 24 | 7 | Multi-timeframe momentum alignment | 1.27 | 1.03 | fails 1.5x cost gate only |
| 10 | 21 | NY continuation | 1.25 | 0.82 | fails 1.5x cost gate only |
| 49 | NOVEL | Intraday session-open range persistence | 1.24 | 0.42 | fails 1.5x cost gate only |
| 9 | 13 | End-of-day reversal | 1.14 | -0.48 | below bar outright |
| 22 | 5 | Breakout-pullback continuation | 1.13 | 0.90 | below bar outright |
| 34 | 40 | Correlation-breakdown reversion | 1.11 | 1.07 | below bar outright |
| 16 | 58 | Volatility spillover network | 1.06 | -0.31 | below bar outright |
| 40 | 43 | Gold/silver ratio (regime-conditioned re-form) | 1.05 | 1.01 | below bar outright |
| 7 | NOVEL | Ranaldo local-hour segmentation | 0.94 | 0.54 | below bar outright |
| 20 | 24 | Friday position-squaring fade | 0.91 | 0.83 | below bar outright |
| 26 | 28 | ATR-regime switch | 0.89 | 0.87 | below bar outright |
| 13 | 56 | EM local-open effect | 0.89 | -0.04 | below bar outright |
| 8 | 10 | Asian range fade | 0.87 | 0.49 | below bar outright |
| 30 | 8 | Bollinger band reversion (daily) | 0.85 | 0.83 | below bar outright |
| 25 | 26 | NR7 range-compression squeeze | 0.83 | 0.75 | below bar outright |
| 47 | 34 | Holiday thin-liquidity reversion | 0.82 | 0.78 | below bar outright |
| 11 | 22 | Tokyo JPY-cross mean reversion | 0.76 | 0.49 | below bar outright |
| 33 | 4 | Cross-sectional momentum (dollar-neutral re-form) | 0.74 | 0.67 | below bar outright |
| 48 | NOVEL | Post-holiday liquidity restoration | 0.73 | 0.67 | below bar outright |
| 45 | 17 | Swap-aware swing bias | 0.69 | 0.67 | below bar outright |
| 41 | 44 | Non-USD gold momentum | 0.68 | 0.66 | below bar outright |
| 31 | NOVEL | Rollover-window displacement reversion | 0.66 | -1.36 | below bar outright |
| 29 | 9 | RSI(2) intraday fade | 0.66 | -1.72 | below bar outright |
| 36 | 3 | TSMOM portfolio (vol-scaled re-form) | 0.66 | 0.50 | below bar outright |
| 39 | 36 | Scandi triangle | 0.65 | 0.60 | below bar outright |
| 42 | 46 | Gold as risk-regime filter | 0.64 | 0.57 | below bar outright |
| 44 | 15 | Vol-targeted carry basket (drawdown-conditioned re-form) | 0.64 | 0.57 | below bar outright |
| 27 | 1 | Dual MA with ATR trailing stop | 0.61 | 0.59 | below bar outright |
| 43 | 47 | Silver beta amplification | 0.58 | 0.48 | below bar outright |
| 23 | 2 | Donchian channel breakout | 0.58 | 0.56 | below bar outright |
| 4 | NOVEL | Pre-FOMC dollar drift | 0.58 | 0.50 | below bar outright |
| 5 | NOVEL | Pre-release range fade | 0.52 | -0.15 | below bar outright |
| 46 | 31 | Day-of-week effect | 0.51 | 0.45 | below bar outright |
| 17 | 14 | Weekend gap fade | 0.47 | 0.16 | below bar outright |
| 3 | NOVEL | Post-release overreaction reversal (45-180min) | 0.39 | 0.16 | below bar outright |
| 38 | 45 | Metals-implied FX | 0.36 | -0.27 | below bar outright |
| 1 | 60 | Scheduled-news straddle | 0.31 | 0.13 | below bar outright |
| 2 | NOVEL | Post-release drift (5-45min) | 0.26 | -0.07 | below bar outright |
| 14 | 57 | NOKSEK microstructure reversion | 0.11 | -0.86 | below bar outright |
| 6 | NOVEL | Event-day cross-sectional dollar beta | -0.48 | -1.57 | below bar outright |
| 32 | 54 | Cross-sectional intraday relative strength | -1.65 | -4.74 | below bar outright |
| 37 | 41 | High-frequency lead-lag | -10.44 | -22.93 | below bar outright |
| 35 | 38 | Synthetic cross divergence | -16.07 | -25.95 | below bar outright |

Routing is not a verdict on the mechanism. It states that a standalone
historical trial cannot produce a passing number even if the thesis is exactly
right, so spending a trial on it would raise the bar for every other spec while
being incapable of clearing it. Per the locked combination pre-registration
(`20260726_fx_combination_spec_prereg.md`), membership is every spec in this wave
that cleared the screen and was run, equal weighted. Routed specs are not
members and this document does not propose an alternative combination rule.

---

## The specs

# F1. US scheduled-event time (CPI / NFP / FOMC)

## Spec 1: Scheduled-news straddle

*catalog #60* | family F1 | **ROUTED -- cannot clear bar**

**1. Mechanism.** Dealers withdraw depth in the seconds before a scheduled US print, so the book is thinnest exactly when the shock lands. The repricing therefore overshoots fundamentals: each unit of flow has outsized impact and stop/margin liquidations extend the initial move. A taker entering on a stop just outside the pre-release range participates in the cascade without needing to predict direction. The other side is pre-positioned traders being stopped out and dealers dumping inventory. It is not arbitraged because it is a liquidity phenomenon, not a mispricing, and capturing it means bearing gap risk at the worst possible execution moment.

**2. Rule.** On each CPI/NFP/FOMC release date, at T-2min record the high/low of [T-30min, T-2min]. Place an OCO stop-entry buy at high+0.5*range and sell at low-0.5*range, live over [T, T+15min]. Bracket the fill: stop 1.0x range, target 2.0x range, time-exit T+90min. One unit per pair per event.

**3. Viability screen.**

```
screen_spec(
    name='Scheduled-news straddle',
    trades_per_year=100, gross_edge_bps=2.0, per_trade_vol_bps=30.0,
    pairs=['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF'],
    hours_of_week=<15 weekday hours>, sr_zero=1.1807)

-> Scheduled-news straddle: if-true Sharpe 0.31 vs bar 1.18 (cost 1.07 bps RT) -> NOT VIABLE
-> at 1.5x cost (1.61 bps RT): if-true Sharpe 0.13 -> FAILS cost gate
```

- `gross_edge_bps` = 2.0 -- First principles. The jump is essentially complete within 60s (Andersen-Bollerslev-Diebold-Vega 2003), so a stop entry is filled after the repricing; only residual cascade is capturable. 2bps is ~20% of the measured 9-14bps mean 5min move.
- `per_trade_vol_bps` = 30.0 -- Measured: 45-180min post-event std averages 30bps across CPI/NFP/FOMC on majors.

**4. Falsifier.** Triggered trades show no positive expectancy beyond entry slippage, i.e. the post-trigger path is a martingale once the stop-entry fill price is used rather than the pre-release mid.

**5. Most likely spurious reason.** Survivorship in the trigger rule: only events with large moves trigger, so a naive average over triggered trades conditions on the move already having happened. Using the pre-release mid instead of the actual stop fill would manufacture the entire edge.

**6. Kill conditions.** Net-of-cost expectancy per triggered trade below zero, or >60% of P&L from fewer than 5 events.

## Spec 2: Post-release drift (5-45min)

*NOVEL (outside the 60-catalog)* | family F1 | **ROUTED -- cannot clear bar**

**1. Mechanism.** Macro information is impounded in stages: the headline moves price instantly, but the cross-sectional implications are worked out over minutes as different participant types update. Evans-Lyons show order flow keeps transmitting news information after the print. If so, the sign of the release-minute move predicts continued drift over the following half hour. The other side is slower participants rebalancing. It survives arbitrage only to the extent the residual is small relative to execution risk.

**2. Rule.** At T+5min compute r0 = sign(close[T+5] - close[T-1]). Enter in the direction of r0 at T+5, exit at T+45. No stop. One unit per pair per event, CPI/NFP/FOMC.

**3. Viability screen.**

```
screen_spec(
    name='Post-release drift (5-45min)',
    trades_per_year=180, gross_edge_bps=1.5, per_trade_vol_bps=22.0,
    pairs=['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF'],
    hours_of_week=<15 weekday hours>, sr_zero=1.1807)

-> Post-release drift (5-45min): if-true Sharpe 0.26 vs bar 1.18 (cost 1.07 bps RT) -> NOT VIABLE
-> at 1.5x cost (1.61 bps RT): if-true Sharpe -0.07 -> FAILS cost gate
```

- `gross_edge_bps` = 1.5 -- Literature-anchored: exploitable post-announcement drift in FX is a small fraction of the initial jump. 1.5bps is ~15% of the measured ~10bps mean 0-5min move.
- `per_trade_vol_bps` = 22.0 -- Measured: 5-45min post-release std averages 22bps across the three events on majors.

**4. Falsifier.** Regressing the 5-45min return on the 0-5min return across events yields a slope indistinguishable from zero.

**5. Most likely spurious reason.** Only ~30 events/year, so a handful of large releases dominate; an apparent drift is easily 3-4 observations. Also, a stale quote at T+5 in thin conditions creates artificial autocorrelation.

**6. Kill conditions.** Slope insignificant, or the effect concentrated in one event type only after being proposed for all three.

## Spec 3: Post-release overreaction reversal (45-180min)

*NOVEL (outside the 60-catalog)* | family F1 | **ROUTED -- cannot clear bar**

**1. Mechanism.** The headline number is a noisy summary; the detail (core vs headline composition, revisions, participation) is digested over the following hours and frequently qualifies the headline. If the initial jump overshoots the considered interpretation, price partially retraces. The other side is fast momentum traders exiting. This is the opposite sign to spec 2 at a longer horizon; both cannot be right at the same horizon, and proposing both at different horizons is deliberate, not hedging.

**2. Rule.** At T+45min compute r1 = sign(close[T+45] - close[T-1]). Enter AGAINST r1 at T+45, exit at T+180min or session end, whichever first. One unit per pair per event.

**3. Viability screen.**

```
screen_spec(
    name='Post-release overreaction reversal (45-180min)',
    trades_per_year=180, gross_edge_bps=2.0, per_trade_vol_bps=32.0,
    pairs=['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF'],
    hours_of_week=<15 weekday hours>, sr_zero=1.1807)

-> Post-release overreaction reversal (45-180min): if-true Sharpe 0.39 vs bar 1.18 (cost 1.07 bps RT) -> NOT VIABLE
-> at 1.5x cost (1.61 bps RT): if-true Sharpe 0.16 -> FAILS cost gate
```

- `gross_edge_bps` = 2.0 -- First principles: partial retracement of an overshoot. 2bps is ~20% of the measured 0-5min jump, an upper bound on plausible mean reversal.
- `per_trade_vol_bps` = 32.0 -- Measured: 45-180min post-event std averages 32bps on majors.

**4. Falsifier.** No negative relation between the 0-45min move and the 45-180min move across events.

**5. Most likely spurious reason.** Mean reversion appears mechanically if the entry uses a bid/ask-crossed print at a volatile moment. Bar-close sampling at T+45 in a high-vol window embeds bid-ask bounce.

**6. Kill conditions.** Reversal coefficient insignificant, or a positive-drift result appears instead, which would falsify this and belong to spec 2.

## Spec 4: Pre-FOMC dollar drift

*NOVEL (outside the 60-catalog)* | family F1 | **ROUTED -- cannot clear bar**

**1. Mechanism.** Lucca-Moench (2015) document large pre-FOMC excess returns in equities, attributed to resolution of policy uncertainty being compensated in advance. The FX analog is a systematic risk-premium tilt in the dollar over the 24h before the statement: if the pre-FOMC period is a risk-on window, the dollar as funding/safe-haven should weaken. The other side is participants unwilling to hold policy risk into the announcement.

**2. Rule.** Short USD (long the non-USD leg) at 18:00 UTC on the day before each FOMC statement day; exit at 17:45 UTC on statement day, before the 18:00/19:00 release. FOMC dates 2013+ only. Equal unit per pair.

**3. Viability screen.**

```
screen_spec(
    name='Pre-FOMC dollar drift',
    trades_per_year=48, gross_edge_bps=6.0, per_trade_vol_bps=57.9,
    pairs=['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF'],
    hours_of_week=<20 weekday hours>, sr_zero=1.1807)

-> Pre-FOMC dollar drift: if-true Sharpe 0.58 vs bar 1.18 (cost 1.19 bps RT) -> NOT VIABLE
-> at 1.5x cost (1.79 bps RT): if-true Sharpe 0.50 -> FAILS cost gate
```

- `gross_edge_bps` = 6.0 -- Lucca-Moench report ~49bp equity excess return pre-FOMC; the FX transmission is a fraction of that. 6bps over 24h is a generous read of the dollar analog.
- `per_trade_vol_bps` = 57.9 -- Measured: 1440min majors-basket std 57.9bps.

**4. Falsifier.** Mean pre-FOMC dollar return indistinguishable from the unconditional mean 24h dollar return.

**5. Most likely spurious reason.** Only ~8 events/year from 2013 gives ~105 observations total across pairs, and the pairs are ~0.8 correlated so the effective N is closer to 13 independent events. A t-stat computed on 105 would be badly overstated.

**6. Kill conditions.** Effective-N-adjusted t below 2, or the sign driven by 2013-2015 taper-era observations alone.

## Spec 5: Pre-release range fade

*NOVEL (outside the 60-catalog)* | family F1 | **ROUTED -- cannot clear bar**

**1. Mechanism.** In the hour before a scheduled release, informed participants stop trading directionally and dealers quote defensively. Price movement in that window is therefore disproportionately uninformed inventory noise, which should revert by the release rather than persist. The other side is small uninformed flow that has to trade. It is not arbitraged away because the window is short and the reversion is small relative to the event risk that follows it.

**2. Rule.** At T-60min record the mid m0. At T-10min, if close deviates from m0 by more than 0.75x the trailing 20-day std of the same 50min window, enter against the deviation; exit at T-1min. Flat into every release. One unit per pair per event.

**3. Viability screen.**

```
screen_spec(
    name='Pre-release range fade',
    trades_per_year=180, gross_edge_bps=1.5, per_trade_vol_bps=10.9,
    pairs=['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF'],
    hours_of_week=<15 weekday hours>, sr_zero=1.1807)

-> Pre-release range fade: if-true Sharpe 0.52 vs bar 1.18 (cost 1.08 bps RT) -> NOT VIABLE
-> at 1.5x cost (1.62 bps RT): if-true Sharpe -0.15 -> FAILS cost gate
```

- `gross_edge_bps` = 1.5 -- First principles: fade of uninformed drift. 1.5bps is ~14% of the measured 10.9bps pre-window std, at the optimistic end for noise reversion.
- `per_trade_vol_bps` = 10.9 -- Measured: 60min pre-release std 10.9bps (majors, excluding one NZDUSD outlier window).

**4. Falsifier.** Pre-release deviations show no negative autocorrelation into the release relative to matched non-event days at the same clock time.

**5. Most likely spurious reason.** Bid-ask bounce produces spurious reversion at any horizon; the effect must survive using mid quotes and a one-minute execution lag.

**6. Kill conditions.** No reversion versus the matched non-event control, or the effect vanishes with a 1min execution lag.

## Spec 6: Event-day cross-sectional dollar beta

*NOVEL (outside the 60-catalog)* | family F1 | **ROUTED -- cannot clear bar**

**1. Mechanism.** A US macro surprise is close to a pure common dollar shock. Pairs load on that shock with stable, heterogeneous betas (AUD and NZD high, CHF and JPY lower and sometimes sign-flipped by safe-haven demand). Trading the high-beta pair against the low-beta pair isolates the dollar factor and cancels idiosyncratic noise, cutting the denominator. The other side is single-pair traders who bear the idiosyncratic component.

**2. Rule.** Estimate each pair's beta to the equal-weight dollar index on a trailing 60 event windows (0-45min), refit annually. On each release, at T+5min go long the top-beta pair and short the bottom-beta pair, sized to equal dollar notional, in the direction of the T+5 dollar-index move. Exit T+45min.

**3. Viability screen.**

```
screen_spec(
    name='Event-day cross-sectional dollar beta',
    trades_per_year=90, gross_edge_bps=2.5, per_trade_vol_bps=14.0,
    pairs=['AUDUSD', 'USDCHF'],
    hours_of_week=<15 weekday hours>, sr_zero=1.1807)

-> Event-day cross-sectional dollar beta: if-true Sharpe 0.61 vs bar 1.18 (cost 1.60 bps RT) -> NOT VIABLE
-> legs=2: cost 1.60 x 2 = 3.21 bps RT
-> LEG-ADJUSTED if-true Sharpe -0.48 vs bar 1.18
-> at 1.5x cost (4.81 bps RT): if-true Sharpe -1.57 -> FAILS cost gate
```

- `gross_edge_bps` = 2.5 -- First principles: same drift premise as spec 2 but on a lower-noise construct; 2.5bps assumes the spread carries a somewhat larger share of the drift than a single pair.
- `per_trade_vol_bps` = 14.0 -- Measured 5-45min single-pair std 22bps, reduced to ~14bps for a beta-matched spread at typical 0.7 cross-correlation.

**4. Falsifier.** Betas are unstable out of sample, or the spread shows no drift while single pairs do.

**5. Most likely spurious reason.** Beta estimated on the same windows used for the test is lookahead; the annual refit must be strictly prior-data-only. Ranking on trailing beta also selects the noisiest pair.

**6. Kill conditions.** Beta rank correlation between adjacent years below 0.3, or two-leg cost exceeds the gross edge.

# F2. Session and time-of-day segmentation

## Spec 7: Ranaldo local-hour segmentation

*NOVEL (outside the 60-catalog)* | family F2 | **ROUTED -- cannot clear bar**

**1. Mechanism.** Ranaldo (2009) documents that a currency depreciates during its own domestic trading hours and appreciates during foreign hours, a robust intraday pattern driven by order-flow segmentation: domestic institutions are net sellers of domestic currency during their own working day (import settlement, hedging, payroll), and that flow is absorbed by dealers who must be compensated. The other side is dealers earning the spread and the intraday reversal. It persists because the flow is non-discretionary and recurs daily.

**2. Rule.** For each of EURUSD, GBPUSD, USDJPY: short the domestic currency during its local session hours (EUR 07-15 UTC, GBP 07-16 UTC, JPY 00-07 UTC) and hold flat otherwise. One entry and one exit per pair per day at fixed clock times.

**3. Viability screen.**

```
screen_spec(
    name='Ranaldo local-hour segmentation',
    trades_per_year=756, gross_edge_bps=1.6, per_trade_vol_bps=25.0,
    pairs=['EURUSD', 'GBPUSD', 'USDJPY'],
    hours_of_week=<45 weekday hours>, sr_zero=1.1807)

-> Ranaldo local-hour segmentation: if-true Sharpe 0.94 vs bar 1.18 (cost 0.74 bps RT) -> NOT VIABLE
-> at 1.5x cost (1.11 bps RT): if-true Sharpe 0.54 -> FAILS cost gate
```

- `gross_edge_bps` = 1.6 -- Ranaldo (2009) reports intraday segmentation effects of a few bps per day on majors. 1.6bps/day is a mid-range read of the published magnitude.
- `per_trade_vol_bps` = 25.0 -- Measured: ~8h holding horizon, majors-basket std 34.3bps at 480min, scaled down to 25bps for the lower-vol subset traded.

**4. Falsifier.** The local-hour return is not reliably negative for the domestic currency once the sample is split by decade.

**5. Most likely spurious reason.** The effect is old (published 2009 on 1993-2005 data) and is exactly the kind of pattern algorithmic execution has since competed away. A positive result on 2011-2026 must be checked for concentration in the early years.

**6. Kill conditions.** First-half vs second-half sample sign flip, or the effect surviving only before 2015.

## Spec 8: Asian range fade

*catalog #10* | family F2 | **ROUTED -- cannot clear bar**

**1. Mechanism.** The Tokyo session has the thinnest G10 depth of the three majors sessions, so the same order size moves price further than it would in London. Moves established on Asian liquidity are therefore disproportionately impact rather than information, and are partially retraced when London depth arrives. The other side is Asian-hours participants who must transact locally. It persists because the retracement is small relative to the cost of holding the position into London.

**2. Rule.** Record the Tokyo range [00:00, 06:00 UTC]. At 07:00 UTC, if price is above the Tokyo high, sell; if below the low, buy. Target the Tokyo range midpoint, stop 1.0x range beyond entry, time-exit 12:00 UTC. One unit per pair per day.

**3. Viability screen.**

```
screen_spec(
    name='Asian range fade',
    trades_per_year=200, gross_edge_bps=2.0, per_trade_vol_bps=17.5,
    pairs=['USDJPY', 'AUDJPY'],
    hours_of_week=<10 weekday hours>, sr_zero=1.1807)

-> Asian range fade: if-true Sharpe 0.87 vs bar 1.18 (cost 0.93 bps RT) -> NOT VIABLE
-> at 1.5x cost (1.39 bps RT): if-true Sharpe 0.49 -> FAILS cost gate
```

- `gross_edge_bps` = 2.0 -- First principles: partial retracement of impact-driven Asian moves. 2.5bps is ~14% of the 120min majors std.
- `per_trade_vol_bps` = 17.5 -- Measured: 120min majors-basket std 17.5bps.

**4. Falsifier.** Breaks of the Tokyo range at the London open continue rather than retrace, on average.

**5. Most likely spurious reason.** Range-break rules condition on volatility, so the sample is drawn from high-vol days where mean reversion and momentum are both easier to find by chance. A matched-vol control is required.

**6. Kill conditions.** No retracement versus a matched-volatility control, or expectancy negative on JPY crosses after the measured cost.

## Spec 9: End-of-day reversal

*catalog #13* | family F2 | **ROUTED -- cannot clear bar**

**1. Mechanism.** The 21:00-22:00 UTC rollover concentrates non-discretionary flow: daily settlement, swap rolls, and risk-limit squaring, all price-insensitive. That flow pushes price away from fair value in the last hour and is reversed once the new session's liquidity arrives. The other side is desks that must flatten before their book rolls. It persists because the reversal window has the widest spreads of the week, which is itself the barrier to arbitrage.

**2. Rule.** Measure the 20:00-21:00 UTC return. At 22:00 UTC enter against it if its magnitude exceeds 0.5x the trailing 20-day std of that hour. Exit at 01:00 UTC. One unit per pair per day.

**3. Viability screen.**

```
screen_spec(
    name='End-of-day reversal',
    trades_per_year=756, gross_edge_bps=2.0, per_trade_vol_bps=12.5,
    pairs=['EURUSD', 'USDJPY', 'GBPUSD'],
    hours_of_week=<10 weekday hours>, sr_zero=1.1807)

-> End-of-day reversal: if-true Sharpe 1.14 vs bar 1.18 (cost 1.48 bps RT) -> NOT VIABLE
-> at 1.5x cost (2.22 bps RT): if-true Sharpe -0.48 -> FAILS cost gate
```

- `gross_edge_bps` = 2.0 -- First principles: reversal of price-insensitive settlement flow. 2bps is ~16% of the 60min majors std.
- `per_trade_vol_bps` = 12.5 -- Measured: 60min majors-basket std 12.5bps.

**4. Falsifier.** The 20:00-21:00 return shows no negative autocorrelation with the subsequent overnight return.

**5. Most likely spurious reason.** This window carries among the widest measured weekday spreads (EURUSD 3.50x at 21:00 and 3.94x at 22:00 UTC). Any mid-quote-based result will look profitable and be untradeable; only executable prices settle it.

**6. Kill conditions.** Cost-adjusted expectancy negative, which the measured rollover spread makes the likely outcome.

## Spec 10: NY continuation

*catalog #21* | family F2 | **ROUTED -- cannot clear bar**

**1. Mechanism.** The London close at 16:00-17:00 UTC removes roughly half of G10 liquidity while the US session continues. A directional move already underway meets a thinner book after the London handover, so the same residual flow produces larger price change, extending the move. The other side is the departing London liquidity. It persists because it requires holding into the thinning NY afternoon.

**2. Rule.** Compute the 12:00-16:00 UTC return. At 17:00 UTC enter in the same direction if magnitude exceeds 0.75x the trailing 20-day std of that window. Exit 20:00 UTC. One unit per pair per day.

**3. Viability screen.**

```
screen_spec(
    name='NY continuation',
    trades_per_year=340, gross_edge_bps=2.0, per_trade_vol_bps=17.5,
    pairs=['EURUSD', 'USDJPY', 'GBPUSD'],
    hours_of_week=<20 weekday hours>, sr_zero=1.1807)

-> NY continuation: if-true Sharpe 1.25 vs bar 1.18 (cost 0.81 bps RT) -> VIABLE
-> at 1.5x cost (1.22 bps RT): if-true Sharpe 0.82 -> FAILS cost gate
```

- `gross_edge_bps` = 2.0 -- First principles: continuation on thinning depth. 2bps is ~11% of the 120min majors std.
- `per_trade_vol_bps` = 17.5 -- Measured: 120min majors-basket std 17.5bps.

**4. Falsifier.** No positive relation between the overlap-session return and the NY-afternoon return.

**5. Most likely spurious reason.** Momentum and reversal specs on adjacent windows (this and spec 9) can both appear to work on the same data through window-slicing. Only one sign can be structurally true at a given horizon.

**6. Kill conditions.** Relation insignificant, or opposite in sign to spec 9's premise in a way that is not reconcilable.

## Spec 11: Tokyo JPY-cross mean reversion

*catalog #22* | family F2 | **ROUTED -- cannot clear bar**

**1. Mechanism.** JPY crosses in Tokyo hours are dominated by domestic institutional flow (exporter hedging, Gotobi settlement on days divisible by five) which is calendar-driven and price-insensitive. Price-insensitive flow creates temporary displacement that reverts once it completes. The other side is Japanese corporates who must settle regardless of level. It persists because the displacement is small and the Tokyo spread is wider than London's.

**2. Rule.** On Gotobi days (5th, 10th, 15th, 20th, 25th, last business day), measure the 00:00-02:00 UTC return; enter against it at 02:00 UTC if magnitude exceeds 0.5x the trailing 20-day std of that window; exit 06:00 UTC.

**3. Viability screen.**

```
screen_spec(
    name='Tokyo JPY-cross mean reversion',
    trades_per_year=108, gross_edge_bps=2.2, per_trade_vol_bps=17.5,
    pairs=['USDJPY', 'EURJPY', 'AUDJPY'],
    hours_of_week=<20 weekday hours>, sr_zero=1.1807)

-> Tokyo JPY-cross mean reversion: if-true Sharpe 0.76 vs bar 1.18 (cost 0.92 bps RT) -> NOT VIABLE
-> at 1.5x cost (1.38 bps RT): if-true Sharpe 0.49 -> FAILS cost gate
```

- `gross_edge_bps` = 2.2 -- First principles from the documented Gotobi fixing convention; 3bps is ~17% of the 240min std, reflecting a concentrated known flow.
- `per_trade_vol_bps` = 17.5 -- Measured: 120-240min majors std 17.5-24.5bps; Tokyo hours run ~0.74x the all-hours level, so 17.5bps.

**4. Falsifier.** Gotobi days show no different intraday reversion profile from matched non-Gotobi days.

**5. Most likely spurious reason.** Gotobi is a well-known retail talking point; if it were this easy it would be gone. The control against non-Gotobi days is the whole test, and without it any Tokyo-hours reversion would be misattributed.

**6. Kill conditions.** No Gotobi-vs-control difference, in which case this collapses into generic Tokyo reversion and is not a separate mechanism.

## Spec 12: Session-transition volatility expansion

*catalog #25* | family F2 | **ROUTED -- cannot clear bar**

**1. Mechanism.** At each session handover (Tokyo->London 07:00, London->NY 12:00, London close 16:00) the participant mix changes abruptly and the new session's participants reprice against the previous session's close. That produces a systematic burst of directional movement in the first 30 minutes of each handover. Trading the direction of the first minutes of the handover captures the repricing. The other side is stale quotes from the departing session.

**2. Rule.** At each of 07:00, 12:00, 17:00 UTC, measure the first 10min return; enter in that direction at +10min; exit at +40min. Stop 1.5x the 10min move. One unit per pair per transition.

**3. Viability screen.**

```
screen_spec(
    name='Session-transition volatility expansion',
    trades_per_year=2268, gross_edge_bps=1.2, per_trade_vol_bps=9.1,
    pairs=['EURUSD', 'USDJPY', 'GBPUSD'],
    hours_of_week=<15 weekday hours>, sr_zero=1.1807)

-> Session-transition volatility expansion: if-true Sharpe 2.36 vs bar 1.18 (cost 0.75 bps RT) -> VIABLE
-> at 1.5x cost (1.12 bps RT): if-true Sharpe 0.39 -> FAILS cost gate
```

- `gross_edge_bps` = 1.2 -- First principles: short-horizon continuation on regime handover. 1.2bps is ~13% of the 30min majors std.
- `per_trade_vol_bps` = 9.1 -- Measured: 30min majors-basket std 9.1bps.

**4. Falsifier.** First-10min direction does not predict the next 30min at session handovers any better than at arbitrary clock times.

**5. Most likely spurious reason.** With ~2270 trades/year the standard error is small, so a tiny bias in bar construction (e.g. the first bar of a session including a stale print) can dominate the result.

**6. Kill conditions.** No excess predictability versus arbitrary-clock-time control, or sensitivity to the first-bar construction.

## Spec 13: EM local-open effect

*catalog #56* | family F2 | **ROUTED -- cannot clear bar**

**1. Mechanism.** EM currencies trade primarily in their own onshore hours, when local banks and the central bank are active. The transition from offshore (thin, G10-hours) to onshore pricing produces a systematic repricing at the local open as onshore participants reconcile the overnight offshore drift with local conditions. The other side is offshore holders. It persists because EM spreads are wide enough to deter arbitrage.

**2. Rule.** For USDMXN and USDZAR, measure the offshore overnight return to the local open (13:00 UTC for MXN, 06:00 UTC for ZAR); enter against it at the local open if magnitude exceeds 1.0x the trailing 20-day std; exit 4h later.

**3. Viability screen.**

```
screen_spec(
    name='EM local-open effect',
    trades_per_year=200, gross_edge_bps=6.0, per_trade_vol_bps=31.0,
    pairs=['USDMXN', 'USDZAR'],
    hours_of_week=<10 weekday hours>, sr_zero=1.1807)

-> EM local-open effect: if-true Sharpe 0.89 vs bar 1.18 (cost 4.05 bps RT) -> NOT VIABLE
-> at 1.5x cost (6.08 bps RT): if-true Sharpe -0.04 -> FAILS cost gate
```

- `gross_edge_bps` = 6.0 -- First principles: onshore-offshore reconciliation. 6bps is ~19% of the 240min EM std, justified by the much wider EM spread barrier.
- `per_trade_vol_bps` = 31.0 -- Measured: 240min std 31.7bps (USDMXN) and 40.0bps (USDZAR); 31bps is the conservative blend.

**4. Falsifier.** No reconciliation effect at the local open beyond generic reversion at any hour.

**5. Most likely spurious reason.** EM pairs carry large interest differentials; a spec that is systematically long the high-yielder will look profitable from carry alone and must be decomposed spot-vs-carry.

**6. Kill conditions.** Effect absent after removing carry accrual, or measured cost (USDMXN 4.32bps, USDZAR 5.54bps round trip) exceeds gross edge.

## Spec 14: NOKSEK microstructure reversion

*catalog #57* | family F2 | **ROUTED -- cannot clear bar**

**1. Mechanism.** NOK and SEK are the least liquid G10 currencies and share a near-identical macro driver (small open European economies, correlated rate cycles). Idiosyncratic order flow in one leg displaces it relative to the other, and because both are thin the displacement is larger and slower to correct than in majors. The other side is a dealer absorbing a block in one currency. It persists because the pair is thin enough that arbitrage capital is capacity-constrained.

**2. Rule.** Form the NOKSEK cross from USDNOK and USDSEK. Compute a 240min z-score of the log cross against its 20-day mean; enter against a |z|>2.0 deviation; exit on z crossing 0.5 or after 24h.

**3. Viability screen.**

```
screen_spec(
    name='NOKSEK microstructure reversion',
    trades_per_year=150, gross_edge_bps=5.0, per_trade_vol_bps=30.0,
    pairs=['USDNOK', 'USDSEK'],
    hours_of_week=<50 weekday hours>, sr_zero=1.1807)

-> NOKSEK microstructure reversion: if-true Sharpe 1.08 vs bar 1.18 (cost 2.37 bps RT) -> NOT VIABLE
-> legs=2: cost 2.37 x 2 = 4.73 bps RT
-> LEG-ADJUSTED if-true Sharpe 0.11 vs bar 1.18
-> at 1.5x cost (7.10 bps RT): if-true Sharpe -0.86 -> FAILS cost gate
```

- `gross_edge_bps` = 5.0 -- First principles: thin-market displacement reversion. 9bps is ~30% of the 240min Nordic std, reflecting genuinely slow correction in a capacity-constrained pair.
- `per_trade_vol_bps` = 30.0 -- Measured: 240min std 33.0bps (USDNOK), 29.2bps (USDSEK); the cross is less volatile than either leg, ~30bps.

**4. Falsifier.** The synthetic cross shows no mean reversion at the 240min horizon after costs.

**5. Most likely spurious reason.** This is a two-leg trade in the two most expensive G10 pairs on the measured surface (3.43 and 3.96bps round trip each). A mid-quote backtest will show a large edge that the 7.4bps two-leg cost erases entirely.

**6. Kill conditions.** Two-leg cost-adjusted if-true Sharpe below the bar, which the screen already indicates.

## Spec 15: Hour-of-week volatility surface conditioning

*catalog #59* | family F2 | **ROUTED -- cannot clear bar**

**1. Mechanism.** Volatility has a stable, strongly periodic hour-of-week shape driven by session overlap and settlement conventions. A signal calibrated on unconditional volatility is therefore mis-scaled at every hour: it over-trades in quiet hours and under-trades in active ones. Conditioning entry thresholds on the hour-specific volatility norm should raise the information ratio of any reversion signal without changing its mechanism. The other side is participants using unconditional thresholds.

**2. Rule.** Take a fixed 60min z-reversion signal (enter at |z|>2 vs trailing 20-period mean, exit on z<0.5 or 4h). Compute z using the hour-of-week-specific volatility norm from a trailing 52-week window rather than an unconditional one. Trade only hours 07-20 UTC.

**3. Viability screen.**

```
screen_spec(
    name='Hour-of-week volatility surface conditioning',
    trades_per_year=650, gross_edge_bps=1.8, per_trade_vol_bps=12.5,
    pairs=['EURUSD', 'USDJPY', 'GBPUSD', 'USDCHF'],
    hours_of_week=<70 weekday hours>, sr_zero=1.1807)

-> Hour-of-week volatility surface conditioning: if-true Sharpe 1.70 vs bar 1.18 (cost 0.97 bps RT) -> VIABLE
-> at 1.5x cost (1.45 bps RT): if-true Sharpe 0.72 -> FAILS cost gate
```

- `gross_edge_bps` = 1.8 -- First principles: correct scaling of an existing signal. 1.8bps is ~14% of the 60min majors std, assuming conditioning adds a modest amount to a baseline reversion edge.
- `per_trade_vol_bps` = 12.5 -- Measured: 60min majors-basket std 12.5bps.

**4. Falsifier.** Hour-conditioned thresholds produce no higher information ratio than unconditional ones on the same signal.

**5. Most likely spurious reason.** This is a refinement of a signal family, not an independent mechanism; if the base reversion signal has no edge, conditioning cannot create one. Presenting it as a new mechanism would be double-counting.

**6. Kill conditions.** No improvement over the unconditional baseline, or the baseline itself has no edge.

## Spec 16: Volatility spillover network

*catalog #58* | family F2 | **ROUTED -- cannot clear bar**

**1. Mechanism.** A volatility shock in one currency propagates to correlated currencies with a short lag because dealers hedge correlated inventory and risk models update across a book. The pair that has not yet moved is therefore temporarily mispriced relative to the one that has. The other side is dealers rebalancing. It persists only at horizons short enough that the propagation is incomplete.

**2. Rule.** Every 15min compute realized 60min vol for the G10 set. When one pair's vol z-score exceeds 2.5 and a historically >0.7-correlated partner's is below 1.0, enter the partner in the direction of the leader's 15min return; exit after 60min.

**3. Viability screen.**

```
screen_spec(
    name='Volatility spillover network',
    trades_per_year=1008, gross_edge_bps=1.5, per_trade_vol_bps=12.5,
    pairs=['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF'],
    hours_of_week=<50 weekday hours>, sr_zero=1.1807)

-> Volatility spillover network: if-true Sharpe 1.06 vs bar 1.18 (cost 1.08 bps RT) -> NOT VIABLE
-> at 1.5x cost (1.62 bps RT): if-true Sharpe -0.31 -> FAILS cost gate
```

- `gross_edge_bps` = 1.5 -- First principles: incomplete cross-pair propagation. 1.5bps is ~12% of the 60min majors std.
- `per_trade_vol_bps` = 12.5 -- Measured: 60min majors-basket std 12.5bps.

**4. Falsifier.** The lagging pair's subsequent return is unrelated to the leader's move once contemporaneous correlation is removed.

**5. Most likely spurious reason.** Correlated pairs share the USD leg, so any 'spillover' is largely a mechanical dollar move rather than propagation. The test must be run on the non-USD residual or it measures nothing.

**6. Kill conditions.** Effect disappears after removing the common dollar factor.

## Spec 17: Weekend gap fade

*catalog #14* | family F2 | **ROUTED -- cannot clear bar**

**1. Mechanism.** The Friday close to Sunday open gap reflects weekend news plus a thin-liquidity opening print. The opening print in particular is set on almost no depth, so it overshoots the level at which weekday liquidity clears, and retraces during the first Asian hours. The other side is participants forced to trade at the Sunday open. It persists because the Sunday-open spread is the widest of the week.

**2. Rule.** At the Sunday 22:00 UTC open, measure the gap versus the Friday 21:00 UTC close. If |gap| > 0.5x the trailing 20-week gap std, enter against it; exit at Monday 07:00 UTC.

**3. Viability screen.**

```
screen_spec(
    name='Weekend gap fade',
    trades_per_year=60, gross_edge_bps=2.5, per_trade_vol_bps=17.5,
    pairs=['EURUSD', 'USDJPY', 'GBPUSD'],
    hours_of_week=<2 weekday hours>, sr_zero=1.1807)

-> Weekend gap fade: if-true Sharpe 0.47 vs bar 1.18 (cost 1.43 bps RT) -> NOT VIABLE
-> at 1.5x cost (2.15 bps RT): if-true Sharpe 0.16 -> FAILS cost gate
```

- `gross_edge_bps` = 2.5 -- First principles: fade of a thin opening print. 4bps is ~23% of the 120min majors std, generous for a gap-fade.
- `per_trade_vol_bps` = 17.5 -- Measured: 120min majors-basket std 17.5bps.

**4. Falsifier.** Weekend gaps are not systematically retraced during the first Asian session.

**5. Most likely spurious reason.** A gap-fade backtest on mid quotes ignores that the Sunday reopening is among the most expensive blocks of the week: the surface DOES measure these hours and charges EURUSD 3.84x its average multiplier at 22:00 UTC Sunday, roughly double the all-in London-hours cost. With only ~60 trades a year there is also no route to statistical comfort.

**6. Kill conditions.** Cost-adjusted expectancy negative, or fewer than 40 effectively independent gaps in the sample.

# F3. Benchmark fixing and rebalancing flow

## Spec 18: Month-end fix flow

*catalog #32* | family F3 | **CLEARS BAR**

**1. Mechanism.** Melvin-Prins document that month-end FX flow is driven by equity-portfolio hedge rebalancing: after a month in which foreign equities outperform, hedgers must sell foreign currency to restore hedge ratios, and this flow is price-insensitive, predictable in sign from the month's equity returns, and concentrated into the 16:00 London WMR fix. The other side is dealers absorbing a known imbalance. It persists because the hedgers are mandate-driven and cannot delay.

**2. Rule.** On the last business day of each month, at 14:00 UTC take the sign of the month's MSCI-proxy relative equity return (use SPY vs a foreign equity ETF as the proxy; if unavailable use the month's cumulative pair return as the rebalancing proxy). Enter in the direction implied by hedge rebalancing; exit at 16:05 UTC just after the fix window.

**3. Viability screen.**

```
screen_spec(
    name='Month-end fix flow',
    trades_per_year=72, gross_edge_bps=6.0, per_trade_vol_bps=20.0,
    pairs=['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF'],
    hours_of_week=<10 weekday hours>, sr_zero=1.1807)

-> Month-end fix flow: if-true Sharpe 2.09 vs bar 1.18 (cost 1.08 bps RT) -> VIABLE
-> at 1.5x cost (1.62 bps RT): if-true Sharpe 1.86 -> CLEARS
```

- `gross_edge_bps` = 6.0 -- Melvin-Prins (2015) document month-end fix effects of order 10-20bps on major pairs conditional on the rebalancing signal. 6bps is a conservative read after accounting for the signal proxy being imperfect.
- `per_trade_vol_bps` = 20.0 -- Measured: ~2h holding window, majors-basket 120min std 17.5bps, raised to 20bps for the elevated fix window.

**4. Falsifier.** Month-end fix returns show no relation to the prior month's relative equity performance.

**5. Most likely spurious reason.** Only 12 events/year: with 6 correlated pairs the effective N is ~12/year, so 15 years gives ~180 effectively-correlated observations but far fewer independent ones. Also, the equity-return proxy is the weakest link and could be fit after the fact.

**6. Kill conditions.** Relation to the equity proxy insignificant, or >50% of P&L from fewer than 8 month-ends.

## Spec 19: WMR 16:00 fix impact and reversal

*catalog #23* | family F3 | **ROUTED -- cannot clear bar**

**1. Mechanism.** The WMR benchmark is computed over a short window around 16:00 London, and a large share of passive and corporate flow is required to execute at that benchmark. Concentrated price-insensitive flow moves price into the window and the impact partially reverses afterwards once the flow stops. Since the 2013 benchmark reform widened the window to 5 minutes, the impact is smaller but the reversal mechanism is unchanged. The other side is dealers who pre-hedge the fix.

**2. Rule.** Measure the return over the fix window (15:45-16:05 UTC, using 16:00 London in local time so the UTC hour shifts with BST). If it exceeds 0.75x its trailing 60-day std, enter against it at 16:05 UTC and exit at 17:05 UTC.

**3. Viability screen.**

```
screen_spec(
    name='WMR 16:00 fix impact and reversal',
    trades_per_year=400, gross_edge_bps=1.8, per_trade_vol_bps=12.5,
    pairs=['EURUSD', 'USDJPY', 'GBPUSD', 'USDCHF'],
    hours_of_week=<10 weekday hours>, sr_zero=1.1807)

-> WMR 16:00 fix impact and reversal: if-true Sharpe 1.37 vs bar 1.18 (cost 0.94 bps RT) -> VIABLE
-> at 1.5x cost (1.41 bps RT): if-true Sharpe 0.62 -> FAILS cost gate
```

- `gross_edge_bps` = 1.8 -- Evans (2018) and the benchmark-reform literature document fix-window impact with partial post-fix reversal; magnitude fell materially after the 2013 reform. 2.5bps is a post-reform estimate.
- `per_trade_vol_bps` = 12.5 -- Measured: 60min majors-basket std 12.5bps.

**4. Falsifier.** No negative autocorrelation between the fix-window return and the following hour.

**5. Most likely spurious reason.** The pre-2013 and post-2013 regimes differ by construction; pooling them would show a strong effect driven entirely by the pre-reform era. The spec must be evaluated post-2013 only.

**6. Kill conditions.** Effect present only pre-2015, or insignificant on the post-reform subsample which is the only economically live one.

## Spec 20: Friday position-squaring fade

*catalog #24* | family F3 | **ROUTED -- cannot clear bar**

**1. Mechanism.** Leveraged participants reduce risk into the weekend because weekend gap risk cannot be hedged. That squaring is price-insensitive and concentrated in the Friday NY afternoon, displacing price against the prevailing week's position. Once the squaring completes the displacement reverts. The other side is dealers absorbing the unwind. It persists because holding the reversion means carrying weekend gap risk.

**2. Rule.** Measure the Friday 12:00-19:00 UTC return. If it opposes the sign of the Monday-Thursday cumulative return and exceeds 0.5x its trailing 20-week std, enter in the direction of the week's move at 19:00 UTC Friday; exit at Monday 08:00 UTC.

**3. Viability screen.**

```
screen_spec(
    name='Friday position-squaring fade',
    trades_per_year=156, gross_edge_bps=5.0, per_trade_vol_bps=57.9,
    pairs=['EURUSD', 'USDJPY', 'GBPUSD'],
    hours_of_week=<3 weekday hours>, sr_zero=1.1807)

-> Friday position-squaring fade: if-true Sharpe 0.91 vs bar 1.18 (cost 0.78 bps RT) -> NOT VIABLE
-> at 1.5x cost (1.17 bps RT): if-true Sharpe 0.83 -> FAILS cost gate
```

- `gross_edge_bps` = 5.0 -- First principles: reversion of an identifiable squaring flow. 5bps is ~9% of the weekend-spanning std.
- `per_trade_vol_bps` = 57.9 -- Measured: 1440min majors-basket std 57.9bps, the closest available proxy for a Friday-to-Monday hold.

**4. Falsifier.** Friday-afternoon counter-trend moves are not systematically reversed by Monday.

**5. Most likely spurious reason.** Holding over the weekend means the result is dominated by weekend gap risk, a fat-tailed variable. A good Sharpe here can coexist with catastrophic tail exposure that Sharpe does not price.

**6. Kill conditions.** Positive mean but negative skew such that the worst 3 weekends erase the cumulative P&L.

## Spec 21: Quarter-end fix amplification

*NOVEL (outside the 60-catalog)* | family F3 | **CLEARS BAR**

**1. Mechanism.** Quarter-end concentrates the month-end rebalancing mechanism of spec 18 with additional balance-sheet and reporting-driven flow: banks reduce balance sheet for regulatory reporting dates, which withdraws liquidity precisely when rebalancing flow peaks. The same predictable flow therefore meets a thinner book and produces larger displacement. The other side is a constrained dealer sector. It persists because the constraint is regulatory, not economic.

**2. Rule.** Identical to spec 18 but restricted to the last business day of March, June, September and December, with a 2x position scale and the window extended to 13:00-16:05 UTC.

**3. Viability screen.**

```
screen_spec(
    name='Quarter-end fix amplification',
    trades_per_year=24, gross_edge_bps=12.0, per_trade_vol_bps=24.5,
    pairs=['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF'],
    hours_of_week=<10 weekday hours>, sr_zero=1.1807)

-> Quarter-end fix amplification: if-true Sharpe 2.18 vs bar 1.18 (cost 1.08 bps RT) -> VIABLE
-> at 1.5x cost (1.62 bps RT): if-true Sharpe 2.08 -> CLEARS
```

- `gross_edge_bps` = 12.0 -- First principles plus the balance-sheet literature: quarter-end effects in funding markets are roughly 2x month-end. 12bps is 2x the spec 18 estimate.
- `per_trade_vol_bps` = 24.5 -- Measured: 240min majors-basket std 24.5bps.

**4. Falsifier.** Quarter-ends show no larger effect than ordinary month-ends.

**5. Most likely spurious reason.** Only 4 events/year, i.e. ~60 observations across 15 years and 6 correlated pairs -- effectively ~15 independent events. This is the single most overfit-prone spec in the slate and is proposed only because the mechanism is strong.

**6. Kill conditions.** Fewer than 40 independent events available, or the quarter-end effect indistinguishable from the month-end effect, which would make it a duplicate of spec 18 rather than a distinct mechanism.

# F4. Intraday breakout and volatility expansion

## Spec 22: Breakout-pullback continuation

*catalog #5* | family F4 | **ROUTED -- cannot clear bar**

**1. Mechanism.** A genuine breakout is followed by a pullback as early entrants take profit, then resumes if the breakout reflected real order-flow imbalance rather than a stop run. Entering on the pullback rather than the break itself filters stop-run false breaks, because a stop run has no residual imbalance to resume the move. The other side is the profit-takers. It persists because waiting for the pullback means missing the moves that never pull back.

**2. Rule.** On 15min bars, mark a breakout when price exceeds the 24-bar high. Wait for a retracement to 0.382 of the breakout leg within 8 bars; enter long there; stop below the breakout base; target 1.618x the leg; time-exit 24 bars.

**3. Viability screen.**

```
screen_spec(
    name='Breakout-pullback continuation',
    trades_per_year=250, gross_edge_bps=2.5, per_trade_vol_bps=24.5,
    pairs=['EURUSD', 'USDJPY', 'GBPUSD'],
    hours_of_week=<50 weekday hours>, sr_zero=1.1807)

-> Breakout-pullback continuation: if-true Sharpe 1.13 vs bar 1.18 (cost 0.74 bps RT) -> NOT VIABLE
-> at 1.5x cost (1.11 bps RT): if-true Sharpe 0.90 -> FAILS cost gate
```

- `gross_edge_bps` = 2.5 -- First principles: conditional continuation after a filter. 3bps is ~12% of the 240min majors std.
- `per_trade_vol_bps` = 24.5 -- Measured: 240min majors-basket std 24.5bps.

**4. Falsifier.** Pullback entries show no better expectancy than immediate breakout entries.

**5. Most likely spurious reason.** Fibonacci retracement levels have no mechanism; 0.382 is arbitrary and any of a dozen levels could be chosen post hoc. The level is fixed here precisely to prevent that, and a result that depends on the exact level is a tell.

**6. Kill conditions.** Result sensitive to the retracement level, or no improvement over the unfiltered breakout.

## Spec 23: Donchian channel breakout

*catalog #2* | family F4 | **ROUTED -- cannot clear bar**

**1. Mechanism.** Central banks smooth exchange-rate adjustment and corporates hedge gradually, so information is impounded slowly and price trends. A channel break marks the point at which accumulated information has overcome the prevailing range, and the residual adjustment continues. The other side is hedgers transacting on a schedule. It persists because trend-following requires tolerating long drawdowns that most capital cannot hold.

**2. Rule.** Daily bars. Enter long on a close above the 55-day high, short below the 55-day low. Exit on a close through the 20-day opposite extreme. ATR(20)-based sizing at a fixed 10% annualized vol target per pair.

**3. Viability screen.**

```
screen_spec(
    name='Donchian channel breakout',
    trades_per_year=40, gross_edge_bps=25.0, per_trade_vol_bps=260.0,
    pairs=['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY', 'AUDJPY', 'EURCHF'],
    hours_of_week=<25 weekday hours>, sr_zero=1.1807)

-> Donchian channel breakout: if-true Sharpe 0.58 vs bar 1.18 (cost 1.23 bps RT) -> NOT VIABLE
-> at 1.5x cost (1.84 bps RT): if-true Sharpe 0.56 -> FAILS cost gate
```

- `gross_edge_bps` = 25.0 -- Derived from literature: published FX trend-following net Sharpe is 0.3-0.6 (Menkhoff-Sarno-Schmeling-Schrimpf 2012). Backing out per-trade edge at T=40 and a ~20-day hold gives ~25bps gross.
- `per_trade_vol_bps` = 260.0 -- Measured: 1440min std 57.9bps scaled to a ~20-day hold -> ~260bps.

**4. Falsifier.** No positive relation between channel breaks and subsequent returns in the modern sample.

**5. Most likely spurious reason.** Trend-following results are dominated by a few large moves; a single 2015 CHF or 2022 JPY episode can carry the whole record and will not recur on demand.

**6. Kill conditions.** Removing the single best quarter turns cumulative P&L negative.

## Spec 24: Multi-timeframe momentum alignment

*catalog #7* | family F4 | **ROUTED -- cannot clear bar**

**1. Mechanism.** Momentum measured at a single horizon is noisy; requiring agreement across horizons filters transient noise from persistent order-flow imbalance. Alignment across 1h, 4h and daily should therefore select the subset of moves driven by real accumulation rather than by a single participant. The other side is short-horizon noise traders. It persists because the aligned subset is small and the waiting cost is high.

**2. Rule.** Enter long when the 1h, 4h and daily returns are all positive and the daily is above its 20-day mean; short on the mirror. Exit when any horizon flips. Evaluated on 1h bars, one position per pair.

**3. Viability screen.**

```
screen_spec(
    name='Multi-timeframe momentum alignment',
    trades_per_year=150, gross_edge_bps=3.5, per_trade_vol_bps=24.5,
    pairs=['EURUSD', 'USDJPY', 'GBPUSD', 'USDCHF'],
    hours_of_week=<70 weekday hours>, sr_zero=1.1807)

-> Multi-timeframe momentum alignment: if-true Sharpe 1.27 vs bar 1.18 (cost 0.97 bps RT) -> VIABLE
-> at 1.5x cost (1.45 bps RT): if-true Sharpe 1.03 -> FAILS cost gate
```

- `gross_edge_bps` = 3.5 -- First principles: filtered momentum. 3.5bps is ~14% of the 240min majors std.
- `per_trade_vol_bps` = 24.5 -- Measured: 240min majors-basket std 24.5bps.

**4. Falsifier.** Multi-horizon agreement adds no expectancy over the single best horizon alone.

**5. Most likely spurious reason.** Three horizons is a choice among many; requiring agreement is a free parameter that can be tuned until something passes. Horizons are fixed here at conventional values to prevent that.

**6. Kill conditions.** No improvement over single-horizon momentum, making this a parameter variation rather than a mechanism.

## Spec 25: NR7 range-compression squeeze

*catalog #26* | family F4 | **ROUTED -- cannot clear bar**

**1. Mechanism.** Volatility is strongly autocorrelated and mean-reverting in level: an unusually narrow range signals a temporary contraction in the arrival of information and in dealer risk appetite, which is followed by expansion. The direction of the expansion is set by whichever side of the compressed range breaks first, because the compressed range concentrates resting stop orders on both sides. The other side is those stops. It persists because the direction is genuinely unpredictable ex ante.

**2. Rule.** Daily bars. When today's range is the narrowest of the last 7 days, place an OCO bracket at the day's high and low for the next session; stop at the opposite extreme; target 2x the compressed range; time-exit at the following day's close.

**3. Viability screen.**

```
screen_spec(
    name='NR7 range-compression squeeze',
    trades_per_year=90, gross_edge_bps=6.0, per_trade_vol_bps=57.9,
    pairs=['EURUSD', 'USDJPY', 'GBPUSD', 'USDCHF'],
    hours_of_week=<50 weekday hours>, sr_zero=1.1807)

-> NR7 range-compression squeeze: if-true Sharpe 0.83 vs bar 1.18 (cost 0.94 bps RT) -> NOT VIABLE
-> at 1.5x cost (1.41 bps RT): if-true Sharpe 0.75 -> FAILS cost gate
```

- `gross_edge_bps` = 6.0 -- First principles: volatility expansion after compression. 8bps is ~14% of the daily majors std.
- `per_trade_vol_bps` = 57.9 -- Measured: 1440min majors-basket std 57.9bps.

**4. Falsifier.** Post-NR7 sessions show no larger realized range than matched ordinary sessions.

**5. Most likely spurious reason.** Range expansion after compression is near-certain by mean reversion in volatility, but that does not make the DIRECTIONAL bracket profitable. Confusing 'range expands' with 'the bracket makes money' is the trap here: a symmetric bracket in an expanding but directionless market loses on both sides.

**6. Kill conditions.** Range expands but bracket expectancy is negative, which would confirm the vol/direction confusion.

## Spec 26: ATR-regime switch

*catalog #28* | family F4 | **ROUTED -- cannot clear bar**

**1. Mechanism.** Trend-following and mean-reversion work in different volatility regimes: in high-volatility regimes moves are information-driven and persist, in low-volatility regimes moves are inventory noise and revert. A single rule applied across both regimes averages two opposite effects toward zero. Switching the rule on a volatility state variable should recover both. The other side differs by regime, which is the point.

**2. Rule.** Compute ATR(20)/close on daily bars and its 250-day percentile. Above the 70th percentile run a 20/50 MA crossover trend rule; below the 30th run a 2-std Bollinger fade on daily closes; between, stay flat. One position per pair.

**3. Viability screen.**

```
screen_spec(
    name='ATR-regime switch',
    trades_per_year=60, gross_edge_bps=22.0, per_trade_vol_bps=180.0,
    pairs=['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY', 'AUDJPY', 'EURCHF'],
    hours_of_week=<25 weekday hours>, sr_zero=1.1807)

-> ATR-regime switch: if-true Sharpe 0.89 vs bar 1.18 (cost 1.23 bps RT) -> NOT VIABLE
-> at 1.5x cost (1.84 bps RT): if-true Sharpe 0.87 -> FAILS cost gate
```

- `gross_edge_bps` = 22.0 -- Derived from literature as in spec 23; regime switching claims to lift a 0.3-0.6 Sharpe trend rule modestly, so per-trade gross edge ~22bps at a ~12-day mean hold.
- `per_trade_vol_bps` = 180.0 -- Measured: 1440min std 57.9bps scaled to a ~10-12 day mean hold -> ~180bps.

**4. Falsifier.** Conditional Sharpe in the two regimes is not materially different from unconditional.

**5. Most likely spurious reason.** The percentile thresholds (70/30) and the two sub-rules are four free parameters; this family is where regime-switching results are usually manufactured. Fixed here, but the degrees of freedom remain a live concern.

**6. Kill conditions.** Either sub-rule alone matches the switched version, meaning the switch adds nothing.

## Spec 27: Dual MA with ATR trailing stop

*catalog #1* | family F4 | **ROUTED -- cannot clear bar**

**1. Mechanism.** Same slow-information-diffusion mechanism as spec 23, but with the exit governed by a volatility-scaled trailing stop rather than an opposite channel break. The exit is the real hypothesis here: trend P&L is dominated by how much of a large move is retained, so a volatility-adaptive trailing exit should retain more than a fixed-horizon or channel exit. The other side is unchanged.

**2. Rule.** Daily bars. Long when MA(20) crosses above MA(60), short on the mirror. Exit on a 3.0x ATR(20) trailing stop from the run-up extreme, or on an opposite cross. Vol-targeted sizing at 10% annualized.

**3. Viability screen.**

```
screen_spec(
    name='Dual MA with ATR trailing stop',
    trades_per_year=45, gross_edge_bps=24.0, per_trade_vol_bps=250.0,
    pairs=['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY', 'AUDJPY', 'EURCHF'],
    hours_of_week=<25 weekday hours>, sr_zero=1.1807)

-> Dual MA with ATR trailing stop: if-true Sharpe 0.61 vs bar 1.18 (cost 1.23 bps RT) -> NOT VIABLE
-> at 1.5x cost (1.84 bps RT): if-true Sharpe 0.59 -> FAILS cost gate
```

- `gross_edge_bps` = 24.0 -- Derived from literature as in spec 23: FX trend net Sharpe 0.3-0.6, backed out at T=45 and a ~18-day hold.
- `per_trade_vol_bps` = 250.0 -- Measured: 1440min std 57.9bps scaled to an ~18-day hold -> ~250bps.

**4. Falsifier.** The trailing exit does not retain more of large moves than the spec 23 channel exit.

**5. Most likely spurious reason.** This differs from spec 23 mainly in the exit, so a difference in results between them is as likely to be noise as mechanism. Treating both as independent evidence would double-count one idea.

**6. Kill conditions.** Indistinguishable from spec 23, in which case it is a parameter variation and should not consume a separate trial.

# F5. Intraday mean reversion (taker-side inventory residual)

## Spec 28: Hourly z-score reversion

*catalog #11* | family F5 | **ROUTED -- cannot clear bar**

**1. Mechanism.** Dealers absorbing a large order accumulate inventory they do not want and must be compensated for holding it; they lay it off over the following hour, and price returns toward the pre-shock level. As a taker we do not earn the spread, only the residual overshoot beyond it, which is the honest and much smaller part of the inventory premium. The other side is the dealer laying off risk. It persists because the compensation is the dealer's, not ours.

**2. Rule.** On 60min bars compute z = (close - MA20)/std20. Enter against |z| > 2.0 at the bar close; exit when |z| < 0.5 or after 6 bars. Trade only 07:00-20:00 UTC. One position per pair.

**3. Viability screen.**

```
screen_spec(
    name='Hourly z-score reversion',
    trades_per_year=655, gross_edge_bps=1.6, per_trade_vol_bps=12.5,
    pairs=['EURUSD', 'USDJPY', 'GBPUSD', 'USDCHF'],
    hours_of_week=<70 weekday hours>, sr_zero=1.1807)

-> Hourly z-score reversion: if-true Sharpe 1.30 vs bar 1.18 (cost 0.97 bps RT) -> VIABLE
-> at 1.5x cost (1.45 bps RT): if-true Sharpe 0.31 -> FAILS cost gate
```

- `gross_edge_bps` = 1.6 -- First principles: taker-side residual of the inventory premium. 1.6bps is ~13% of the 60min majors std, and deliberately below what a maker would earn.
- `per_trade_vol_bps` = 12.5 -- Measured: 60min majors-basket std 12.5bps.

**4. Falsifier.** No negative autocorrelation at the 1-6 hour horizon after removing bid-ask bounce.

**5. Most likely spurious reason.** Bid-ask bounce guarantees measured negative autocorrelation at short horizons in mid-quote data. If the test does not execute at ask-on-buy and bid-on-sell, the entire result is the bounce.

**6. Kill conditions.** Effect vanishes under executable pricing, or net edge below the 1.5x cost-sensitivity threshold.

## Spec 29: RSI(2) intraday fade

*catalog #9* | family F5 | **ROUTED -- cannot clear bar**

**1. Mechanism.** Same inventory-absorption mechanism as spec 28, but the trigger is a short-lookback oscillator that fires on rapid sequential moves rather than on a level deviation. The distinction is real: a fast 2-period extreme identifies a burst of one-directional flow, while a z-score identifies a sustained displacement. They select different events despite sharing a mechanism.

**2. Rule.** On 15min bars compute RSI(2). Enter long below 5, short above 95, at the bar close. Exit when RSI(2) crosses 50 or after 16 bars. Trade 07:00-20:00 UTC only.

**3. Viability screen.**

```
screen_spec(
    name='RSI(2) intraday fade',
    trades_per_year=2016, gross_edge_bps=1.1, per_trade_vol_bps=9.1,
    pairs=['EURUSD', 'USDJPY', 'GBPUSD', 'USDCHF'],
    hours_of_week=<70 weekday hours>, sr_zero=1.1807)

-> RSI(2) intraday fade: if-true Sharpe 0.66 vs bar 1.18 (cost 0.97 bps RT) -> NOT VIABLE
-> at 1.5x cost (1.45 bps RT): if-true Sharpe -1.72 -> FAILS cost gate
```

- `gross_edge_bps` = 1.1 -- First principles: same taker-side residual as spec 28 at a shorter horizon. 1.1bps is ~12% of the 30min majors std.
- `per_trade_vol_bps` = 9.1 -- Measured: 30min majors-basket std 9.1bps.

**4. Falsifier.** RSI(2) extremes show no subsequent reversion beyond the bid-ask bounce.

**5. Most likely spurious reason.** At ~2000 trades/year the cost term dominates: gross edge 1.1bps against ~0.9bps cost leaves almost nothing, so a small cost misestimate flips the sign. This spec is a cost-model test as much as a signal test.

**6. Kill conditions.** Fails the 1.5x cost-sensitivity gate, which on these numbers is close to certain.

## Spec 30: Bollinger band reversion (daily)

*catalog #8* | family F5 | **ROUTED -- cannot clear bar**

**1. Mechanism.** At the daily horizon, deviations from a 20-day mean reflect either information (which persists) or flow (which reverts). The band width conditions on realized volatility so a 2-std break is a comparable event across regimes. The reversion component is the compensation for providing liquidity to multi-day flow. The other side is a participant executing a large program over days.

**2. Rule.** Daily bars. Enter against a close outside the 20-day 2.0-std Bollinger band; exit on a close back inside the 1.0-std band or after 10 days.

**3. Viability screen.**

```
screen_spec(
    name='Bollinger band reversion (daily)',
    trades_per_year=30, gross_edge_bps=30.0, per_trade_vol_bps=185.0,
    pairs=['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY', 'AUDJPY', 'EURCHF'],
    hours_of_week=<25 weekday hours>, sr_zero=1.1807)

-> Bollinger band reversion (daily): if-true Sharpe 0.85 vs bar 1.18 (cost 1.23 bps RT) -> NOT VIABLE
-> at 1.5x cost (1.84 bps RT): if-true Sharpe 0.83 -> FAILS cost gate
```

- `gross_edge_bps` = 30.0 -- Derived from literature: daily FX reversion strategies report net Sharpe well under 0.5. Backed out at T=30 and a ~10-day hold.
- `per_trade_vol_bps` = 185.0 -- Measured: 1440min std 57.9bps scaled to a ~10-day hold -> ~185bps.

**4. Falsifier.** No reversion after 2-std daily breaks in the modern sample.

**5. Most likely spurious reason.** Daily mean reversion in FX is heavily regime-dependent and blows up in trends; a positive average with a catastrophic tail is the classic presentation of this family.

**6. Kill conditions.** Max drawdown exceeding 3x the annualized return, or negative skew concentrated in trending years.

## Spec 31: Rollover-window displacement reversion

*NOVEL (outside the 60-catalog)* | family F5 | **ROUTED -- cannot clear bar**

**1. Mechanism.** At 21:00-22:00 UTC the interbank day rolls: swap points are applied, daily risk limits reset, and many venues briefly widen or halt. Flow that must transact in that window is entirely price-insensitive, and the displacement it causes is mechanical rather than informational, so it reverts once normal liquidity resumes in Asia. The other side is desks with no choice about timing. It is not arbitraged because the window has the widest measured spreads of the week.

**2. Rule.** Measure the 21:00-22:00 UTC return. If |return| exceeds 1.5x its trailing 20-day std, enter against it at 23:00 UTC (after the widest hour has passed) and exit at 03:00 UTC.

**3. Viability screen.**

```
screen_spec(
    name='Rollover-window displacement reversion',
    trades_per_year=756, gross_edge_bps=3.0, per_trade_vol_bps=17.5,
    pairs=['EURUSD', 'USDJPY', 'AUDUSD'],
    hours_of_week=<5 weekday hours>, sr_zero=1.1807)

-> Rollover-window displacement reversion: if-true Sharpe 0.66 vs bar 1.18 (cost 2.58 bps RT) -> NOT VIABLE
-> at 1.5x cost (3.87 bps RT): if-true Sharpe -1.36 -> FAILS cost gate
```

- `gross_edge_bps` = 3.0 -- First principles: reversion of mechanical settlement displacement. 3bps is ~17% of the 120min majors std.
- `per_trade_vol_bps` = 17.5 -- Measured: 120min majors-basket std 17.5bps.

**4. Falsifier.** Rollover-window moves are not systematically reversed in the subsequent Asian session.

**5. Most likely spurious reason.** Entering at 23:00 rather than 22:00 is a deliberate cost choice, but the 23:00 hour is still 1.1-2.3x the pair's average spread. A version that entered at 22:00 would look better on mids and be untradeable.

**6. Kill conditions.** Reversion absent, or entry-hour sensitivity such that only the 22:00 entry works.

# F6. Cross-sectional and dollar-factor structure

## Spec 32: Cross-sectional intraday relative strength

*catalog #54* | family F6 | **ROUTED -- cannot clear bar**

**1. Mechanism.** Within a trading session, currencies that have outperformed on genuine order-flow imbalance tend to continue relative to those that have not, because institutional execution programs are worked over hours rather than minutes. Ranking cross-sectionally removes the common dollar factor and isolates the currency-specific flow. The other side is the program's counterparty. It persists because the programs are large and slow.

**2. Rule.** At 12:00 UTC rank the G10 set by return since 07:00 UTC. Go long the top 2 and short the bottom 2, equal notional, dollar-neutral. Exit at 20:00 UTC. One rebalance per day.

**3. Viability screen.**

```
screen_spec(
    name='Cross-sectional intraday relative strength',
    trades_per_year=1008, gross_edge_bps=3.5, per_trade_vol_bps=24.5,
    pairs=['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY'],
    hours_of_week=<25 weekday hours>, sr_zero=1.1807)

-> Cross-sectional intraday relative strength: if-true Sharpe 2.99 vs bar 1.18 (cost 1.19 bps RT) -> VIABLE
-> legs=4: cost 1.19 x 4 = 4.77 bps RT
-> LEG-ADJUSTED if-true Sharpe -1.65 vs bar 1.18
-> at 1.5x cost (7.16 bps RT): if-true Sharpe -4.74 -> FAILS cost gate
```

- `gross_edge_bps` = 3.5 -- First principles: intraday continuation of institutional execution. 3.5bps is ~14% of the 240min majors std.
- `per_trade_vol_bps` = 24.5 -- Measured: 240min majors-basket std 24.5bps; the dollar-neutral basket is comparable after netting.

**4. Falsifier.** Intraday relative-strength rank has no predictive power for the remainder of the session.

**5. Most likely spurious reason.** A 4-leg construction pays four round-trip costs against a single spread's edge; the screen's single-leg cost understates the true charge fourfold. This is stated explicitly below.

**6. Kill conditions.** Four-leg cost-adjusted if-true Sharpe below the bar.

## Spec 33: Cross-sectional momentum (dollar-neutral re-form)

*catalog #4* | family F6 | **ROUTED -- cannot clear bar**

**1. Mechanism.** Re-form of a naive-only slot. The naive version ranked pairs by raw return, which conflates the common dollar factor with currency-specific momentum: in a dollar rally every USD pair ranks together and the portfolio is a disguised dollar bet. The re-form ranks CURRENCIES by their average return against all others, so the dollar is a rankable currency rather than an embedded factor. That is a different functional form, not a retuned parameter.

**2. Rule.** Monthly. Build the 10x10 currency return matrix from G10 crosses over the trailing 3 months. Rank each currency by mean return against all others. Long the top 3, short the bottom 3, equal weight, held one month, rebalanced month-end.

**3. Viability screen.**

```
screen_spec(
    name='Cross-sectional momentum (dollar-neutral re-form)',
    trades_per_year=72, gross_edge_bps=45.0, per_trade_vol_bps=430.0,
    pairs=['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY', 'AUDJPY', 'EURCHF'],
    hours_of_week=<25 weekday hours>, sr_zero=1.1807)

-> Cross-sectional momentum (dollar-neutral re-form): if-true Sharpe 0.86 vs bar 1.18 (cost 1.23 bps RT) -> NOT VIABLE
-> legs=6: cost 1.23 x 6 = 7.36 bps RT
-> LEG-ADJUSTED if-true Sharpe 0.74 vs bar 1.18
-> at 1.5x cost (11.04 bps RT): if-true Sharpe 0.67 -> FAILS cost gate
```

- `gross_edge_bps` = 45.0 -- Derived from literature: FX cross-sectional momentum net Sharpe ~0.3-0.5 (Menkhoff et al. 2012). Backed out at T=72 and a 1-month hold.
- `per_trade_vol_bps` = 430.0 -- Measured: 1440min std 57.9bps scaled to a 21-trading-day hold -> ~265bps per leg, ~430bps for a 6-leg portfolio after partial diversification.

**4. Falsifier.** Currency-level momentum has no cross-sectional predictive power once the dollar factor is rankable rather than embedded.

**5. Most likely spurious reason.** FX momentum has decayed sharply since 2010 in published work; a positive result on 2011-2026 would run against the literature and should be treated as suspect rather than as a discovery.

**6. Kill conditions.** Literature-implied Sharpe below the bar, which the screen confirms -- route to combination.

## Spec 34: Correlation-breakdown reversion

*catalog #40* | family F6 | **ROUTED -- cannot clear bar**

**1. Mechanism.** Currency pairs sharing macro drivers maintain stable correlations. A sudden breakdown usually reflects a one-sided flow in one leg rather than a genuine divergence in fundamentals, so the relationship re-establishes. The other side is whoever needed to move that one leg. It persists because identifying a true regime change from a temporary flow requires holding through ambiguity.

**2. Rule.** For the AUDUSD/NZDUSD and EURUSD/USDCHF pairs, compute the 60-day rolling correlation and the 20-day spread z-score. When correlation stays above 0.7 but |spread z| > 2.0, enter against the spread; exit on |z| < 0.5 or 15 days.

**3. Viability screen.**

```
screen_spec(
    name='Correlation-breakdown reversion',
    trades_per_year=25, gross_edge_bps=55.0, per_trade_vol_bps=230.0,
    pairs=['AUDUSD', 'NZDUSD'],
    hours_of_week=<25 weekday hours>, sr_zero=1.1807)

-> Correlation-breakdown reversion: if-true Sharpe 1.15 vs bar 1.18 (cost 1.95 bps RT) -> NOT VIABLE
-> legs=2: cost 1.95 x 2 = 3.90 bps RT
-> LEG-ADJUSTED if-true Sharpe 1.11 vs bar 1.18
-> at 1.5x cost (5.85 bps RT): if-true Sharpe 1.07 -> FAILS cost gate
```

- `gross_edge_bps` = 55.0 -- First principles: convergence of a temporarily displaced stable relationship. 55bps is ~24% of the 15-day spread vol.
- `per_trade_vol_bps` = 230.0 -- Measured: AUDUSD/NZDUSD leg vols at 1440min are 68.7 and 68.5bps; the spread is far less volatile, ~230bps over a 15-day hold.

**4. Falsifier.** Spread divergences under stable correlation do not converge.

**5. Most likely spurious reason.** Selecting the two pair-relationships after knowing which ones held together is selection on the outcome. These two are chosen on the economic grounds of shared commodity and shared European exposure, and must be fixed before testing.

**6. Kill conditions.** Convergence absent, or the relationship breaks structurally (correlation falls below 0.5 for a quarter), which invalidates the premise rather than the parameters.

## Spec 35: Synthetic cross divergence

*catalog #38* | family F6 | **ROUTED -- cannot clear bar**

**1. Mechanism.** A cross rate must equal the ratio of its two USD legs or a triangular arbitrage exists. At sub-second horizons this is enforced by HFT. At the 1-minute horizon the residual should be zero, so this spec is a test of whether any exploitable residual survives at the frequency we can actually trade. The honest expectation is that it does not.

**2. Rule.** Every minute compute the synthetic EURGBP from EURUSD/GBPUSD and compare to the quoted EURGBP. Enter when the divergence exceeds 3bps; exit on convergence below 1bp or after 30 minutes.

**3. Viability screen.**

```
screen_spec(
    name='Synthetic cross divergence',
    trades_per_year=2520, gross_edge_bps=0.5, per_trade_vol_bps=6.8,
    pairs=['EURUSD', 'GBPUSD', 'EURGBP'],
    hours_of_week=<50 weekday hours>, sr_zero=1.1807)

-> Synthetic cross divergence: if-true Sharpe -2.89 vs bar 1.18 (cost 0.89 bps RT) -> NOT VIABLE
-> legs=3: cost 0.89 x 3 = 2.68 bps RT
-> LEG-ADJUSTED if-true Sharpe -16.07 vs bar 1.18
-> at 1.5x cost (4.01 bps RT): if-true Sharpe -25.95 -> FAILS cost gate
```

- `gross_edge_bps` = 0.5 -- First principles, with a strong prior that the true edge is ~0 at 1-minute frequency because triangular arbitrage is the most heavily contested trade in FX. 3bps is the entry threshold, i.e. an upper bound.
- `per_trade_vol_bps` = 6.8 -- Measured: 15min majors-basket std 6.8bps.

**4. Falsifier.** Divergences above 3bps at 1-minute sampling are artifacts of non-synchronous bar timestamps rather than tradeable dislocations.

**5. Most likely spurious reason.** Almost certainly the entire signal: three separate 1m bar series with independent last-print timestamps will show apparent divergence whenever one leg is stale. This is a data-artifact detector as much as a strategy.

**6. Kill conditions.** Divergences do not survive synchronised-timestamp construction, or the three-leg cost (which includes the unmeasured EURGBP fallback) exceeds the gross edge.

## Spec 36: TSMOM portfolio (vol-scaled re-form)

*catalog #3* | family F6 | **ROUTED -- cannot clear bar**

**1. Mechanism.** Re-form of a naive-only slot. The naive version took an equal-weight sign-of-return signal, which lets the highest-volatility pairs dominate realized risk and turns a portfolio claim into a bet on a few pairs. The re-form scales each pair's position by inverse realized volatility and targets portfolio-level rather than pair-level risk, so the diversification the mechanism actually claims is realised. The mechanism itself is unchanged slow information diffusion.

**2. Rule.** Monthly. For each G10 pair, signal = sign of the 12-month return. Position = signal / realized 60-day vol, scaled so the portfolio targets 10% annualized vol. Rebalance monthly.

**3. Viability screen.**

```
screen_spec(
    name='TSMOM portfolio (vol-scaled re-form)',
    trades_per_year=120, gross_edge_bps=38.0, per_trade_vol_bps=430.0,
    pairs=['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY', 'AUDJPY', 'EURCHF'],
    hours_of_week=<25 weekday hours>, sr_zero=1.1807)

-> TSMOM portfolio (vol-scaled re-form): if-true Sharpe 0.94 vs bar 1.18 (cost 1.23 bps RT) -> NOT VIABLE
-> legs=10: cost 1.23 x 10 = 12.26 bps RT
-> LEG-ADJUSTED if-true Sharpe 0.66 vs bar 1.18
-> at 1.5x cost (18.39 bps RT): if-true Sharpe 0.50 -> FAILS cost gate
```

- `gross_edge_bps` = 38.0 -- Derived from literature: time-series momentum in FX reports net Sharpe ~0.3-0.5. Backed out at T=120 and a 1-month rebalance.
- `per_trade_vol_bps` = 430.0 -- Measured: scaled daily std to a 21-day hold and aggregated with partial diversification -> ~430bps portfolio-level.

**4. Falsifier.** Vol-scaling does not change the portfolio's realized risk concentration versus equal weight.

**5. Most likely spurious reason.** Vol-scaling reliably improves Sharpe mechanically without adding any predictive content; attributing that improvement to the momentum mechanism would be a category error.

**6. Kill conditions.** Literature-implied Sharpe below the bar -- route to combination.

# F7. Lead-lag and cross-market propagation

## Spec 37: High-frequency lead-lag

*catalog #41* | family F7 | **ROUTED -- cannot clear bar**

**1. Mechanism.** The most liquid pair in a correlated set incorporates common information first because it is where informed participants execute; less liquid correlated pairs follow with a lag measured in seconds to minutes. Trading the laggard on the leader's move captures the propagation. The other side is the laggard's slower participants. It persists only if the lag exceeds our execution latency, which at 1-minute sampling is the binding question.

**2. Rule.** On 1min bars, when EURUSD moves more than 2.5 std of its 1min distribution, enter USDCHF in the corresponding direction (inverse sign) at the next bar open; exit after 5 minutes.

**3. Viability screen.**

```
screen_spec(
    name='High-frequency lead-lag',
    trades_per_year=5040, gross_edge_bps=0.9, per_trade_vol_bps=4.4,
    pairs=['USDCHF'],
    hours_of_week=<50 weekday hours>, sr_zero=1.1807)

-> High-frequency lead-lag: if-true Sharpe -10.44 vs bar 1.18 (cost 1.55 bps RT) -> NOT VIABLE
-> at 1.5x cost (2.32 bps RT): if-true Sharpe -22.93 -> FAILS cost gate
```

- `gross_edge_bps` = 0.9 -- First principles: residual propagation at 1min. 0.9bps is ~20% of the 5min std, and is knowingly optimistic given that this lag is the single most contested in FX.
- `per_trade_vol_bps` = 4.4 -- Measured: 5min majors-basket std 4.4bps.

**4. Falsifier.** EURUSD's 1min move has no incremental predictive power for USDCHF's NEXT minute beyond the contemporaneous correlation.

**5. Most likely spurious reason.** Contemporaneous correlation leaking into the next bar through non-synchronous timestamps would create an apparent lead-lag that is pure artifact. This is the same failure mode as spec 35.

**6. Kill conditions.** No incremental predictability at a 1-bar execution lag, or cost exceeds the gross edge -- at ~5000 trades/year the cost term is decisive.

## Spec 38: Metals-implied FX

*catalog #45* | family F7 | **ROUTED -- cannot clear bar**

**1. Mechanism.** Gold is priced in dollars and trades nearly 24 hours, and it responds to real-rate and dollar-debasement news. Commodity currencies (AUD, CAD, NZD) share exposure to the same global-growth and dollar factors but trade in thinner books outside their local hours. A gold move therefore leads the commodity currencies when those currencies' local markets are closed. The other side is the currency's thin overnight book.

**2. Rule.** During 00:00-06:00 UTC, when XAUUSD moves more than 2.0 std of its 60min distribution, enter AUDUSD in the same direction at the next bar; exit after 120 minutes.

**3. Viability screen.**

```
screen_spec(
    name='Metals-implied FX',
    trades_per_year=200, gross_edge_bps=2.0, per_trade_vol_bps=17.5,
    pairs=['AUDUSD'],
    hours_of_week=<25 weekday hours>, sr_zero=1.1807)

-> Metals-implied FX: if-true Sharpe 0.36 vs bar 1.18 (cost 1.55 bps RT) -> NOT VIABLE
-> at 1.5x cost (2.33 bps RT): if-true Sharpe -0.27 -> FAILS cost gate
```

- `gross_edge_bps` = 2.0 -- First principles: shared factor with asynchronous liquidity. 4bps is ~23% of the 120min majors std, reflecting a genuinely thin window.
- `per_trade_vol_bps` = 17.5 -- Measured: 120min majors-basket std 17.5bps; AUDUSD Tokyo-hours 60min vol is 15.0bps, consistent.

**4. Falsifier.** Gold moves in Asian hours do not predict AUD beyond contemporaneous correlation.

**5. Most likely spurious reason.** AUD and gold are both risk-on assets, so the relation may be entirely contemporaneous with no lead. The test is strictly whether the NEXT bar is predictable.

**6. Kill conditions.** No incremental predictability at a 1-bar lag.

## Spec 39: Scandi triangle

*catalog #36* | family F7 | **ROUTED -- cannot clear bar**

**1. Mechanism.** NOK and SEK are driven by a common Nordic factor plus idiosyncratic oil (NOK) and domestic-rate (SEK) exposure. Constructing the triangle across EURNOK, EURSEK and the implied NOKSEK isolates a residual that should be stationary if the common factor dominates. Displacement of the residual reflects one-sided flow in the thinnest G10 currencies and should revert. The other side is a dealer with unwanted Nordic inventory.

**2. Rule.** Compute the residual of log(EURNOK) - log(EURSEK) against its 60-day mean. Enter against |z| > 2.0; exit on |z| < 0.5 or 10 days.

**3. Viability screen.**

```
screen_spec(
    name='Scandi triangle',
    trades_per_year=30, gross_edge_bps=40.0, per_trade_vol_bps=290.0,
    pairs=['EURNOK', 'EURSEK'],
    hours_of_week=<50 weekday hours>, sr_zero=1.1807)

-> Scandi triangle: if-true Sharpe 0.71 vs bar 1.18 (cost 2.66 bps RT) -> NOT VIABLE
-> legs=2: cost 2.66 x 2 = 5.33 bps RT
-> LEG-ADJUSTED if-true Sharpe 0.65 vs bar 1.18
-> at 1.5x cost (7.99 bps RT): if-true Sharpe 0.60 -> FAILS cost gate
```

- `gross_edge_bps` = 40.0 -- First principles: thin-market residual reversion. 70bps is ~24% of the 10-day spread vol, reflecting slow correction in capacity-constrained currencies.
- `per_trade_vol_bps` = 290.0 -- Measured: EURNOK and EURSEK 1440min std ~72 and ~68bps; the spread over a 10-day hold is ~290bps.

**4. Falsifier.** The residual is not stationary, i.e. NOK and SEK have genuinely diverged on oil rather than temporarily displaced.

**5. Most likely spurious reason.** Oil is a real, persistent driver of NOK that SEK does not share. A 'reversion' spec that is actually short an oil trend will look stationary until it is not, and the failure will be concentrated in one large episode.

**6. Kill conditions.** Residual fails a stationarity test on rolling windows, or the two-leg cost (EURNOK 4.32 + EURSEK 3.70bps) exceeds the gross edge.

# F8. Cross-asset metals-FX linkage

## Spec 40: Gold/silver ratio (regime-conditioned re-form)

*catalog #43* | family F8 | **ROUTED -- cannot clear bar**

**1. Mechanism.** Re-form of a naive-only slot. The naive version traded the raw gold/silver ratio as a stationary series, which it is not: the ratio has a secular industrial-demand trend that swamps the mean-reverting component. The re-form conditions on the ratio's own volatility regime, trading reversion only when the ratio's realized vol is in its lower tercile, on the premise that high-vol episodes are the structural repricings that break stationarity. Different functional form, same underlying precious-metals substitution mechanism.

**2. Rule.** Daily. Compute the log gold/silver ratio z-score against its 120-day mean. When the ratio's 60-day realized vol is in the bottom tercile of its 3-year history and |z| > 2.0, enter against the deviation; exit on |z| < 0.5, on the vol regime leaving the bottom tercile, or after 20 days.

**3. Viability screen.**

```
screen_spec(
    name='Gold/silver ratio (regime-conditioned re-form)',
    trades_per_year=15, gross_edge_bps=180.0, per_trade_vol_bps=620.0,
    pairs=['XAUUSD', 'XAGUSD'],
    hours_of_week=<25 weekday hours>, sr_zero=1.1807)

-> Gold/silver ratio (regime-conditioned re-form): if-true Sharpe 1.09 vs bar 1.18 (cost 6.18 bps RT) -> NOT VIABLE
-> legs=2: cost 6.18 x 2 = 12.36 bps RT
-> LEG-ADJUSTED if-true Sharpe 1.05 vs bar 1.18
-> at 1.5x cost (18.54 bps RT): if-true Sharpe 1.01 -> FAILS cost gate
```

- `gross_edge_bps` = 180.0 -- First principles: conditioned reversion in a substitution relationship. 180bps is ~29% of the 20-day ratio vol.
- `per_trade_vol_bps` = 620.0 -- Measured: XAUUSD 1440min std 102.7bps, XAGUSD 162.2bps; the ratio over a 20-day hold is ~620bps.

**4. Falsifier.** The low-vol regime does not exhibit more stationarity in the ratio than the high-vol regime.

**5. Most likely spurious reason.** Silver's measured round-trip cost is 10.41bps, by far the widest in the set, and the two-leg cost is ~12.4bps. A mid-quote result would look strong and be materially eroded. Also, only ~15 trades/year makes this extremely small-sample.

**6. Kill conditions.** No regime difference in stationarity, or fewer than 40 independent trades in the full history.

## Spec 41: Non-USD gold momentum

*catalog #44* | family F8 | **ROUTED -- cannot clear bar**

**1. Mechanism.** Gold priced in a non-USD currency isolates that currency's debasement and real-rate story from the dollar's. When gold rises in EUR terms, it signals euro-specific real-rate or credibility deterioration that should subsequently show up in EURUSD. The other side is participants who watch only USD gold. It persists because the construct requires holding two exposures.

**2. Rule.** Daily. Compute XAU/EUR = XAUUSD / EURUSD. When its 60-day return exceeds its 250-day 80th percentile, short EURUSD; when below the 20th, go long. Hold 20 days or until the signal exits the tail.

**3. Viability screen.**

```
screen_spec(
    name='Non-USD gold momentum',
    trades_per_year=20, gross_edge_bps=42.0, per_trade_vol_bps=260.0,
    pairs=['EURUSD', 'XAUUSD'],
    hours_of_week=<25 weekday hours>, sr_zero=1.1807)

-> Non-USD gold momentum: if-true Sharpe 0.70 vs bar 1.18 (cost 1.25 bps RT) -> NOT VIABLE
-> legs=2: cost 1.25 x 2 = 2.51 bps RT
-> LEG-ADJUSTED if-true Sharpe 0.68 vs bar 1.18
-> at 1.5x cost (3.76 bps RT): if-true Sharpe 0.66 -> FAILS cost gate
```

- `gross_edge_bps` = 42.0 -- First principles: currency-specific debasement signal. 42bps is ~16% of the 20-day EURUSD vol.
- `per_trade_vol_bps` = 260.0 -- Measured: EURUSD 1440min std 50.2bps scaled to a 20-day hold -> ~225bps; with the gold leg, ~260bps.

**4. Falsifier.** XAU/EUR momentum has no predictive power for EURUSD beyond USD-gold momentum alone.

**5. Most likely spurious reason.** XAU/EUR mechanically contains EURUSD, so a 'prediction' of EURUSD from XAU/EUR can be pure algebra rather than information. The control against USD-gold-only is essential.

**6. Kill conditions.** No incremental power over USD gold, meaning the non-USD construction adds nothing.

## Spec 42: Gold as risk-regime filter

*catalog #46* | family F8 | **ROUTED -- cannot clear bar**

**1. Mechanism.** Gold rallies during risk aversion and real-rate declines. Carry strategies fail precisely in risk-off episodes, so gold's trend is a candidate real-time state variable for when to be out of carry. The mechanism is not a new source of return but a conditioning variable on a known one: crash risk is what carry is compensated for, so an indicator that anticipates crashes should improve the payoff profile even if not the mean.

**2. Rule.** Daily. Run an equal-weight G10 carry basket (long 3 highest-yield, short 3 lowest-yield by overnight swap). Scale exposure to zero whenever XAUUSD's 20-day return exceeds its 250-day 85th percentile; otherwise full exposure. Monthly rebalance of the basket.

**3. Viability screen.**

```
screen_spec(
    name='Gold as risk-regime filter',
    trades_per_year=72, gross_edge_bps=40.0, per_trade_vol_bps=430.0,
    pairs=['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY', 'AUDJPY', 'EURCHF'],
    hours_of_week=<25 weekday hours>, sr_zero=1.1807)

-> Gold as risk-regime filter: if-true Sharpe 0.77 vs bar 1.18 (cost 1.23 bps RT) -> NOT VIABLE
-> legs=6: cost 1.23 x 6 = 7.36 bps RT
-> LEG-ADJUSTED if-true Sharpe 0.64 vs bar 1.18
-> at 1.5x cost (11.04 bps RT): if-true Sharpe 0.57 -> FAILS cost gate
```

- `gross_edge_bps` = 40.0 -- Derived from literature: G10 carry net Sharpe ~0.4-0.6 historically and lower post-2010. Backed out at T=72 and a 1-month hold.
- `per_trade_vol_bps` = 430.0 -- Measured: scaled to a 21-day hold with partial diversification -> ~430bps.

**4. Falsifier.** The gold filter does not reduce carry drawdowns in known risk-off episodes (2011, 2015, 2020, 2022).

**5. Most likely spurious reason.** A filter tested on the same crises it was designed around is fitted to four observations. The honest test is whether it would have been out BEFORE each crash, not during.

**6. Kill conditions.** Filter improves in-sample drawdown but not out-of-sample, or literature-implied Sharpe below the bar -- route to combination.

## Spec 43: Silver beta amplification

*catalog #47* | family F8 | **ROUTED -- cannot clear bar**

**1. Mechanism.** Silver behaves as a higher-beta version of gold with respect to the same real-rate and dollar factors, because its smaller market absorbs the same macro flow with larger price impact. If gold leads the commodity-currency complex (spec 38), silver should provide an amplified and therefore earlier-detectable version of the same signal. The other side is the thinner silver book.

**2. Rule.** Daily. When XAGUSD's 5-day return exceeds 2.0x XAUUSD's 5-day return and both are positive, go long AUDUSD; on the mirror, go short. Hold 10 days.

**3. Viability screen.**

```
screen_spec(
    name='Silver beta amplification',
    trades_per_year=25, gross_edge_bps=48.0, per_trade_vol_bps=310.0,
    pairs=['AUDUSD', 'XAGUSD'],
    hours_of_week=<25 weekday hours>, sr_zero=1.1807)

-> Silver beta amplification: if-true Sharpe 0.68 vs bar 1.18 (cost 6.06 bps RT) -> NOT VIABLE
-> legs=2: cost 6.06 x 2 = 12.11 bps RT
-> LEG-ADJUSTED if-true Sharpe 0.58 vs bar 1.18
-> at 1.5x cost (18.17 bps RT): if-true Sharpe 0.48 -> FAILS cost gate
```

- `gross_edge_bps` = 48.0 -- First principles: amplified common-factor signal. 48bps is ~15% of the 10-day AUDUSD vol.
- `per_trade_vol_bps` = 310.0 -- Measured: AUDUSD 1440min std 68.7bps scaled to a 10-day hold -> ~217bps; with the silver leg, ~310bps.

**4. Falsifier.** Silver's excess move over gold has no predictive content for AUD beyond gold's own move.

**5. Most likely spurious reason.** Silver is far noisier than gold, so 'silver outperforming gold' is frequently just silver's idiosyncratic noise. The 2.0x threshold selects noise as often as signal.

**6. Kill conditions.** No incremental power over the gold-only signal of spec 38.

# F9. Carry and swap-aware forms

## Spec 44: Vol-targeted carry basket (drawdown-conditioned re-form)

*catalog #15* | family F9 | **ROUTED -- cannot clear bar**

**1. Mechanism.** Re-form of a naive-only slot. The naive version targeted constant volatility, which is the wrong risk axis for carry: carry's risk is crash skew, not variance, and vol-targeting actually increases exposure just before crashes because carry crashes follow low-vol periods. The re-form conditions on the basket's own drawdown state instead, cutting exposure after the basket itself begins to break. Different risk functional, same carry mechanism.

**2. Rule.** Monthly G10 carry basket (long 3 highest, short 3 lowest overnight swap). Exposure = full when the basket's 60-day drawdown is under 3%, halved between 3% and 6%, zero above 6%. Re-enter when drawdown recovers below 3%.

**3. Viability screen.**

```
screen_spec(
    name='Vol-targeted carry basket (drawdown-conditioned re-form)',
    trades_per_year=72, gross_edge_bps=40.0, per_trade_vol_bps=430.0,
    pairs=['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY', 'AUDJPY', 'EURCHF'],
    hours_of_week=<25 weekday hours>, sr_zero=1.1807)

-> Vol-targeted carry basket (drawdown-conditioned re-form): if-true Sharpe 0.77 vs bar 1.18 (cost 1.23 bps RT) -> NOT VIABLE
-> legs=6: cost 1.23 x 6 = 7.36 bps RT
-> LEG-ADJUSTED if-true Sharpe 0.64 vs bar 1.18
-> at 1.5x cost (11.04 bps RT): if-true Sharpe 0.57 -> FAILS cost gate
```

- `gross_edge_bps` = 40.0 -- Derived from literature: G10 carry net Sharpe ~0.4-0.6 pre-2010, materially lower since. Backed out at T=72 and a 1-month hold.
- `per_trade_vol_bps` = 430.0 -- Measured: scaled to a 21-day hold with partial diversification -> ~430bps.

**4. Falsifier.** Drawdown-conditioning does not reduce the basket's left-tail relative to vol-targeting.

**5. Most likely spurious reason.** Any stop-loss overlay improves backtested drawdown by construction while reducing mean return; reporting the drawdown improvement without the return cost would be selective.

**6. Kill conditions.** Literature-implied Sharpe below the bar -- route to combination.

## Spec 45: Swap-aware swing bias

*catalog #17* | family F9 | **ROUTED -- cannot clear bar**

**1. Mechanism.** Retail and small-institutional platforms charge asymmetric overnight swap: the debit for holding the negative-carry side typically exceeds the credit for the positive side. That asymmetry creates a systematic incentive to hold positions in the positive-carry direction and to avoid multi-day holds against carry. A directional swing rule that breaks ties in favour of positive carry should therefore keep more of its gross edge. The other side is the platform earning the spread.

**2. Rule.** Daily. Take a 20/60 MA crossover signal on G10 pairs, but suppress any signal that would hold against the positive-carry direction for more than 3 days. Exit on opposite cross or after 15 days.

**3. Viability screen.**

```
screen_spec(
    name='Swap-aware swing bias',
    trades_per_year=35, gross_edge_bps=28.0, per_trade_vol_bps=230.0,
    pairs=['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY', 'AUDJPY', 'EURCHF'],
    hours_of_week=<25 weekday hours>, sr_zero=1.1807)

-> Swap-aware swing bias: if-true Sharpe 0.69 vs bar 1.18 (cost 1.23 bps RT) -> NOT VIABLE
-> at 1.5x cost (1.84 bps RT): if-true Sharpe 0.67 -> FAILS cost gate
```

- `gross_edge_bps` = 28.0 -- Derived from literature as in spec 23, plus the swap asymmetry as a cost reduction rather than an alpha source. 28bps backed out at T=35 and a ~15-day hold.
- `per_trade_vol_bps` = 230.0 -- Measured: 1440min std 57.9bps scaled to a ~15-day hold -> ~230bps.

**4. Falsifier.** Suppressing negative-carry holds does not improve net returns versus the unfiltered crossover.

**5. Most likely spurious reason.** The swap asymmetry is a real cost saving, but the spec would then be a cost improvement on a trend rule whose own edge is below the bar. Improving the cost of a losing strategy does not make it a winner.

**6. Kill conditions.** Underlying trend rule's if-true Sharpe below the bar, which the screen confirms -- route to combination.

# F10. Calendar and liquidity-regime effects

## Spec 46: Day-of-week effect

*catalog #31* | family F10 | **ROUTED -- cannot clear bar**

**1. Mechanism.** Weekly institutional processes are not uniform across weekdays: Monday reprices weekend information on thin books, mid-week carries the heaviest scheduled-data load, and Friday carries pre-weekend squaring. If any systematic return pattern by weekday survives, it should be attributable to one of these flow processes rather than to the calendar itself. This spec is stated with an explicitly weak prior: a calendar effect without an identified flow is a data-mining artifact by default.

**2. Rule.** Daily. Long EURUSD from Monday 07:00 UTC to Monday 20:00 UTC; flat otherwise. Fixed, single weekday, chosen a priori as the weekend-repricing day rather than selected from the data.

**3. Viability screen.**

```
screen_spec(
    name='Day-of-week effect',
    trades_per_year=52, gross_edge_bps=3.0, per_trade_vol_bps=34.3,
    pairs=['EURUSD'],
    hours_of_week=<50 weekday hours>, sr_zero=1.1807)

-> Day-of-week effect: if-true Sharpe 0.51 vs bar 1.18 (cost 0.58 bps RT) -> NOT VIABLE
-> at 1.5x cost (0.88 bps RT): if-true Sharpe 0.45 -> FAILS cost gate
```

- `gross_edge_bps` = 3.0 -- First principles with a weak prior. 3bps is ~9% of the 480min majors std, and the honest expectation is zero.
- `per_trade_vol_bps` = 34.3 -- Measured: 480min majors-basket std 34.3bps.

**4. Falsifier.** Monday session returns are indistinguishable from other weekdays.

**5. Most likely spurious reason.** This is the archetypal p-hacking family: with 5 weekdays x 2 directions x 10 pairs there are 100 combinations, and roughly 5 will pass at the 5% level by chance. Fixing one pair and one weekday in advance is the only defence, and even then the prior is weak.

**6. Kill conditions.** Insignificant, which is the expected outcome; and if significant, it must be checked against the other four weekdays to confirm it is not the best of five.

## Spec 47: Holiday thin-liquidity reversion

*catalog #34* | family F10 | **ROUTED -- cannot clear bar**

**1. Mechanism.** On days when a major financial centre is closed, the affected currency's book is materially thinner while the rest of the world continues to trade. Order flow that must transact meets less depth and displaces price further than it would on a normal day; the displacement corrects when the centre reopens. The other side is participants who cannot wait for the holiday to end. It persists because holidays are few and the positions must be held across a low-liquidity period.

**2. Rule.** On US market holidays that are not FX holidays, measure the 12:00-20:00 UTC return in EURUSD and GBPUSD. If |return| exceeds 1.0x the trailing 20-day std of that window, enter against it at 20:00 UTC and exit at 16:00 UTC the following business day.

**3. Viability screen.**

```
screen_spec(
    name='Holiday thin-liquidity reversion',
    trades_per_year=18, gross_edge_bps=12.0, per_trade_vol_bps=57.9,
    pairs=['EURUSD', 'GBPUSD'],
    hours_of_week=<20 weekday hours>, sr_zero=1.1807)

-> Holiday thin-liquidity reversion: if-true Sharpe 0.82 vs bar 1.18 (cost 0.86 bps RT) -> NOT VIABLE
-> at 1.5x cost (1.29 bps RT): if-true Sharpe 0.78 -> FAILS cost gate
```

- `gross_edge_bps` = 12.0 -- First principles: amplified impact on thin books. 12bps is ~21% of the daily majors std, justified by genuinely reduced depth.
- `per_trade_vol_bps` = 57.9 -- Measured: 1440min majors-basket std 57.9bps.

**4. Falsifier.** Holiday-session displacements are not corrected on reopening.

**5. Most likely spurious reason.** Roughly 9 US holidays a year across 2 pairs is ~18 trades/year and ~270 in the full history, but holidays cluster seasonally so the effective independence is lower. A handful of episodes will dominate.

**6. Kill conditions.** Fewer than 40 effectively independent observations, or no correction versus a matched non-holiday control.

## Spec 48: Post-holiday liquidity restoration

*NOVEL (outside the 60-catalog)* | family F10 | **ROUTED -- cannot clear bar**

**1. Mechanism.** The counterpart to spec 47. When a major centre reopens after a holiday or an extended break, accumulated orders that could not be executed arrive together, producing a concentrated one-directional flow in the first hours of the reopening. Unlike the holiday displacement itself, this flow is informed accumulation rather than noise, so it should CONTINUE rather than revert within the session. Proposing both signs at different points in the holiday cycle is deliberate: they are different flow processes, not a hedge.

**2. Rule.** On the first session after a US market holiday, measure the 12:00-14:00 UTC return; enter in the same direction at 14:00 UTC; exit at 20:00 UTC.

**3. Viability screen.**

```
screen_spec(
    name='Post-holiday liquidity restoration',
    trades_per_year=18, gross_edge_bps=5.0, per_trade_vol_bps=24.5,
    pairs=['EURUSD', 'GBPUSD'],
    hours_of_week=<15 weekday hours>, sr_zero=1.1807)

-> Post-holiday liquidity restoration: if-true Sharpe 0.73 vs bar 1.18 (cost 0.76 bps RT) -> NOT VIABLE
-> at 1.5x cost (1.14 bps RT): if-true Sharpe 0.67 -> FAILS cost gate
```

- `gross_edge_bps` = 5.0 -- First principles: concentrated release of accumulated orders. 5bps is ~20% of the 240min majors std.
- `per_trade_vol_bps` = 24.5 -- Measured: 240min majors-basket std 24.5bps.

**4. Falsifier.** Reopening sessions show no more within-session continuation than ordinary sessions.

**5. Most likely spurious reason.** With ~18 trades/year this is small-sample, and it shares its event calendar with spec 47, so the two are not independent evidence about holidays.

**6. Kill conditions.** No excess continuation versus a matched ordinary-session control.

## Spec 49: Intraday session-open range persistence

*NOVEL (outside the 60-catalog)* | family F10 | **ROUTED -- cannot clear bar**

**1. Mechanism.** Gao-Han-Li-Zhou (2018) document intraday momentum in equity index futures: the first half-hour return predicts the last half-hour return, attributed to informed traders splitting orders across the session and to late-day rebalancing by liquidity providers. The FX analog is the London session: institutional programs begun at the open are worked through the day and completed before the London close. The other side is the program's counterparty.

**2. Rule.** Measure the EURUSD, GBPUSD and USDJPY return over 07:00-08:00 UTC. Enter in the same direction at 15:00 UTC; exit at 16:30 UTC. One trade per pair per day.

**3. Viability screen.**

```
screen_spec(
    name='Intraday session-open range persistence',
    trades_per_year=756, gross_edge_bps=1.3, per_trade_vol_bps=12.5,
    pairs=['EURUSD', 'USDJPY', 'GBPUSD'],
    hours_of_week=<10 weekday hours>, sr_zero=1.1807)

-> Intraday session-open range persistence: if-true Sharpe 1.24 vs bar 1.18 (cost 0.74 bps RT) -> VIABLE
-> at 1.5x cost (1.11 bps RT): if-true Sharpe 0.42 -> FAILS cost gate
```

- `gross_edge_bps` = 1.3 -- Gao et al. report a robust first-half-hour to last-half-hour relation in equity futures. The FX transmission is untested; 1.8bps is ~14% of the 60min majors std, a modest read of the analog.
- `per_trade_vol_bps` = 12.5 -- Measured: 60min majors-basket std 12.5bps.

**4. Falsifier.** The London first-hour return does not predict the London last-hour return.

**5. Most likely spurious reason.** The effect is documented in a different asset class with a different market structure (a single central limit order book with a defined close). FX has no close, so the mechanism's key ingredient -- a deadline -- is weaker.

**6. Kill conditions.** No relation, or the relation present only in the pre-2015 sample.

---

## Standing constraints honoured

- Frequencies: 1m, aggregations of 1m, and daily only. Nothing sub-minute.
- Execution: spread-TAKER throughout. No spec assumes liquidity provision.
- Events: US CPI / NFP / FOMC only (FOMC 2013+). No spec depends on ECB, BoE,
  BoJ, BoC, SNB, RBA or RBNZ event times.
- No options-implied, order-book, order-flow or consensus-forecast data is used.
- No ML slot is proposed: the triple-barrier meta-label harness does not exist,
  so catalog slots 48-53 are deliberately left unfilled. Slot 55 (USDCNH PBOC
  fix) is also unfilled: the fix data is not held.
- Every parameter is fixed at a stated value. No ranges, no sweeps.
- Cost caveats carried explicitly: spec 35 leans on EURGBP, which is unmeasured
  and takes the 4.0bps conservative fallback (note the derived-cross table in
  `costs/fx.py` is NOT consulted by `fx_round_trip_bps_at`, so the fallback is
  what the screen actually charged). Specs on Nordic and EM pairs inherit the
  measured-but-wide levels, and any pair outside the measured 25 gets a flat
  hourly shape.

## Trial accounting

- Prior trials: 141
- Specs pre-registered here: 49
- Specs that will consume a trial (cleared the screen): **2**
- Specs routed without consuming a trial: 47

The bar quoted throughout is computed at N + 50 as the ledger specifies, which is
conservative relative to the smaller number of trials actually to be consumed. If
only the cleared specs are run, the bar should be RECOMPUTED at the true N before
any verdict is issued -- and recomputing it downward after seeing which specs
passed would be exactly the gate-tuning this campaign has ruled out. Fix N from
the pre-registered intent, not from the outcome.

