# FX Campaign: Final Synthesis - 2026-07-26

Supersedes `20260719_fx_catalog_campaign_synthesis.md`, which predates the
apparatus sweep, the corrected bar, and wave 3.

## The result in one line

**141 pre-registered trials, zero passes, and a 50th spec that will never run
because the bar has outgrown what a single FX factor can produce.**

## Two findings, and the second is the important one

### 1. The tested slice carries no deflation-surviving edge

Daily, spot, spread-taker, G10 and EM, 2011-2026. Twelve-plus gated strategies
across trend, cross-sectional momentum, carry (naive and filtered), value,
session breakout, metals ratio, PCA residual, macro regime, seasonality,
cointegration relative value, speculative positioning, commodity terms-of-trade
and range-based signals. All fail net of costs, robust to an
IBKR-optimistic cost assumption. Best genuine out-of-sample result: **+0.087**.

This is a real, scoped negative and it cost roughly nothing in live capital.

### 2. The bar has outgrown the asset class's single-factor edge

This is the finding that actually closes the book, and it is arithmetic rather
than empirical.

- Deflated bar: **SR_zero = 1.1372** at N=141, rising with every trial
- Realistic ceiling for one daily G10 factor: **0.3-0.6** net Sharpe

Wave 3 made this explicit. 49 specs were generated blind, screened before any
data was touched, and **41 could not clear the bar even if entirely correct** --
their if-true Sharpe is *derived from* the literature Sharpe, so it sits below
the bar by construction. Not "probably fails". Cannot pass.

That is venue-independent, cost-independent, and permanent, because N never
shrinks. No amount of further FX search changes it. The only routes past such a
bar are a genuinely higher-edge mechanism class, or combining many
near-uncorrelated edges -- and the combination spec voided itself for want of
members.

**Most of the 141 trials were arithmetically incapable of passing when they were
run.** Each still raised the bar for everything after it. That is the campaign's
most expensive lesson and the reason the viability screen now exists.

## What wave 3 cost: nothing

49 specs pre-registered, 49 not run, **0 trials consumed, bar unmoved**.

| outcome | specs | venue-dependent |
|---|---:|---|
| Cannot clear the bar even if true | 41 | no |
| Clear at 1x cost, fail the 1.5x gate | 6 | marginally |
| Clear both, killed by the $2 per-order commission minimum | 2 | yes |

The last two are an **access** constraint, not a market one: a cash-only account
at $50k puts ~$8,333 behind each of six concurrent orders, where a fixed $2
minimum dominates the entire gross edge. They are routed to forward paper.

## The apparatus was systematically permissive

Twelve defects were found and fixed. The pattern matters more than any one:
**almost every silent default moved in the permissive direction.**

| defect | direction |
|---|---|
| Trial count silently fell N=141 -> 40 when the registry was locked | gate 35% softer |
| Two runners deflated against SR_zero = 0.0000 | no bar at all |
| PSR/DSR computed with annualized Sharpe against daily n | ~16x inflated |
| Ledger's `.get(..., "OPEN")` served a dead slot as fresh inventory | invites re-proposing a failure |
| Cost model missing the $2 per-order commission minimum | under-charged small orders |
| Intraday cost model hour-blind (real dispersion 34x) | wrong in both directions |
| `spread_model` artifact: a constant dressed as a surface, zero consumers | misleading |
| Event calendars were rule proxies (CPI exact in 14% of months) | silent non-events |
| PBO computed on 65 of ~260 OOS days | optimistic |
| Same-bar fills (no execution lag) | optimistic |
| Two data-layer lookaheads (spike cleaning, FRED publication lag) | optimistic |
| Degenerate filter ran under a pre-registered name (EM seatbelt) | untested mechanism, reported as tested |

A research apparatus whose failure modes are one-directional is not neutral. It
is a machine for producing false positives, and the only reason this campaign did
not produce any is that the underlying signals were weak enough to fail anyway.

## What is closed, and at exactly what scope

**Closed:** retail, cash-account, spot, spread-taking FX at daily and intraday
frequency, for single-factor mechanisms, at a deflated bar of ~1.14.

**Not closed, and must not be claimed:**
- Whether these mechanisms are real. Month-end fix flow was never backtested.
- FX as an asset class. Profitable participants exist; they operate as liquidity
  providers, at latencies we cannot reach, or with flow we cannot see.
- Liquidity provision. Bounded, not tested: majors are not maker-viable for us,
  wide crosses show a positive gross bound of 1-2 bps per fill, and closing the
  gap needs L2 queue data.

## Retired without building, each on arithmetic

| item | why |
|---|---|
| Non-US CB calendars | event specs need 21x more events; all 8 central banks gives ~3x |
| ML meta-label harness (6 slots) | meta-labelling filters direction, cannot create it; best case ~1.04 vs 1.18 |
| More measured spread pairs | nothing is blocked on them |
| Spot maker backtest | bound-yes, backtest-no without queue data |

Four builds avoided by doing the arithmetic first. That is the viability screen
generalising beyond specs.

## Durable assets

Spot / intraday / spread engines; the reusable intraday day-loop harness;
session clock; measured hour-of-week cost surface (27 pairs); authoritative
US event calendar validated against our own data; S&P benchmark harness; CPCV /
DSR / PBO gate; the statistical-viability screen; the degenerate-signal
tripwire; registry duplicate detection; the forward-paper log; and the
blind-generation ledger. All asset-class portable except the FX data itself.

## What remains live

1. **Forward paper**, running: two fix specs, 12 and 4 events a year, zero trials.
2. **CME FX futures**, if pursued: dissolves the access constraint but not the
   arithmetic one, and only 6E has a measured fee (the other seven roots sit at a
   blanket estimate 3-5x apart from it).
3. **Asset-class pivot**, with the trigger substantially fired.

The honest read: FX at this account size and this bar is finished. The apparatus
built to prove it is the asset worth carrying forward.
