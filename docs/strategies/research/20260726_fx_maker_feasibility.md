# Maker / Liquidity-Provision Feasibility Study - 2026-07-26

**Verdict: a maker BACKTEST is NOT defensible on minute bars. A maker BOUND is.**
Recommendation: do not build a maker strategy; use the bound to scope future work.

## The question

Every one of the 141 FX specs paid the spread as a TAKER. The North Star names
the cost side as a verdict-flipping mis-specification: a large share of real FX
edge is EARNED as a liquidity provider, where the spread flips from a cost to
revenue, replaced by adverse selection. Phase 2 asked whether we can test that
honestly with the data we have (minute bid/ask, no order book).

## Method (measured, not assumed)

Post a passive BUY at bar t's bid; fill only if bar t+1 trades strictly THROUGH
the level (a touch is not a fill -- that would assume queue priority we cannot
verify). P&L per fill = mid(t+1+k) - fill price, in bps, which already contains
the half-spread captured and the subsequent adverse drift. EURUSD / EURNOK /
USDMXN, June 2024, ~29k minute quotes each.

## Result 1: tight majors are NOT maker-viable (robust)

| pair | half-spread | net@1m | net@5m | net@30m |
|---|---:|---:|---:|---:|
| EURUSD | 0.13 bps | -0.048 | -0.065 | -0.156 |

Adverse selection exceeds the entire half-spread, and it gets worse with horizon.
This holds at EVERY fill-strictness margin tested (-0.03 to -0.11 bps across
margins 0-4x half-spread). A robust negative, and an intuitive one: at a 0.13 bps
spread there is nothing to compensate being run over.

## Result 2: wide crosses show a persistent positive, but it is an UPPER BOUND

EURNOK, varying how strictly price must trade through the posted level:

| margin | fills | fill rate | net@5m |
|---:|---:|---:|---:|
| 0.0 | 24,917 | 86.7% | +2.534 |
| 0.5 | 9,385 | 32.6% | +1.587 |
| 1.0 | 5,243 | 18.2% | +1.369 |
| 2.0 | 1,825 | 6.3% | +1.259 |
| 4.0 | 271 | 0.9% | +2.135 (only 271 fills -- noise) |

The positive does NOT collapse as fills get realistic; it stabilises near
+1.3-1.6 bps at 6-18% fill rates. USDMXN behaves similarly (+1.9 to +2.8).

## Why this is still not backtestable

The 86.7% fill rate at margin 0 is the tell: no real passive order fills that
often. Tightening the rule fixes the fill RATE but not the fundamental gap --
**we cannot model queue position**, and queue position is the mechanism through
which adverse selection actually operates. At the top of book you fill only when
the size ahead of you is exhausted, which is precisely when flow is informed
against you. Our rule instead assumes front-of-queue at every fill.

So the measured +1.3-1.6 bps is an **upper bound under a front-of-queue
assumption**, not an achievable edge. A strategy backtest built on it would
inherit that assumption invisibly and could show a spectacular, entirely fake
Sharpe -- the exact failure this study existed to prevent. Note what would have
happened had we skipped it: an EURNOK maker backtest would have "worked".

## What we can and cannot claim

CAN: majors are not maker-viable for us at any plausible fill assumption; wide
crosses have a positive gross bound of order 1-2 bps per fill before queue
effects; the bound is larger where the spread is larger, as theory predicts.

CANNOT: that any of it is capturable. Closing the gap needs L2 / queue data and
realistic latency -- which is the genuinely inaccessible category the North Star
says to name honestly rather than pretend to replicate.

## Recommendation

1. Do NOT build a maker strategy backtest on minute bars. Any gate result would
   be an artifact of the fill assumption.
2. Record the bound as the scoped answer to the cost-side axis: the taker
   negatives across 141 specs are NOT explained away by "we should have been
   makers" for majors, since maker economics are negative there too.
3. If the maker axis is ever to be tested properly it requires an L2/tick data
   acquisition, which is a separate project with its own cost/benefit.
4. The remaining accessible axis is therefore FREQUENCY (intraday taker, 21
   catalog strategies on an already-built engine) or a different ASSET CLASS.
