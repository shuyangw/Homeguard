# FX Wave 3 Resolution: zero runnable specs, zero trials consumed - 2026-07-26

**Outcome: the wave does not run on spot FX.** Not because the mechanisms
failed, but because the account cannot reach the order size at which they are
tradeable. No backtest was run. **N stays at 141** and the deflated bar is
unmoved.

## What happened

49 specs were pre-registered across 10 mechanism families
(`20260726_fx_wave3_slate_prereg.md`), generated blind. The viability screen
routed 47 before any data was touched. Of the 2 that cleared, both were then
killed by a cost term modelled the same day: IBKR's **$2 per-order commission
minimum**, confirmed from the account's own schedule.

The minimum stops binding above $100,000 of notional per order. Both surviving
specs trade 6 majors concurrently, so per-order notional is capital divided by
six, and the account is **cash-only** (no ECP status, therefore no leveraged
spot FX at IBKR).

Cash spot, 1x, at the 1.5x cost gate against a bar of 1.1807:

Bar provenance, stated once: the trial-Sharpe dispersion is v=0.4293 over 130
observed trials. SR_zero = 1.1372 at N=141 (the count today) and
1.1807 at N=141+50 (the count a 50-spec wave would face, which is the
bar every wave-3 spec was screened against).

| capital | notional/order | cost RT | #18 Month-end | #21 Quarter-end |
|---:|---:|---:|---:|---:|
| $25,000 | $4,167 | 10.28 bps | -4.00 | -0.68 |
| $50,000 | $8,333 | 5.48 bps | -0.94 | 0.76 |
| $100,000 | $16,667 | 3.08 bps | 0.59 | 1.48 |

Capital required to clear on cash spot: **$163,855** for #18, **$70,932** for
#21. At $50k of cash spot #18 is deeply negative -- commission alone exceeds its
entire gross edge -- and #21 reaches only 0.76 against a 1.18 bar.

CORRECTION (same day): an earlier version of this table used inferred
parameters for #21 (edge 9.0 / vol 20.0) instead of its pre-registered ones
(edge 12.0 / vol 24.5). The headline is unchanged -- both fail at $50k cash
spot -- but #21's figures were understated, and it does clear at $100k, which
the earlier table denied. Read parameters from the locked spec, never infer
them from surrounding prose.

The pre-registered combination spec is **VOID** by its own locked rule, which
required K >= 3 members. That clause was written before any component existed
precisely so it could not be renegotiated afterwards. It fired as designed.

## What this finding is, and what it is not

**It IS:** a scoped statement that the surviving fix-flow mechanisms are not
tradeable as *cash spot FX at retail account size*, because a fixed $2 per-order
commission dominates at achievable notionals. This is an **access and size**
constraint, not a market one.

**It is NOT** evidence about whether the mechanisms are real. Nothing was
backtested. Month-end fix rebalancing flow remains a documented effect with a
structural driver; we simply cannot reach the order size that makes it
survivable after costs on this venue.

Stating it the wrong way round would be the error the North Star warns about:
"month-end fix flow does not work" would be an over-generalisation from what is
actually a constraint on our own account.

## The screen did its whole job

49 specs proposed, 49 not run, **0 trials consumed, bar unmoved at 1.1807**.

Under the previous regime every one of those specs would have been a backtest,
and each would have raised the bar for everything after it. The two failure
modes it caught are different and both matter:

1. **47 specs could not clear the bar even if entirely correct** -- their if-true
   Sharpe was below SR_zero by construction. Daily G10 factor specs are the
   clearest case: their if-true Sharpe is *derived from* the 0.3-0.6 literature
   Sharpe, so it is definitionally under a 1.18 bar.
2. **2 specs could clear on signal but not at our order size.** That is only
   visible because the cost model now carries a per-order minimum. It did not
   yesterday.

## The live alternative: CME FX futures

Futures dissolve the exact constraint that killed spot. One 6E contract is
125,000 EUR (about $135,000), already above the commission-minimum threshold, at
roughly 0.185 bps/side against cash spot's 2.40 bps at $8,333. They are
leveraged by construction and carry no ECP gate.

The repo already carries the apparatus:

| component | state |
|---|---|
| All 8 CME FX futures contract specs | present (`src/data/futures/contract_specs.py`) |
| 6E commission + spread in the cost model | present (`src/backtesting/costs/futures.py`) |
| Asset-class mapping 6A/6B/6C/6E/6J/6M/6N/6S -> fx | present |
| Futures backtest engine + portfolio simulator | present |
| Walk-forward runner (`run_carver_walkforward.py`) | present |

Two honest caveats. **Micros do not help**: M6E at $13,500 notional costs about
1.85 bps/side, as bad as small spot orders, so this needs full contracts. And
six full contracts on $50k is 11.6x notional leverage ($578,375 of actual
per-root notional, not 6 x 6E); margin permits it (about $15k initial) but the binding constraint becomes risk, not cost.

This is a **different instrument, so it needs its own pre-registration and its
own trials.** The locked spot specs must not be quietly re-pointed at futures.

## Recommended next step

Do not chase leverage for these two specs. Regenerate against futures, where the
cost structure is genuinely favourable rather than marginal, using the corrected
cost model. Several of the 47 routed specs were routed on costs that are now
more accurate, and a futures-denominated slate would screen differently.

Before that: verify the CME/IBKR futures fee assumption (0.85 + 1.60 per
contract per side is indicative and UNVERIFIED), since it is the input the whole
comparison rests on, and getting it wrong here would repeat the error this wave
just caught.
