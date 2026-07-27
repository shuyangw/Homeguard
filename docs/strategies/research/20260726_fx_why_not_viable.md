# Why FX Is Not Viable Right Now: the complete record

**Date:** 2026-07-26
**Status:** Final. Supersedes the interim syntheses of 2026-07-19.
**Scope of the claim:** retail, cash-account, spot FX, spread-taking, daily and
intraday, at a deflated bar of ~1.14. Read Section 8 before quoting this
anywhere, because most of what people would want to conclude from it is not
supported.

---

## 1. The verdict

**141 pre-registered trials, zero passes. A 49-spec wave 3 that consumed zero
further trials because 41 of its specs could not have passed even if entirely
correct.**

FX is not viable *right now* for three reasons, and they are not equally
permanent:

| # | reason | permanent? |
|---|---|---|
| 1 | The deflated bar has outgrown what a single FX factor can produce | **Yes** -- N never shrinks |
| 2 | A cash-only account cannot reach the order size the survivors need | No -- capital or venue fixes it |
| 3 | Fixed per-order commission dominates at retail order sizes | No -- same fix |

Reason 1 is the one that closes the book. Reasons 2 and 3 only matter because
reason 1 left so little standing that the survivors were marginal to begin with.

---

## 2. Everything that was tested

### 2.1 Gated strategies, all waves

Eighteen strategies reached a walk-forward gate. Every one failed.

| # | strategy | mechanism family | verdict |
|---|---|---|---|
| 3 | TSMOM portfolio | time-series momentum | FAIL-naive |
| 4 | Cross-sectional momentum | relative momentum | FAIL-naive |
| 6 | ADX-gated trend | trend + regime filter | FAIL |
| 12 | Keltner reversion | range mean-reversion | FAIL |
| 15 | Vol-targeted carry basket | carry | FAIL-naive |
| 16 | Carry-momentum filter | filtered carry | FAIL-enh |
| 18 | EM carry (EM7) | carry, emerging markets | FAIL |
| 19 | Carry-unwind detector | crash-filtered carry | FAIL-enh |
| 20 | London open breakout | intraday session breakout | FAIL (cost-robust) |
| 27 | Bandwidth squeeze | volatility compression | FAIL |
| 29 | Vol-spike fade | volatility mean-reversion | FAIL |
| 30 | Relative-vol pair | volatility relative value | REJECT |
| 33 | Turn-of-month USD | calendar seasonality | REJECT |
| 35 | AUD/NZD pairs | cointegration relative value | REJECT |
| 37 | Cointegration scanner | systematic pairs | REJECT |
| 39 | PCA dollar-factor residual | statistical residual | REJECT |
| 42 | RORO regime spread | macro regime | WEAK |
| 43 | Gold/silver ratio | metals ratio | FAIL-naive |

Plus, not all separately tracked: an 8-leg PPP-style **value** test, a **COT
speculative positioning** wave (3 specs, both signs, time-series and
cross-sectional), a **Kalman dynamic hedge-ratio** diagnostic, a **commodity
terms-of-trade** wave (3 specs), and a **cost-sensitivity re-gate** of six
prior failures at an IBKR-optimistic assumption.

`FAIL-naive` means only the naive form was tested and an enhanced form remained
open. `REJECT` means the idea was killed across forms. `WEAK` means it produced
a positive gross number that died after cost and deflation.

### 2.2 The three results that look like leads and are not

**Kalman hedge ratio, +0.417 OOS.** The highest raw out-of-sample Sharpe of the
entire campaign, and not alpha. The rolling-OLS baseline it beat had a hedge
ratio wandering over [0.013, 1.656] -- a beta of 0.013 on two cointegrated
currencies is an outright directional bet wearing a pairs label. The Kalman
filter acted as heavy shrinkage on a near-collinear regression. Ten of 3,185
out-of-sample days produced 94% of the P&L. PBO 0.893, DSR effectively zero. The
correct and narrower takeaway: hedge-ratio mis-specification is eliminated as
the explanation for the pairs failure.

**TOT-OIL, +0.0505, surviving a 1.5x cost stress.** The campaign's only
cost-robust positive. The oil-to-CAD/NOK transmission is real and correctly
signed (pre-check correlation -0.79), but its daily-sampled tradable content is
about 1/22 of the deflated bar. Economically trivial rather than
cost-destroyed. It also traded USDNOK under a cost model later found 17x too
cheap on Nordic crosses, so even this figure flatters itself.

**OHLC vol-spike fade, +0.087.** The best genuine number in the campaign, from
594 sparse fills with per-window Sharpe ranging -1.90 to +1.93. A Probabilistic
Sharpe Ratio of 0.62 says it plainly: indistinguishable from noise.

### 2.3 Wave 3: 49 specs, none run

Generated blind in a separate session fed only a leak-checked ledger, with every
result withheld. Ten mechanism families:

| family | mechanism | specs | cleared the bar |
|---|---|---:|---:|
| F1 | US scheduled-event time (CPI / NFP / FOMC) | 6 | 0 |
| F2 | Session and time-of-day segmentation | 11 | 0 |
| F3 | Benchmark fixing and rebalancing flow | 4 | **2** |
| F4 | Intraday breakout and volatility expansion | 6 | 0 |
| F5 | Intraday mean reversion | 4 | 0 |
| F6 | Cross-sectional and dollar-factor structure | 5 | 0 |
| F7 | Lead-lag and cross-market propagation | 3 | 0 |
| F8 | Cross-asset metals-FX linkage | 4 | 0 |
| F9 | Carry and swap-aware forms | 2 | 0 |
| F10 | Calendar and liquidity-regime effects | 4 | 0 |

Outcome, and the split is the finding:

| outcome | specs | venue-dependent? |
|---|---:|---|
| Could not clear the bar **even if entirely correct** | 41 | no |
| Cleared at 1x cost, failed the mandatory 1.5x gate | 6 | marginally |
| Cleared both gates, killed by the per-order commission minimum | 2 | yes |

---

## 3. Reason 1: the arithmetic wall

This is the finding that ends the campaign, and it is not empirical.

- Deflated bar: **SR_zero = 1.1372** at N=141, rising with every trial
- Realistic ceiling for a single daily G10 factor: **0.3-0.6** net Sharpe

Bar provenance, stated once: the trial-Sharpe dispersion is v=0.4293 over 130
observed trials. SR_zero = 1.1372 at N=141 (the count today) and
1.1807 at N=141+50 (the count a 50-spec wave would face, which is the
bar every wave-3 spec was screened against).

The Deflated Sharpe Ratio raises the bar as the search widens, because the best
of many trials is expected to look good by chance. After 141 honestly-counted
trials the bar sits near 1.14. A single factor drawn from the FX literature
tops out around half that.

**So most of the 141 trials were arithmetically incapable of passing at the
moment they were run**, whatever the data said. And each one raised the bar for
everything after it. The campaign spent its significance budget on specs that
could not have used it.

Wave 3 made this explicit rather than implicit. Its viability screen computes,
for each spec, the Sharpe it would achieve *if its thesis were exactly right*:

```
if_true_sharpe = sqrt(trades_per_year) * (gross_edge_bps - cost_bps) / per_trade_vol_bps
```

For 41 of 49 specs that number was below the bar. Not "probably fails" --
cannot pass. The clearest case is any daily G10 factor: its if-true Sharpe is
*derived from* the 0.3-0.6 literature estimate, so it is below a 1.14 bar by
construction.

This is why the wall is permanent. It does not depend on venue, on costs, on
account size, or on data quality. **N never shrinks**, so the bar never falls.
The only routes past it are a mechanism class with structurally higher edge, or
combining many near-uncorrelated edges.

The combination route was pre-registered before any component existed, and
voided itself: its rule required at least 3 members and only 2 specs survived.

---

## 4. Reason 2: the access constraint

The 2 survivors were month-end and quarter-end WM/R fix rebalancing flow -- the
one family where the edge-to-volatility ratio is structurally large, because the
flow is mechanical, calendar-known and price-insensitive.

Both were then killed by a term the cost model had never carried: **IBKR charges
0.20 bps of trade value per side with a $2 minimum per order.**

The minimum stops binding above **$100,000 of notional per order**:

| notional / order | commission / side | vs headline rate |
|---:|---:|---:|
| $100,000+ | 0.20 bps | 1.0x |
| $50,000 | 0.40 bps | 2.0x |
| $25,000 | 0.80 bps | 4.0x |
| $10,000 | 2.00 bps | 10.0x |

Both specs trade 6 majors concurrently, so per-order notional is capital divided
by six. The account is cash-only: no Eligible Contract Participant status, and
US retail leveraged spot FX at IBKR requires it (a $10M asset test, not a
setting that can be enabled).

At the 1.5x cost gate against a 1.1807 bar:

| capital | notional / order | cost RT | #18 Month-end | #21 Quarter-end |
|---:|---:|---:|---:|---:|
| $25,000 | $4,167 | 10.28 bps | -4.00 | -0.68 |
| $50,000 | $8,333 | 5.48 bps | -0.94 | 0.76 |
| $100,000 | $16,667 | 3.08 bps | 0.59 | 1.48 |

At $50k, commission alone exceeds #18's entire gross edge. Capital at which each
becomes gate-able on cash spot: **$163,855** for #18, **$70,932** for #21.

This constraint is contingent, not permanent. More capital fixes it. So does a
venue with larger contract units.

---

## 5. Reason 3: cost structure at retail size

Worth separating from reason 2, because it is the more general point.

FX cost is not a fixed percentage. It has a spread component that scales with
notional and a commission component with a fixed floor. At institutional size
the floor is irrelevant and the spread dominates; at retail size the floor
dominates and swamps everything.

Measured hour-of-week round-trip spread, EURUSD, from real tick data:

- liquid London/NY hours: **0.18 bps** round trip
- widest weekday hour: **1.92 bps** round trip (11x)
- raw dispersion in pips across the full week, including the unquoted weekend:
  0.30 to 10.20 pips, **34x**

Against that, a $2 minimum on an $8,333 order is 2.40 bps per side, or 4.80 bps
round trip -- **twenty-six times** the liquid-hour spread it is supposed to
accompany. The
strategy is no longer paying for liquidity; it is paying a toll.

This is why many-small-trades strategies are structurally disadvantaged at
retail size, and it is why the 6 cost-marginal specs in wave 3 -- net edges of a
few tenths of a bp against 0.6-1.0 bps of cost -- were routed rather than run.
Running them would have spent six trials testing the cost model rather than the
signal.

### Does switching to CME FX futures fix it?

Partly, and less than expected.

| 6-major basket, round trip | bps |
|---|---:|
| CME futures, 1 contract each | 1.397 |
| spot at institutional size | 1.064 |

Futures cost **1.31x** what large-order spot costs. Their advantage is not
efficiency but *access to size*: one 6E contract is 125,000 EUR (about
$135,000), already above the commission-minimum threshold, on a $50k account.

But that means futures address reasons 2 and 3 while leaving reason 1 untouched.
They would restore the same 2 specs, in the same single mechanism family, with
no combination test -- at $578,375 of notional on $50k, or **11.6x leverage**.
The binding constraint moves from cost to risk.

Caveats if it is ever pursued: micros do not help (M6E at $13,500 notional costs
~1.85 bps/side, as bad as small spot), and only 6E has a measured fee. The other
seven FX roots sit at a blanket $2.50/side estimate, 3-5x away from 6E in bps
terms, so a 6-major futures basket would be dominated by its worst legs -- the
same trap that killed the spot specs.

---

## 6. The apparatus was systematically permissive

Twelve defects were found and fixed. The individual bugs matter less than the
pattern: **almost every silent default moved in the permissive direction.**

| defect | effect |
|---|---|
| Trial count silently fell N=141 -> 40 when the registry was locked | gate 35% softer, no warning |
| Two runners deflated against SR_zero = 0.0000 | DSR gate reduced to "is the Sharpe positive" |
| PSR/DSR used annualized Sharpe against a daily observation count | z-score inflated ~16x |
| Ledger's `.get(..., "OPEN")` served a tested-and-failed slot as fresh | invited re-proposing a dead spec |
| Cost model missing the $2 per-order commission minimum | under-charged every small order |
| Intraday cost model hour-blind, real dispersion 34x | wrong in both directions |
| `spread_model` artifact: a constant dressed as a surface, zero consumers | read as calibrated, was not |
| Event calendars were recurring-rule proxies (CPI exact in 14% of months) | trades booked on non-events |
| PBO computed on 65 of ~260 out-of-sample days | optimistic |
| Same-bar fills, no execution lag | optimistic |
| Two data-layer lookaheads (spike cleaning at t+1; FRED rates visible 1-2 months early, and those ARE the carry signal) | optimistic |
| A degenerate filter ran under a pre-registered name (EM carry seatbelt, identically zero) | untested mechanism reported as tested |

A research apparatus whose failure modes are one-directional is not neutral. It
is a machine for producing false positives. **The only reason this campaign
produced none is that the underlying signals were weak enough to fail anyway** --
which is luck, not process.

Every one of these is now covered by a regression test, and several by a
tripwire that raises rather than warns.

---

## 7. What would change the verdict

Stated concretely, so this document can be falsified rather than merely believed.

| change | fixes | does not fix |
|---|---|---|
| Capital to ~$164k, cash spot | reasons 2, 3 | reason 1 |
| CME FX futures at current capital | reasons 2, 3 | reason 1, adds 11.6x leverage risk |
| A mechanism class with if-true Sharpe > 1.14 | reason 1 | -- |
| 3+ near-uncorrelated surviving components | reason 1, via combination | -- |
| L2 / queue data enabling a maker strategy | possibly reason 1 | requires a data acquisition |

Note what is absent: "more search". Additional FX trials make reason 1 strictly
worse, because each one raises the bar it would have to clear.

---

## 8. What this document does NOT claim

Every one of these would be an over-generalisation from what was actually
tested, and the campaign's own operating principles forbid them.

- **Not** that FX has no edge. Profitable FX participants demonstrably exist.
  They operate as liquidity providers, at latencies we cannot reach, with
  order-flow we cannot see, or on financing terms we cannot obtain.
- **Not** that the surviving mechanisms are unreal. Month-end fix rebalancing
  flow was **never backtested**. It is documented in the literature with a
  structural driver. We simply cannot reach the order size at which it survives
  costs on this venue.
- **Not** that liquidity provision fails. It was bounded, not tested: majors are
  not maker-viable for us at any plausible fill assumption, and wide crosses show
  a positive gross bound of 1-2 bps per fill under a front-of-queue assumption we
  cannot verify.
- **Not** that intraday FX is exhausted. 21 intraday slots were screened, not
  run. They failed an arithmetic bar, which is a statement about our bar, not
  about the market.
- **Not** that the 41 screened-out specs would have lost money. Many might have
  been mildly profitable. They could not have cleared a 1.14 deflated bar, which
  is a different and much stronger requirement.

The honest one-line scope: **this retail, cash-account, spot, spread-taking
construction of the FX factor catalog carries no deflation-surviving edge at a
bar of ~1.14, and cannot be made to at this account size.**

---

## 9. What survives, and carries forward

**Still running:** two fix-flow specs on forward paper, zero trials, first
observations 2026-07-31 and 2026-09-30. Multi-year by construction, at 12 and 4
events a year.

**Retired without building, each on arithmetic rather than effort:** non-US
central bank calendars (event specs need 21x more events; all 8 central banks
gives ~3x), the ML meta-label harness (meta-labelling filters direction and
cannot create it; best case ~1.04 against a 1.18 bar), further spread
measurement, and a spot maker backtest.

**Apparatus, all asset-class portable:** the statistical-viability screen; the
measured hour-of-week cost surface with a per-order commission floor; the
degenerate-signal tripwire; registry duplicate detection; corrected PSR/DSR
units; purge/embargo walk-forward with CPCV; the reusable intraday day-loop
harness; the blind-generation ledger and its leak checks; the forward-paper log
with its forward-only guard; and an authoritative event calendar validated
against our own 1-minute data at 24-27x the day's median minute.

That apparatus, and the discipline of counting every trial honestly, is what
this campaign actually produced. The FX verdicts are disposable. The machine that
produced them is not.
