# M7 -- Smoothed IV Surface: Build and Validation Report

**Date:** 2026-07-30
**Status:** Built and validated over the full buildable scope. This is a data/model build -- **no backtest, no P&L, no positions, zero trials consumed.**
**Pre-registration:** `20260730_m7_prereg.md`, committed to git BEFORE the first fit was run (commit `6271424`). Every choice below that could have been made after seeing results was fixed in advance there.

## Headline

M7 is **built over 100% of the buildable scope** -- all 270 `options_chain_eod`
partitions (SPY 2017-01..2025-12, QQQ 2012-06..2025-12), zero failures.
**100,507 of 126,777 session-expiry slices fit (79.3%)**; the remaining 20.7%
are refused with a reason code and a null `iv_smooth`, never a guessed number.

**The registered D-047/030 integrity gate FAILS on both roots.** It fails on
criterion (a) -- the surface lands inside the raw quote's own bid-ask IV band
only **27.5% (SPY) / 55.1% (QQQ)** of the time at 0.05 delta, against a
registered floor of 90%. It passes criterion (b) with a wide margin: median
divergence from the quote mid is **0.154 (SPY) / 0.122 (QQQ) vol points**
against a registered ceiling of 1.5.

Those two facts are not in tension, and the reconciliation is the single most
useful thing this build learned: **the surface is very close to the market, but
the market at these strikes is quoted far tighter than any five-parameter form
can thread.** The median bid-ask IV band at 0.05 delta is only **0.09-0.35 vol
points wide**. A smooth global fit through ~100 points quoted to a tenth of a
vol point will sit outside that band most of the time while still being within
a sixth of a vol point of the mid.

**The registered consequence therefore binds, and I am applying it as written:
OPT-047 and OPT-030 must not use `iv_smooth` as a price MARK.** Raw deep-OTM
quotes exist at 0.05 delta in 100% of sessions (wave-0 census), and where a
valid quote exists the quote is the better mark. What M7 is *for* is strike
selection, delta, and curvature -- where it performs well: at a 0.05-delta
target it picks the **same strike as the shipped greeks 74.3% (SPY) / 87.2%
(QQQ)** of the time, median strike difference zero.

**Self-audit.** The gate threshold was fixed in git before the first fit was
run and has **not** been adjusted after seeing the result. Criterion (a) turned
out to be far more demanding than I anticipated when I set it -- I expected
deep-OTM IV bands to be wide, and they are not. That is a fact about my prior,
not a licence to move the line. The gate is reported FAIL.

---

## 1. What was built

| Artifact | Path |
|---|---|
| Fitter | `src/data/options/iv_surface.py` |
| Materializer | `src/data/options/iv_surface_build.py` |
| Parallel build driver | `scripts/data/build_m7_surface.py` |
| Validation battery | `scripts/data/validate_m7_surface.py` |
| Tests (31, TDD) | `tests/data/test_options/test_iv_surface.py` |
| Surface params table | `<storage>/options/options_iv_surface/root=/year=/month=/data.parquet` |
| Smoothed marks table | `<storage>/options/options_iv_smooth/root=/year=/month=/data.parquet` |
| Validation CSVs | `output/m7_validation/` |

`options_chain_eod` is **not mutated**. M7 writes two join tables, so the
canonical chain stays immutable and M7 stays re-runnable.

**`options_iv_surface`** -- one row per `(root, session_date, expiry)`: the five
SVI parameters, the parity-implied forward and discount, `n_points`, RMSE, the
fitted `k` range, the butterfly violation rate, and the **reason code**. Refused
slices are present with null parameters; a refusal is a positive record, not a
missing row.

**`options_iv_smooth`** -- one row per contract-session: `iv_smooth`,
`delta_smooth`, `k`, `extrapolated`, `iv_source`, `surface_reason`. P1 joins
this to `options_chain_eod` on `(session_date, expiry, strike, right)`. The join
is exact and total: on the reference partition, 68,615 chain rows -> 68,615
joined rows, zero unmatched.

## 2. The parameterization choice, and why

**Registered choice: raw SVI**, `w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))`,
in total variance `w = iv^2 * T` against `k = log(K/F)`, fitted per
`(root, session, expiry)`.

The whole reason M7 exists is trustworthiness at |delta| ~ 0.05 -- in the wings.
A spline's wing behaviour is set by knot placement and end conditions, not by
any financial constraint, so it can manufacture curvature artifacts exactly
where OPT-047 reads the surface, and it has no defensible extrapolation. Raw
SVI is **linear in `k` in both wings by construction** -- the correct
large-strike asymptotic, with slope bounded by Lee's moment formula -- and its
five parameters admit closed-form static-no-arbitrage constraints. A wing that
is wrong under SVI is wrong by a bounded, inspectable slope. A wing that is
wrong under a spline is unbounded.

The cost of that rigidity was stated in advance: raw SVI cannot represent every
observed smile, so it shows larger residuals than a flexible spline would. That
is the intended trade -- misfit that is *measured* beats misfit that is
*absorbed*.

## 3. Dividend-awareness: how it is actually delivered

The registered spec requires the surface be "dividend-aware for single names".
The task brief proposed satisfying that by fetching a realized dividend history
from yfinance and caching it.

**I did not do that, deliberately, and this is the most important divergence in
this report.** Using a *realized, later-paid* dividend series to build a forward
as of date `t` injects information not knowable at `t` -- for any dividend not
yet declared, that is a straightforward lookahead leak, and spec v2 Section 1.3
forbids exactly this class of error.

Instead the forward comes from **put-call parity on the session's own quotes**:
for each strike quoted on both sides, `F_k = K + (C - P)/D`, with `F` the robust
median over the paired strikes nearest spot. This embeds **the market's own
dividend expectation as priced at the 15:45 snapshot**, is point-in-time by
construction, needs no external dividend series, and is what dealers use. The
discount factor `D` comes from the repo's existing FRED reader
(`src/data/rates/fred_reader.py`), log-linearly interpolated across
`DGS1MO/DGS3MO/DGS6MO/DGS1/DGS2` -- all already on disk. **No new data source
was introduced and no yfinance dividend wrapper was added.**

**The forward validates itself.** On SPY 2024-01-16 the implied dividend yield
`q = r - log(F/S)/T` converges to **+1.09% to +1.20%** across the 94-350 DTE
expiries -- SPY's actual trailing dividend yield in January 2024 was ~1.3%. The
surface recovers the dividend from option quotes alone, with no dividend input.

Spot delta then falls out with no extra assumption: `dC/dS = exp(-qT)*N(d1)`
and `exp(-qT) = D*F/S` is already known from the parity fit.

## 4. Coverage and refusal rates

All 270 partitions built, 0 failures. Slice = one `(root, session, expiry)`.

| root | slices | fitted | refused |
|---|---|---|---|
| SPY | 59,148 | 50,070 (**84.65%**) | 15.35% |
| QQQ | 67,629 | 50,437 (**74.58%**) | 25.42% |

**Refusal reasons (both roots, 26,270 refused slices):**

| reason | n | dominant location |
|---|---|---|
| `NO_FORWARD` | 9,630 | QQQ 181+ DTE (39.4% of that bucket) -- fewer than 4 paired call/put strikes |
| `ARB_VIOLATION` | 9,445 | short DTE: SPY 1-7 (16.2%), SPY 8-30 (14.7%), QQQ 1-7/8-30 (10.3%) |
| `ZERO_DTE` | 3,070 | dte 0, 100% of that bucket (correctly labelled after the D-M10 fix) |
| `DTE_OUT_OF_RANGE` | 1,873 | QQQ only -- LEAPS beyond the registered 400-day cap |
| `TOO_FEW_STRIKES` | 1,050 | QQQ 1-7 DTE (8.7%) |
| `IMPLAUSIBLE_FORWARD` | 739 | SPY 1-7 DTE (5.6%) |
| `ONE_SIDED` | 463 | diffuse, <1% everywhere |

**Coverage by year -- this is the headline limitation.** SPY is uniformly good;
QQQ's early years are not.

| year | QQQ fitted | SPY fitted |
|---|---|---|
| 2012 | **45.6%** | -- |
| 2013 | **50.6%** | -- |
| 2014 | 54.4% | -- |
| 2015 | **41.7%** | -- |
| 2016 | 50.6% | -- |
| 2017 | 66.8% | 75.0% |
| 2018 | 80.6% | 85.8% |
| 2019 | 75.8% | 79.2% |
| 2020 | 83.7% | 86.9% |
| 2021 | 88.5% | 87.0% |
| 2022 | 88.1% | 81.7% |
| 2023 | 87.4% | 88.1% |
| 2024 | 90.3% | 90.9% |
| 2025 | 88.6% | 82.2% |

**Diagnosis, and it is a data fact rather than a fitter defect.** Early-QQQ EOD
chains are far thinner: median **22** valid contracts per session-expiry in
2012-06 against SPY's **103** in 2024-01, with a 5th percentile of **3**. A
five-parameter fit needing >= 8 OTM points and >= 4 paired strikes for the
forward simply cannot be run on many of those slices.

**Consequence for consumers:** any M7-dependent study on QQQ before ~2017
is running on roughly half the expiry surface, and that must be reported as a
binding constraint, not averaged away. SPY 2017 (75.0%) additionally overlaps
the known SPY 2017-H1 truncation.

**Coverage by DTE** -- best where the slate actually lives. SPY 61-90 DTE
**98.3%**, SPY 91-180 **98.5%**, SPY 31-60 **91.1%**, QQQ 61-90 **93.4%**,
QQQ 31-60 **90.0%**. OPT-047's 3-month rungs and OPT-006/027's 30-45 day legs
sit in the best-covered part of the surface.

## 5. Validation 1 -- fit residuals by moneyness bucket

Residual = smoothed IV minus the IV implied by the contract's own quote mid, in
absolute vol (0.0100 = 1 vol point). OTM side only -- the fit's own domain.
Distributions, not just means, as registered.

| root | \|delta\| bucket | n | p50 | p95 | p99 | max | signed p50 | median quoted IV band | extrapolated |
|---|---|---|---|---|---|---|---|---|---|
| SPY | 0.00-0.05 | 83,795 | 0.00236 | 0.0660 | **0.1715** | 0.925 | -0.00053 | 0.00346 | **30.5%** |
| SPY | 0.05-0.15 | 64,052 | 0.00116 | 0.0059 | 0.0106 | 1.633 | +0.00035 | 0.00093 | 0.3% |
| SPY | 0.15-0.35 | 75,690 | 0.00162 | 0.0071 | 0.0115 | 6.683 | -0.00038 | 0.00081 | 0.1% |
| SPY | 0.35-0.65 | 46,986 | 0.00129 | 0.0055 | 0.0084 | 4.239 | +0.00003 | 0.00095 | 0.0% |
| QQQ | 0.00-0.05 | 34,989 | 0.00250 | 0.0811 | **0.2368** | 1.037 | -0.00055 | 0.00624 | **32.3%** |
| QQQ | 0.05-0.15 | 29,728 | 0.00080 | 0.0055 | 0.0106 | 1.249 | +0.00037 | 0.00173 | 0.3% |
| QQQ | 0.15-0.35 | 37,522 | 0.00102 | 0.0052 | 0.0089 | 5.100 | -0.00007 | 0.00133 | 0.1% |
| QQQ | 0.35-0.65 | 24,261 | 0.00082 | 0.0041 | 0.0077 | 6.656 | -0.00026 | 0.00141 | 0.0% |

**Read this the right way.** The median fit is excellent -- **0.08 to 0.25 vol
points** everywhere, and essentially unbiased (signed medians are ~5e-4, i.e.
0.05 vol points, with no consistent sign). But:

1. **The far wing carries a heavy tail.** In the `|delta| < 0.05` bucket the p99
   is **17.1 (SPY) / 23.7 (QQQ) vol points** and the max approaches 100. The
   median is fine and the tail is terrible. Roughly **31%** of contracts in that
   bucket sit outside the fitted `k` range, i.e. the surface is extrapolating.
   **This is the region OPT-047 reads, and it is the least trustworthy region of
   the surface.**
2. **The mid-buckets show large maxima** (up to 6.7 vol points) on a p99 of ~1
   vol point -- a thin population of pathological slices, not a broad problem.
3. **`inside_band_frac` is low in every bucket** (0.23-0.55), not just the wings
   -- the same tight-quote effect that drives the gate result in Section 8. It
   is a property of fitting a 5-parameter form to a very finely quoted smile,
   not a wing-specific failure.

## 6. Validation 2 -- static no-arbitrage

**Not repaired anywhere.** Measured and reported, per the registered rule.

### Butterfly (Gatheral `g(k) >= 0`, i.e. non-negative risk-neutral density)

Evaluated at every actual contract's `k`, not on a synthetic grid.

| root | \|delta\| bucket | n | violations | rate |
|---|---|---|---|---|
| SPY | 0.00-0.05 | 1,242,461 | 22,574 | **1.82%** |
| SPY | 0.05-0.15 | 929,073 | 2,732 | 0.29% |
| SPY | 0.15-0.35 | 1,086,202 | 1,103 | 0.10% |
| SPY | 0.35-0.65 | 1,256,966 | 252 | **0.02%** |
| SPY | 0.65-1.00 | 1,368,670 | 11,654 | 0.85% |
| QQQ | 0.00-0.05 | 791,278 | 10,267 | **1.30%** |
| QQQ | 0.05-0.15 | 688,720 | 855 | 0.12% |
| QQQ | 0.15-0.35 | 897,869 | 966 | 0.11% |
| QQQ | 0.35-0.65 | 1,099,904 | 475 | **0.04%** |
| QQQ | 0.65-1.00 | 1,126,520 | 7,288 | 0.65% |

The U-shape is the expected one: the density is cleanest near ATM (0.02-0.04%)
and degrades into both wings, worst in the deepest OTM bucket (**1.3-1.8%**).
Note `min_g` reaches large negative values (-683, -2909) in mid-buckets; these
come from very short-dated slices where total variance `w` is tiny and `g`'s
`1/w` term explodes. They are numerically extreme rather than economically
large, but they are genuine constraint violations and are reported as such.

### Calendar (total variance must not fall with maturity at fixed `k`)

**M7 fits each expiry independently, so nothing enforces calendar consistency.**
This is the honest structural weakness of a per-expiry parameterization and it
shows up directly.

| root | near-expiry bucket | adjacent pairs | pairs violating | grid points violating | worst crossing (vol pts) |
|---|---|---|---|---|---|
| SPY | near <= 7 | 5,917 | **57.3%** | 20.0% | 13.39 |
| SPY | near 8-30 | 15,650 | 44.8% | 10.6% | 9.53 |
| SPY | near 31-90 | 11,289 | 36.9% | 7.4% | 7.67 |
| SPY | near 91+ | 15,059 | **17.3%** | 1.8% | **1.66** |
| SPY | ALL | 47,915 | 35.8% | 8.2% | 13.39 |
| QQQ | near <= 7 | 5,638 | **55.0%** | 16.0% | 15.43 |
| QQQ | near 8-30 | 15,018 | 46.0% | 9.5% | 10.10 |
| QQQ | near 31-90 | 13,459 | 32.0% | 4.6% | 5.27 |
| QQQ | near 91+ | 12,932 | **10.5%** | 0.9% | **1.79** |
| QQQ | ALL | 47,047 | 33.3% | 6.5% | 15.43 |

The aggregate (33-36% of adjacent pairs cross somewhere) looks alarming and is
mostly short-dated: **57%/55% at <= 7 DTE, falling to 17%/10% at 91+ DTE**,
where the worst crossing is only ~1.7 vol points. The median worst-crossing is
**0.00 vol points** in every bucket except `near <= 7`, meaning fewer than half
of pairs cross at all and most crossings that do occur are numerically trivial.
**For OPT-047's 3-month region the calendar behaviour is acceptable; for any
short-dated term-structure signal (OPT-032's M2/M1 slope) it is not, and a
calendar-coupled fit would be required.**

## 7. Validation 3 -- day-over-day stability

Session-over-session change in smoothed IV at fixed log-moneyness, consecutive
sessions only, same expiry (n = 94,966 pairs, both roots).

| metric | p50 | p95 | p99 | share of jumps > 5 vol pts | median \|spot return\| on those jumps | median \|spot return\| overall |
|---|---|---|---|---|---|---|
| ATM (`k = 0`) | 0.00545 | 0.0354 | 0.0743 | **2.40%** | **2.61%** | 0.58% |
| Wing (`k = -0.20`) | 0.00677 | 0.1391 | 0.8229 | **10.64%** | **0.71%** | 0.58% |

**The ATM surface is stable and its discontinuities are real.** Median session
change 0.55 vol points; only 2.4% of sessions jump more than 5 vol points, and
on those sessions the underlying moved a median **2.61%** against a 0.58%
baseline -- a 4.5x concentration. Large ATM moves are vol events, not fit noise.
This is the registered check passing.

**The wing is not.** At `k = -0.20`, 10.6% of consecutive sessions jump more
than 5 vol points, and on those sessions the underlying moved a median
**0.71%** -- barely above the 0.58% baseline. **Wing jumps are not explained by
market moves; they are fit instability.** The p99 session-over-session wing
change is 82 vol points. This is the clearest quantitative statement of where
the surface should not be trusted, and it corroborates Section 5's tail.

## 8. Validation 4 -- the D-047/030 integrity gate

Registered in advance (pre-reg Section 6): SPY and QQQ, expiries 21-45 DTE, per
session the contract whose *smoothed* `|delta|` is nearest 0.05, both rights,
seeded sample of 200 sessions per root (`seed = 20260730`).

**PASS required both:** (a) `iv_smooth` inside the raw `[IV(bid), IV(ask)]` band
in >= 90% of sampled contract-sessions; (b) median `|iv_smooth - IV(mid)|` <= 1.5
vol points.

| root | n | (a) inside band | (b) median abs diff | p90 | p99 | signed median | extrapolated | (a) | (b) | **VERDICT** |
|---|---|---|---|---|---|---|---|---|---|---|
| SPY | 2,346 | **27.45%** | **0.00154** | 0.00582 | 0.01304 | +0.00057 | 0.30% | FAIL | PASS | **FAIL** |
| QQQ | 1,560 | **55.13%** | **0.00122** | 0.00424 | 0.01039 | +0.00020 | 1.60% | FAIL | PASS | **FAIL** |

**Measured divergence: 0.154 (SPY) / 0.122 (QQQ) vol points at the median,
0.58 / 0.42 at p90, 1.30 / 1.04 at p99.** Bias is negligible and slightly
positive (the surface reads ~0.05 vol points rich). Only 0.3%/1.6% of the
sampled 0.05-delta contracts required extrapolation -- **at 21-45 DTE, 0.05
delta is inside the quoted strike range, so this is interpolation error, not
extrapolation error.**

**Why (a) fails.** The bid-ask IV band at these strikes has a median width of
about **0.09-0.35 vol points**. The surface sits within 0.15 vol points of the
mid but the band is often narrower than that, so the fit falls outside it. The
failure is "the market is quoted tighter than a global 5-parameter fit can
follow", not "the surface is far from the market".

**Applying the registered consequence as written** ("if the surface diverges
materially at 0.05 delta, both 047 and 030 need a different mark source"):

- **OPT-047 and OPT-030 must take their price MARKS from the raw quote, not
  from `iv_smooth`.** Wave-0 established a valid two-sided quote at ~0.05 delta
  in 100.0% of sessions for both roots, so a better mark source exists and
  costs nothing.
- **`iv_smooth` and `delta_smooth` remain the right source for strike
  SELECTION and for curvature/skew signals** -- which is what P1's low-delta
  rule is actually about. Section 9 quantifies that this use holds up.
- The gate is **not** a verdict on any strategy and consumed **zero trials**.

## 9. Supplementary -- strike-selection agreement (not one of the four registered checks)

P1 uses the surface to *select* a strike. Comparing the strike chosen by
smoothed delta against the one chosen by the vendor's shipped delta:

| root | target delta | n | same strike | median \|strike diff\| | p95 \|strike diff\| |
|---|---|---|---|---|---|
| SPY | 0.05 | 2,346 | **74.3%** | 0.0 | 5.0 |
| SPY | 0.10 | 2,346 | 76.1% | 0.0 | 2.0 |
| SPY | 0.25 | 2,346 | 73.2% | 0.0 | 1.0 |
| QQQ | 0.05 | 1,558 | **87.2%** | 0.0 | 2.0 |
| QQQ | 0.10 | 1,558 | 84.3% | 0.0 | 1.5 |
| QQQ | 0.25 | 1,558 | 81.5% | 0.0 | 1.0 |

Agreement is high and, notably, **no worse at 0.05 delta than at 0.25** -- so
the shipped greeks are not visibly degrading at low delta in a way that changes
strike choice. A separate direct comparison on SPY 2024-01 found median
`|delta_smooth - delta_shipped|` of **0.0008** for `|delta| < 0.05`, rising
monotonically to **0.0187** for `|delta| > 0.65`.

**This is worth stating plainly because it partly undercuts M7's original
motivation.** The registered rationale for the low-delta rule was deep-OTM error
in the vendor's dividends-off Black-Scholes greeks. Measured, that error is
**smallest** at low delta and **largest deep ITM -- the opposite end**. The
deep-ITM divergence is real and is exactly where a dividends-off model should
fail (it is OPT-006's 0.80-delta LEAPS leg that is most exposed, not OPT-047's
0.05-delta ladder). M7 is still the right source for OPT-006 and for curvature,
but the specific fear that motivated P1's `|delta| < 0.10` rule is not strongly
supported by the data.

## 10. Honest limitations

1. **The far wing is the weakest region, and it is the region OPT-047 depends
   on.** `|delta| < 0.05`: p99 residual 17-24 vol points, ~31% extrapolated,
   butterfly violations 1.3-1.8%, and wing-level day-over-day jumps that do not
   track the market (Section 7). Treat wing IV as indicative; do not treat it as
   authoritative, and do not mark off it.
2. **No calendar coupling.** Expiries are fit independently; 33-36% of adjacent
   pairs cross somewhere. Acceptable at 91+ DTE, not at short DTE.
3. **Early QQQ is half-covered** (41.7-54.4% of slices in 2012-2016) because the
   chains are genuinely thin, not because the fitter fails.
4. **`IMPLAUSIBLE_FORWARD` is a mis-calibrated registered threshold.** It refuses
   on implied `q = r - log(F/S)/T` outside [-0.10, +0.15]; at 1-7 DTE that
   quantity is `1/T`-amplified noise, so it fires on 5.6% of SPY short-dated
   slices for a reason that is an artifact of the diagnostic rather than a bad
   forward. **I registered it blind and am not retuning it after the fact.** It
   should be re-registered for a v2 as a DTE-aware bound on `F/S` directly. Its
   practical cost is confined to <= 7 DTE, which no M7 consumer uses.
5. **`ARB_VIOLATION` at short DTE is high** (14-16% of SPY 1-30 DTE slices).
   Some of this is the Lee wing-slope bound biting on genuinely steep
   short-dated smiles that raw SVI cannot represent arbitrage-free. A more
   flexible short-dated parameterization (e.g. SSVI with a calendar-consistent
   `theta`) would recover some of these. Registered as a limitation, not fixed.
6. **`DTE_OUT_OF_RANGE` discards 1,873 QQQ LEAPS slices** beyond the registered
   400-day cap. OPT-006's 12-18 month long leg **exceeds 400 days at entry** and
   will therefore find no surface. This is a registered-parameter consequence
   that needs an explicit amendment before OPT-006 runs -- flagging it rather
   than silently widening the cap.
7. **No cross-vendor validation.** ORATS was deferred, so every number here is
   internally consistent with the ThetaData quotes and nothing external
   corroborates the level. The forward is self-validating (Section 3); the IV
   level is not.
8. **Single names are out of scope** and cannot simply be added: V7 established
   strikes are AS-REPORTED with non-standard deliverables after splits, and the
   adjustment layer does not exist. A per-expiry fit across a split date would
   mix two contract populations.
9. **`T` is calendar-day** (`dte/365`). A trading-day or business-time clock
   would change short-dated fits materially. Registered choice, not tested
   against alternatives (testing alternatives would be a search).
10. **The build is slow**: ~75 minutes for 270 partitions at the mandated
    `--jobs 8` on a 16-physical-core machine. It is CPU-bound in
    `scipy.least_squares`, not I/O-bound, so it would roughly halve at
    `--jobs 16`.

## 11. What consumers should do

| consumer | use `iv_smooth` for | do NOT use it for |
|---|---|---|
| **P1** (`|target_delta| < 0.10`) | `delta_smooth` for strike selection; `iv_smooth` for the IV reported alongside | the price mark |
| **OPT-047** (0.05-delta ladder) | selecting the rung strike | marking the rung -- use the raw quote, available 100% of sessions |
| **OPT-030 / OPT-027** | curvature and skew signals at 21-45 DTE, where coverage is 90-98% | short-dated (<= 7 DTE) curvature, where refusal and arb rates are worst |
| **OPT-006** (0.80-delta LEAPS) | the ITM leg -- this is where shipped greeks are genuinely worst (median delta error 0.0187) | anything beyond 400 DTE until the cap is amended |

Every consumer must branch on `surface_reason`: a null `iv_smooth` means **no
surface here**, and must not be silently coerced to a shipped value.

### End-to-end check of OPT-047's actual selection

An OPT-047-shaped query -- 3-month (80-100 DTE) puts, strike nearest 0.05
smoothed delta -- run against four partitions spanning both roots and the
2019/2022 stress years:

| partition | selections | median \|delta\| achieved | median `iv_smooth` | extrapolated |
|---|---|---|---|---|
| SPY 2019-03 | 19 | 0.0499 | 0.2398 | **0.0%** |
| SPY 2022-10 | 21 | 0.0500 | 0.4151 | **0.0%** |
| QQQ 2014-07 | 17 | 0.0514 | 0.2084 | **0.0%** |
| QQQ 2023-05 | 16 | 0.0500 | 0.3368 | **0.0%** |

The join is exact on all four (row count unchanged, zero unmatched), the
selection hits the 0.05 target essentially exactly, and the IV level responds
correctly to regime (0.24 in Mar-2019 vs 0.42 in Oct-2022).

**Important qualifier on the 31% extrapolation figure in Section 5:** that is
the share across *all* contracts with `|delta| < 0.05`, which includes far
deeper strikes than any consumer selects. At OPT-047's actual selection point --
3-month, 0.05 delta -- extrapolation was **0.0%** on every partition sampled.
The wing caveat is real but it binds beyond the ladder's strike, not at it.

---

## Appendix A -- Divergence log

Things this brief or the doc chain asserted that turned out to be false, plus
choices where I departed from the brief.

| # | Asserted | Reality |
|---|---|---|
| D-M1 | Column names in `options_chain_eod` are `iv_shipped, delta_shipped, theta_shipped, vega_shipped, spot`, keyed by `ticker`/`trade_date`, with `day_volume` | **Wrong on every name.** Actual: `implied_vol, delta, theta, vega, underlying_px`, keyed by `root`/`session_date`, with `volume`. There is no `day_volume` and no `iv_smooth` column -- `iv_smooth` had to be a new join table, not a column update. Verified against the arrow schema of `root=SPY/year=2024/month=01`. |
| D-M2 | "Canonical reader: `src/data/options/canonical.py` (99 tests pass)" | `tests/data/test_options/test_canonical.py` holds **27** tests; the whole `tests/data/test_options/` directory held **76** before this work (114 collected after adding 31, of which some are parametrized). The "99" figure does not correspond to any collection I can produce. All existing tests do pass. |
| D-M3 | "Dividends... `src/data/yfinance/fundamentals.py` exists but does not expose dividends -- add thin wrappers there... Cache them locally" | **Declined, deliberately.** A realized dividend series used to build a historical forward is lookahead. The parity-implied forward is point-in-time, needs no series, and validates to within ~0.2pp of SPY's actual dividend yield. See Section 3. |
| D-M4 | "Risk-free rate: find what the repo already uses" | Found: `src/data/rates/fred_reader.get_fred_series`, with the full Treasury curve (`DGS1MO/3MO/6MO/1/2/5/10/30`) already downloaded to `<storage>/alt_data/fred/`. **Also found two places that hardcode `RISK_FREE_RATE = 0.05`** -- `src/data/options/thetadata_adapter.py:59` and `src/strategies/options/csp/mark_to_market.py:48,75`. Those are a latent inconsistency with M7 and with each other; flagged, not changed (out of scope). |
| D-M5 | The worktree assigned to this task was ready to build in | It sat on a branch **diverged from `main` by 18 unrelated commits** (RAMP/Grafana work) and contained **neither `canonical.py` nor the options-slate docs**. I branched `feat/m7-iv-surface` fresh off `main` and left the divergent ref untouched. |
| D-M6 | The doc chain reads as if `20260728_options_wave0_diagnostics.md` is part of the record | It is **not committed to `main`** -- it exists only in the shared working tree. The other options-slate docs are committed. Whoever owns that file should commit it; conclusions in this report that depend on it are cited to an uncommitted source. |
| D-M7 | "M7... populates `iv_smooth`" implying the column exists to populate | No `iv_smooth` column exists anywhere, and no SVI/spline/smoothing code existed in the repo (confirmed by search). Built from zero. |
| D-M8 | (My own bug, found against real data) | The SVI `b` box bound was `4/T` while the Lee no-arb check tests `b*(1+\|rho\|) <= 4/T`. The optimizer could settle where the box allowed but the check rejected, spuriously refusing good long-dated slices (SPY 2024-01-16 dte 339 and 350) and landing on a degenerate `rho=+0.96` optimum -- the wrong sign for equity skew. Bound is now `2/T`, which is sufficient. Regression-tested. |
| D-M9 | "diverges materially" (the registered D-047/030 gate text) | **Undefined anywhere in the doc chain.** Operationalized in the pre-registration before measuring, and flagged there as a researcher degree of freedom. |
| D-M10 | (My own bug, found by reading the refusal census) | The builder solved the forward *before* testing the expiry structurally. At `dte == 0` the implied dividend yield `q = r - log(F/S)/T` divides by `T -> 0` and blows up, so the forward check fired first and stamped **695 SPY 0DTE slices `IMPLAUSIBLE_FORWARD`** when the true reason is `ZERO_DTE`; some >400-DTE slices likewise got `NO_FORWARD` instead of `DTE_OUT_OF_RANGE`. The slices were correctly *refused* either way and no `iv_smooth` value was affected -- but the reason code is the deliverable. Fixed (`structural_refusal` now runs first), regression-tested, and **the entire 270-partition store was rebuilt to relabel**. |
| D-M11 | (My own bug, latent) | `np.interp` was used to seed the ATM total variance without guaranteeing `k` was sorted. Inputs arrive strike-ordered today so it was inactive; verified a bit-exact no-op against the built data before committing the defensive sort. |

**Confirmed as stated** (no divergence): the 270-partition scope (SPY 2017-01..2025-12 = 108, QQQ 2012-06..2025-12 = 162); the NaN-vs-NULL trap (greeks are NaN-valued float64 -- though M7 never reads them, so it is immune); `quote_valid == False` rows already absent from `options_chain_eod`; mixed Float32/Float64 across EOD partitions (handled with a widening cast on read); DuckDB single-writer (not hit -- M7 writes parquet only).
