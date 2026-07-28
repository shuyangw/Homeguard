# Options Slate — Phase 1 V-Battery Report

**Date:** 2026-07-27
**Scope:** Phase 1a canonicalization + Phase 1b verification battery V1-V13 over the owned
`options_combined/` store, per
`2026-07-25_options_phase01_cc_work_order.md` Sections 2 and 3.
**Status:** V4, V6, V7, V8, V10, V12, V13 COMPLETE. V1/V2/V3/V5/V9/V11 sweep IN PROGRESS
(definitions validated on SPY 2024-01; full 31-root sweep running).

**No strategy backtest has been run. No P&L has been computed or observed.**
Per work order Section 5, this report states measurements and gate outcomes only. It draws
**no conclusions about strategy viability**.

Per work-order ground rule 1: where this report and the doc chain disagree, **this report
wins** and the divergence is stated rather than silently reconciled.

---

## 0. Escalation summary — read this first

Work order Section 6 requires stopping and reporting, without adapting, on the following.
**Five escalation conditions have fired.**

| # | Escalation | Registered consequence |
|---|---|---|
| **ESC-1** | **V7: 7 of 7 corporate actions are AS-REPORTED.** Not one is back-adjusted | "As-reported => adjustment layer required before **ANY** single-name work" |
| **ESC-2** | **V4: a 16-column schema variant exists (323 partitions, ~1.0B rows, 17 roots, 2012-2016) carrying NO `implied_vol`, `delta`, `theta`, `vega`, or `underlying_px`** | V1's delta-bucketed gate is **structurally unmeasurable** on those partitions; V5 is 0% non-null there by construction => those root-years are **UNTRUSTED** per the registered V5 gate |
| **ESC-3** | **V8: `root=META` for 2021-07 .. 2022-01 is a DIFFERENT ISSUER** (Meta Materials, underlying ~USD 14-16), and Meta Platforms has a **7.4-month hole** (2021-10-30 .. 2022-06-08) | The assumed "FB -> META splice" does not close the series and would splice in a penny stock |
| **ESC-4** | **V8: `root=SPX` contains NO SPXW.** 31/31 sampled expirations are AM-settled third-Friday | Weekly / 0DTE SPX work is **not testable** on this store |
| **ESC-5** | **The data materially differs from the brief's description.** The store's usable edge is **2025-12-26**, not 2026-02; and greeks are absent pre-2017 for SPY/IWM/SPX and the ETFs | Work order Section 6: "discovery that the data materially differs from Section 2's description" |

A sixth item is a **correction to a registered ruling's rationale**, not an escalation — see V6.

---

## 1. Phase 1a — Canonicalization (COMPLETE)

### 1.1 What was built

| Artifact | Path |
|---|---|
| Canonical layer | `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\src\data\options\canonical.py` |
| Tests (36, all passing) | `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\tests\data\test_options\test_canonical.py` |
| EOD builder CLI | `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\scripts\data\build_options_chain_eod.py` |

**Table A `options_chain_1m`** — a streaming READ layer (no bulk copy of 233 GB).
`timestamp`/`expiration` parsed from ISO-8601 strings, localized US/Eastern, converted to the
repo standard `Datetime[us, UTC]`. `right` normalized `PUT->P` / `CALL->C`, raising on any
unexpected value. Derived `mid`, `spread_abs`, `spread_rel`, `dte`, `dte_trading`,
`quote_valid`.

**Table B `options_chain_eod`** — materialized at the registered snapshot minute, one row per
contract per session.

### 1.2 Binding rules — verification evidence

| Rule | Verified how | Result |
|---|---|---|
| Snapshot minute is 15:45:00 ET, one guarded function | Behavioural check on materialized SPY 2024-01 (68,615 rows) | **0 snapshots after 15:45 ET.** 67,766 exact 15:45 hits; 849 fallbacks with `snapshot_fallback=true` and true `snapshot_ts` |
| Marks from quotes, never trade prints | `close`/`vwap` pass through but are never used to derive `mid` | Enforced; module docstring states it |
| Invalid quotes excluded, never repaired | `mid`/`spread_*` NULL exactly where `quote_valid` false | Unit-tested; EOD table `quote_valid` all true, `mid` null count 0 |
| `_eod` suffix preserved | Column assertion on materialized output | Columns include `gamma_eod`, `oi_eod`, `oi_eod_lag1`, `gamma_eod_lag1`. **No bare `open_interest`/`gamma`** — the `data_loader.py:20-26` footgun is not repeated |
| Intraday convention: decide bar *t*, fill bar *t+1* | Registered in the module; no intraday execution built this phase | Registered |
| `get_eod_chain()`'s 16:00 default NOT reused | Independent implementation | Confirmed |

### 1.3 Two canonicalization findings the chain did not anticipate

- **DST verified, both directions.** Naive-ET interpretation holds; sessions correctly map to
  13:30 UTC (EDT) and 14:30 UTC (EST).
- **The vendor pads early-close sessions out to 16:00 with stale repeated quotes.** On
  half-days (e.g. 2024-07-03, close 13:00 ET) bars continue to 16:00 carrying a frozen quote.
  A naive "last bar of session" mark would therefore read a **post-close** price. The
  canonical cutoff is clamped to `min(15:45, real session close)` via the NYSE calendar.
  This is a genuine leakage hazard that neither the work order nor Phase 0 identified.

### 1.4 Prerequisites materialized

| Artifact | Coverage | Note |
|---|---|---|
| VIX index spot | 1990-01-02 -> 2026-07-24, 9,204 rows | Was previously **fetched** at call time from yfinance `^VIX` — not reproducible. Now materialized with a provenance sidecar (source + snapshot timestamp). |
| `regime_state_daily` | 2012-06-01 -> 2026-02-27, 3,455 sessions | Causal replay via `analyze_regime_history()`. Sidecar records the VIX snapshot consumed, SPY source, and the data-vintage caveat verbatim. |

The regime causality regression test passes **with a negative control** (a deliberately
non-causal variant is confirmed to fail the same assertion), so the test has demonstrated
power rather than passing vacuously.

**Data-vintage caveat, to be stated on every downstream result:** replay uses TODAY's SPY/VIX
series. If that data was revised or backfilled since, the recomputed state differs from what
was known at the time.

---

## 2. Phase 1b — V-battery results

Gates are restated **verbatim** from work order Section 3 so this report is auditable against
what was registered. No threshold was altered after measurement.

### V1 — Quote population · IN PROGRESS

> **Registered gate:** "Per root x year, fraction of regular-trading-hours listed-contract-minutes with `bid>0 AND ask>=bid AND both finite`, bucketed by `|delta|`: [0.05,0.15], [0.15,0.35], [0.35,0.65]. Index roots (SPX, SPY, QQQ, IWM), [0.05,0.15] bucket: **>=95%** per year = PASS; 90-95% = PARTIAL; **any year <90% = FAIL**. Single names: 90% / 85-90% / <85%. FAIL => stop, report; do **not** substitute another vendor."

Validated on SPY 2024 (full month 2024-01, 27,460,364 rows):

| `|delta|` bucket | RTH rows | quote-valid | fraction |
|---|---|---|---|
| [0, 0.05) | 7,222,732 | 6,319,127 | 87.489% |
| **[0.05, 0.15)** | 3,945,593 | 3,945,050 | **99.986%** |
| [0.15, 0.35) | 4,405,217 | 4,405,178 | 99.999% |
| [0.35, 0.65) | 4,664,922 | 4,664,920 | 100.000% |
| [0.65, 1] | 7,221,900 | 7,197,761 | 99.666% |

The gated [0.05,0.15] bucket clears 95% comfortably on SPY 2024. **This does not generalize
for free** — the registered gate is per root x year and the full sweep is still running.

**Structural limitation (ESC-2):** on the 323 sixteen-column partitions, `delta` does not
exist, so V1 **cannot be computed as registered** there. Those root-years are reported as
NOT MEASURABLE, not as PASS.

### V2 — Row semantics · IN PROGRESS

> **Registered gate:** ">=20 sampled sessions, >=1 per year 2013-2025 plus >=3 high-volatility sessions. Measure (a) rows/day vs listed contracts x RTH minutes, (b) fraction of rows with `volume=0` **and** valid quote. **>=60%** of rows `volume=0` with valid quote => quote-bearing confirmed (PASS). **<10%** => these are trade bars => **FAIL**, all intraday-quote-dependent work blocked. 10-60% => PARTIAL, report distribution."

Preliminary (SPY 2024-01, 614,400-row sample): 77.4% — PASS. Full sweep pending.

### V3 — Quote semantics · IN PROGRESS

> **Registered gate:** "Crossed (`bid>ask`) and locked (`bid==ask`) frequency; zero-bid frequency by moneyness; establish whether `bid_close`/`ask_close` are NBBO at minute end. Crossed **<0.1%** of quoted contract-minutes = PASS. Crossed rows excluded from marks **by rule regardless of rate**. Zero-bid handling documented, not repaired."

SPY 2024-01, full month (27,460,364 rows):

| Measure | Value |
|---|---|
| Crossed (`bid>ask`) | 9 rows = **0.00003%** — PASS |
| Locked (`bid==ask`, both >0) | 731 = 0.003% |
| Zero-bid | 928,319 = **3.381%** |
| Non-finite quote | 0 |

Zero-bid by moneyness `log(K/S)`: deep ITM-put side (<=-0.10) 4.24%, ATM (+/-0.02) 0.91%,
OTM-call side (>0.10) 6.45%. Documented, **not repaired**.

**DIVERGENCE:** the execution plan records zero-bid at **11.7%**. The full-month figure is
**3.38%**. The 11.7% came from a 614,400-row sample and is a sampling artifact. The
correct figure is 3.38%.

### V4 — Schema reconciliation · COMPLETE · REPORT-ONLY (gate satisfied) · **ESCALATION**

> **Registered gate:** "20 vs 21 columns (leading `symbol`); full dtype audit against the `Datetime[us, UTC]` standard. **Report only** — output is the canonicalization mapping."

Metadata-only sweep of **all 4,510 partitions**, 31 roots, **24,078,079,007 rows**,
249,992,314,784 bytes. Two independent implementations agree.

**The 20-vs-21 framing was a false binary. There are 9 distinct variants by dtype, and 3 by
column count:**

| ncols | first col | partitions | roots | years |
|---|---|---|---|---|
| 20 | `timestamp` | 2,784 | 31 | 2012-2026 |
| 21 | `symbol` | 1,403 | 22 | 2012-2026 |
| **16** | `symbol` | **323** | **17** | **2012-2016** |

- **Both the brief (20 cols) and `DATA_INVENTORY.md:248` (21 cols) are correct — about
  different partitions.** The 21st column is `symbol`, which is redundant (one distinct value
  per file, equal to the partition key). Mapping action: DROP.
- **The 16-column variant was undocumented by every source.** It is missing
  **`implied_vol`, `delta`, `theta`, `vega`, `underlying_px`** (ESC-2).
- **`timestamp` conforms to the `Datetime[us, UTC]` standard in 0 of 4,510 partitions (0%).**
  It is `large_string` (3,579) or `string` (931), always tz-naive.
- Further forks: `expiration` is `date32[day]` in 100 partitions; `right` is a `dictionary`
  in 100; `volume`/`trade_count` vary int32/int64; **274 partitions (2.18B rows) store OHLC,
  bid/ask and all four greeks as `float32`** — precision lost at write time, not recoverable
  by casting; `gamma_eod` is an all-null typed column in 318 partitions.

Practical consequence, confirmed by two crashes during this session: a reader that globs
across months and assumes a uniform schema **fails**
(`SchemaError: expiration: String != Date`; `volume: Int32 != Int64`). Partitions must be
read individually and cast.

**Canonicalization mapping delivered:** `output/vbattery/v04/v04_canonicalization_mapping.csv`.

#### The 16-column variant, scoped

323 partitions, 2012-2016 only, zero after 2016. Affected roots are **entirely ETFs/indices**
(plus AMD); **every single name has greeks throughout**.

| Root | no-greek partitions | Greeks reliable from |
|---|---|---|
| SPY | 35 / 164 | **2017** |
| GLD, IWM, EEM, SLV, XLV, FXI, SMH, XLK, XLE, TLT, DIA, XLI, XLF, SPX | 10-26 each | **2017** |
| VIX | 4 | 2016 |
| AMD | 11 | 2015 |
| AAPL, AMZN, MSFT, NVDA, TSLA, GOOGL, AVGO, MSTR, META, PLTR, COIN, IBIT, FB, **QQQ** | 0 | present throughout |

### V5 — IV/greeks population · IN PROGRESS

> **Registered gate:** "Null rate for `implied_vol`, `delta`, `theta`, `vega` per root x year. Sanity: `IV in (0.01, 5.0)`, `|delta| <= 1`, delta sign correct per `right`. **>=90%** non-null per root-year = PASS. Below => that root-year is flagged **untrusted**, routed to later recomputation. Do **not** recompute now."

SPY 2024-01: non-null 100% for all four fields — PASS on the gate as registered.

**But "non-null" overstates coverage.** Measured caveats:

| Measure | Value |
|---|---|
| `implied_vol` non-null | 100.0% |
| `implied_vol` **plausible** (registered screen `IV in (0.01,5.0)`) | **97.60%** |
| `implied_vol` **== 0.5 exactly** (solver fallback sentinel) | 1.59% (436,829 rows) |
| ... of which, share of **invalid-quote** rows that are IV==0.5 | **44.7%** |
| `|delta| <= 1` | 100.0% |
| delta sign correct per `right` | 100.0% (0 violations) |

The `IV == 0.5` sentinel concentrates in rows with no valid quote — it is a solver fallback,
not a measured IV. The registered plausibility screen is therefore **load-bearing**, and the
non-null rate alone should not be used to establish trust.

**ESC-2 consequence:** on the 323 sixteen-column partitions the four fields are **absent**,
i.e. 0% non-null. Per the registered gate those root-years are **UNTRUSTED** and routed to
later recomputation. No recomputation performed — not authorized this phase.

### V6 — EOD-join leakage · COMPLETE · **RULING'S RATIONALE CORRECTED**

> **Registered gate:** "HARD GATE. (a) read the join in `combine_options_data.py`; (b) verify `open_interest_eod` is constant across all minutes of a session (if it varies intraday the join is broken); (c) establish whether the value on session *t* is OI as-of *t*'s close or *t-1*'s close. Same-day join => **all OI/gamma features must lag >=1 session**, enforced in the primitive layer, not per-strategy. If the answer cannot be established from code + data => **all OI/gamma work BLOCKED**."

**(a) Same-day join — CONFIRMED at code level (independently re-read).**
`combine_options_data.py` derives `_date = pl.col("timestamp").str.slice(0, 10)` (the row's
own session) and left-joins the EOD frame on `["_date","expiration","strike","right"]`.

**(b) OI constant intraday — CONFIRMED.** 0 of 144,250 SPY contract-sessions show intraday
variation in `open_interest_eod`.

**(c) Which session's OI — the prior conclusion is NOT supported.**

Phase 0 and the execution plan both concluded: *"session t's rows carry t's END-OF-DAY OI,
published the following morning. Reading `oi_eod` at the 15:45 snapshot on session t is a
hard lookahead leak."* That claim was inferred from the join being same-day; it was never
measured. **Measured, it is contradicted.**

Discriminator: OI changes because of the trading in a session. If the date-*t* stamp is *t*'s
CLOSING OI, then `OI[t]-OI[t-1]` tracks `volume[t]`. If it is START-OF-DAY OI (what exchanges
actually publish on the morning of *t*, reflecting *t-1*'s close), then `OI[t+1]-OI[t]` tracks
`volume[t]`.

| Root/period | corr(vol_t, OI_t - OI_t-1) | corr(vol_t, OI_t+1 - OI_t) | mean abs dOI / vol (same) | (next) | frac. new contracts with OI=0 on first traded session |
|---|---|---|---|---|---|
| SPY 2024 | 0.094 | **0.534** | 7.80 | **1.02** | **94.9%** |
| QQQ 2021 | 0.164 | **0.286** | 7.34 | **1.03** | **91.8%** |
| AAPL 2020 | 0.448 | **0.477** | 1.90 | **0.87** | **93.5%** |
| TSLA 2022 | 0.480 | 0.451 | 1.49 | **0.69** | **84.5%** |
| SPX 2024 | 0.269 | **0.598** | 12.46 | **1.66** | 23.8% |
| SPY 2013 | 0.179 | 0.148 | 12.39 | **1.76** | 0.0% |

Three independent signals agree for the modern era: the ratio `mean|dOI|/volume` sits at
**~1.0 for the next-day alignment** (exactly what a correct alignment predicts) versus 1.5-12
for same-day; the correlation favours next-day; and **85-95% of newly-listed contracts show
OI = 0 on their first traded session**, which is only possible if the stamp is start-of-day.

**Therefore: the date-*t* `open_interest_eod` is START-OF-DAY OI (= *t-1*'s close). The
same-day join is NOT the lookahead leak it was described as.**

SPY 2013 dissents on the new-listing test (0.0%) while agreeing on the ratio test. Its
first-observed-session sample is small (1,396) and probably window-truncated rather than
genuinely newly-listed, but this could not be proven.

**`gamma_eod` is NOT resolved.** It is well-formed data (mean gamma peaks correctly at
`|delta|~0.5`: 0.0245 at [0.45,0.55) falling to 0.0031 at [0,0.05)), but no test performed
distinguished which session's close it was computed at.

**Registered-gate outcome: PARTIAL.** The answer is established for OI in the modern era, not
uniformly across the store, and not at all for gamma. Per the gate's own terms
("if the answer cannot be established ... BLOCKED"), **the >=1-session lag stands and is
implemented** (`oi_eod_lag1`, `gamma_eod_lag1`, unit-tested). It is conservative and safe
under *both* hypotheses.

**What changes is the rationale, not the rule.** The lag survives as belt-and-braces, not as
a leak correction. Relaxing it is a decision for the principal, not for this phase — and it
should not be relaxed for `gamma_eod` on any current evidence.

### V7 — Corporate actions · COMPLETE · **FAIL / ESCALATION (ESC-1)**

> **Registered gate:** "Strike-grid behavior across: AAPL 2020-08-31 (4:1), TSLA 2020-08-31 (5:1) and 2022-08-25 (3:1), NVDA 2021-07-20 (4:1) and 2024-06-10 (10:1), AMZN 2022-06-06 (20:1), GOOGL 2022-07-18 (20:1). Report adjusted vs as-reported. **As-reported => adjustment layer required before ANY single-name work.**"

**7 of 7 AS-REPORTED. Zero back-adjusted.**

| Event | Factor | underlying pre -> post | ratio | median strike ratio | max strike pre -> post |
|---|---|---|---|---|---|
| AAPL 2020-08-31 | 4:1 | 501.57 -> 129.45 | 3.87 | 3.91 | 655 -> 170 |
| TSLA 2020-08-31 | 5:1 | 2246.53 -> 482.27 | 4.66 | 5.13 | 3000 -> 640 |
| TSLA 2022-08-25 | 3:1 | 900.65 -> 293.90 | 3.06 | 3.02 | 1180 -> 391.67 |
| NVDA 2021-07-20 | 4:1 | 749.36 -> 184.58 | 4.06 | 4.02 | 985 -> 243.75 |
| NVDA 2024-06-10 | 10:1 | 1202.40 -> 121.96 | 9.86 | 9.95 | 1580 -> 159 |
| AMZN 2022-06-06 | 20:1 | 2443.53 -> 125.53 | 19.47 | 19.30 | 3230 -> 167.5 |
| GOOGL 2022-07-18 | 20:1 | 2233.02 -> 111.43 | 20.04 | 19.88 | 4500 -> 225 |

Both `underlying_px` and the strike grid jump discontinuously by the split factor. Nothing is
restated onto the post-split grid.

**Non-standard deliverables are present**, the expected as-reported artifact. Post-split modal
strike increments land on the split fraction: TSLA 2022 increment **1.67** (=5/3) with **all
120 strikes off-grid** on 08-25 (124 on 08-26, 116 on 08-29); NVDA 2021 increment 1.25 (=5/4)
with 42/136 off-grid persisting for at least three sessions. AAPL 8 off-grid, GOOGL 6; AMZN
and TSLA-2020 clean (their factors divide the 5.00 grid evenly).

**Registered consequence: an adjustment layer is required before ANY single-name work.**
Not built — work order Section 4 does not authorize it.

### V8 — Root continuity · COMPLETE · PARTIAL · **ESCALATIONS (ESC-3, ESC-4)**

> **Registered gate:** "FB (2017-2021) -> META splice; SPX vs SPXW composition and AM/PM settlement; VIX expiry convention. Report + registered splice/filter rules per root."

**1. FB -> META — the splice premise is wrong (ESC-3).**

- `root=FB`: 2017-01-03 -> 2021-10-29, 58 partitions, no missing months.
- `root=META`: 2021-07-08 -> 2026-02-04, 52 partitions, **missing 2022-02 through 2022-05**.

The nominal 2021-07..2021-10 "overlap" is not overlapping content. Independently verified:

| Root/month | median `underlying_px` | strike range |
|---|---|---|
| META 2021-08 | **14.70** | 10-20 |
| **FB 2021-08** (same month) | **361.88** | 5-700 |
| META 2022-01 | 13.65 | 3-32 |
| META 2022-06 | 164.17 | 5-700 |
| META 2023-01 | 136.27 | 5-700 |

`root=META` for **2021-07 .. 2022-01 (7 months, 2,044,148 rows) is Meta Materials, a different
issuer** — ~25x apart in price from contemporaneous FB, with a disjoint strike grid. Meta
Platforms data begins **2022-06-09**.

Consequence: FB stops 2021-10-29 and Meta Platforms starts 2022-06-09, leaving a **~7.4-month
hole**. A naive FB+META concatenation would both splice in a penny stock and silently bridge
that gap.

**2. SPX — no SPXW (ESC-4).** 12 months sampled (2025-03..2026-02), 31 distinct expirations:
Friday 28, Thursday 3 (holiday shifts). `is_third_friday` 28/31. Decisive settlement
discriminator: **`exp_day_rows == 0` for all 31 expirations** — every expiration's last traded
session is the session *before* the expiration date, the textbook AM-settlement signature.
**AM-like 31/31 = 100%. PM-like 0.** No weeklies, no Monday/Wednesday, no same-day expirations.

Independently spot-checked on 2024 Q1: 2024-01-19 expiry last traded 2024-01-18; 2024-02-16
last traded 2024-02-15; 2024-03-15 last traded 2024-03-14. Corroborated.

So "SPX vs SPXW composition" resolves to: **not commingled, because only one population
exists**. The AM/PM distinguishability question is moot for the same reason.

**3. VIX.** 24 months sampled, 32 expirations: Wednesday 29, Tuesday 3 (2024-06-18, 2025-03-18,
2026-05-19). Consistent with the Wednesday convention plus the 30-days-before-SPX-expiry rule
and holiday shifts. Convention, not anomaly — the proposed rule explicitly does not drop them.

Five proposed rules (FB-01, META-01 quarantine, META-02 splice, SPX-01 scope, VIX-01 filter)
are recorded in `output/vbattery/v08/v08_proposed_rules.csv`, all marked
`PROPOSED -- not applied`. **None applied anywhere.**

**Coverage limitation:** SPX sampled over the last 12 of 164 partitions, VIX the last 24 of
165. Findings are firm for those windows; earlier eras — particularly the pre-2017 16-column
era — are **unmeasured**.

### V9 — Partition integrity · IN PROGRESS · **preliminary: ESC-5**

> **Registered gate:** "Sessions present vs expected per root-month 2012-2026; quantify the 2026-03 -> present gap; half-day handling. Any root-month with **<90%** of expected sessions flagged. Refresh executed only if the subscription supports it."

Refresh is **NOT executable** — the ThetaData subscription is cancelled. No download attempted.
The gap stands as a live-edge caveat.

Metadata inventory of all 4,510 partitions flags **98 truncated partitions** (<90% of the
spanned month). Two systematic patterns:

**(i) The store's real edge is 2025-12-26, not 2026-02.** Verified directly:

| Root | 2025-12 | 2026-01 | 2026-02 |
|---|---|---|---|
| SPY | 2025-12-01 -> **2025-12-26** (19/22) | **1 session** (01-02 only) | **3 sessions** (02-02..02-04) |
| QQQ | same | 1 session | 3 sessions |
| IWM | same | 1 session | 3 sessions |

Every doc in the chain describes the window as "2012-06 -> 2026-02". The **last complete month
is 2025-12**; 2026-01 and 2026-02 are 1- and 3-session stubs. This is a material divergence
from the described data.

**(ii) SPY is heavily truncated across 2017-H1 and 2018.** Verified directly: SPY 2017-01
contains **2 sessions** (Jan 3-4, 1.25M rows); 2017-03 stops Mar 8; 2017-06 stops Jun 22.
QQQ and IWM are complete in the same months, so this is SPY-specific, not a store-wide outage.

**Combined with ESC-2, the honest greek-bearing, non-truncated window for SPY is
approximately 2017-09 -> 2025-12 (~8.3 years), not the 13.7 years the chain assumes.**
SPY is the underlying of the majority of slate candidates. This is a data property; its
consequences for statistical power are not assessed here.

Half-day handling is implemented (NYSE calendar early closes; a half-day is not a missing
session) and the vendor's 16:00 padding of half-days is handled in the canonical layer.

### V10 — Provenance mix · COMPLETE · **UNIFORMLY THETADATA** (gate consequence not triggered)

> **Registered gate:** "`DATA_INVENTORY.md` lists ThetaData **+ IBKR chains**; the download scripts are ThetaData. Which rows, if any, are IBKR-sourced; are populations distinguishable. If mixed and indistinguishable, V1/V3 statistics must be computed conservatively over the whole."

Three independent lines of evidence, all negative for IBKR:

- **Schema:** union of distinct column names across all 4,510 partitions is exactly 21 names.
  **No `source`/`vendor`/`feed`/`origin`/`venue`/`exchange` column exists anywhere.** The only
  candidate token, `symbol`, is just the root ticker.
- **Writers:** 14 options modules inspected — **0 mention IBKR/ib_async/ib_insync/TWS**;
  8 mention ThetaData. All 5 partition-writing scripts are ThetaData-sourced.
- **Logs:** all 34 files in `H:\Stock_Data\options\_logs` (217 MB) scanned line-by-line:
  **0 IBKR lines, 173 ThetaData lines.**

`DATA_INVENTORY.md:248`'s "ThetaData + IBKR options chains" is **not supported by any artifact
in the store**. The population is not mixed, so V1/V3 need not be computed conservatively on
provenance grounds.

**Stated honestly:** this is an absence-of-evidence argument. Nothing in the data *marks*
provenance, so IBKR rows written by a since-deleted script leaving no log would be
undetectable. What is assertable: every surviving writer, log, and schema is ThetaData.

### V11 — Spread census · IN PROGRESS (deliverable, no gate)

> **Registered gate:** "Quoted-width distribution by root x moneyness x DTE bucket x year x volatility state, including event windows and 0DTE. **No gate — this is a deliverable**: a parquet of width statistics that later parameterizes the cost model."

Cells: root x year x moneyness x DTE bucket {0DTE, 1-7, 8-30, 31-60, 61-90, 91-180, 181+} x
`|delta|` bucket, over valid quotes only, carrying excluded counts and `mid` percentiles.

**Volatility state is a stated causal proxy**, not `regime_state_daily`: trailing-20-session
realized vol of per-session last `underlying_px`, expanding-percentile ranked, strictly
backward-looking. Labelled as such in the output schema so it is not mistaken for the
classifier state.

Percentiles are reservoir-sampled (cap 1,500/cell) — an estimate, disclosed in the metadata.

### V12 — Legacy store · COMPLETE · **MOOT / CLOSED**

> **Registered gate:** "`options/options_1min/` (17 roots, 2024-11 -> 2025-12) vs `options_combined` overlap equivalence. Report equivalence. **Delete nothing.**"

Direct listing of `H:\Stock_Data\options`:

| Entry | Files | Bytes |
|---|---|---|
| `_logs` | 34 | 217,329,575 |
| `chains` | **0** | **0** |
| `gex_daily` | **0** | **0** |
| `options_combined` | 4,510 | 249,992,314,784 |

`options_1min/` checked at three plausible locations — all **absent**. Overlap-equivalence is
therefore **unrunnable, not failed**, and was correctly skipped rather than faked.
`combine_options_data.py` documents joining `options_1min/` with `options_eod/`, so the legacy
store did once exist; it is gone now.

**Nothing deleted. Nothing modified.**

### V13 — PIT universe coverage · COMPLETE · **FAIL / NOT MEASURABLE**

> **Registered gate:** "For every point-in-time-defined universe (rank/weight-based membership), fraction of sessions where **all** required members exist on disk; report the earliest date from which coverage is complete. **<95% of sessions fully covered => that universe is NOT usable as registered**; report the first-full-coverage date and the implied usable window."

**Primary finding: no point-in-time membership source exists in this repo.**
`config/universes/` holds only `sp500-2025.csv`, `russell1000-2025.csv`,
`russell2000-2025.csv` — **today snapshots**, survivorship-biased by construction, with no
weights and no `as_of` column. Using one as membership at a 2013 date is exactly the bias
spec v2 rule 7 forbids.

| Universe | PIT source? | Coverage | Verdict |
|---|---|---|---|
| `U_INDEX` {SPY,QQQ,IWM} | n/a (fixed) | **94.42%** | **FAIL** (<95%) |
| `U_TOP6_SPY` | **NO** | — | **NOT MEASURABLE AS REGISTERED** |
| `U_MEGA10` | **NO** | — | **NOT MEASURABLE AS REGISTERED** |
| `U_MEGA20` | **NO** | — | **NOT MEASURABLE AS REGISTERED** |
| `U_TIER1_100` | **NO** | — | **NOT MEASURABLE AS REGISTERED** |

No optimistic today-snapshot substitute was reported as if it were PIT.

`U_MEGA10/20` and `U_TIER1_100` are additionally unconstructible in principle from this store:
they rank by option volume across the whole US market, and only 31 roots are on disk. Ranking
within the 31 and calling the result `U_MEGA10` would be a different universe wearing the same
name.

**`U_INDEX` fails at 94.42%**, 0.58pp under the gate, driven by the universal recent-ingest
hole and per-root gaps. The gate is registered at 95% and was **not** adjusted to accommodate
the miss. Reported as FAIL.

---

## 3. Divergence log

Everything the brief, the work order, or the Phase-0 chain asserted that measurement
contradicted.

| # | Asserted | Reality | Evidence |
|---|---|---|---|
| D1 | `timestamp` needs tz normalization from a datetime type | `timestamp`/`expiration` are **strings** (`large_string` or `string`), tz-naive. **0 of 4,510** partitions comply with the `Datetime[us,UTC]` standard | V4 sweep |
| D2 | `right in {C,P}` (spec v2 §1.2) | Values are **`PUT`/`CALL`** | SPY 2024-01 |
| D3 | V5 "IV 100% non-null" implies coverage | Non-null but **`IV == 0.5` exactly** on 1.59% of rows / **44.7% of invalid-quote rows** — a solver sentinel. Plausible-IV rate is 97.60% | V5 shard |
| D4 | V4 is a binary 20-vs-21-column question | **3 column counts, 9 dtype variants.** Both the brief and `DATA_INVENTORY.md` are right about different partitions | V4 sweep |
| D5 | (nobody asserted) | **16-column variant, 323 partitions, no greeks and no `underlying_px`** — undocumented everywhere | V4 sweep |
| D6 | Store spans 2012-06 -> **2026-02** | Last **complete** month is **2025-12** (edge 2025-12-26). 2026-01 = 1 session, 2026-02 = 3 sessions | Footer inventory |
| D7 | (nobody asserted) | **SPY truncated across 2017-H1/2018**; 2017-01 has 2 sessions. QQQ/IWM complete in the same months | Footer inventory |
| D8 | V6: "session t carries t's END-OF-DAY OI => hard lookahead leak" | **Contradicted.** Modern-era evidence says the stamp is **start-of-day** (= t-1 close). The lag rule stands on conservatism, not on this rationale | V6 attribution |
| D9 | V7 unknown | **7/7 AS-REPORTED**, with non-standard deliverables present | V7 shard |
| D10 | "FB -> META splice" closes the series | `root=META` 2021-07..2022-01 is **Meta Materials**; 7.4-month Meta Platforms hole remains | V8 + independent check |
| D11 | "SPX vs SPXW composition ... AM/PM settlement" presumes both present | **No SPXW at all**; 31/31 AM-settled third-Friday | V8 + independent check |
| D12 | `DATA_INVENTORY.md`: source is "ThetaData + IBKR chains" | No IBKR trace in schema, 14 writers, or 34 logs | V10 |
| D13 | Zero-bid frequency is **11.7%** (execution plan) | **3.38%** full-month. The 11.7% was a 614k-row sampling artifact | V3 shard |
| D14 | VIX spot needed materializing from 2012-06 | Available from **1990-01-02**; materialized 9,204 rows to 2026-07-24 | VIX sidecar |
| D15 | (nobody asserted) | **Vendor pads early-close sessions to 16:00 with stale quotes** — a real post-close mark hazard; canonical cutoff clamps to the true close | Canonical layer |
| D16 | `DATA_INVENTORY.md`: "24.1B rows, 4,510 files, ~250 GB" | **CONFIRMED**: 24,078,079,007 rows, 4,510 files, 249,992,314,784 bytes | V4 sweep |

---

## 4. Limitations

- **The V1/V2/V3/V5/V9/V11 sweep is incomplete at time of writing.** Definitions are validated
  against SPY 2024-01 and reconcile with the known preliminary values. Root-years not yet swept
  are reported as NOT MEASURED, never as PASS.
- **V8 SPX/VIX are windowed** (last 12 and 24 partitions of ~165). Pre-2017 eras unmeasured.
- **V7 covers 3 sessions each side** of each event — sufficient to discriminate adjusted vs
  as-reported, not to enumerate every non-standard deliverable.
- **V10 cannot prove a negative.**
- **V6 `gamma_eod` date attribution is unresolved.**
- **float32 precision loss in 274 partitions** is stated as fact; the resulting error magnitude
  was not quantified.
- The SPY-2013 V6 dissent could not be conclusively attributed to window truncation.

---

## 5. Artifacts

**Code**
- `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\src\data\options\canonical.py`
- `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\tests\data\test_options\test_canonical.py`
- `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\scripts\data\build_options_chain_eod.py`
- `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\scripts\data\build_vix_spot.py`
- `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\scripts\data\build_regime_state_daily.py`
- `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\tests\data\test_vix_spot_and_regime.py`
- `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\scripts\data\vbattery\` — V4, V6, V7, V8, V10,
  V12, V13, schema/coverage inventory, and the V1/V2/V3/V5/V9/V11 sweep

**Measurement shards** — `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\output\vbattery\`
(`v04/`, `v06/`, `v07/`, `v08/`, `v10/`, `v12/`, `v13/`, `schema_coverage/`, `sweep/shards/`)

**Run-status** — `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\output\run_status\`

*No strategy backtest has been run. No P&L has been observed.*
