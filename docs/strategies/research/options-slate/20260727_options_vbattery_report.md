# Options Slate -- Phase 1 V-Battery Report

**Date:** 2026-07-27
**Scope:** Phase 1a canonicalization + Phase 1b verification battery V1-V13 over the owned
`options_combined/` store, per
`2026-07-25_options_phase01_cc_work_order.md` Sections 2 and 3.
**Status:** **COMPLETE.** All of V1-V13 measured. The V1/V2/V3/V5/V9/V11 sweep covered
**4,510 of 4,510 partitions, 31 of 31 roots, 415 root-years, 24,078,079,007 rows, 93,179
sessions** -- full coverage, verified by reconciling per-root partition counts on disk against
the per-partition schema records (zero mismatches).

**No strategy backtest has been run. No P&L has been computed or observed.**
Per work order Section 5, this report states measurements and gate outcomes only. It draws
**no conclusions about strategy viability**.

Per work-order ground rule 1: where this report and the doc chain disagree, **this report
wins** and the divergence is stated rather than silently reconciled.

---

## 0. Escalation summary -- read this first

Work order Section 6 requires stopping and reporting, without adapting, on the following.
**Seven escalation conditions have fired.**

| # | Escalation | Registered consequence |
|---|---|---|
| **ESC-1** | **V7: 7 of 7 corporate actions are AS-REPORTED.** Not one is back-adjusted | "As-reported => adjustment layer required before **ANY** single-name work" |
| **ESC-2** | **V4: a 16-column schema variant exists (323 partitions, ~1.0B rows, 17 roots, 2012-2016) carrying NO `implied_vol`, `delta`, `theta`, `vega`, or `underlying_px`** | V1's delta-bucketed gate is **structurally unmeasurable** on those partitions; V5 is 0% non-null there by construction => those root-years are **UNTRUSTED** per the registered V5 gate |
| **ESC-3** | **V8: `root=META` for 2021-07 .. 2022-01 is a DIFFERENT ISSUER** (Meta Materials, underlying ~USD 14-16), and Meta Platforms has a **7.4-month hole** (2021-10-30 .. 2022-06-08) | The assumed "FB -> META splice" does not close the series and would splice in a penny stock |
| **ESC-4** | **V8: `root=SPX` contains NO SPXW.** 31/31 sampled expirations are AM-settled third-Friday | Weekly / 0DTE SPX work is **not testable** on this store |
| **ESC-5** | **The data materially differs from the brief's description.** Last complete month is **2025-12**; the store's last session is **2026-02-04**; and greeks are absent pre-2017 for SPY/IWM/SPX and the ETFs | Work order Section 6: "discovery that the data materially differs from Section 2's description" |
| **ESC-6** | **V1 FAILs on 12 non-index root-years**, and is **UNGRADEABLE on 82 root-years** (no `delta` column). No index root-year fails | "FAIL => stop, report; do **not** substitute another vendor" |
| **ESC-7** | **The preliminary V-battery figures that motivated this phase were a 6-minute sample, not a month.** They do not reproduce | See D1 -- the premise "all four pass by wide margins" was measured on the opening 6 minutes of one session |

Two further items are **not** escalations but must be read before Phase 2: a **correction to a
registered ruling's rationale** (V6), and a **repo bug found and fixed** (Section 4).

---

## 1. Phase 1a -- Canonicalization (COMPLETE)

### 1.1 What was built

| Artifact | Path |
|---|---|
| Canonical layer | `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\src\data\options\canonical.py` |
| Tests (36, all passing) | `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\tests\data\test_options\test_canonical.py` |
| EOD builder CLI | `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\scripts\data\build_options_chain_eod.py` |

**Table A `options_chain_1m`** -- a streaming READ layer (no bulk copy of 233 GB).
`timestamp`/`expiration` parsed from ISO-8601 strings, localized US/Eastern, converted to the
repo standard `Datetime[us, UTC]`. `right` normalized `PUT->P` / `CALL->C`, raising on any
unexpected value. Derived `mid`, `spread_abs`, `spread_rel`, `dte`, `dte_trading`,
`quote_valid`.

**Table B `options_chain_eod`** -- materialized at the registered snapshot minute, one row per
contract per session.

### 1.2 Binding rules -- verification evidence

| Rule | Verified how | Result |
|---|---|---|
| Snapshot minute is 15:45:00 ET, one guarded function | Behavioural check on materialized SPY 2024-01 (68,615 rows) | **0 snapshots after 15:45 ET.** 67,766 exact 15:45 hits; 849 fallbacks with `snapshot_fallback=true` and true `snapshot_ts` |
| Marks from quotes, never trade prints | `close`/`vwap` pass through but are never used to derive `mid` | Enforced; module docstring states it |
| Invalid quotes excluded, never repaired | `mid`/`spread_*` NULL exactly where `quote_valid` false | Unit-tested; EOD table `quote_valid` all true, `mid` null count 0 |
| `_eod` suffix preserved | Column assertion on materialized output | Columns include `gamma_eod`, `oi_eod`, `oi_eod_lag1`, `gamma_eod_lag1`. **No bare `open_interest`/`gamma`** -- the `data_loader.py:20-26` footgun is not repeated |
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
| VIX index spot | 1990-01-02 -> 2026-07-24, 9,204 rows | Was previously **fetched** at call time from yfinance `^VIX` -- not reproducible. Now materialized with a provenance sidecar (source + snapshot timestamp). |
| `regime_state_daily` | 2012-06-01 -> 2026-02-27, 3,455 sessions | Causal replay via `analyze_regime_history()`. Sidecar records the VIX snapshot consumed, SPY source, and the data-vintage caveat verbatim. |

The regime causality regression test passes **with a negative control** (a deliberately
non-causal variant is confirmed to fail the same assertion), so the test has demonstrated
power rather than passing vacuously.

**Data-vintage caveat, to be stated on every downstream result:** replay uses TODAY's SPY/VIX
series. If that data was revised or backfilled since, the recomputed state differs from what
was known at the time.

---

## 2. Phase 1b -- V-battery results

Gates are restated **verbatim** from work order Section 3 so this report is auditable against
what was registered. No threshold was altered after measurement.

### V1 -- Quote population -- COMPLETE -- **PASS (index) / PARTIAL (store-wide)** -- ESC-6

> **Registered gate:** "Per root x year, fraction of regular-trading-hours listed-contract-minutes with `bid>0 AND ask>=bid AND both finite`, bucketed by `|delta|`: [0.05,0.15], [0.15,0.35], [0.35,0.65]. Index roots (SPX, SPY, QQQ, IWM), [0.05,0.15] bucket: **>=95%** per year = PASS; 90-95% = PARTIAL; **any year <90% = FAIL**. Single names: 90% / 85-90% / <85%. FAIL => stop, report; do **not** substitute another vendor."

**Classification applied, stated not hidden:** SPX/SPY/QQQ/IWM under the index thresholds. All
27 remaining roots -- **including the ETFs (GLD, SLV, TLT, EEM, FXI, SMH, XLE/XLF/XLI/XLK/XLV),
DIA, and VIX** -- were graded under the **single-name** thresholds (90 / 85-90 / <85). The gate
names only "index roots" and "single names"; rather than invent a third threshold, the looser of
the two was applied to everything not explicitly named.

**Bucket edges, declared before measuring:** `[0,0.05)`, `[0.05,0.15)`, `[0.15,0.35)`,
`[0.35,0.65]`, `(0.65,1]`, `null/out-of-range` -- half-open below so no row is double-counted.

| Class | PASS | PARTIAL | FAIL | UNGRADEABLE (no `delta`) |
|---|---|---|---|---|
| **INDEX (SPY/SPX/QQQ/IWM)** | **45** | **0** | **0** | **15** |
| Non-index (single-name thresholds) | 267 | 9 | **12** | 67 |

**Index roots PASS on every gradeable year**, with enormous margin -- lowest is QQQ 2015 at
99.03%, everything else 99.86-100.00%. The gate binds hardest here and it clears cleanly.

Index [0.05,0.15] quote-valid %, by year (blank = greeks absent, ungradeable):

| year | IWM | QQQ | SPX | SPY |
|---|---|---|---|---|
| 2012-2016 | -- | 99.03-99.98 | -- | -- |
| 2017 | 99.96 | 99.95 | 100.00 | 99.99 |
| 2018 | 99.95 | 99.94 | 99.99 | 99.97 |
| 2019 | 100.00 | 99.99 | 100.00 | 99.95 |
| 2020 | 99.88 | 99.95 | 99.95 | 99.94 |
| 2021 | 99.92 | 99.99 | 99.97 | 99.97 |
| 2022 | 99.99 | 99.99 | 99.99 | 99.97 |
| 2023 | 100.00 | 99.99 | 100.00 | 99.98 |
| 2024 | 99.89 | 99.94 | 99.99 | 99.98 |
| 2025 | 99.86 | 99.92 | 99.99 | 99.95 |
| 2026 | 99.90 | 99.99 | 100.00 | 99.99 |

**Every FAIL (<85%), enumerated -- not summarized away:**

| root | year | n rows | frac |
|---|---|---|---|
| VIX | 2017 | 681,149 | **0.3924** |
| MSTR | 2015 | 81,428 | 0.5913 |
| MSTR | 2016 | 95,363 | 0.6038 |
| MSTR | 2014 | 69,903 | 0.6751 |
| AMD | 2015 | 412,615 | 0.7501 |
| META | 2021 | 281,902 | 0.7549 |
| VIX | 2018 | 872,186 | 0.7586 |
| MSTR | 2013 | 142,662 | 0.8000 |
| AVGO | 2012 | 61,251 | 0.8008 |
| MSTR | 2018 | 64,793 | 0.8031 |
| MSTR | 2017 | 128,916 | 0.8339 |
| EEM | 2024 | 2,878,758 | 0.8420 |

**Every PARTIAL (85-90%):** XLV 2025 (0.8534), AMD 2016 (0.8610), FXI 2025 (0.8610), VIX 2019
(0.8624), XLV 2026 (0.8670), MSTR 2019 (0.8737), AVGO 2013 (0.8813), XLK 2026 (0.8928), FXI
2026 (0.8988).

Note META 2021 (0.7549) is a FAIL, and per V8 that root-year is **Meta Materials, not Meta
Platforms** -- the two findings compound.

**Structural limitation (ESC-2):** on the 323 sixteen-column partitions `delta` does not exist,
so V1 **cannot be computed at the registered bucket**. Those **82 root-years** -- including the
entire 2012-2016 era for SPY/SPX/IWM -- are reported as **UNGRADEABLE**, a category of their own.
They are not folded into PASS and not into FAIL.

Full table: `output/vbattery/sweep/v1_gate_by_root_year.csv`.

### V2 -- Row semantics -- COMPLETE -- **PASS**

> **Registered gate:** ">=20 sampled sessions, >=1 per year 2013-2025 plus >=3 high-volatility sessions. Measure (a) rows/day vs listed contracts x RTH minutes, (b) fraction of rows with `volume=0` **and** valid quote. **>=60%** of rows `volume=0` with valid quote => quote-bearing confirmed (PASS). **<10%** => these are trade bars => **FAIL**, all intraday-quote-dependent work blocked. 10-60% => PARTIAL, report distribution."

**High-volatility sessions, chosen before any measurement:** 2015-08-24 (ETF flash-crash open),
2018-02-05 (Volmageddon/XIV), 2020-03-16 (COVID limit-down), 2024-08-05 (yen-carry unwind).

**(a) Rows/day vs listed contracts x RTH minutes:** median grid density **1.0000**, p05 0.9615.
The store is a **complete dense grid** -- every listed contract has a row for every RTH minute.

**(b) `volume=0` AND valid quote: 91.78% store-wide** over 24.08bn rows / 93,179 sessions
(`volume=0` alone is 95.20%). Per-root range **87.40% (AAPL) to 95.34% (SPX)** -- all 31 roots
clear the 60% gate.

Registered SPY sample (30 sessions: 2/year 2013-2025 + the 4 pre-specified stress days): min
0.8391, median 0.9048, max 0.9335 -- **30/30 above the gate**. Stress days across all roots:
2015-08-24 = 0.8733, 2018-02-05 = 0.9234, 2020-03-16 = 0.8990, 2024-08-05 = 0.9138.

**Verdict: PASS, decisively** -- cleared by ~32 percentage points, and corroborated
independently by the dense-grid result. These are quote-bearing bars, not trade bars.

**Distribution caveat, reported not smoothed:** 201 of 93,179 sessions fall below 60% and 67
below 10%. These are not noise -- they are whole sessions with **100% null `bid_close`**,
verified directly (COIN 2022-01-03: 276,828 rows, `bid_close` null on every one). Concentrated
in COIN 2022-01 (9 sessions), MSTR (67 sessions, 2013-2022), SPX 2015 (85), AMZN 2015 (20).

### V3 -- Quote semantics -- COMPLETE -- **PASS**

> **Registered gate:** "Crossed (`bid>ask`) and locked (`bid==ask`) frequency; zero-bid frequency by moneyness; establish whether `bid_close`/`ask_close` are NBBO at minute end. Crossed **<0.1%** of quoted contract-minutes = PASS. Crossed rows excluded from marks **by rule regardless of rate**. Zero-bid handling documented, not repaired."

Store-wide over **24,078,079,007** RTH contract-minutes:

| Measure | Count | Rate |
|---|---|---|
| **Crossed** (`bid>ask`) | 894,830 | **0.003716%** -- **PASS** |
| Locked (`bid==ask`, both >0) | 779,664 | 0.0032% |
| Zero-bid | 844,215,406 | 3.5062% |
| Non-finite quote | 53,183,735 | 0.2209% |
| `quote_valid` | -- | 96.4901% |

**Crossed passes by a factor of 27.** Zero of 415 root-years reach 0.1%; the worst is XLE 2013
at 0.0400%. Crossed rows are excluded from marks **by rule regardless of rate**, implemented in
the single registered `quote_valid` predicate (`ask >= bid`) applied at every call site.

Zero-bid by moneyness `log(K/S)` -- **documented, not repaired:**

| bucket | n | zero-bid |
|---|---|---|
| `<= -0.10` | 5,583,196,635 | 4.438% |
| `-0.10..-0.05` | 2,560,199,031 | 2.584% |
| `-0.05..-0.02` | 2,206,953,728 | 1.525% |
| **`-0.02..0.02` (ATM)** | 3,650,263,495 | **0.646%** |
| `0.02..0.05` | 2,054,342,193 | 2.494% |
| `0.05..0.10` | 2,128,549,676 | 4.113% |
| `> 0.10` | 3,428,292,610 | 5.636% |
| undefined (no `underlying_px`) | 2,466,281,639 | 5.719% |

A clean U-shape with its minimum at the money -- the expected microstructural signature, and
independent evidence the quotes are real rather than synthetic.

**NBBO-at-minute-end: INFERRED, NOT PROVEN.** This could not be established from the store
alone; no vendor spec is available and the subscription is cancelled. Three consistent
indications: the `_close` suffix matches the OHLC bar convention; 91.78% of rows carry a valid
quote with zero volume, which only a quote snapshot produces; and the moneyness U-shape matches
live NBBO behaviour. **Treat as an end-of-minute quote snapshot, provenance unconfirmed.**

**Structural finding -- the 09:30 bar carries no quote at all.** Independently verified on SPY
2024-01: **all 70,239 rows at 09:30 have `bid_close == 0 AND ask_close == 0`** (literal zero,
not null), versus 68,022 of 70,239 valid at 09:31. The opening minute is systematically
unquoted, contributing ~0.256% of RTH contract-minutes to the zero-bid population. Correctly
excluded by the registered rule; flagged because **any strategy referencing the 09:30 bar as a
mark will get nothing.**

### V4 -- Schema reconciliation -- COMPLETE -- REPORT-ONLY (gate satisfied) -- **ESCALATION**

> **Registered gate:** "20 vs 21 columns (leading `symbol`); full dtype audit against the `Datetime[us, UTC]` standard. **Report only** -- output is the canonicalization mapping."

Metadata-only sweep of **all 4,510 partitions**, 31 roots, **24,078,079,007 rows**,
249,992,314,784 bytes. Two independent implementations agree.

**The 20-vs-21 framing was a false binary. There are 9 distinct variants by dtype, and 3 by
column count:**

| ncols | first col | partitions | roots | years |
|---|---|---|---|---|
| 20 | `timestamp` | 2,784 | 31 | 2012-2026 |
| 21 | `symbol` | 1,403 | 22 | 2012-2026 |
| **16** | `symbol` | **323** | **17** | **2012-2016** |

- **Both the brief (20 cols) and `DATA_INVENTORY.md:248` (21 cols) are correct -- about
  different partitions.** The 21st column is `symbol`, which is redundant (one distinct value
  per file, equal to the partition key). Mapping action: DROP.
- **The 16-column variant was undocumented by every source.** It is missing
  **`implied_vol`, `delta`, `theta`, `vega`, `underlying_px`** (ESC-2).
- **`timestamp` conforms to the `Datetime[us, UTC]` standard in 0 of 4,510 partitions (0%).**
  It is `large_string` (3,579) or `string` (931), always tz-naive.
- Further forks: `expiration` is `date32[day]` in 100 partitions; `right` is a `dictionary`
  in 100; `volume`/`trade_count` vary int32/int64; **274 partitions (2.18B rows) store OHLC,
  bid/ask and all four greeks as `float32`** -- precision lost at write time, not recoverable
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

### V5 -- IV/greeks population -- COMPLETE -- **PARTIAL** (331 PASS / 84 UNTRUSTED)

> **Registered gate:** "Null rate for `implied_vol`, `delta`, `theta`, `vega` per root x year. Sanity: `IV in (0.01, 5.0)`, `|delta| <= 1`, delta sign correct per `right`. **>=90%** non-null per root-year = PASS. Below => that root-year is flagged **untrusted**, routed to later recomputation. Do **not** recompute now."

**331 of 415 root-years PASS. 84 flagged UNTRUSTED.** No recomputation performed.

Of the 84 untrusted, **82 are 0.0% non-null because the columns do not exist in the file
schema** (ESC-2) -- SPY/SPX/IWM/DIA/EEM/FXI/GLD/SLV/SMH/TLT/VIX/XLE/XLF/XLI/XLK/XLV 2012-2016
and AMD 2012-2014. **Only 2 are genuinely partially-null: AMZN 2015 (88.69%) and GOOGL 2015
(82.58%).** This distinction matters -- the gate's remedy ("routed to later recomputation") is
feasible for the 2 but not for the 82, where there is no input to recompute from at all.

Sanity screens where greeks exist:

| Screen | Result |
|---|---|
| `|delta| <= 1` | **100%** |
| delta sign correct per `right` | **100%** -- `delta_sign_wrong_n` is **0** across all 24bn rows |
| `IV in (0.01, 5.0)` | 97.187% |

**The `IV == 0.5` sentinel -- reported separately, and it moves no gate outcome.** The gate binds
on non-null, reported as such above.

- `implied_vol == 0.5` exactly: **1.05%** of all rows where greeks exist.
- On **invalid-quote rows specifically**, the sentinel share runs **~47-85%** by root-year --
  confirming it is a solver fallback emitted when there is no quote to solve against, not a
  measured IV.
- Worst root-years by sentinel share: PLTR 2022 (4.00%), PLTR 2023 (3.79%), AMD 2017 (3.62%),
  IBIT 2026 (3.25%), AMD 2023 (3.22%).

**"Non-null" materially overstates greek coverage.** The registered plausibility screen is
load-bearing; the non-null rate alone must not be used to establish trust.

Full table: `output/vbattery/sweep/v5_gate_by_root_year.csv`.

### V6 -- EOD-join leakage -- COMPLETE -- **RULING'S RATIONALE CORRECTED**

> **Registered gate:** "HARD GATE. (a) read the join in `combine_options_data.py`; (b) verify `open_interest_eod` is constant across all minutes of a session (if it varies intraday the join is broken); (c) establish whether the value on session *t* is OI as-of *t*'s close or *t-1*'s close. Same-day join => **all OI/gamma features must lag >=1 session**, enforced in the primitive layer, not per-strategy. If the answer cannot be established from code + data => **all OI/gamma work BLOCKED**."

**(a) Same-day join -- CONFIRMED at code level (independently re-read).**
`combine_options_data.py` derives `_date = pl.col("timestamp").str.slice(0, 10)` (the row's
own session) and left-joins the EOD frame on `["_date","expiration","strike","right"]`.

**(b) OI constant intraday -- CONFIRMED.** 0 of 144,250 SPY contract-sessions show intraday
variation in `open_interest_eod`.

**(c) Which session's OI -- the prior conclusion is NOT supported.**

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
a leak correction. Relaxing it is a decision for the principal, not for this phase -- and it
should not be relaxed for `gamma_eod` on any current evidence.

### V7 -- Corporate actions -- COMPLETE -- **FAIL / ESCALATION (ESC-1)**

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
Not built -- work order Section 4 does not authorize it.

### V8 -- Root continuity -- COMPLETE -- PARTIAL -- **ESCALATIONS (ESC-3, ESC-4)**

> **Registered gate:** "FB (2017-2021) -> META splice; SPX vs SPXW composition and AM/PM settlement; VIX expiry convention. Report + registered splice/filter rules per root."

**1. FB -> META -- the splice premise is wrong (ESC-3).**

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
issuer** -- ~25x apart in price from contemporaneous FB, with a disjoint strike grid. Meta
Platforms data begins **2022-06-09**.

Consequence: FB stops 2021-10-29 and Meta Platforms starts 2022-06-09, leaving a **~7.4-month
hole**. A naive FB+META concatenation would both splice in a penny stock and silently bridge
that gap.

**2. SPX -- no SPXW (ESC-4).** 12 months sampled (2025-03..2026-02), 31 distinct expirations:
Friday 28, Thursday 3 (holiday shifts). `is_third_friday` 28/31. Decisive settlement
discriminator: **`exp_day_rows == 0` for all 31 expirations** -- every expiration's last traded
session is the session *before* the expiration date, the textbook AM-settlement signature.
**AM-like 31/31 = 100%. PM-like 0.** No weeklies, no Monday/Wednesday, no same-day expirations.

Independently spot-checked on 2024 Q1: 2024-01-19 expiry last traded 2024-01-18; 2024-02-16
last traded 2024-02-15; 2024-03-15 last traded 2024-03-14. Corroborated.

So "SPX vs SPXW composition" resolves to: **not commingled, because only one population
exists**. The AM/PM distinguishability question is moot for the same reason.

**3. VIX.** 24 months sampled, 32 expirations: Wednesday 29, Tuesday 3 (2024-06-18, 2025-03-18,
2026-05-19). Consistent with the Wednesday convention plus the 30-days-before-SPX-expiry rule
and holiday shifts. Convention, not anomaly -- the proposed rule explicitly does not drop them.

Five proposed rules (FB-01, META-01 quarantine, META-02 splice, SPX-01 scope, VIX-01 filter)
are recorded in `output/vbattery/v08/v08_proposed_rules.csv`, all marked
`PROPOSED -- not applied`. **None applied anywhere.**

**Coverage limitation:** SPX sampled over the last 12 of 164 partitions, VIX the last 24 of
165. Findings are firm for those windows; earlier eras -- particularly the pre-2017 16-column
era -- are **unmeasured**.

### V9 -- Partition integrity -- COMPLETE -- **113 root-months flagged** -- ESC-5

> **Registered gate:** "Sessions present vs expected per root-month 2012-2026; quantify the 2026-03 -> present gap; half-day handling. Any root-month with **<90%** of expected sessions flagged. Refresh executed only if the subscription supports it."

Refresh is **NOT executable** -- the ThetaData subscription is cancelled. No download attempted.
The gap stands as a live-edge caveat.

Expected sessions come from the repo's existing calendar -- **reused, not forked**:
`src/backtesting/utils/market_calendar.py` (`MarketCalendar('NYSE')`), including its
`market_close` column for early-close detection.

**113 of 4,510 root-months flagged (<90% of expected sessions).** Three groups:

1. **The live edge -- 62 flagged root-months.** Every root's 2026-01 and 2026-02 are partial.
   The store's **last session is 2026-02-04** for all 30 live roots.
2. **Genuine historical holes -- ~40 root-months.** Largest is **SPY 2017-01 .. 2017-08**, a
   severe ramp: 2017-01 **2/20 sessions (10%)**, 2017-02 3/19, 2017-03 6/23, 2017-04 7/19,
   2017-05 10/22, 2017-06 16/22, 2017-07 16/20, 2017-08 17/23. Also MSTR 2017-2020 (17 months),
   META 2021-07 (16/21) and 2022-06 (15/21), COIN 2021-04 (9/21), GLD 2015-09 (13/21),
   IBIT 2024-11 (8/20), EEM/FXI/GLD/IWM 2022-01 (11/20).
3. **2025-12 (19/22)** across ~20 roots -- a uniform 3-session shortfall.

Two systematic patterns follow.

**(i) The last COMPLETE month is 2025-12; the last SESSION is 2026-02-04.** Verified directly:

| Root | 2025-12 | 2026-01 | 2026-02 |
|---|---|---|---|
| SPY | 2025-12-01 -> **2025-12-26** (19/22) | **1 session** (01-02 only) | **3 sessions** (02-02..02-04) |
| QQQ | same | 1 session | 3 sessions |
| IWM | same | 1 session | 3 sessions |

Every doc in the chain describes the window as "2012-06 -> 2026-02" and the work order refers to
"the 2026-03 -> present gap". Both are wrong: the **last complete month is 2025-12**, 2026-01
and 2026-02 are 1- and 3-session stubs, and **the gap actually begins 2026-02-05** -- about a
month earlier than stated.

**(ii) SPY is heavily truncated across 2017-H1 and 2018.** Verified directly: SPY 2017-01
contains **2 sessions** (Jan 3-4, 1.25M rows); 2017-03 stops Mar 8; 2017-06 stops Jun 22.
QQQ and IWM are complete in the same months, so this is SPY-specific, not a store-wide outage.

**Combined with ESC-2, the honest greek-bearing, non-truncated window for SPY is
approximately 2017-09 -> 2025-12 (~8.3 years), not the 13.7 years the chain assumes.**
SPY is the underlying of the majority of slate candidates. This is a data property; its
consequences for statistical power are not assessed here.

**(iii) Half-day handling -- a significant finding, independently verified.** Half-days are NOT
missing sessions, and they are **not truncated either: the store pads them out to a full 391
bars (09:30-16:00) with stale carried-forward quotes.** All **864 of 864** half-day sessions in
the store have `last_bar == 16:00`; **zero** end at their calendar 13:00 close.

Verified directly on SPY 2024-11-29 (NYSE 13:00 early close): 391 distinct minutes, last bar
16:00. Volume collapses to zero after ~13:05, but `quote_valid` stays frozen at **3,397 of
3,548 contracts all the way through 16:00**:

| minute | rows | volume | quote_valid |
|---|---|---|---|
| 12:59 | 3,548 | 40,053 | 3,407 |
| 13:05 | 3,548 | 12,924 | 3,399 |
| 14:00 | 3,548 | **0** | 3,397 |
| **15:45** | 3,548 | **0** | 3,397 |
| 16:00 | 3,548 | **0** | 3,397 |

**~3 hours of non-tradeable stale minutes per half-day are indistinguishable from live minutes
on their own.** Note the registered 15:45 snapshot falls 2h45m AFTER the real close on these
sessions -- which is exactly why the canonical layer clamps the cutoff to
`min(15:45, real session close)`.

**(iv) Off-calendar sessions -- the opposite direction.** 57 root-months have **more** sessions
than the NYSE calendar: **SPX and VIX carry bars on 64 NYSE holidays from 2022 onward**
(Memorial Day, Juneteenth, July 4, Labor Day, Thanksgiving, Christmas, New Year's Day). Row
counts are ~20-50k versus ~1M on a normal session, so these are artifact rows, not real
sessions. Any calendar reconciliation must handle the over-count direction, not just
under-count.

### V10 -- Provenance mix -- COMPLETE -- **UNIFORMLY THETADATA** (gate consequence not triggered)

> **Registered gate:** "`DATA_INVENTORY.md` lists ThetaData **+ IBKR chains**; the download scripts are ThetaData. Which rows, if any, are IBKR-sourced; are populations distinguishable. If mixed and indistinguishable, V1/V3 statistics must be computed conservatively over the whole."

Three independent lines of evidence, all negative for IBKR:

- **Schema:** union of distinct column names across all 4,510 partitions is exactly 21 names.
  **No `source`/`vendor`/`feed`/`origin`/`venue`/`exchange` column exists anywhere.** The only
  candidate token, `symbol`, is just the root ticker.
- **Writers:** 14 options modules inspected -- **0 mention IBKR/ib_async/ib_insync/TWS**;
  8 mention ThetaData. All 5 partition-writing scripts are ThetaData-sourced.
- **Logs:** all 34 files in `H:\Stock_Data\options\_logs` (217 MB) scanned line-by-line:
  **0 IBKR lines, 173 ThetaData lines.**

`DATA_INVENTORY.md:248`'s "ThetaData + IBKR options chains" is **not supported by any artifact
in the store**. The population is not mixed, so V1/V3 need not be computed conservatively on
provenance grounds.

**Stated honestly:** this is an absence-of-evidence argument. Nothing in the data *marks*
provenance, so IBKR rows written by a since-deleted script leaving no log would be
undetectable. What is assertable: every surviving writer, log, and schema is ThetaData.

### V11 -- Spread census -- COMPLETE (deliverable, no gate)

> **Registered gate:** "Quoted-width distribution by root x moneyness x DTE bucket x year x volatility state, including event windows and 0DTE. **No gate -- this is a deliverable**: a parquet of width statistics that later parameterizes the cost model."

**Delivered.**

| | |
|---|---|
| Primary | `H:\Stock_Data\options\derived\spread_census\spread_census.parquet` |
| Repo copy | `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\output\vbattery\sweep\spread_census.parquet` |
| Rows | **181,761** cells, 31 roots, 2012-2026 |
| Coverage | 23,232,968,771 valid quotes, carrying 845,110,236 excluded invalid quotes per cell |

Path rationale: derived datasets under `get_local_storage_dir()` sit beside their source family
(`options/gex_daily`, `options/chains`, `options/options_chain_eod` are all siblings of
`options_combined`; `alt_data/` is reserved for non-market inputs). Path is built from
`get_local_storage_dir()`, never hardcoded.

Cell key: `root x year x moneyness_bucket x dte_bucket x abs_delta_bucket x vol_state_proxy`,
plus `n_valid_quotes`, `n_excluded_invalid_quote`, `n_sampled`, and
`{spread_abs, spread_rel, mid} x {mean, p10, p25, p50, p75, p90, p95, p99}`.
DTE buckets `{0, 1-7, 8-30, 31-60, 61-90, 91-180, 181+}`; moneyness `log(strike/underlying_px)`
at `{-0.10,-0.05,-0.02,0.02,0.05,0.10}`; `|delta|` at `{0.05,0.15,0.35,0.65}`. Valid quotes
only, per the V3 rule.

Representative cells -- **SPY, ATM, 31-60 DTE, |delta| 0.35-0.65**:

| year | n valid | spread_abs p50 | spread_rel p50 | spread_rel p90 | mid p50 |
|---|---|---|---|---|---|
| 2017 | 2,536,865 | 0.0300 | 1.00% | 1.96% | 2.87 |
| 2018 | 7,287,360 | 0.0467 | 1.09% | 1.87% | 4.43 |
| 2019 | 8,606,948 | 0.0333 | 0.72% | 1.44% | 5.00 |
| 2020 | 10,083,476 | 0.0467 | 0.62% | 1.30% | 7.89 |
| 2021 | 11,980,166 | 0.0500 | 0.60% | 1.66% | 8.71 |
| 2022 | 12,823,442 | 0.0600 | 0.52% | 2.02% | 12.06 |
| 2023 | 8,415,918 | 0.0250 | 0.33% | 1.15% | 8.93 |
| 2024 | 10,939,183 | 0.0375 | 0.43% | 1.88% | 9.75 |
| 2025 | 12,483,686 | 0.0433 | 0.39% | 1.10% | 12.35 |
| 2026 | 264,550 | 0.0550 | 0.44% | 1.36% | 13.32 |

SPY 2024 0DTE ATM by vol state: absolute width pinned at the **1-cent tick** in every state;
relative width 1.06% (high vol) to 1.27% (low vol), because the ATM 0DTE mid rises with vol
(1.225 vs 0.855) while the tick floor does not move.

**Two caveats, stated plainly:**

1. **Percentiles are reservoir-sampled estimates, not exact.** Each cell holds an Algorithm-R
   reservoir capped at 1,500 observations (seeded, reproducible). `n_valid_quotes` and the
   excluded counts are **exact**, and `n_sampled` is stored, so any cell's precision is
   auditable.
2. **`vol_state_proxy` is a proxy and is labelled as such -- it is NOT `regime_state_daily`.**
   The materialized regime state did not exist when the sweep ran. The proxy is: trailing
   20-session realized vol of the per-session last `underlying_px`, expanding-percentile-ranked
   against that root's prior history, **strictly backward-looking**, bucketed low/mid/high at
   the 33rd/67th percentiles. `vol_unknown` covers 13.4% of valid quotes (rows before 60 prior
   observations accumulate, including chunk-boundary restarts on the 5 chunked roots). The
   other three states split 19.6% / 26.6% / 40.5%. **If cost parameterization needs the real
   regime, the census must be rebuilt against `regime_state_daily`.**

### V12 -- Legacy store -- COMPLETE -- **MOOT / CLOSED**

> **Registered gate:** "`options/options_1min/` (17 roots, 2024-11 -> 2025-12) vs `options_combined` overlap equivalence. Report equivalence. **Delete nothing.**"

Direct listing of `H:\Stock_Data\options`:

| Entry | Files | Bytes |
|---|---|---|
| `_logs` | 34 | 217,329,575 |
| `chains` | **0** | **0** |
| `gex_daily` | **0** | **0** |
| `options_combined` | 4,510 | 249,992,314,784 |

`options_1min/` checked at three plausible locations -- all **absent**. Overlap-equivalence is
therefore **unrunnable, not failed**, and was correctly skipped rather than faked.
`combine_options_data.py` documents joining `options_1min/` with `options_eod/`, so the legacy
store did once exist; it is gone now.

**Nothing deleted. Nothing modified.**

### V13 -- PIT universe coverage -- COMPLETE -- **FAIL / NOT MEASURABLE**

> **Registered gate:** "For every point-in-time-defined universe (rank/weight-based membership), fraction of sessions where **all** required members exist on disk; report the earliest date from which coverage is complete. **<95% of sessions fully covered => that universe is NOT usable as registered**; report the first-full-coverage date and the implied usable window."

**Primary finding: no point-in-time membership source exists in this repo.**
`config/universes/` holds only `sp500-2025.csv`, `russell1000-2025.csv`,
`russell2000-2025.csv` -- **today snapshots**, survivorship-biased by construction, with no
weights and no `as_of` column. Using one as membership at a 2013 date is exactly the bias
spec v2 rule 7 forbids.

| Universe | PIT source? | Coverage | Verdict |
|---|---|---|---|
| `U_INDEX` {SPY,QQQ,IWM} | n/a (fixed) | **94.42%** | **FAIL** (<95%) |
| `U_TOP6_SPY` | **NO** | -- | **NOT MEASURABLE AS REGISTERED** |
| `U_MEGA10` | **NO** | -- | **NOT MEASURABLE AS REGISTERED** |
| `U_MEGA20` | **NO** | -- | **NOT MEASURABLE AS REGISTERED** |
| `U_TIER1_100` | **NO** | -- | **NOT MEASURABLE AS REGISTERED** |

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
| D2 | `right in {C,P}` (spec v2 Section 1.2) | Values are **`PUT`/`CALL`** | SPY 2024-01 |
| D3 | V5 "IV 100% non-null" implies coverage | Non-null but **`IV == 0.5` exactly** on 1.05% of rows store-wide, and **47-85% of invalid-quote rows** by root-year -- a solver sentinel. Plausible-IV rate 97.19% | V5 shard |
| **D13a** | **The preliminary V-battery figures in the execution plan (zero-bid 11.7%, vol0-with-quote 77.4%, IV==0.5 1.29%/31%) -- the evidence that "all four gates pass by wide margins"** | **They were a head-of-file sample spanning the first ~6 MINUTES of 2024-01-02, not a month.** Full-month: zero-bid **3.38%**, vol0-with-quote **88.99%**, IV==0.5 **1.59% / 44.7%**. Row groups 0-4 alone reproduce the old numbers (zb 8.5%, vol0&qv 78.9%) and span only 09:30-09:36 -- the widest-spread minutes of the day. **My own initial 2.46M-row probe was biased the same way and its numbers are likewise superseded.** Two independent pipelines (pandas/numpy and polars) agree to the row on the corrected figures | Sweep D1 |
| D4 | V4 is a binary 20-vs-21-column question | **3 column counts, 9 dtype variants.** Both the brief and `DATA_INVENTORY.md` are right about different partitions | V4 sweep |
| D5 | (nobody asserted) | **16-column variant, 323 partitions, no greeks and no `underlying_px`** -- undocumented everywhere | V4 sweep |
| D6 | Store spans 2012-06 -> **2026-02** | Last **complete** month is **2025-12** (edge 2025-12-26). 2026-01 = 1 session, 2026-02 = 3 sessions | Footer inventory |
| D7 | (nobody asserted) | **SPY truncated across 2017-H1/2018**; 2017-01 has 2 sessions. QQQ/IWM complete in the same months | Footer inventory |
| D8 | V6: "session t carries t's END-OF-DAY OI => hard lookahead leak" | **Contradicted.** Modern-era evidence says the stamp is **start-of-day** (= t-1 close). The lag rule stands on conservatism, not on this rationale | V6 attribution |
| D9 | V7 unknown | **7/7 AS-REPORTED**, with non-standard deliverables present | V7 shard |
| D10 | "FB -> META splice" closes the series | `root=META` 2021-07..2022-01 is **Meta Materials**; 7.4-month Meta Platforms hole remains | V8 + independent check |
| D11 | "SPX vs SPXW composition ... AM/PM settlement" presumes both present | **No SPXW at all**; 31/31 AM-settled third-Friday | V8 + independent check |
| D12 | `DATA_INVENTORY.md`: source is "ThetaData + IBKR chains" | No IBKR trace in schema, 14 writers, or 34 logs | V10 |
| D13 | Zero-bid frequency is **11.7%** (execution plan) | **3.38%** full-month SPY 2024-01; **3.51%** store-wide. See D13a for the cause | V3 shard |
| D17 | "Bars run 09:30 -> 16:00 inclusive" | True, with two material qualifications: **half-days are padded to 16:00 with stale quotes** (864/864 sessions), and **the 09:30 bar has `bid==ask==0` universally** (70,239/70,239 rows on SPY 2024-01) so the opening minute carries no quote at all | Independent verification |
| D18 | Sessions present == NYSE trading days | **SPX and VIX carry bars on 64 NYSE holidays from 2022 onward** (~20-50k rows vs ~1M normal) | V9 sweep |
| D19 | Work order: "the 2026-03 -> present gap" | The gap begins **2026-02-05**; last session is 2026-02-04 | V9 sweep |
| D20 | (repo bug, not a doc error) | **`src/utils/run_status.py` had a tmp-file race** that killed 2 of 8 parallel sweep jobs. Found, reproduced with a failing test, and **fixed** -- see Section 4 | `tests/utils/test_run_status.py` |
| D14 | VIX spot needed materializing from 2012-06 | Available from **1990-01-02**; materialized 9,204 rows to 2026-07-24 | VIX sidecar |
| D15 | (nobody asserted) | **Vendor pads early-close sessions to 16:00 with stale quotes** -- a real post-close mark hazard; canonical cutoff clamps to the true close | Canonical layer |
| D16 | `DATA_INVENTORY.md`: "24.1B rows, 4,510 files, ~250 GB" | **CONFIRMED**: 24,078,079,007 rows, 4,510 files, 249,992,314,784 bytes | V4 sweep |

---

## 4. Repo bug found and fixed -- `RunStatus` tmp-file race

`src/utils/run_status.py` wrote every status update to a **single fixed** scratch path,
`self.path.with_suffix(".tmp")`. That path is shared by the caller's thread (an explicit
`heartbeat()`) and the background heartbeat thread, so the two race two ways: both write the
same file at once (`PermissionError`), or the first `replace()` consumes it and the second finds
nothing (`FileNotFoundError`, WinError 2). The existing 5-attempt retry loop **cannot** recover
from the second case -- it retries the rename, but the source file is already gone -- so the
exception propagates and aborts the run.

**This killed 2 of 8 parallel jobs in the first sweep wave.** It is exactly the class of failure
`RunStatus` exists to prevent, so it is worth fixing rather than working around.

Reproduced with a failing regression test (4 threads hammering `heartbeat()` against the live
heartbeat thread) which surfaced both error modes, then fixed by making the tmp path unique per
writer (`.tmp.<pid>.<thread_id>`) and unlinking the scratch file on failure so none leak. All 4
`RunStatus` tests pass.

- `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\src\utils\run_status.py`
- `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\tests\utils\test_run_status.py`

This is an apparatus correction, not a specification change.

---

## 5. Limitations

- **V8 SPX/VIX are windowed** (last 12 and 24 partitions of ~165). Pre-2017 eras unmeasured.
- **`bid_close`/`ask_close` = NBBO at minute end is INFERRED, not established.** No vendor spec
  is available and the subscription is cancelled, so there is no authority to check against.
- **V11 percentiles are reservoir-sampled**, not exact; `vol_state_proxy` is not the regime state.
- **Why greeks are absent for 82 root-years was not determined** -- only that the columns are
  missing from the file schema. That is a provenance question, not a measurement available from
  the store.
- The 09:30 zero-quote effect and half-day padding were verified on samples (SPY 2024-01,
  SPY 2024-11-29) plus the store-wide `last_bar` census (864/864); a store-wide per-minute
  quantification of the 09:30 effect was not run.
- **V7 covers 3 sessions each side** of each event -- sufficient to discriminate adjusted vs
  as-reported, not to enumerate every non-standard deliverable.
- **V10 cannot prove a negative.**
- **V6 `gamma_eod` date attribution is unresolved.**
- **float32 precision loss in 274 partitions** is stated as fact; the resulting error magnitude
  was not quantified.
- The SPY-2013 V6 dissent could not be conclusively attributed to window truncation.

---

## 6. Artifacts

**Code**
- `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\src\data\options\canonical.py`
- `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\tests\data\test_options\test_canonical.py`
- `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\scripts\data\build_options_chain_eod.py`
- `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\scripts\data\build_vix_spot.py`
- `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\scripts\data\build_regime_state_daily.py`
- `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\tests\data\test_vix_spot_and_regime.py`
- `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\scripts\data\vbattery\` -- V4, V6, V7, V8, V10,
  V12, V13, schema/coverage inventory, and the V1/V2/V3/V5/V9/V11 sweep

**Measurement shards** -- `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\output\vbattery\`
(`v04/`, `v06/`, `v07/`, `v08/`, `v10/`, `v12/`, `v13/`, `schema_coverage/`, `sweep/shards/`)

**Run-status** -- `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\output\run_status\`

*No strategy backtest has been run. No P&L has been observed.*

---

# CORRECTION ADDENDUM -- 2026-07-28 (main loop)

Two corrections to this report, both established by measurement after it was written.

## C1. V5 missed the `ALL_NAN` partition class -- 580 partitions

This report enumerated the **323 `NO_COLUMN`** partitions (ESC-2) but did not identify a second,
larger failure class: partitions where the greek columns are **present but entirely NaN**.

Full-store census (4,510 partitions, `implied_vol` tested with `np.isnan` on values):

| Status | Partitions | Share |
|---|---|---|
| **OK** (usable) | **3,600** | **79.8%** |
| **ALL_NAN** | **580** | **12.9%** |
| NO_COLUMN | 323 | 7.2% |
| PARTIAL_NAN | 7 | 0.2% |

Per-partition detail: `20260728_options_greek_coverage_census.csv`.

**Root cause of the miss -- the NaN-vs-NULL trap.** The greek columns are **NaN-valued float64,
not SQL NULL**. Arrow/parquet therefore report **`null_count = 0` for a column that is 100%
NaN**. Any V5-style "non-null rate >= 90%" gate computed on `null_count` **passes a completely
empty column**. The same trap caught two independent analyses in the main loop before it was
identified.

**Registered rule going forward:** greek usability is tested with `np.isnan()` on the values.
Column presence and `null_count` are both invalid tests.

**Consequence for V5's verdict:** the registered V5 gate ("**>= 90%** non-null per root-year =
PASS") must be re-evaluated on usable-value rate. Root-years that are ALL_NAN previously scored
as passing and are in fact **0% usable**.

## C2. ESC-2's implied cause was wrong -- this is a vendor boundary, not a bad download

The main loop initially concluded the missing greeks were a recoverable download artifact
(rate-limiting on the separate greeks endpoint, whose error is discarded at the call site and
whose absence `_merge_data` tolerates). **That conclusion was wrong**, and the correction
matters because it was about to justify paying to re-download.

Three measurements kill the transient hypothesis:

1. **Same downloader, same days, different outcome by root.** QQQ/AAPL/MSFT 2012-2016 were
   written across the *same* mtime range (2026-01-29 .. 2026-02-22) as SPY/IWM/SPX 2012-2016
   and got usable greeks.
2. **184 partitions written during the suspected "bad week" (2026-02-14..20) have OK greeks.**
   The pipeline was working that week.
3. Write-date clustering explains only *which failure mode was recorded* (`NO_COLUMN` on one
   code path, `ALL_NAN` on another), not the root cause.

**Finding: ThetaData does not serve IV/greeks for the ETF/index roots before 2017-01.** Every
such root -- SPY, IWM, SPX, DIA, GLD, SLV, TLT, EEM, FXI, SMH, XLE, XLF, XLI, XLK, XLV, VIX --
has first-fully-usable year **2017**. **QQQ is the sole exception**, fully usable 2012-2026.
Single names run further back (GOOGL 2014+, AMD 2015+, the rest full).

**A re-download will not recover it. Do not spend money on a re-pull for this.**

### Consequences

- The honest window for SPY-based candidates is **~2017-2025 (~8.3 y)**, permanently. sigma_SR
  ~ 0.35 rather than 0.27, roughly doubling the DSR hurdle relative to the doc chain's
  registered arithmetic.
- **ORATS (~$399, 2007+) is now the only route to pre-2017 SPY history**, and bundles the 2008
  GFC window and dividend-aware smoothed surfaces. It moves from optional extension to the only
  available fix for the slate's biggest structural weakness.
- **QQQ is the only index root with a full 13.7-year usable series.** Re-anchoring from SPY to
  QQQ now has a genuine data-availability justification, but it is a spec change to a different
  underlying and requires a **logged amendment**, not a silent swap.
- The download bug found while investigating is still real and worth fixing regardless: the
  greeks error is discarded (`greeks_df, _ = ...`), `_merge_data` treats greeks as optional, and
  there is **no post-download schema or usability validation anywhere**. A future re-pull that
  hits rate limits would now produce null-padded columns that pass a column-count check.

*No strategy backtest has been run. No P&L has been observed.*
