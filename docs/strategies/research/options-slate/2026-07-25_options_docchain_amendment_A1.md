# Options Doc-Chain Amendment A1 — Data-Inventory Correction (2026-07-25)

**Date:** 2026-07-25
**Trigger:** CC direct-disk inventory of `H:\Stock_Data\options\` (provided by Shuyang, 2026-07-25), superseding the data-availability assumptions in the doc chain.
**Amends:** `2026-07-24_options_strategy_slate_v1.md` · `2026-07-24_options_slate_feasibility_screen_v1.md` · `2026-07-24_options_slate_cc_handoff_spec_v1.md`
**Instrument:** logged amendment per the chain's own registered rules ("deviations require a logged amendment before the affected test runs"; superseded versions are retired with their own ledger rows, **never overwritten in place**). The three v1 documents stand unmodified in the record; this document supersedes the specific sections enumerated in §3.
**Status:** still blind. No strategy backtest from this slate has run. Every change below is driven by **data-availability facts**, not by any performance observation — amending on availability is legitimate prior-free information and is not results-conditioning.

---

## TL;DR

- **The feasibility screen's central premise was false at write time.** It assumed "options-side data is the open decision variable; nothing above EOD chains is assumed." Disk truth: **~233 GB of 1-minute intraday options data, 31 underlyings, 2012-06 → 2026-02, ThetaData-sourced, with per-minute OHLCV + bid/ask close + IV + delta/theta/vega, plus EOD-joined gamma and open interest.** The stale premise came from working notes (RAMP-CSP "blocked pending options data") and an old inventory note conflating the small legacy `options_1min/` set (17 symbols, 13 months) with the real `options_combined/` set. This is the project's own "code/disk is ground truth" lesson operating on our doc chain.
- **Consequences run in the favorable direction, and the conditional structure contained the damage.** Because every candidate was tagged with data-class requirements (D1–D6) rather than assumed-available sources, verdicts port cleanly: the D-classes now resolve against disk instead of vendors. **No candidate definition, parameter, prior tier, cost-viability verdict, or decay conclusion changes.** What changes: the purchase ladder, the data-screen component of ~8 conditions, the leakage/mark conventions, the DSR window arithmetic, and CC Phase 1 (which becomes *canonicalize and verify*, ~$0, instead of *purchase and ingest*).
- **ORATS is no longer the gate — but it is still probably worth $399**, for a different reason: the on-disk window starts 2012-06 and therefore **contains no 2008-class crisis**. For a slate whose center of mass is short volatility, validating on a window that right-censors the worst modern short-vol regime is a permanent caveat; ORATS 2007+ is the cheap fix, plus dividend-aware smoothed surfaces and cross-vendor validation. Decision is Shuyang's; nothing blocks on it.
- **Two decisions are required before the affected tests run** (§7): (1) OPT-021 exit version — the T+1-open exit (v1.0) is now data-feasible, so keep v1.1 (near-close) or restore v1.0, one only; (2) universe treatment for breadth-dependent candidates — the disk covers ~13 liquid single names, not `U_MEGA20`/`U_TIER1_100`, so either narrow the registered universes (logged amendment) or top-up download via the ThetaData subscription.
- **New finding from the repo docs:** the shelved OpEx pinning strategy (2025-12-30) was blocked on exactly the gamma/OI fields that now exist as EOD-joined columns — and it **already ran a backtest** (18 trades, 11.1% win rate, −$415, on estimated gamma/OI). That is a prior tested trial on overlapping data: it enters the lifetime ledger, and OPT-043 must be reconciled against that prior art (reusable GEX machinery, 152 passing tests) rather than specified as greenfield.

---

## 1. How corrections happen (the procedural answer)

1. **Conversation-chain docs (the three v1 documents):** corrected by *this amendment*, not by editing. Each superseded claim is enumerated in §3 with its replacement. The ledger gains amendment rows (`amendment_log` entries referencing A1) for every candidate whose conditions changed. Rationale-before-test discipline is preserved: nothing amended here has been tested.
2. **In-repo documentation:** corrected by CC in place (repo docs are living documents, not pre-registrations) — but from **disk truth, not from this amendment**. §4 lists the specific stale claims found. CC's correction commits should cite the inventory evidence (paths, row counts, schema dumps), consistent with the Phase-0 reporting standard.
3. **Memory/working notes:** the "17 roots / 13 months," "EOD only," and "options data is a constraint / RAMP-CSP blocked on data" notes are retired. The long-standing reconciliation failure (24.1B rows ≠ 17 roots × 13 months) is **resolved, not merely corrected**: the two numbers described *different datasets* — the legacy `options_1min/` store (17 symbols, Nov 2024–Dec 2025, per the OpEx status doc) and the real `options_combined/` store (31 roots, 2012–2026, per `DATA_INVENTORY.md`). The notes collapsed them into one.

---

## 2. Canonical data record (as of 2026-07-25, pending §5 verification)

**Primary store:** `<local_storage_dir>/options/options_combined/`, Hive-partitioned `root={SYMBOL}/year={YYYY}/month={MM}/data.parquet`. ~233 GB on disk (repo doc: 24.1B rows, 4,510 files). Read by `src/strategies/options/data_loader.py` (`OptionsDataLoader`). Provenance: ThetaData Standard subscription via Theta Terminal (`scripts/data/download_options*.py`), EOD fields joined by `combine_options_data.py`; `DATA_INVENTORY.md` additionally lists "IBKR options chains" as a source — **provenance mix to verify (V10)**.

**Granularity:** 1-minute bars per contract. One SPY-month ≈ 305 MB / ~27.4M rows — consistent with near-full-chain × every-minute coverage (~3,300 contracts × 390 min × 21 days), i.e., rows appear to exist independent of trades. **This is the load-bearing property (V1–V3): if true, these are quote-bearing minute records, not trade-print bars, and the chain's trade-bar prohibition is satisfied by the disk set itself.**

**Schema (per contract per minute):** `timestamp, expiration, strike, right(C/P), open, high, low, close, volume, trade_count, vwap, bid_close, ask_close, implied_vol, delta, theta, vega, underlying_px, gamma_eod, open_interest_eod` (CC: 20 cols; `DATA_INVENTORY.md`: 21 with leading `symbol` — reconcile, likely partition-key materialization; V4). Minute-level greeks are **first-order only**; gamma and open interest are **EOD-joined daily fields** with known join-leakage risk (V6) and a known history: the 1-minute feed's native gamma/OI were 0% populated (OpEx doc), which is what shelved that strategy.

**Coverage:**

| Group | Roots | Span |
|---|---|---|
| Index/vol | SPX, VIX, SPY, QQQ, IWM, DIA | 2012–2026 |
| Sector/theme ETFs | XLE, XLF, XLI, XLK, XLV, SMH | 2012–2026 |
| Commodity/bond/intl ETFs | GLD, SLV, TLT, EEM, FXI | 2012–2026 |
| Mega-cap singles | AAPL, AMD, AMZN, AVGO, MSFT, MSTR, NVDA, TSLA | 2012–2026 |
| Later-listed | GOOGL (2014), PLTR (2020), COIN (2021), META (2021), IBIT (2024) | partial |
| Delisted | FB (2017–2021, → META; splice required, V8) | — |

Largest stores: SPY 32 GB, QQQ 24 GB, AMZN 20 GB, SPX 18 GB, TSLA 15 GB.

**Known gaps and hazards (feeding §5):** window starts 2012-06 (**no GFC**); ends 2026-02 (**~5-month refresh gap** to present); ThetaData-computed IV/greeks carry the already-registered caveats (Black-Scholes, dividends ignored by default, deep-ITM/OTM IV error) — those caveats were written against a vendor we might buy from and now apply to data we own; `bid_close/ask_close` population across the full history is unverified (the old note "EOD quote fields may not exist pre-Dec-2023" concerned a different endpoint but mandates checking the minute-level analogues); SPX root composition (SPX vs SPXW, AM/PM settlement) unverified; two dead-path layouts exist (`chains/`, `gex_daily/` — empty, still pointed at by `src/data/options/options_store.py`).

**Secondary store (legacy):** `options/options_1min/` — 17 symbols, Nov 2024–Dec 2025, the OpEx-era download. Disposition to decide: superseded by `options_combined` (verify overlap equivalence, then archive/delete) — V12.
---

## 3. Errata — conversation chain (superseded → replacement)

### 3.1 Slate v1 (`2026-07-24_options_strategy_slate_v1.md`)

| Location | Superseded claim | Replacement |
|---|---|---|
| Scope intake, assumption 2 | "Options-side data is the open decision variable… nothing above EOD chains is assumed" | Options-side data exists on disk per §2. The D1–D6 requirement tags per candidate are **unchanged** (they are requirements, not availability claims) and now resolve against disk |
| Data-class census (closing section) | "D3/D4-dependent (intraday options data required — the open vendor question): 6" | The vendor question is closed: D3 is on disk; D4 (NBBO quotes) is on disk **conditional on V1–V3** confirming quote-bearing rows for illiquid strikes |
| — | Candidate definitions, fixed parameters, families, priors, ablation ladders, +50 ledger delta | **Unchanged. The slate is untouched.** |

### 3.2 Feasibility screen v1 (`2026-07-24_options_slate_feasibility_screen_v1.md`)

| Location | Superseded claim | Replacement |
|---|---|---|
| TL;DR / §2 purchase ladder | "ORATS near-EOD (~$399) unlocks 21 GO candidates… ThetaData Value/Databento unlocks the intraday handful" | Ladder v2 (§6). The 21 GO candidates were already unlocked on disk; Rung 2a/2b are **superseded** by the owned set; ORATS' role changes from *gate* to *extension/surface/breadth/cross-validation* |
| §2 rule 1 | Trade-print OHLC bars prohibited as OTM marks; "quote bars or nothing" via purchase | **Principle retained verbatim.** Application changes: the on-disk set appears to be quote-bearing minute records; V1–V3 decide whether it satisfies the rule. If OTM quote coverage fails the gate, the prohibition re-binds exactly as written |
| §2 rule 2 | Marks = ORATS ~15:46 near-close snapshot | Marks = **self-derived snapshot at a registered minute (15:45:00 ET close)** from the minute data. We control the snapshot, so option marks, underlying signals, and hedge fills can share one timestamp |
| §2 rule 3 | "T+1-open mark is structurally unreliable" at EOD rungs → event exits amended to near-close | **Partially retired.** Open-adjacent marks are now *measurable* from minute quotes. Open-auction spread realism still applies (stressed cost tier). Consequence for OPT-021/025/040 in §7 |
| §2 rule 4 | ThetaData greeks caveat, "ORATS smoothed preferred deep-OTM" | **Caveat retained and now applies to owned data.** The smoothed-surface question becomes a build-vs-buy choice: ORATS purchase, or an own fitted surface (new module M7) from on-disk quotes. Shipped deep-OTM/ITM greeks remain disfavored either way |
| History-depth table | "19.5 y (ORATS)… 6.5 y (ThetaData Value)… intraday trials cost ≈0.47 at N=5" | Superseded by §8. Primary window is **13.7 y for everything, EOD and intraday alike** — the intraday-specific short-history penalty is retired and replaced by a slate-wide missing-2008 caveat (curable via ORATS for EOD-level candidates) |
| §4 ("what the 1m underlying asset is for") | Framed 1m *underlying* as the only owned 1m asset | Extended: owned 1m **options** data additionally enables empirical spread measurement (cost model v2), intraday touch quantification, and the D3/D4 candidates — without changing the four flagship estimator candidates |
| §6.1 purchase decisions | "Buy ORATS now; it gates everything. Rung 2 deferred…" | Nothing gates. §6 of this amendment |
| Per-candidate data screens | 8 candidates' conditions referenced purchases | §7 delta table |

### 3.3 CC handoff spec v1 (`2026-07-24_options_slate_cc_handoff_spec_v1.md`)

| Location | Superseded | Replacement |
|---|---|---|
| §1.1 rung ladder | Rungs 1/2a/2b as purchases | Ladder v2 (§6): Rung 0 now contains the 233 GB set as **primary source**; ORATS optional extension; 2a/2b dead |
| §1.2 canonical schema | Single `options_chain_eod` built from vendor payload | **Dual-table design:** `options_chain_1m` (native, canonicalized from the 20/21-col layout) + `options_chain_eod` (derived view: registered snapshot minute 15:45). Column mapping: `bid_close→bid`, `ask_close→ask`, `right→right`, `expiration→expiry`, mid materialized, `snapshot_ts` carried |
| §1.3 rule 1 (15:45 truncation) | Built to neutralize the ORATS 15:46-vs-signal basis | Reframed as the **snapshot-symmetry rule**: signals, option marks, and hedge marks share the registered snapshot minute; the 14-minute lookahead trap is gone because we control the clock. The OPT-015 "4-minute hedge basis" note dissolves (hedge and mark at the same minute) |
| §1.3 (new) | — | **Intraday execution convention (registered):** decide on bar *t* close, fill at bar *t+1* quotes. Applies to 014/041/042 and any intraday exit |
| §1.3 rule 2 (OI) | "OI is next-morning data; use T−1" | Retained **and extended**: `open_interest_eod` / `gamma_eod` are *joined* fields — V6 must establish which date's value lands on row-date *t* before any OI/gamma-conditioned logic runs (this is the exact leakage flagged in the working notes, now a named gate) |
| §2 cost model | Width assumptions table from research-cited typicals | `cost_model_v2`: widths **parameterized from the empirical spread census (V11)** on own data, by root/moneyness/DTE/regime. Fill-fraction conventions (25% single-leg / 3–6% combo / stressed tier) remain assumptions until live fills. The honesty-block line "the cost model is the weakest quantitative input" is materially improved |
| §5 Wave 0 | Group B "after the ORATS pull" | Group B runs **after Phase-1 canonicalization** ($0). **New Group C = the V-battery (§5)**, which precedes everything |
| §6 candidate specs | See §7 | §7 |
| §6.1 OPT-016 note | "Touch detection at daily snapshots misses intraday touches… a legitimate later use of Rung-2 data" | The bias is now **quantifiable immediately** from on-disk minute data — as a *measurement reported alongside results*, not a spec change (the registered spec still evaluates exits daily) |
| §7.2 DSR table | 19.5 y / 6.5 y arithmetic | §8 |
| §9 Phase 1 | "Purchase and ingest ORATS (Wave-1 universe)" | **"Canonicalize and verify the owned set"** (§9): dual tables, V-battery, refresh 2026-03→present, optional parallel ORATS |

---

## 4. Errata — in-repo documentation (CC correction items, from disk truth)

| Doc | Stale claim | Correction |
|---|---|---|
| `docs/reference/DATA_INVENTORY.md` (options section) | Mostly accurate on `options_combined` (24.1B rows, ~250 GB, schema) but lacks the coverage table and lists "+ IBKR options chains" provenance without qualification | Add per-root coverage spans (§2 table), largest-store sizes, provenance clarification after V10, and the `options_1min` legacy-store disposition |
| `docs/methodology/backtesting.md` §10.5 | Options row: "`options/{chains,gex_daily,options_combined}/` — **EOD**, varies" | Granularity is **1-minute** (plus EOD-joined fields); `chains/` and `gex_daily/` are **empty dead paths** |
| `docs/architecture/infra_patterns.md` | `OptionsDataLoader` "key columns: strike, expiry, delta, bid, ask, mid_price, implied_vol, open_interest, days_to_expiry, option_type, underlying_price" | Column names do not match disk (`bid_close/ask_close`, `expiration`, `right`, `open_interest_eod`…). Determine whether the loader renames/derives (then document the mapping) or the doc is fictional (then rewrite from the loader's actual output) |
| `docs/strategies/20251230_OPEX_PINNING_STRATEGY_STATUS.md` | "SHELVED — pending proper options data"; gamma/OI 0% null; describes only `options_1min/` (17 symbols, 13 months) | The blocking data now exists (`gamma_eod`, `open_interest_eod` in `options_combined`, 31 roots, 2012+). Status doc needs a dated addendum: blocker resolved **pending V6 leakage verification**; the strategy's disposition routes through the OPT-043 reconciliation (§7) |
| `src/data/options/options_store.py` | `OptionsDataStore` points at empty `chains/`/`gex_daily/` layouts | Code-level divergence (CC already found it): deprecate/redirect the class or delete the dead layout — decide in Phase 0.9, do not leave two truths standing |

---

## 5. Verification battery (Wave-0 Group C — precedes all diagnostics and trials; zero `n_trials`)

Each item is a data-property measurement with a pre-registered gate. These decide whether the owned set actually delivers what §2 claims; several were offered by CC's inventory turn and are formalized here.

| ID | Measurement | Gate / consequence |
|---|---|---|
| **V1 — Quote population** | Fraction of contract-minutes with valid quotes (bid > 0, ask ≥ bid, not zero-width where implausible), by root × year × moneyness bucket, with |delta| < 0.15 broken out | OTM bucket must clear a high coverage bar on index roots for D4-dependent candidates (014/041/042). Failure re-binds the trade-bar prohibition and re-opens Databento CBBO-1m as the fallback |
| **V2 — Row semantics** | Rows per day vs listed-chain contract count (sample days vs OCC reference); fraction of rows with volume = 0 but valid quotes | Confirms rows exist independent of trades (quote-bearing records). If rows only exist on traded minutes, D3/D4 verdicts revert to the screen's original CONDITIONAL |
| **V3 — Quote semantics** | What `bid_close/ask_close` are (NBBO at minute end?); crossed/locked frequency; zero-bid handling on far OTM | Defines the mark rule for `options_chain_eod`; crossed-quote rows are excluded from marks by rule |
| **V4 — Schema reconciliation** | 20 vs 21 columns (leading `symbol`); dtype audit vs the repo's `[us, UTC]` standard (inventory notes options dtypes are "mixed; check before assuming") | Canonicalization spec for `options_chain_1m` |
| **V5 — IV/greeks population by year** | Null rates for `implied_vol/delta/theta/vega` across 2012→2026 per root (the OpEx doc measured 95–98% populated on the *legacy* 13-month set; the full-history figure is unknown) | Years/roots below threshold get greeks recomputed from quotes (M7 path) rather than trusted |
| **V6 — EOD-join leakage** | For `gamma_eod`/`open_interest_eod`: which date's value is joined onto row-date *t*? Trace `combine_options_data.py` + spot-check against a known OI print | **Hard gate for OPT-043 and any OI/gamma feature.** If join is same-day (t's end-of-day value on t's rows), all uses must lag explicitly; document the lag rule in P-primitives |
| **V7 — Corporate actions** | Strike/contract adjustment behavior across known splits (AAPL 2020, TSLA 2020/2022, NVDA 2021/2024, AMZN 2022, GOOGL 2015/2022) | If unadjusted ("exactly as reported"), build the adjustment layer before any single-name candidate runs — was already a Phase-1 verify item; now testable on owned data |
| **V8 — Root continuity** | FB (2017–2021) → META (2021+) splice; SPX root composition (SPX vs SPXW; AM vs PM settlement); VIX expiry conventions (Wednesdays) | Registered splice/filter rules per root; SPX/VIX are signal-side assets for now (§7) so this is documentation, not a blocker |
| **V9 — Session/partition integrity** | Partition gaps 2012–2026; half-days; the 2026-03→present refresh gap; Theta Terminal re-run path | Refresh executed in Phase 1 if the subscription is active (Phase 0.10); otherwise the live-edge gap is a standing caveat for current-premium sizing (015/019) |
| **V10 — Provenance mix** | `DATA_INVENTORY.md` lists ThetaData **+ IBKR chains** as sources; CC found ThetaData scripts. Which rows, if any, are IBKR-sourced, and are the two populations distinguishable? | If mixed and indistinguishable, quote-quality conclusions from V1–V3 must be computed per sub-population or conservatively on the whole |
| **V11 — Empirical spread census** | Quoted-width distributions by root × moneyness × DTE × regime state × era (incl. event windows, 0DTE) | Parameterizes `cost_model_v2`; replaces the assumed width table — the single largest strengthening of the chain's weakest input |
| **V12 — Legacy-store disposition** | `options_1min/` (17 roots, 2024-11→2025-12) vs `options_combined` overlap equivalence | If equivalent, archive/delete the legacy store and retire its documentation; if not, understand why before deleting anything |

---

## 6. Purchase ladder v2

| Rung | Product | Status | Role |
|---|---|---|---|
| 0 | **Owned:** `options_combined` (233 GB, 31 roots, 1-min, 2012-06→2026-02) · Alpaca 1m underlying · CBOE VIX term · classifier history · shelved OpEx GEX machinery | **Primary source** pending V-battery | Everything: D2 (via derived EOD snapshot), D3, D4 (pending V1–V3), D5 (with caveats/M7), OI (pending V6) |
| 1 | ORATS near-EOD historical (~$399 one-time, 2007+) | **Optional — recommended, non-blocking** | (i) 2007–2012 extension: the *only* cure for the missing-GFC caveat at EOD granularity; (ii) dividend-aware smoothed SMV surfaces (skew/curvature/deep-OTM candidates 027/030/047 and LEAPS marks for 006) without building M7; (iii) breadth beyond 31 roots; (iv) independent cross-vendor validation of owned quotes/IV (upgrades D-047/030 from internal-consistency to true cross-check) |
| ~~2a~~ | ~~ThetaData Value~~ | **Dead** | Superseded by owned data + (if active) the existing Standard subscription for incremental pulls |
| ~~2b~~ | Databento OPRA CBBO-1m | Dormant fallback | Only if V1–V3 fail badly on quote quality |
| 3 | Earnings-calendar API | Unchanged (deferred until 020/021 front of queue) | Approximate dates + BMO/AMC |
| — | True PIT consensus | Unchanged: never at retail | OPT-023 stays NO-GO |

**New build-vs-buy fork (decide at Phase 2):** smoothed IV surface. Buy = ORATS SMV. Build = **M7**, an own surface fitter (e.g., per-expiry SVI or spline on quote mids at the snapshot, dividend-aware for single names) with its own validation battery. If ORATS is purchased anyway for the GFC extension, buy wins on effort; M7 remains the long-run self-sufficiency path.

---

## 7. Verdict and condition deltas (data-screen column only; economics, priors, and decay conclusions unchanged)

| ID | Was | Now |
|---|---|---|
| **014** Opening-range gap verticals | COND: underlying pre-test → **Rung-2 purchase** | COND: underlying pre-test (D-014/042, unchanged) → **V1–V3 pass**. History for the eventual test: 13.7 y, incl. 2012–2019 pre-0DTE era; note SPY expiry-availability regimes (Fri weeklies full-history; M/W ~2016+; daily ~2022+) |
| **041** 0DTE post-OR condor | COND: Rung-2 + strict cost bar; "never justifies the purchase alone" | COND: **V1–V3 + the same strict cost bar.** The purchase objection is moot; the economics objection stands verbatim (research-to-disprove; MARGINAL-to-FAIL cost). Backtest must segment by expiry-availability era |
| **042** 0DTE first-hour trend | COND: shared pre-test → Rung 2 | COND: shared pre-test → V1–V3. Expected DROP on the post-2023 slice, unchanged |
| **040** Weekend theta | COND: D-040b diagnostic; Monday-09:45 exit not Rung-1-compatible | COND: **D-040b unchanged and still expected to kill it.** Monday-09:45 exit now feasible as registered — the forced-amendment discussion is retired |
| **043** OpEx pin/charm | COND: OI acquisition + D-043 pre-test | COND: **V6 leakage gate + D-043 pre-test** (OI is on disk). **New reconciliation duty:** the repo's shelved OpEx pinning strategy is prior art — reuse its GEX/calendar machinery where sound, and record its 2025 backtest (18 trades, −$415, estimated gamma/OI) as a **prior tested trial in the lifetime ledger**. OPT-043's spec (short straddle at T−1 max-OI strike, monthly OpEx) remains distinct from the OpEx strategy's GEX-directional design — related family, different hypothesis; both count trials |
| **021** Earnings crush | v1.1 registered (T+1 near-close exit) because T+1 open was unmeasurable | **Decision required, one version only, before any earnings data is touched:** (a) keep v1.1, or (b) restore v1.0 (T+1 open exit) now that minute quotes make the open measurable — with stressed-tier open costs. **Recommendation: restore v1.0** — the pure-crush hypothesis (minimal drift exposure) is the cleaner mechanism and was only amended away for data reasons that no longer hold. Universe issue below applies |
| **025** Macro-event straddle | Amended to T+0 near-close | Amendment stands (it matches the original 15:50-style exit anyway); intraday exit marks now verifiable. Wave-4 null-anchor status unchanged |
| **020/026/045/031/021-universe** | Universes reference `U_MEGA10/20`, "liquid names with weeklies," sector peers, squeeze names | **Universe decision required:** disk covers ~13 liquid singles (≈`U_MEGA10`+). Options: (a) narrow registered universes to on-disk roots — logged amendment; materially shrinks 021's event count and 026/045's pair space; (b) top-up download via the ThetaData subscription (Phase 0.10 determines feasibility/marginal cost); (c) ORATS breadth for EOD-level needs. `U_MEGA10` candidates (020) proceed with a PIT-ranking check only |
| **006/030/047** | "ORATS smoothed mandatory" for LEAPS/deep-OTM marks | "Smoothed source = ORATS **or** M7; shipped ThetaData greeks remain disfavored deep-OTM/ITM." D-047/030 integrity check upgrades to cross-vendor if ORATS bought |
| **015/019** | Current-premium sizing implicitly to present | Data edge is 2026-02; refresh (V9) or carry a live-edge caveat |
| **016** | Touch-frequency bias caveat, quantifiable "later" | Quantify **now** from minute data; report alongside results; spec unchanged |
| All P5-gated | Classifier PIT log (Phase 0.8) | Unchanged — this amendment does not resolve 0.8; it remains the other existential verification |

**Not added:** no new candidates. SPX/VIX/IBIT availability invites new families (VIX term trades, SPX European variants, crypto-ETF vol). Per the generator's invariant, expanding the search space is a **deliberate, recorded act** — noted as available, not exercised. SPX/VIX data may be used **signal-side** (e.g., OPT-013's VIX-term trigger now computable from owned options rather than CBOE indices — keep the registered CBOE source to avoid a silent spec change; owned data serves as cross-check).

---

## 8. Revised DSR arithmetic

Primary window 2012-06→2026-02 ≈ **13.7 y** ⇒ σ_SR ≈ 0.27 (was 0.226 on the assumed 19.5 y). Null best-of-N (Bailey–López de Prado expected max):

| Stage | N | E[max SR | null], 13.7 y | with ORATS ext. (19.5 y) |
|---|---|---|---|
| Wave 1 | 9 | ≈ **0.41** | ≈ 0.34 |
| + Wave 2 | ≈19 | ≈ **0.51** | ≈ 0.42 |
| + Waves 3–4 | ≈31–38 | ≈ **0.56–0.59** | ≈ 0.47–0.49 |

The intraday-specific penalty ("N=5 on 6.5 y ≈ 0.47") is **retired** — intraday and EOD candidates now share the 13.7 y window. It is replaced by two slate-wide caveats: (1) **missing-2008**: no GFC-class regime in-window; for the short-vol core this right-censors the loss distribution — curable at EOD granularity by the ORATS extension, permanent for intraday-native candidates; (2) **per-root windows**: later-listed singles shrink further (PLTR 2020 ≈ 5.7 y → σ ≈ 0.42; COIN 2021 ≈ 4.7 y → σ ≈ 0.46) — DSR per candidate uses **its** window, and mixed-window baskets (020) report the binding shortest root. Lifetime-N reconciliation (Phase 4) now explicitly includes the **OpEx 2025 backtest** and any other options-era trials found in the repo ledger.

---

## 9. CC phase-plan deltas

- **Phase 0 additions:** **0.9** resolve the `OptionsDataStore` (empty layouts) vs `OptionsDataLoader` (`options_combined`) divergence — one truth; **0.10** confirm ThetaData subscription status + tier (data starts 2012-06, which the research pass associated with Pro-tier first-access, vs CC's "Standard" label — determines marginal cost of root top-ups and the 2026-03+ refresh); **0.11** locate and read `download_options*.py` / `combine_options_data.py` as the refresh + join-semantics ground truth (feeds V6/V9).
- **Phase 1 rewritten:** *Canonicalize and verify the owned set* (~$0): build `options_chain_1m` + derived `options_chain_eod` (snapshot 15:45), run V1–V12, refresh 2026-03→present, execute the ORATS decision in parallel if taken. The Wave-1 ingest-scope restriction ("SPY+QQQ+U_MEGA20 only") is void — data is on disk; *processing* scope is still staged sensibly.
- **Phase 2:** add **M7** (surface fitter) to the build-vs-buy fork; P6's stressed-slippage schedule can now be cross-checked against measured options-spread behavior in stressed eras (V11).
- **Phases 3–5:** unchanged in structure; Group B diagnostics run post-Phase-1 at $0; Wave 1 remains nine trials, stop at Wave 1.

## 10. What does not change

Candidate definitions and fixed parameters (slate v1, verbatim) · prior tiers · cost-viability *economics* and every decay conclusion (VRP compression, 0DTE crowding, earnings-crush name-dependence, dead PEAD, priced weekends) · wave structure, kill switches, prohibited-responses list · diagnostics D-003…D-050b (only their *precondition* moves from "post-ORATS" to "post-Phase-1") · the ledger schema and the +50-emitted delta · OPT-023's NO-GO · the reiterate criteria · the rule that STOP decisions on capital are Shuyang's.

## 11. Open items & honesty block

**Requires Shuyang's decision:** (1) OPT-021 v1.0-restore vs v1.1-keep (recommendation: restore v1.0); (2) universe treatment for breadth-dependent candidates (narrow vs top-up vs ORATS); (3) the ORATS $399 purchase (recommendation: yes, for GFC + surface + cross-validation; nothing blocks on it); (4) legacy `options_1min/` disposition after V12.

**Meta-lesson, recorded:** the doc chain preached "code is ground truth; docs diverge" and then inherited a stale data premise from working notes without a disk check — the feasibility screen's §"Scope intake" even flagged the assumption as correctable, which is what saved the structure. The corrective is procedural: **any future feasibility screen's data-availability section requires a direct-inspection inventory as an input artifact, not a recollection.** This amendment, the CC inventory it cites, and the V-battery results become that artifact set for the options chain.

Nothing in this amendment observed a strategy return. The favorable direction of the correction (more data than assumed) does not upgrade any prior: better feasibility is not better edge. Zero survivors after validation remains a legitimate, pre-registered outcome.

*End of Amendment A1. Ledger action: append A1 rows to every candidate listed in §7; retire the superseded purchase-ladder and mark-convention sections of the two v1 process docs; slate v1 untouched.*
