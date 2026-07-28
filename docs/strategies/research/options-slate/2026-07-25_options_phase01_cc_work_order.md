# Options Data Layer — CC Work Order, Phase 0 + Phase 1

**Date:** 2026-07-25
**Scope:** repo reconciliation and data-layer canonicalization/verification **only**. No strategy work.
**Authority:** this document is the complete instruction set for this phase. It is self-contained — you do not need the slate, the feasibility screen, or Amendment A1 to execute it. Those exist as background and are listed in §7 if you need context for a judgment call, but nothing here depends on reading them.
**Supersedes:** any Phase-0/Phase-1 instruction in `2026-07-24_options_slate_cc_handoff_spec_v1.md` (retired).

---

## 0. Ground rules

1. **Code and disk are ground truth.** This document asserts things about the repo and about `options_combined/`. Several assertions are second-hand and may be wrong. Where this document and reality disagree, **reality wins and you report the divergence** — you do not silently reconcile, and you do not build against the assertion.
2. **Measure, do not adapt.** Every gate below has a registered numeric threshold. You do not adjust a threshold after seeing a measurement. A failed gate is a reportable result, not a problem to engineer around.
3. **Stop conditions are real.** If a hard gate fails, stop that workstream and report. Do not substitute a different data source, do not impute, do not proceed to dependent work.
4. Report divergences and failures **as you find them**, not only in a final summary.

---

## 1. Phase 0 — Repo reconciliation (blocking; nothing in Phase 1 starts until this is reported)

Eleven integration points are **asserted and unverified**. For each: state what the repo actually does, cite file + line, and mark `CONFIRMED` / `DIVERGENT` / `ABSENT`.

| # | Assertion to verify | Acceptance criterion |
|---|---|---|
| 0.1 | Strategy registry format and how a strategy registers | Exact registration mechanism documented with file:line; note whether `RAMPSignals`-style omissions exist |
| 0.2 | `OptionsDataLoader` (`src/strategies/options/data_loader.py`) is the live reader of `options_combined/` | Its actual output schema — **column names it emits**, not the ones the docs claim |
| 0.3 | Provider/data-access abstraction: is there one, what is its interface | Whether a new options store plugs in or needs a new seam |
| 0.4 | Experiment/trial ledger: where it lives, its schema, how `n_trials` is recorded | Path + schema; if absent, say so plainly |
| 0.5 | `config/trading/strategy_toggle.yaml` structure | Format for adding a toggled-off strategy |
| 0.6 | Yang-Zhang / realized-vol estimator: does an implementation already exist | Path if yes; do not write a second one |
| 0.7 | S3 / DuckDB / parquet storage + partitioning conventions | The convention a new table must follow |
| 0.8 | Cost model: existing module, its interface, its assumptions | Path + whether options are representable |
| **0.9** | **`src/data/options/options_store.py` (`OptionsDataStore`) points at `chains/` and `gex_daily/`, which are empty.** `OptionsDataLoader` points at `options_combined/`, which holds the data | Determine which is dead. **Do not leave two truths standing** — propose deprecate-or-delete, do not execute the deletion yet |
| **0.10** | **ThetaData subscription: active? which tier?** Data starts 2012-06, which is associated with a higher tier than the "Standard" label in prior notes | Status + tier + whether incremental pulls (new roots, 2026-03→present) are possible and at what marginal cost |
| **0.11** | **`scripts/data/download_options*.py` and `combine_options_data.py`** — read them | The **join semantics** for `gamma_eod` / `open_interest_eod` (feeds V6) and the **refresh procedure** (feeds V9) |

**Also verify:** the classifier point-in-time state log (whether the regime classifier's historical state is reconstructable point-in-time, or only recomputable). This is tracked separately as an existential blocker for a different candidate family. Report status; do not attempt to fix.

---

## 2. Phase 1a — Canonicalization

Build **two tables** over the owned store. Do not bulk-copy 233 GB; these are read-layer canonicalizations plus a materialized derived table.

**Source:** `<local_storage_dir>/options/options_combined/root={SYM}/year={YYYY}/month={MM}/data.parquet`

**Table A — `options_chain_1m`** (canonicalized view over native minute data)

| Source column | Canonical name | Notes |
|---|---|---|
| `timestamp` | `ts` | normalize to `Datetime[us, UTC]`; confirm source tz semantics first |
| `root` (partition) / `symbol` | `root` | resolve the 20-vs-21-column question (V4) |
| `expiration` | `expiry` | date type |
| `strike`, `right` | `strike`, `right` | `right ∈ {C,P}` |
| `bid_close`, `ask_close` | `bid`, `ask` | |
| — | `mid` | `(bid+ask)/2`, null where quote invalid per V3 rule |
| — | `spread_abs`, `spread_rel` | `ask-bid`; `spread_abs/mid` |
| `open/high/low/close/volume/trade_count/vwap` | unchanged | **trade-derived; never used as a mark** |
| `implied_vol`, `delta`, `theta`, `vega` | unchanged | vendor-computed; carries known caveats |
| `underlying_px` | `underlying_px` | |
| `gamma_eod`, `open_interest_eod` | unchanged | **do not use until V6 passes** |
| — | `dte` | calendar and trading-day variants |
| — | `quote_valid` | boolean per V3 rule |

**Table B — `options_chain_eod`** (derived, materialized)
Single record per contract per session, taken at the **registered snapshot minute 15:45:00 ET**. Same columns. This is the EOD table for daily-cadence work — it is *derived from owned minute data*, not from a vendor snapshot.

**Binding rules:**
- **Snapshot symmetry.** 15:45:00 ET is the one clock. Option marks, any signal derived from minute bars (truncated at that minute inclusive), and daily hedge execution all use it. Implement as a single guarded function; nothing downstream slices bars itself.
- **Marks come from quotes, never from trade prints.** `close`/`vwap` are not marks. Out-of-the-money contracts trade rarely; a trade-print mark is stale or absent exactly where it matters.
- **Invalid quotes are excluded, never repaired.** No imputation, no forward-fill, no interpolation.
- **Intraday convention** (for later intraday work, register now): decide on bar *t* close, fill at bar *t+1*.

---

## 3. Phase 1b — Verification battery (V1–V13)

Numeric gates are registered here **before measurement**. Do not alter them.

| ID | Measurement | Registered gate |
|---|---|---|
| **V1** Quote population | Per root × year, fraction of regular-trading-hours listed-contract-minutes with `bid>0 ∧ ask≥bid ∧ both finite`, bucketed by `\|delta\|`: [0.05,0.15], [0.15,0.35], [0.35,0.65] | Index roots (SPX, SPY, QQQ, IWM), [0.05,0.15] bucket: **≥95%** per year = PASS; 90–95% = PARTIAL; **any year <90% = FAIL**. Single names: 90% / 85–90% / <85%. FAIL ⇒ stop, report; do **not** substitute another vendor |
| **V2** Row semantics | ≥20 sampled sessions, ≥1 per year 2013–2025 plus ≥3 high-volatility sessions. Measure (a) rows/day vs listed contracts × RTH minutes, (b) fraction of rows with `volume=0` **and** valid quote | **≥60%** of rows `volume=0` with valid quote ⇒ quote-bearing confirmed (PASS). **<10%** ⇒ these are trade bars ⇒ **FAIL**, all intraday-quote-dependent work blocked. 10–60% ⇒ PARTIAL, report distribution |
| **V3** Quote semantics | Crossed (`bid>ask`) and locked (`bid==ask`) frequency; zero-bid frequency by moneyness; establish whether `bid_close`/`ask_close` are NBBO at minute end | Crossed **<0.1%** of quoted contract-minutes = PASS. Crossed rows excluded from marks **by rule regardless of rate**. Zero-bid handling documented, not repaired |
| **V4** Schema reconciliation | 20 vs 21 columns (leading `symbol`); full dtype audit against the `Datetime[us, UTC]` standard | Report only; output is the canonicalization mapping in §2 |
| **V5** IV/greeks population | Null rate for `implied_vol`, `delta`, `theta`, `vega` per root × year. Sanity: `IV ∈ (0.01, 5.0)`, `\|delta\| ≤ 1`, delta sign correct per `right` | **≥90%** non-null per root-year = PASS. Below ⇒ that root-year is flagged **untrusted**, routed to later recomputation. Do **not** recompute now |
| **V6** EOD-join leakage | **HARD GATE.** (a) read the join in `combine_options_data.py`; (b) verify `open_interest_eod` is constant across all minutes of a session (if it varies intraday the join is broken); (c) establish whether the value on session *t* is OI as-of *t*'s close or *t−1*'s close | Same-day join ⇒ **all OI/gamma features must lag ≥1 session**, enforced in the primitive layer, not per-strategy. If the answer cannot be established from code + data ⇒ **all OI/gamma work BLOCKED** |
| **V7** Corporate actions | Strike-grid behavior across: AAPL 2020-08-31 (4:1), TSLA 2020-08-31 (5:1) and 2022-08-25 (3:1), NVDA 2021-07-20 (4:1) and 2024-06-10 (10:1), AMZN 2022-06-06 (20:1), GOOGL 2022-07-18 (20:1) | Report adjusted vs as-reported. As-reported ⇒ adjustment layer required before **any** single-name work |
| **V8** Root continuity | FB (2017–2021) → META splice; SPX vs SPXW composition and AM/PM settlement; VIX expiry convention | Report + registered splice/filter rules per root |
| **V9** Partition integrity | Sessions present vs expected per root-month 2012–2026; quantify the 2026-03→present gap; half-day handling | Any root-month with **<90%** of expected sessions flagged. Refresh executed only if 0.10 says the subscription supports it |
| **V10** Provenance | `DATA_INVENTORY.md` lists ThetaData **+ IBKR chains**; the download scripts are ThetaData. Which rows, if any, are IBKR-sourced; are populations distinguishable | If mixed and indistinguishable, V1/V3 statistics must be computed conservatively over the whole |
| **V11** Spread census | Quoted-width distribution by root × moneyness × DTE bucket × year × volatility state, including event windows and 0DTE | No gate — this is a **deliverable**: a parquet of width statistics that later parameterizes the cost model |
| **V12** Legacy store | `options/options_1min/` (17 roots, 2024-11→2025-12) vs `options_combined` overlap equivalence on shared root-months | Report equivalence. **Delete nothing** |
| **V13** PIT universe coverage | For every point-in-time-defined universe (rank/weight-based membership), fraction of sessions where **all** required members exist on disk; report the earliest date from which coverage is complete | **<95%** of sessions fully covered ⇒ that universe is **not usable as registered**; report the first-full-coverage date and the implied usable window |

V13 exists because at least one registered universe (`U_TOP6_SPY`, six largest SPY constituents *at date t*) has a membership that in 2012–2018 consists largely of roots not on disk. Assume other rank-based universes share the defect until measured.

---

## 4. Prohibited in this phase

- **No strategy backtest of any kind.** Not exploratory, not "a quick sanity check," not on a subset. Any P&L computation is out of scope.
- **No threshold selection or adjustment after seeing data.** Gates are in §3.
- **No imputation, forward-fill, interpolation, or smoothing** of quotes, IV, or greeks.
- **No silent row exclusion.** Quarantine, count, report.
- **No recomputation of greeks or IV surfaces.** Not authorized this phase.
- **No scope expansion** — no new roots, no new candidates, no new data sources. A failed V1 does **not** authorize evaluating an alternative vendor.
- **No edits to the pre-registration documents** (slate, screen, spec, amendment).
- **No deletion** of `options_1min/`, `chains/`, or `gex_daily/`.
- **No silent reconciliation** of doc-vs-code divergence.

---

## 5. Deliverables

1. `docs/research/2026-07-XX_options_phase0_reconciliation.md` — the eleven items: asserted / actual / status / evidence (file:line).
2. `docs/research/2026-07-XX_options_vbattery_report.md` — per item: **measured value, registered gate, PASS / PARTIAL / FAIL, evidence**. Gates restated verbatim from §3 so the report is auditable against what was registered.
3. Canonicalization code for `options_chain_1m` + `options_chain_eod`, with tests, following the conventions found in 0.7.
4. `spread_census.parquet` (V11 output) at the conventional storage path.
5. A short divergence log — everything this work order asserted that turned out false.

Reports state measurements and gate outcomes. **They do not draw conclusions about strategy viability**; that judgment happens elsewhere.

---

## 6. Escalation

Stop and report, without adapting, on: any V1/V2 FAIL · V6 unresolvable · V7 as-reported strikes · any Phase-0 item `ABSENT` where downstream work assumes it exists · discovery that the data materially differs from §2's description.

## 7. Background (not required reading)

`2026-07-25_options_slate_cc_handoff_spec_v2.md` (full build spec, Phase 2+) · `2026-07-25_options_slate_feasibility_screen_v2.md` (verdicts and reasoning) · `2026-07-25_options_docchain_amendment_A1.md` (why the data premise changed) · `2026-07-24_options_strategy_slate_v1.md` (candidate definitions — note its scope-intake assumption about data availability is retired).
