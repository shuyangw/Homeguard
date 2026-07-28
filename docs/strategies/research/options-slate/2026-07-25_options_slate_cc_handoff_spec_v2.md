# Equity Options Slate — CC Implementation Handoff Spec (v2)

**Date:** 2026-07-24 · **Amended:** 2026-07-25 — **Amendment A1** (data-inventory correction) incorporated throughout. v1 is retired; where v1 and v2 differ, v2 governs.
**Author context:** Homeguard, spec-first workflow. Claude = spec producer / reviewer. CC = execution agent.
**Chain:** `2026-07-25_options_strategy_slate_v1_1.md` (candidate definitions, priors) → `2026-07-25_options_slate_feasibility_screen_v2.md` (data/cost screens, verdicts, waves) → `2026-07-25_options_docchain_amendment_A1.md` (correction record + V-battery) → **this document** (buildable spec).
**Scope:** the **46 candidates still in play** — 21 GO, 18 CONDITIONAL, 7 DEFER — with full implementation detail. OPT-023 (NO-GO) is excluded. The 3 DROP candidates (010, 012, 022) appear in Appendix C as a *do-not-resurrect* record only.
**Status:** Pre-registration of record. Blind. No backtest return has been observed by any stage of this chain.

---

## TL;DR

- **This document is the buildable form of the slate.** It defines the data layer over the **owned `options_combined/` store** (~233 GB, 31 roots, 1-minute quotes/IV/greeks, 2012-06 → 2026-02) with an **optional ORATS extension**, dual canonical chain tables (native 1-minute + derived EOD snapshot), ten shared strategy primitives (P1–P10), a two-tier cost model with an empirical-census upgrade path, seven harness modules, then a per-candidate spec for all 46 live candidates organized in build order (Wave 0 → Wave 4), plus the ledger schema and a five-phase CC implementation plan.
- **CC must not begin Phase 2 (strategy specs) before Phase 0 (repo reconciliation) is complete and reported.** Eleven integration points are asserted and *unverified* — the original eight (registry format, provider abstraction, ledger location, toggles, Yang-Zhang reuse, storage conventions, cost module, classifier PIT log) plus three added by A1: the `OptionsDataStore`-vs-`OptionsDataLoader` divergence, the ThetaData subscription status/tier, and the download/combine scripts' join semantics. Code is ground truth. Do not build against this document's assumptions where the repo disagrees — report the divergence and stop.
- **The leakage discipline is now the snapshot-symmetry rule.** We control the snapshot: the registered minute is **15:45:00 ET** for option marks, for truncation of every 1-minute-derived signal, and for the daily hedge (P6 `daily_1545` — hedge time amended 15:50→15:45 per A1, logged). One clock; v1's vendor-snapshot lookahead trap no longer exists, but the symmetric-truncation guard it motivated remains centrally enforced in P9/P10.
- **Sequencing is the whole design.** Wave 0 now opens with **Group C — the V-battery** (data verification on the owned store, Amendment A1 §5), then costs zero trials killing or promoting roughly a dozen candidates before any strategy backtest runs. Wave 1 is nine trials. Nothing beyond Wave 1 is authorized by this document — later waves are specified so CC can build reusable machinery once, not so they can be run early.

---

## What this document does NOT do

- It does **not** authorize backtests beyond Wave 0 diagnostics and the Wave 1 slate. Waves 2–4 are specified for build reuse; each requires its named gate to open first.
- It does **not** permit parameter tuning. Every parameter here is pre-registered and fixed. If a value proves unimplementable as written, CC **stops and reports**; it does not substitute a nearby value. A substituted parameter is a silent extra trial.
- It does **not** define GO decisions on strategies. It defines pre-registered *criteria*; the harness computes; STOP/GO on capital remains Shuyang's call.
- It does **not** replace the repo. Where this spec and the code disagree, the code is right and this spec is stale.
- It does **not** treat any performance number as existing. None do.

---

## Phase 0 — Repo reconciliation (mandatory first task, blocking)

CC has previously propagated false claims about repo state. Every item below must be verified by **direct inspection** (`cat`, `ls`, `git log`, `rg`) and reported with the evidence inline — file paths, line numbers, commit SHAs. "I checked and it exists" is not a report. If an item does not exist, say it does not exist; do not describe the version you expected.

| # | Assertion to verify | How | If divergent |
|---|---|---|---|
| 0.1 | Research-harness registry ingest format (fields a strategy spec must expose) | Locate the registry module; print the schema/dataclass | Reconcile §7 ledger + §5 spec fields to the real format; report deltas before building |
| 0.2 | Data-provider abstraction vs chain-shaped data — **note: a chain loader already exists** (`src/strategies/options/data_loader.py`, reads `options_combined/`) | Read the provider base class **and** the existing `OptionsDataLoader`; print the loader's actual output columns (the infra doc's column list is suspected fiction) | Chain data is strike × expiry × date-time — decide extend-loader vs new provider type; reconcile with 0.9 |
| 0.3 | Multiple-testing ledger: does it exist, where, what schema | `rg` for the ledger; check research docs dir | If absent, create per §7 and report the chosen path for approval |
| 0.4 | `config/trading/strategy_toggle.yaml` conventions + the known dead `ramp.variant: v11` key | `cat` the file | Do not add options entries until the dead-config question is settled |
| 0.5 | `TransformedDataProvider`, robust z-score, Yang-Zhang, log/rank transforms (Tier-1 toolbelt) | `rg` for each | If Yang-Zhang exists, **reuse it** — P9 must not fork a second implementation |
| 0.6 | S3/local Hive-partition conventions + DuckDB attach patterns actually in use | Inspect an existing dataset path and loader | Match existing convention exactly; §1.2 materialization paths are proposals, not decrees |
| 0.7 | Existing cost-model / slippage module (from RAMP work) | `rg` | Extend rather than fork; RAMP's cost floor lesson lives there |
| 0.8 | Regime classifier interface + historical state log availability | Inspect `MarketRegimeDetector` and any persisted state history | OPT-050 and every regime gate depend on a point-in-time state log. If states were only computed live and never persisted, say so — several gates become unbacktestable as specified |
| **0.9** | **`OptionsDataStore` divergence:** `src/data/options/options_store.py` points at empty `chains/`/`gex_daily/` layouts while `OptionsDataLoader` reads `options_combined/` | `cat` both; `ls` the dead paths | Two truths cannot stand: deprecate/redirect the store class or delete the dead layouts; report the choice with evidence. Do not build a third path |
| **0.10** | **ThetaData subscription status + tier.** Disk history starts 2012-06 (Pro-tier first-access per vendor docs) vs the inventory's "Standard" label | Check Theta Terminal config/account; attempt a metadata query | Determines marginal cost of universe top-up downloads and the 2026-03 → present refresh. If lapsed, refresh + top-ups price like new purchases — report before any universe decision is executed |
| **0.11** | **Download/combine script semantics:** `scripts/data/download_options*.py`, `combine_options_data.py` | Read both; identify the endpoints used and the EOD-join key/date logic | These are ground truth for V6 (gamma/OI join leakage) and V9 (refresh path). Report the join semantics verbatim |

**Phase 0 output:** a written reconciliation report. Phases 1+ are blocked until it is reviewed.

---

# 1. Data layer

## 1.1 Source ladder v2 (what exists, what may be bought, and when)

| Rung | Source | Status | Role |
|---|---|---|---|
| **0 (owned)** | `options_combined/` (233 GB, 31 roots, 1-min OHLCV + bid/ask close + IV + delta/theta/vega + EOD-joined gamma/OI, 2012-06→2026-02) · Alpaca 1m underlying · CBOE VIX term · macro/earnings calendars (free tiers) · classifier history · shelved OpEx GEX machinery | **Primary, pending the V-battery (§5 Group C)** | Everything: derived EOD chains (D2), IV/greeks (D5, caveated), intraday quotes (D3/D4, pending V1–V3), OI (V6-gated) |
| 1 | ORATS near-EOD historical, ~$399 one-time, 2007+ | **Optional — recommended, non-blocking** | (i) 2007–2012 GFC extension (only cure for the missing-2008 caveat at EOD granularity); (ii) dividend-aware smoothed SMV surfaces (P1 low-delta rule, 006/027/030/047) without building M7; (iii) breadth beyond 31 roots; (iv) independent cross-vendor validation of owned quotes/IV |
| ~~2a~~ | ~~ThetaData Value subscription~~ | **Dead** | Superseded by the owned store + (if 0.10 finds it active) the existing subscription for incremental pulls |
| 2b | Databento OPRA CBBO-1m | Dormant fallback | Only if V1–V3 fail on OTM quote quality |
| 3 | Earnings-calendar API (approximate dates + BMO/AMC) | Deferred until 020/021 front the queue | Event candidates |
| — | True point-in-time consensus | Never at retail | OPT-023 stays NO-GO |

**OI note:** `open_interest_eod` is on disk; its use is blocked until V6 establishes the join-date semantics, and every use obeys the T−1-known rule (§1.3).

## 1.2 Canonical internal chain tables (v2 — dual-table design)

The owned store is the source. Canonicalize at ingest; do not let on-disk column names leak into strategy code.

**Table 1 — `options_chain_1m` (canonicalized native).** One row per contract per minute, mapped from the on-disk 20/21-column layout (V4 reconciles the column count and audits dtypes; normalize timestamps to the repo's `[us, UTC]` standard):

| on-disk | canonical | note |
|---|---|---|
| `root` (partition key) | `ticker` | V4 resolves whether a `symbol` column also materializes |
| `timestamp` | `bar_ts` | minute bar end, tz-normalized |
| `expiration`, `strike`, `right` | `expiry`, `strike`, `right` | plus derived `dte`, `dte_trading` |
| `open/high/low/close/volume/trade_count/vwap` | trade-print fields, same names | **never marks for OTM/low-volume strikes** (M2) |
| `bid_close`, `ask_close` | `bid`, `ask` | + `mid` materialized; V3 defines validity (crossed/zero-bid exclusion) |
| `implied_vol` | `iv_shipped` | ThetaData-computed: Black-Scholes, dividends off, deep-ITM/OTM error; V5 gates trust by year/root |
| `delta`, `theta`, `vega` | `delta_shipped`, `theta_shipped`, `vega_shipped` | gamma is absent at minute level by construction |
| `underlying_px` | `spot` | |
| `gamma_eod`, `open_interest_eod` | `gamma_eod`, `oi_eod` | **V6-gated**; T−1 rule (§1.3) |
| — | `ingest_version`, `source` | `source='thetadata'` default; V10 may split provenance |

**Table 2 — `options_chain_eod` (derived).** One row per contract per session: the record at the **registered snapshot minute 15:45:00 ET**, selected from Table 1 (fallback: latest bar ≤ 15:45 with valid quotes, flagged `snapshot_fallback=true`; `snapshot_ts` carries the actual time — never assumed). Fields: `ticker, trade_date, snapshot_ts, snapshot_fallback, expiry, dte, dte_trading, strike, right, bid, ask, mid, iv_shipped, iv_smooth, delta_shipped, theta_shipped, vega_shipped, spot, oi_eod, day_volume, ingest_version, source`. `iv_smooth` is nullable — populated by M7 or by ORATS SMV if purchased; **it is the required source for |delta| < 0.10 work** (P1 rule). If ORATS is bought, its rows land in this same table with `source='orats'` and its smoothed fields populate `iv_smooth`/`smv_value`; the 2007–2012 extension arrives as additional `trade_date` range, not a schema fork.

**Derived tables to materialize once** (recomputing per-strategy is both slow and a divergence risk):
- `atm_iv_daily(ticker, trade_date, dte_bucket, atm_iv)` — dte buckets {7, 14, 30, 45, 60, 90}, interpolated on `iv_smooth` where present, else ATM `iv_shipped`; record which source per row.
- `skew_daily(ticker, trade_date, dte_bucket, iv_25d_put, iv_25d_call, skew_25d = iv_25d_put − iv_25d_call, curvature)` — smoothed source per the P1 rule.
- `term_slope_daily(ticker, trade_date, m1_atm_iv, m2_atm_iv, slope = m2/m1)`.
- `iv_rank_daily(ticker, trade_date, iv_rank_1y, iv_pctile_1y, iv_pctile_2y)` — trailing windows, strictly backward-looking.
- `rv_daily(ticker, trade_date, yz_10d, yz_20d, har_forecast_22d)` — from P9/P10, snapshot-truncated (§1.3).

**Partitioning & scope:** the native store stays in place (`options_combined/root=/year=/month=`); canonical and derived tables materialize per the conventions Phase 0.6 finds. **There is no bulk pull in this plan** — the data is on disk; *processing* is staged (Wave-1 roots first: SPY, QQQ, then the on-disk megacaps for the overwrite candidates), and the 2026-03 → present refresh runs per V9/Phase 0.10.

## 1.3 Point-in-time and leakage rules (binding, enforced centrally)

1. **The snapshot-symmetry rule** (supersedes v1's 15:45-truncation rule, same guard). The registered snapshot minute is **15:45:00 ET**. Same-day option marks come from the snapshot record; every signal derived from 1-minute bars uses bars through the snapshot minute inclusive; the daily hedge executes at the snapshot minute (P6 `daily_1545` — amended from 15:50 per A1, logged before any test). Implemented as a single guarded function in the feature layer; strategies never slice bars themselves. One clock for signals, marks, and hedges — the basis and lookahead traps of vendor-snapshot designs are eliminated by construction, not by bookkeeping.
2. **Intraday execution convention (new, registered).** Intraday candidates (014, 041, 042) and intraday exits (040 Monday 09:45; 021 v1.0's T+1 open if restored): decide on bar *t* close, fill at bar *t+1* quotes. No same-bar fills anywhere.
3. **Open interest and gamma are EOD-joined fields.** Blocked until **V6** establishes which date's value lands on row-date *t* (trace `combine_options_data.py` + spot-check a known OI print). Regardless of the join answer, any OI-conditioned selection (OPT-043) uses **T−1-known** OI; same-session OI at decision time is a hard leak.
4. **Trailing windows are strictly backward-looking and expanding-safe.** IV rank/percentile at date *t* uses only data < *t*. No full-sample percentile computation anywhere.
5. **Earnings dates are an integrity item, not a convenience field.** Dates pulled today may be corrected relative to what was known then. ALFRED-vintage discipline: store the retrieval timestamp, validate a sample against a second source, mark affected results lower-confidence. Applies to OPT-020/021/024/026/045.
6. **Regime-classifier state must be point-in-time.** If Phase 0.8 finds no persisted historical state log, regime gates are recomputed causally and that recomputation is reported as a build artifact — not silently assumed correct.
7. **No survivorship in the universe.** Single-name universes from a point-in-time membership/liquidity snapshot, not today's list; if a PIT option-volume ranking is unavailable, fall back to index membership at date *t* plus a liquidity proxy, and record the compromise.
8. **Corporate actions and root continuity (= V7/V8).** Verify the store's adjustment behavior on known splits (AAPL 2020, TSLA 2020/2022, NVDA 2021/2024, AMZN 2022, GOOGL 2022); if delivered "as reported," build the adjustment layer before any single-name candidate runs. Register the FB→META splice, SPX-vs-SPXW composition, and VIX expiry conventions before those roots feed anything.

## 1.4 Universe definitions (fixed) + on-disk availability

- `U_INDEX` = {SPY, QQQ, IWM} — **all on disk** (DIA additionally available; not in any registered spec).
- `U_MEGA10` = the ten largest US-listed single names by option volume at date *t*, PIT-constructed (rule 7). **≈ the on-disk singles** (AAPL, NVDA, TSLA, AMZN, MSFT, META, AMD, GOOGL, AVGO, MSTR…) — confirm with the PIT ranking rather than assuming.
- `U_MEGA20`, `U_TIER1_100` = same construction, 20 and 100 names; liquidity floor **≥ 1,000 contracts/day trailing 20-session average**. **Not covered on disk** beyond the ~13 singles.
- `U_TOP6_SPY` = six largest SPY constituents by index weight at date *t* (OPT-044) — on disk.

**Universe decision (pending, Shuyang):** candidates whose registered universes exceed the on-disk roots — 021 ("liquid names with weeklies, expected move ≥ 4%"), 026 (sector peers), 031 (squeeze names), 045 (sector pairs), and 001/002/020 to the extent they reach beyond the on-disk megacaps — resolve by one of: (a) **narrow** to on-disk roots (logged amendment; shrinks 021's event count materially), (b) **top-up download** via the existing subscription (feasibility/cost from Phase 0.10), (c) **ORATS breadth** for EOD-level needs. Until decided, Wave-1 candidates proceed on `U_INDEX` + on-disk megacaps; nothing in Wave 1 is blocked by this decision.

---

# 2. Cost model (pre-registered, versioned)

One module, `cost_model_v1`, used by every candidate. No strategy may define its own costs.

**Signature:** `cost(legs, regime_state, width_source, tier) -> (entry_cost, exit_cost, fees)`

**Spread convention** (fraction of quoted width paid beyond mid, per transit):
- Single leg, normal conditions: **25%**
- Native multi-leg combo (2–4 legs), normal: **3–6% per leg**
- **Stressed tier** — event windows, `regime_state ∈ {UNPREDICTABLE, BEAR}`, or realized-vol percentile > 80: treat combos at **single-leg grade (15–25%)**
- 0DTE / same-day: single-leg grade always

**Fee stack, per contract, one way:** commission $0.35–0.65 (tiered/Lite); Options Clearing Corporation clearing $0.025; Options Regulatory Fee ≈ $0.023; Consolidated Audit Trail $0.0003; FINRA Trading Activity Fee $0.00329 (sells only); SEC Section 31 at $20.60 per $1M of sale proceeds from **2026-04-04** (negligible at these premia but modeled for completeness). All-in ≈ **$0.42–0.72/contract one way**.

**Width assumptions (normal conditions), pending replacement by own fills:**

| Class | Quoted width |
|---|---|
| SPY/QQQ ATM, 30–45 DTE | $0.03–0.12 |
| SPY 0DTE ATM | $0.01–0.02 |
| SPY 0.05–0.10Δ (value $0.30–1.50) | $0.02–0.06 |
| Tier-1 single name ATM | $0.05–0.15 |
| Rank 50–100 ATM | $0.10–0.30 |
| Event window | 1.5–3× normal |
| LEAPS (12–18M) | $0.30–1.00 |

**Mandatory:** every backtest reports results at **1.0× and ±50%** cost multipliers. A candidate whose sign flips within that band is reported as cost-indeterminate, not as a result. **Upgrade path (new, per A1):** the width table above is provisional — **V11 (empirical spread census on the owned quote data, by root × moneyness × DTE × regime × era, including event windows and 0DTE)** re-parameterizes it as `cost_model_v2` before Wave 1 runs; measured widths supersede the assumptions above, and the census itself is a Wave-0 deliverable. Fill-fraction conventions (25% / 3–6% / stressed) remain assumptions at v2. **Replacement trigger:** once any options strategy trades live, own IBKR fill statistics supersede fill fractions too, and every prior result is re-run — fills are the residual weakest quantitative input in this chain.

---

# 3. Shared primitives (P1–P10)

Build once, in a primitives module. Candidates reference these by name; duplicated logic is a correctness risk and a silent-divergence risk.

- **P1 `select_strike_by_delta(chain, right, target_delta, dte_window)`** — nearest available |delta| to target within the DTE window, ties to the more liquid strike. **Rule:** for |target_delta| < 0.10, read delta and IV from the **smoothed surface** (`iv_smooth`: ORATS SMV if purchased, else M7) — never from `*_shipped` per-contract greeks or raw quotes (deep-OTM IV error; the caveat applies to the owned ThetaData-computed fields, per A1). Return the strike plus the realized delta actually selected; strategies log both.
- **P2 `select_expiry(chain, dte_min, dte_max, prefer)`** — `prefer ∈ {monthly, any}`. Monthly-only where the spec says OpEx (OPT-043). Never straddle a distribution of expiries silently — log the chosen expiry.
- **P3 `standard_exit(position, profit_take, dte_exit)`** — close at `profit_take` fraction of credit captured **or** at `dte_exit`, whichever first; evaluated once daily at snapshot. Default (001, 002, 004, 016, 017, 018): `profit_take=0.50`, `dte_exit=21`.
- **P4 `size_by_vega_budget(structure, budget_vega, nav)`** — units such that |net vega| ≈ budget. The **single shared budget constant** across F1/F3/F6 makes the ablation ladders comparable; it is fixed at spec time and never tuned. Alternative sizers (`size_by_debit`, `size_by_notional`) exist for debit structures and are named per candidate.
- **P5 `regime_gate(state, allowed_states)`** — from the point-in-time classifier state (rule 5). Gate is evaluated at entry only unless the spec says otherwise; a spec that unwinds on regime change says so explicitly.
- **P6 `hedge_ledger(position, bars_1m, mode, band, cost_bps_schedule)`** — delta hedging in the underlying. `mode ∈ {daily_1545, band}`. Daily mode hedges at the registered snapshot minute, **15:45**, using own bars (amended from 15:50 per A1 — snapshot symmetry; logged before any test). Band mode triggers on ±band σ moves. **Slippage schedule is tiered by realized-vol percentile** — hedge frequency spikes exactly when underlying spreads widen, and a flat bps assumption flatters every gamma strategy. Emits a ledger of every hedge fill; P&L attribution reads the ledger, not a reconstruction.
- **P7 `roll(position, roll_trigger, new_selector)`** — mechanical rolls (OPT-001/006/038/047). Logs each roll as a separate cost event.
- **P8 `iv_rank(ticker, date, window_years)` / `iv_percentile(...)`** — from `iv_rank_daily`, strictly backward-looking.
- **P9 `yang_zhang_rv(bars_1m, window_days)`** — Yang-Zhang realized-volatility estimator. **Reuse the Tier-1 toolbelt implementation if Phase 0.5 finds one.** Enforces the 15:45 truncation internally.
- **P10 `har_rv_forecast(bars_1m, spec=(1,5,22), horizon)`** — Heterogeneous Autoregressive Realized Volatility forecast on P9 inputs. **Spec frozen at (1,5,22); no order selection, no re-fitting schedule tuning.** Fit on an expanding window with no forward data. Enforces 15:45 truncation. Emits the forecast plus its correlation to contemporaneous VIX (reported alongside every result — see OPT-019's spurious-path check).

---

# 4. Harness modules (build once)

- **M1 — Early assignment & pin.** Short in-the-money calls: assign the session before ex-dividend when remaining extrinsic value < dividend. Short options in-the-money by ≥ $0.01 at expiry: auto-exercise per OCC. Pin handling at short strikes. **Required by every short-call/short-put structure** — roughly two-thirds of the slate. Needs a dividend calendar (external source required; ORATS `div_assumption` becomes available only if the extension is purchased — verify against a second source either way).
- **M2 — Mark convention (v2).** All EOD-level marks are the derived 15:45 snapshot record. **Prohibited:** trade-print-derived option OHLC bars as marks for out-of-the-money or low-volume strikes, and raw session "close" prints. Marking uses `mid` from valid quotes (crossed/zero-bid rows excluded per V3, flagged not dropped silently); on low-delta strikes the smoothed surface (`iv_smooth`, when available) is the cross-check — a mid that diverges materially from the surface is flagged, not silently used. Intraday marks (014/041/042/040/021-v1.0) use bar-level quotes under the *t*/*t+1* convention (§1.3 rule 2).
- **M3 — Cost model** (§2), with the sensitivity switch.
- **M4 — P&L decomposition.** Theta / vega / delta-residual / gamma attribution. Required by all of F4 (the calendar falsifiers are stated in these terms) and reused by OPT-038/044. Build it once with F4.
- **M5 — Regime-sliced attribution.** Every result reports P&L and Sharpe **per classifier state** in the standard output. This is not optional reporting polish: the OPT-015/039/048 falsifiers are stated per-regime, and aggregate numbers over 2007–2026 can hide a result manufactured entirely by the 2012–2019 short-vol era.
- **M6 — Validation wrapper.** Combinatorial Purged Cross-Validation (CPCV) with **embargo ≥ maximum holding period** of the candidate (45–60 days for most of this slate; longer for OPT-003/006's quarterly and LEAPS legs). Overlapping expiries make naive purging insufficient — the embargo must cover the full life of any position open at a block boundary. Probability of Backtest Overfitting (PBO) and Deflated Sharpe Ratio (DSR) computed with `n_trials` drawn from the **lifetime ledger** (§7), never from the wave.
- **M7 — Smoothed IV surface (build-vs-buy fork, decide at Phase 2).** Buy = ORATS SMV. Build = per-expiry SVI or spline fit on snapshot quote mids, dividend-aware for single names, with its own validation battery (fit residuals by moneyness, static no-arbitrage checks, day-over-day stability). Required by P1's low-delta rule and by 027/030/047's curvature signals; populates `iv_smooth`. If ORATS is purchased for the GFC extension anyway, buy wins on effort; M7 remains the self-sufficiency path.

---

# 5. Wave 0 — Verification & diagnostics (zero trials, run first)

Diagnostics measure **data properties**. They involve no strategy P&L and no selection on returns, so they do **not** increment `n_trials`. Each has a pre-registered gate whose failure drops the dependent candidate *before* it spends a draw from the finite sample.

**Group C — the V-battery (first; validates the owned store itself; full gates in Amendment A1 §5):**

| ID | Measurement | Consequence on failure |
|---|---|---|
| V1 | Quote population by root × year × moneyness (|Δ| < 0.15 broken out) | Trade-bar prohibition re-binds; Databento fallback for 014/041/042 |
| V2 | Row semantics: rows per day vs listed chain; volume=0-with-quotes fraction | D3/D4 verdicts revert to purchase-conditional |
| V3 | Quote semantics: NBBO-at-minute-end? crossed/locked/zero-bid rules | Defines M2's validity filter |
| V4 | Schema reconciliation (20 vs 21 cols) + dtype audit vs `[us, UTC]` | Canonicalization spec for §1.2 |
| V5 | IV/greeks null rates by year/root | Low-coverage years re-derive greeks via M7 instead of trusting shipped fields |
| V6 | `gamma_eod`/`oi_eod` join semantics (which date lands on row *t*?) | **Hard gate for OPT-043 and any OI/gamma feature**; explicit lag rule documented |
| V7 | Corporate-action handling on known splits | Adjustment layer built before any single-name candidate |
| V8 | Root continuity: FB→META splice; SPX/SPXW; VIX expiries | Registered per-root filters |
| V9 | Partition/session integrity + 2026-03→present refresh | Refresh via Theta Terminal (0.10) or standing live-edge caveat |
| V10 | Provenance mix (ThetaData vs possible IBKR rows) | Quote-quality stats computed per sub-population or conservatively |
| V11 | Empirical spread census → `cost_model_v2` parameterization | Falls back to the assumed width table, flagged provisional |
| V12 | Legacy `options_1min/` overlap-equivalence vs `options_combined` | Archive/delete only after equivalence is understood |

**Group A — runnable today, owned underlying data only, $0:**

| ID | Measurement | Data | Gate |
|---|---|---|---|
| D-014/042 | Gap-continuation effect (open gap ≥ ±0.5%, 30-min confirmation → close) and first-hour-trend continuation (≥ ±0.35% → rest-of-day), SPY/QQQ 2016+; **post-2023 sub-sample reported separately** | 1m bars | Proceed toward the options backtest (V1–V3 permitting) only if underlying effect ≥ **15 bps/trade** net of a 2 bp slippage haircut **in the post-2023 slice**. Else DROP both |
| D-040a | Weekend realized-variance share: Fri-close→Mon-open variance vs 3 trading days' worth | 1m bars | Feeds D-040b |
| D-013/030 | Drawdown-shape census: conditional on regime-downgrade triggers, classify subsequent declines gap vs grind | 1m bars + VIX term | Descriptive; frames both candidates' valley-of-death falsifier |
| D-050a | Realized-vol timing around classifier transitions: RV in the 5 sessions before vs after each transition | 1m bars + classifier log | If RV peaks **before** transitions, the classifier lags and OPT-050 is dead pre-trial |

**Group B — after Phase-1 canonicalization (~$0, on the derived chain tables):**

| ID | Measurement | Gate |
|---|---|---|
| D-003 | Financing ratio distribution: (1M 0.20Δ call premium × 3) ÷ (3M 0.15Δ put cost), monthly, full owned window | Proceed only if ratio ≥ **0.7 in ≥ 60% of months**; else DROP 003 |
| D-005 | Post-pullback put VRP vs unconditional put VRP at 0.30Δ | Proceed only if conditional > unconditional; else DROP 005 |
| D-011 | Entry-day put-IV percentile census at breakdown triggers | Descriptive; if entries systematically land at IV pctile > 80, expect the drift edge to be consumed |
| D-018 | IV-rank state-transition matrix (does IVR>50 persist / lead rising vol?) | Descriptive; frames the conditioning trap |
| D-027 | Skew-percentile state persistence + realized crash frequency conditional on steepness | Descriptive |
| D-029 | Jade-lizard starvation census: frequency the credit>width constraint binds | Proceed only if ≥ **6 qualifying entries/yr** average; else DROP 029 |
| D-033 | Non-overlapping M2/M1 < 0.97 episode count, 2012–2026 (2007–2026 if the ORATS extension is purchased) | **≥ 12 episodes** → Wave 3 backtest; **< 12** → route to forward paper validation, do not spend the historical sample |
| D-037 | Broken-wing-fly starvation census (no-cost entry constraint) | Same 6/yr rule as D-029 |
| D-040b | Friday vs Thursday term-adjusted ATM IV discount | Proceed only if measured Friday discount < **50%** of calendar-day theta differential; else DROP 040 (expected) |
| D-043 | Pin distance: |spot − max-OI strike| behavior, monthly OpEx week vs control weeks (**OI on disk; runs after V6**) | Proceed only if OpEx-week distance is materially smaller; else DROP 043 |
| D-047/030 | Deep-OTM integrity: sample cross-check of smoothed curvature/low-delta marks against owned raw quotes (cross-vendor if ORATS is purchased) | Integrity gate — if the surface diverges materially at 0.05Δ, both candidates need a different mark source |
| D-048 | IV percentile in compressed-RV states (is IV already at its own floor?) | If IV sits at its floor, sellers are not extrapolating → DROP 048 pre-trial |
| D-049 | Non-overlapping episodes where HAR-RV forecast > IV_30 (owned IV, V5-gated), aligned window | **≥ 15 episodes** → Wave 2; fewer → forward paper route |
| D-050b | Entry-day IV percentile at classifier transitions | If mean pctile > **80**, classifier lags the vol → DROP 050 |

---

# 6. Per-candidate specifications

Field key: `structure` / `entry` / `exit` / `sizing` / `gate` / `primitives` / `data` / `cost` / `criteria` (pre-registered GO/NO-GO) / `falsifier` / `notes`. Every candidate contributes **`n_trials = 1`** on execution unless stated. Parameters are FIXED — see §"What this document does NOT do."

**Universal pre-registered criteria** (apply to every strategy backtest unless overridden): report DSR against lifetime N; PBO; regime-sliced attribution (M5); results at 1.0×/±50× cost. A candidate is **not** a survivor if its sign flips inside the cost band, if PBO > 0.5, or if DSR fails at lifetime N. These are necessary conditions, not a GO — capital decisions remain Shuyang's.

## 6.1 Wave 1 — nine trials, authorized after Phase 0 + Phase 1 (canonicalization + Group C)

### OPT-015 — Index Delta-Hedged Short Straddle · F1 · GO · Prior A
- `structure`: short 1 ATM call + 1 ATM put, same expiry, delta-hedged in shares.
- `entry`: P2 select expiry at **30 DTE**; ATM = nearest strike to spot at snapshot.
- `exit`: hold to **7 DTE**; no profit-take, no stop (deliberate — the anchor must be unmanaged to serve as the ladder baseline).
- `sizing`: P4, shared vega budget.
- `gate`: P5 — suspend new entries when state = UNPREDICTABLE.
- `primitives`: P1(ATM), P2, P4, P5, P6(`mode=daily_1545`).
- `data`: derived EOD snapshot (owned) + smoothed/shipped greeks per the P1 rule; own 1m bars for hedges.
- `cost`: 2-leg combo, monthly; hedge slippage from P6's tiered schedule. Modeled drag ~1–2% of premium annually — PASS with margin. Stressed-width exits are the real risk and must use the stressed tier.
- `criteria`: **the anchor.** Report VRP realized per regime state; the ladder comparisons (018, 019) are only interpretable against this. Explicit sub-criterion: report 2012–2019 / 2020–2026 sub-period Sharpes separately (2007–2011 additionally if the ORATS extension is purchased — absent it, the missing-2008 caveat attaches to every short-vol result).
- `falsifier`: index VRP net of hedge slippage and costs ≤ 0 in the current regime.
- `notes`: hedge, marks, and every 1m-derived input share the registered 15:45 snapshot minute — the hedge-time amendment (15:50→15:45, per A1) *is* the registered fix for v1's basis note; do not drift the minute again without a further amendment. M1 required (assignment on the short legs).

### OPT-019 — Forecast-Throttled Short Vol · F1 · GO · Prior A− · **flagship 1m candidate**
- `structure`: OPT-015's structure; units scaled by the VRP estimate.
- `entry`: as OPT-015, with `size_multiplier = clip((IV_30 − HAR_forecast) / IV_30, 0, cap)`, **cap = 2× base vega**. Zero size when the spread ≤ 0.
- `exit` / `gate`: as OPT-015.
- `primitives`: P1, P2, P4, P5, P6(daily), **P9, P10**.
- `data`: chains + greeks + own 1m bars. **Signal window is bounded by own-bar history (2016+), not by the 2012+ owned option history.** Report the aligned window honestly; do **not** backfill the estimator with daily-bar approximations to extend it.
- `cost`: as OPT-015.
- `criteria`: throttled net Sharpe **>** unthrottled OPT-015 on the aligned window, at matched average vega. Also report throttle-vs-VIX correlation (P10 emits it): if the throttle is ~a VIX-level proxy, say so — the "forecast edge" claim fails even if P&L is positive.
- `falsifier`: throttled ≤ unthrottled on the aligned window.
- `notes`: 15:45 truncation is load-bearing here. Frozen HAR spec — no order selection.

### OPT-016 — Iron Condor, Delta-Managed · F1 · GO · Prior A−
- `structure`: short 0.20Δ strangle + long 0.05Δ wings, one expiry.
- `entry`: P2 **30–45 DTE**; P1 for all four strikes.
- `exit`: P3 (50% / 21 DTE) **or** mechanical close on short-strike touch. **No adjustment, ever** — adjustment rules generate researcher degrees of freedom.
- `sizing`: P4. `gate`: none. `primitives`: P1, P2, P3, P4.
- `data`: chains only (cheapest footprint on the slate — no greeks strictly needed beyond strike selection).
- `cost`: 4-leg combo monthly → ~4–8% of collected premium. PASS.
- `criteria`: compare to OPT-015 at matched vega; the wings' cost must be justified by tail reduction, reported as drawdown/CVaR improvement, not Sharpe alone.
- `falsifier`: long-run P&L < OPT-015 at matched vega **with** worse tail statistics.
- `notes`: **known measurement bias — now quantifiable, per A1.** Touch detection at daily snapshots misses intraday touches; the understatement is measurable directly from the owned minute quotes. Report the measured touch-frequency gap alongside results. The registered spec still evaluates exits at the daily snapshot — the measurement is reporting, not a spec change.

### OPT-001 — Delta-Targeted Covered Call Overwrite · F7 · GO · Prior A
- `structure`: long 100 shares/unit + short 1 call at **0.25Δ**.
- `entry`: P2 **30–45 DTE**; P1(C, 0.25).
- `exit`: P3 (50% / 21 DTE), then P7 roll.
- `sizing`: fixed share lots per name. `gate`: P5 — **skip overwrite** when state ∈ {STRONG_BULL, UNPREDICTABLE}.
- `primitives`: P1, P2, P3, P5, P7. `data`: chains + deltas.
- `cost`: 1 leg, ~12 round trips/yr/name → <1% of premium. PASS.
- `criteria`: total return and risk vs buy-and-hold on the same shares; premium captured must exceed upside forfeited over the full regime cycle.
- `falsifier`: realized vol persistently ≥ implied at 0.25Δ strikes; or forfeited upside > premium captured across the cycle.
- `notes`: **M1 is mandatory** — dividend-driven early assignment on short ITM calls is the classic covered-call backtest error. Universe `U_INDEX` + `U_MEGA20`. Book note: overlaps RAMP's long-equity capital; that is a portfolio question, not a research blocker.

### OPT-002 — Cash-Secured Put Wheel · F7 · GO · Prior A
- `structure`: short 1 cash-secured put at **0.30Δ**; on assignment, hold shares and overwrite per OPT-001 (0.25Δ call) until called away; repeat.
- `entry`: P2 30–45 DTE. `exit`: P3 (50% / 21 DTE).
- `sizing`: one unit per name, no averaging down. `gate`: P5 — suspend new put sales in BEAR.
- `primitives`: P1, P2, P3, P5, P7. `data`: chains + deltas.
- `cost`: as OPT-001 plus share transitions. PASS.
- `criteria`: **P&L must be accounted net of the risk-free yield on the secured cash.** The researchable object is premium *in excess of* T-bill yield; an accounting that ignores it flatters the strategy in a positive-rate regime. Report both gross and excess.
- `falsifier`: put-side VRP at 0.30Δ ≤ 0 net of costs and cash yield across a full cycle.
- `notes`: assignment clustering at regime breaks is the risk; the BEAR gate carries the strategy — one registered spec only, no post-hoc with/without comparison.

### OPT-027 — Steep-Skew Put-Spread Harvest · F3 · GO · Prior A−
- `structure`: short 0.25Δ put / long 0.10Δ put, same expiry.
- `entry`: P2 **45 DTE**; only when `skew_25d` > trailing-**2y 70th percentile** (P8-style, backward-looking).
- `exit`: P3 (50% / 21 DTE). `sizing`: P4. `primitives`: P1, P2, P3, P4, P8.
- `data`: chains + `skew_daily`. `cost`: 2-leg index monthly → PASS.
- `criteria`: conditional premium **>** unconditional (same structure, no skew gate) — the gate must earn its place.
- `falsifier`: conditional ≤ unconditional.
- `notes`: D-027 frames this. Skew premium is among the less-decayed premia found in research; the conditioning trap (steep skew = regime label, not mispricing) is the live risk.

### OPT-032 — Contango Calendar Carry · F4 · GO · Prior B+
- `structure`: short 30 DTE ATM straddle / long 60 DTE ATM straddle.
- `entry`: when `term_slope_daily.slope` (M2/M1 ATM IV) > **1.05**.
- `exit`: at front expiry **−7 days**. `sizing`: P4. `primitives`: P1(ATM), P2, P4.
- `data`: chains + greeks, two expiries. `cost`: 4 contract-transits/cycle, index → PASS.
- `criteria`: **requires M4.** Slope-conditional carry must be positive **after** vega mark-to-market — decompose P&L into theta-differential vs vega components and report both. A positive total that is entirely vega is not carry.
- `falsifier`: slope-conditional carry ≤ 0 post-vega-MTM (i.e., slope is compensation, not mispricing).
- `notes`: build M4 here; OPT-038 and the F4 DEFERs reuse it.

### OPT-047 — Rolling Far-OTM Put Ladder · F10 · GO · Prior B+ (as infrastructure)
- `structure`: always-on ladder of three rungs, 3-month **0.05Δ** puts, one new rung per month.
- `entry`: monthly, budget-capped at **40 bps/month** of allocated capital. `exit`/monetization: sell a rung when VIX > **40** or rung delta > **0.30**, then immediately re-strike.
- `sizing`: budget-constrained, not vega-targeted. `primitives`: P1(0.05Δ — **smoothed greeks mandatory**), P2, P7.
- `data`: chains + smoothed low-delta IV. `cost`: ~2.5–4% of premium per rung-roll — small relative to a deliberately budgeted carry cost. PASS.
- `criteria`: **evaluated jointly with the Wave-1 short-vol book, never standalone.** Metric: portfolio Conditional Value-at-Risk improvement per basis point of drag versus the unhedged book. A standalone Sharpe on a tail hedge is a machine for concluding insurance loses money.
- `falsifier`: drag exceeds budget, or monetization triggers never fire across a full cycle containing a crash.
- `notes`: D-047/030 integrity gate applies — verify smoothed 0.05Δ marks against owned raw deep-OTM quotes on a sample before trusting them (cross-vendor if ORATS is purchased). Smoothed source = `iv_smooth` (ORATS SMV or M7). The compressed-VRP finding cuts *for* this candidate: cheaper insurance regime.

### OPT-050 — Regime-Transition Long Vol · F10 · GO · Prior B · **cheapest build on the slate**
- `structure`: long ATM straddle, **21–30 DTE**.
- `entry`: for **5 sessions** following any classifier transition *into* UNPREDICTABLE or *out of* STRONG_BULL.
- `exit`: at 5 sessions or 10 DTE, whichever first. `sizing`: P4, small fixed vega budget.
- `primitives`: P1(ATM), P2, P4, P5. `data`: chains + classifier state log.
- `cost`: episodic 2-leg index → PASS.
- `criteria`: **gated by D-050b** — if mean entry-day IV percentile at transitions > 80, the classifier lags the vol and this is dead pre-trial.
- `falsifier`: as above; or straddle P&L ≤ 0 across the transition census.
- `notes`: this is substantively a test of the classifier's **lead time**, so both outcomes pay: a failure is diagnostic information about RAMP's regime layer. Depends entirely on Phase 0.8 (persisted PIT state log).

## 6.2 Wave 2 — ten candidates, each gated

**Wave 2 opens only if Wave 1 leaves a live short-vol / carry question.** If OPT-015 and OPT-019 jointly establish that index VRP is dead net of costs in the current regime, the F1/F3/F6 block loses its foundation and Wave 2 is cancelled rather than run (see §8).

### OPT-018 — IV-Rank-Gated Short Strangle · F1 · GO · Prior B+ · ladder step 2
- `structure`: short **0.16Δ** strangle, undefined risk. `entry`: P2 **45 DTE**; only when `iv_rank_1y` > **50**. `exit`: P3 (50%/21 DTE).
- `sizing`: **half** the OPT-015 vega budget (undefined risk). `primitives`: P1, P2, P3, P4, P8. `cost`: 2-leg index → PASS.
- `criteria`: gated premium **>** ungated (OPT-015) at matched vega. Clean pre-registered ablation; must run *after* 015.
- `falsifier`: gated ≤ ungated. `notes`: D-018 frames the conditioning trap. Popular practitioner lore with thin rigorous support — informative either way.

### OPT-036 — Range-Gated Iron Condor · F6 · GO · Prior B+ · **flagship 1m candidate**
- `structure`: OPT-016's structure. `entry`: only when (a) state = SIDEWAYS **and** (b) `yz_10d` < 0.8 × ATM IV_30 **and** (c) **≥ 5 sessions from a scheduled CPI/FOMC date** (registered as part of the spec, closing the "IV correctly prices a known event" channel; macro calendar is free).
- `exit`/`sizing`: as OPT-016. `primitives`: P1–P5, **P9**. `data`: chains + own 1m bars + macro calendar.
- `criteria`: gated **>** ungated OPT-016 on the aligned (2016+) window. `falsifier`: gated ≤ ungated.
- `notes`: 15:45 truncation load-bearing. Reuse the Tier-1 Yang-Zhang implementation if it exists.

### OPT-005 — Laddered CSP Pullback Grid · F7 · GO · Prior B · gated by D-005
- `structure`: 3-rung put ladder at **0.30 / 0.20 / 0.10Δ**, 21–35 DTE. `entry`: sell the next rung only after a ≥ **2%** underlying pullback (daily close basis from 1m bars); unwind the ladder in BEAR.
- `exit`: P3. `sizing`: one unit per rung. `primitives`: P1, P2, P3, P5. `cost`: index widths → PASS.
- `criteria`: post-trigger put VRP > unconditional (that is D-005; if the diagnostic fails, this never runs).
- `falsifier`: the trigger is a regime-transition detector, not a mean-reversion signal — visible as ladder deployment into 2008/2020/2022-style declines.

### OPT-006 — Poor Man's Covered Call (LEAPS diagonal) · F7 · GO · Prior B · cost-sensitivity mandatory
- `structure`: long **12–18M 0.80Δ** call + short **30–45 DTE 0.25Δ** call. `entry`/`exit`: short leg per OPT-001; long leg rolled at 6M remaining (P7). `gate`: P5 as OPT-001.
- `primitives`: P1, P2, P3, P5, P7. `data`: chains + greeks — **smoothed source mandatory** (`iv_smooth`: ORATS SMV if purchased, else M7); shipped ThetaData-computed greeks disfavored here (deep-ITM IV error + dividend handling). LEAPS quoted widths are now directly measurable (V11) — the cost sensitivity uses measured, not assumed, widths.
- `cost`: LEAPS round trip ~1.5–3% of option value + monthly short legs → annualized drag ~3–5% worst case. **MARGINAL-PASS — the ±50% sensitivity run is the actual test.**
- `criteria`: net carry **>** OPT-001 on matched notional. `falsifier`: it is not.
- `notes`: vega is the bill for the capital efficiency — a vol crush marks the long leg down exactly when the overwrite is working. Must run alongside OPT-001 (its benchmark).

### OPT-013 — Put Ratio Backspread on Deterioration · F5→F10 · GO · Prior B
- `structure`: short 1 ATM put + long 2 **−1σ** puts, ~zero cost, **60 DTE**. `entry`: on regime downgrade (WEAK_BULL→SIDEWAYS→BEAR transitions) **with** VIX term flattening. `exit`: 21 DTE or regime recovery.
- `sizing`: zero-cost constraint. `primitives`: P1, P2, P5. `data`: chains + greeks + free VIX term structure.
- `criteria`: positive expectancy conditional on the trigger, with the D-013/030 shape census reported alongside.
- `falsifier`: declines are predominantly slow grinds settling in the valley between strikes (the live 2022 pattern).

### OPT-021 — Earnings Vol-Crush Short Iron Fly · F2 · CONDITIONAL · **exit-version decision pending**
- **Version state (per A1):** v1.0 (exit T+1 **open** — the original pure-crush hypothesis) was amended to v1.1 (exit T+1 **near-close**) solely because the open was unmeasurable at the assumed data rung. Owned minute quotes restore v1.0 to feasibility under the *t*/*t+1* fill convention with **stressed-tier open costs**. **Exactly one version enters the queue — decision is Shuyang's, logged before any earnings data is touched. Recommendation: restore v1.0.** The unchosen version is retired in the ledger, not deleted; v1.1 carries day-1 drift exposure v1.0 does not — they are different strategies, and running both is two trials.
- `structure`: short ATM straddle + wings at the **priced expected move**. Defined risk always.
- `entry`: T−1 near-close snapshot, nearest expiry after the print; universe = liquid names with weeklies and priced expected move ≥ **4%** — **universe decision applies** (§1.4): on-disk singles only, or top-up.
- `exit`: per the chosen version. `sizing`: fixed risk per event. `primitives`: P1, P2. `data`: owned chains + Rung-3 earnings dates.
- `cost`: 4 legs × 2 transits, single-name, **stressed tier mandatory** → 10–25% of collected premium. MARGINAL; sensitivity run required.
- `criteria`: **universe filter must be ex-ante** (liquidity, market cap, expected-move threshold). Selecting names by historical beat-the-implied-move rate imports the outcome into the universe and is prohibited.
- `falsifier`: realized moves ≥ implied on average, net of stressed-tier spreads.
- `notes`: research found beat-rates ranging ~25–63% **across names** — the premium is name-dependent and crowded, and published beat-rate statistics are themselves in-sample folklore. Earnings dates are a leakage item (rule 5). Requires M1.

### OPT-044 — Dispersion-Lite · F8 · GO · Prior B+ · cost-sensitivity mandatory
- `structure`: short SPY ATM straddle vs long weight-proportional ATM straddles on `U_TOP6_SPY`, **vega-neutral** at entry.
- `entry`: P2 **30–45 DTE**. `exit`: 21 DTE. `hedging`: P6 `mode=daily_1545` but executed **weekly** per spec. `primitives`: P1, P2, P4, P6.
- `data`: chains + greeks across 7 roots from the owned store. **Availability constraint (A1 addendum, requires decision):** `U_TOP6_SPY` is a *point-in-time* weight ranking, and the 2012–~2018 top-6 (XOM, GE, JNJ, WFC, BRK.B, PG, T, PFE, CVX era) are **largely not on disk** — only the post-~2019 megacap-concentrated composition is fully covered, and META's root starts 2021. Executable window is therefore ~5–6 y, not 13.7 y (σ_SR ≈ 0.41–0.45, not 0.27). Resolve before this candidate runs: (a) re-register the universe as a **fixed on-disk basket** (logged amendment), (b) top-up download the historical constituents, or (c) accept the truncated window and report the binding root. Do not run OPT-044 on a silently-truncated universe.
- `cost`: 14 legs/cycle; 6 tier-1 single-name legs dominate → ~10–20% of premium capture. **MARGINAL-PASS.**
- `criteria`: **report the decomposition** — index-leg P&L vs component-leg P&L separately. With six names this is substantially a bet on six idiosyncratic vol processes plus index concentration; the "dispersion" label must not launder the exposure.
- `falsifier`: implied-realized correlation gap ≤ costs of the 14-leg structure.
- `notes`: research found this among the **least-decayed** premia (long-run gap ≈ 6.9 correlation points) — but correlation → 1 on crash days; the April 2025 near-total-co-movement session is the shape of the tail. Report worst-day P&L prominently.

### OPT-048 — Vol-Compression Breakout Long Straddle · F10 · GO · Prior B · gated by D-048
- `structure`: long ATM straddle, **30–45 DTE**. `entry`: when `yz_10d` < **20th** trailing-2y percentile **and** 20-day price range < **15th** percentile. `exit`: **+75% / −40%** of debit, or 10 DTE.
- `sizing`: `size_by_debit`. `primitives`: P1, P2, P9. `cost`: 2-leg index, episodic → PASS.
- `criteria`: gated by D-048 — if IV in compressed states already sits at its own percentile floor, sellers are not extrapolating and the thesis dies pre-trial.
- `falsifier`: as above; or long-vol bleed exceeds expansion capture (2017-style persistence).
- `notes`: M5 regime slicing mandatory — a long calm regime is the falsifier's natural habitat.

### OPT-049 — Gamma Scalping when Forecast RV > IV · F10 · CONDITIONAL · gated by D-049 (≥15 episodes)
- `structure`: long **30 DTE** ATM straddle, band-hedged. `entry`: when P10's HAR-RV forecast > IV_30 (owned, V5-gated; sign flip of OPT-019's estimator). `exit`: on sign flip back or 10 DTE.
- `hedging`: P6 `mode=band`, band = **±0.25σ** underlying moves, off 1m bars. `primitives`: P1, P2, P9, P10, P6(band).
- `data`: **best structural fit on the slate** — hedge P&L accrues from own 1m bars via the hedge ledger; option marks needed only at entry/exit/terminal, so the owned derived EOD snapshot suffices.
- `cost`: share hedges cheap, **but** hedge frequency spikes exactly when underlying spreads widen — the tiered slippage schedule in P6 is what makes this honest. A flat bps assumption flatters it materially.
- `criteria`: hedge-ledger P&L > theta paid, across the episode census. `falsifier`: it is not; or D-049 finds < 15 episodes → forward-paper route, do not spend the historical sample.
- `notes`: shares P9/P10 machinery with OPT-019 — one build, two candidates.

### OPT-003 — Financed Collar Carry · F7 · CONDITIONAL · gated by D-003 (financing ratio)
- `structure`: long SPY + long 3M **0.15Δ** put + short monthly **0.20Δ** calls. `entry`/`roll`: static calendar, no discretionary adjustment. `primitives`: P1, P2, P7.
- `cost`: ~2 legs/mo + quarterly put roll → PASS. `criteria`: gated by D-003 (ratio ≥ 0.7 in ≥ 60% of months).
- `falsifier`: collar drag > unhedged drawdown improvement across a bear regime. `notes`: research reinforced drafting-time skepticism — financing ratios frequently fail. Expect D-003 to be the end of this candidate.

## 6.3 Wave 3 — twelve candidates (build machinery now, run only on gates)

- **OPT-011 — Bear Put Debit Spread on Breakdown** · F5 · GO · B. Short: mirror of OPT-007 — buy ATM put / sell −1σ put, 45–60 DTE, on state ∈ {BEAR} + 20-day-low breakdown; exits at 21 DTE or +100%/−50% of debit; universe = bottom-decile momentum, max 10. **Design constraint: the signal layer must NOT reuse RAMP's selection machinery** — RAMP's BEAR failure was long-side *selection*, and importing it re-imports the failure. `criteria`: entry-IV richness (D-011) must not consume the drift edge. Genuine diversifier vs the F7/F1 book.
- **OPT-020 — Single-Name VRP Basket** · F1 · CONDITIONAL (parent 015 + earnings-date QA). OPT-015 replicated per name on `U_MEGA10` (≈ the on-disk singles; PIT-ranking check per rule 7), equal vega, earnings windows excluded. `cost`: MARGINAL (8–15% of premium on wider single-name spreads — measurable per name via V11). `criteria`: basket VRP net of spreads **>** index VRP, else OPT-015 dominates and this dies. **QA condition:** validate exclusion dates against a second source on a sample; a single mis-dated print inside an "excluded" window contaminates the ambient-premium claim.
- **OPT-028 — Skew-Extreme Risk-Reversal RV** · F3 · GO · B. 60 DTE; skew > 90th pctile → sell put/buy call at 0.25Δ; < 10th → reverse; P6 daily hedge (15:45). `criteria`: **verify realized residual delta stays small** — otherwise the result is a directional trade mislabeled as a skew trade. Needs the hedge-ledger module from Wave 1.
- **OPT-030 — Crash-Zone Broken-Wing Put Butterfly** · F3 · GO · B. 60–90 DTE; long 1×−1σ, short 2×−1.5σ, long 1×−2.5σ, ~zero cost, rolled monthly. **Smoothed low-delta source mandatory (`iv_smooth`: ORATS SMV or M7; D-047/030 integrity gate).** `cost`: MARGINAL — 4 OTM legs, exit friction in stressed tapes is the bill (now measurable in-era via V11). `falsifier`: terminal distributions concentrate at −1.5σ (2022-style grind) rather than bimodal.
- **OPT-033 — Backwardation Reverse Calendar** · F4 · CONDITIONAL · gated by D-033 (≥12 episodes, 2012–2026 owned window; 2007+ with ORATS). Long 30 DTE / short 60 DTE ATM straddles when slope < **0.97**; exit on re-normalization or front −7d. If < 12 episodes → **forward paper**, per the saturation rule.
- **OPT-038 — Double Diagonal Income** · F6 · GO · B−. Short 30 DTE 0.25Δ strangle / long 60 DTE 0.15Δ strangle; short legs rolled monthly, long legs re-struck quarterly. Reuses M4. `criteria`: must beat OPT-016 at matched risk.
- **OPT-039 — Band-Recentered Short Straddle** · F6 · GO · B−. 30 DTE short ATM straddle; recenter when the underlying exits a **±0.75σ** band on a **daily close** basis (deliberately not intraday — a registered design choice, retained now that it is no longer a data constraint); **max 1 recenter**, then hard stop. `criteria`: must beat OPT-015 **within SIDEWAYS-classified periods specifically** — M5 delivers this directly. The recenter cap is the discipline separating this from martingale adjustment folklore.
- **OPT-043 — Expiration-Week Pin/Charm Short Straddle** · F9 · CONDITIONAL · gated by **V6 → D-043**. Monthly OpEx only (weekly pinning is diluted); short straddle at the **T−1-known** max-OI strike Wednesday, exit Friday 15:00. OI is on disk (`oi_eod`); **blocked until V6 reports the join semantics — same-session OI at decision time is a hard leak regardless.** **Prior-art duty (per A1):** reconcile against the shelved OpEx pinning strategy — reuse its GEX/calendar machinery where sound (152 passing tests), and record its 2025 backtest (18 trades, −$415, estimated gamma/OI) as a prior tested trial in the lifetime ledger. OPT-043's max-OI-strike hypothesis is distinct from that GEX-directional design; both count trials.
- **OPT-041 — 0DTE Post-Opening-Range Iron Condor** · F9 · CONDITIONAL · **V1–V3 + the strict cost bar.** Enter 10:30 after the opening range (decide bar *t*, fill bar *t+1*), wings beyond ±1× opening range, hold to 15:55 or short-strike touch. Data on disk 2012+ — **segment and report by expiry era** (Fri weeklies full-history → M/W ~2016 → daily ~2022; the daily-expiry regime is only ~4 y). `cost`: worst profile on the slate — 8 contract-transactions/day, modeled drag ~10–15% of daily credit, every day. **MARGINAL-to-FAIL.** Explicitly a **research-to-disprove** candidate: the valuable output is a clean measurement of how dead intraday VRP is after costs. The longer window now *contains* Aug 2015, Feb 2018, Mar 2020 — a positive aggregate that evaporates in those eras is the expected signature of a dead edge.
- **OPT-007 / OPT-008 / OPT-009 — the long-momentum trio** · F5 · CONDITIONAL · **gated on the PR6 plain-momentum re-baseline** (already RAMP's registered Phase-4C go/no-go). If PR6 fails at matched cadence, all three DROP together with zero trials spent. Specs: 007 = bull call debit spread, ATM/+1σ, 45–60 DTE, top-decile 12-1 momentum + 20-day-high breakout, max 10 positions, exit 21 DTE or +100%/−50%; 008 = ZEBRA, buy 2× 0.70Δ calls / sell 1 ATM call, 60–90 DTE, exit on regime downgrade; 009 = risk reversal, sell 0.25Δ put / buy 0.25Δ call, 45 DTE, max 5 names. `falsifier` (shared): spread P&L < delta-equivalent stock P&L on matched signals — i.e. the wrapper only added cost. 009 is the undefined-risk member and the first to cut if only some survive.

## 6.4 Wave 4 / shelf — eight candidates (specified, not queued)

Each has a named unlock; **none justifies spend on its own.**

- **OPT-014 — Opening-Range Gap Same-Day Verticals** · unlock: D-014/042 post-2023 slice ≥ 15 bps **then** V1–V3 pass. Enter ~10:00 on gap > ±0.5% with 30-min confirmation (decide bar *t*, fill bar *t+1*), 1-wide debit vertical, hard stop 15:45. Quote-valid bars only; segment by expiry era.
- **OPT-042 — 0DTE First-Hour-Trend Debit Spread** · unlock: same shared diagnostic, then V1–V3. First-hour return ≥ |0.35%| → 1-wide vertical at 10:30, exit 15:45. Expected per research: post-2023 decay shows in the slice → DROP.
- **OPT-040 — Weekend Theta Capture** · unlock: D-040b (Friday discount < 50% of calendar-day theta differential). **Expected DROP** — research indicates Friday IV generally pre-discounts the weekend. Both registered legs (Friday 15:45 entry, **Monday 09:45 exit**) are feasible on owned minute quotes as written; v1's forced-amendment note is retired.
- **OPT-024 — Post-Crush Overshoot Short Strangle** · unlock: parents 020 ∧ 021 both clear. T+1 close entry, 21–30 DTE 0.16Δ strangle. `falsifier`: post-print VRP ≤ ambient single-name VRP (then 020 dominates) — uninterpretable before 020 exists.
- **OPT-025 — Macro-Event Index Straddle** · **registered amendment stands:** exit T+0 **near-close** (CPI 08:30 / FOMC 14:00 both resolve intraday; the exit mark is now directly verifiable from owned minute quotes). Demoted to null-anchor — research reiterates implied event moves ≥ realized on average. Regime-conditional slicing of the result is reporting, not a new trial.
- **OPT-026 — Sector-Sympathy IV Fade** · **research-to-disprove; universe decision applies** (peer sets are thin at 31 roots). The falsifier is currently live: 2023–26 AI-capex tapes show peer moves on leader prints are frequently real information transfer, not sympathy overpricing. A confirming result would be surprising and must be checked hardest.
- **OPT-031 — Inverted-Call-Skew Fade** · unlock: **live IBKR fill data** establishing achievable spreads in squeeze names **and the universe decision** (squeeze names are mostly off-disk). The mechanism (lottery-preference overpricing of OTM single-name calls) is documented; the cost model, not the mechanism, is the blocker — modeled drag frequently exceeds the 0.15Δ spread's premium.
- **OPT-045 — Same-Sector IV-Percentile Pairs** · unlock: OPT-044 clears; **universe decision applies** (same-industry liquid pairs barely exist at 31 roots). 80/50 IV-percentile gates, delta-hedged both legs, no leg within 10 days of either print. If the clean index-vs-component version of the correlation premium fails, the noisier pairwise version is not the rescue.

## 6.5 DEFER — seven candidates (sequential testing under pre-registered rules)

These were flagged as near-duplicates in slate v1 and are **not** independent hypotheses. Each runs **only** if its parent(s) clear validation; on execution each increments `n_trials` normally. The rules are fixed now, before any result — that is what makes this legitimate sequential testing rather than results-conditioned selection.

| ID | Parent(s) | Marginal question it answers | Extra gate |
|---|---|---|---|
| OPT-004 Covered Strangle | 001 ∧ 002 | Does the joint structure with its own regime kill-rules beat the sum of the parents? | — |
| OPT-017 Iron Butterfly | 016 | Profit-zone width vs touch frequency at max theta | — |
| OPT-029 Jade Lizard | 002 ∧ 027 | Does the credit>width constraint create a genuinely better skew+carry blend? | D-029 starvation (≥6 entries/yr) |
| OPT-034 Slope-Switched Program | 032 ∧ 033 | Is term slope a tradable factor rather than two episodic trades? | Defers with 033 if 033 routes to paper |
| OPT-035 Double Calendar | 032 | Do extra legs buy real path tolerance, or just cost? (strictly cost-worse: 8 transits/cycle) | — |
| OPT-037 Directional-Tilt BWB | 016/017 | Does the drift tilt add anything vs a symmetric fly? | D-037 starvation |
| OPT-046 Index-vs-Single-Name VRP Switch | 015 ∧ 020 | Does the hysteresis-banded switch beat static 50/50? | Dies automatically if either parent dies |

---

# 7. Ledger & DSR protocol

## 7.1 Schema

```
strategy_trial_ledger:
  id                     str      # OPT-0XX
  spec_version           str      # e.g. "v1", "v1.1" (021, 025 are amended)
  mechanism_family       str      # F1..F10
  emitted_at             date     # 2026-07-24 for all 50
  status                 enum     # emitted | queued | diagnostic_only | tested | deferred | dropped | no_go | paper_forward
  wave                   int
  data_window_start      date     # actual window the trial consumed
  data_window_end        date
  data_rung              str      # rung_1 | rung_2a | ...
  n_trials_contribution  int      # 0 until tested; 1 on execution
  tested_at              date
  parent_ids             list     # DEFER lineage
  gate_ids               list     # D-0XX diagnostics that gated it
  amendment_log          list     # registered spec changes with timestamps and rationale
  notes                  str
```

**Rules:** (1) `emitted ≠ tested`. All 50 are emitted today; DSR's N counts trials **evaluated against the overlapping data window**, so generated-but-never-tested candidates are not draws from the sample. (2) `n_trials_contribution` flips 0→1 **on test execution**, not on queueing. (3) Diagnostics (`diagnostic_only`) contribute **0** — they measure data properties, not strategy returns, and select nothing on P&L. (4) Any later parameter sweep on a candidate multiplies its contribution by the sweep cardinality and is re-appended. (5) Amendments are logged before the affected test runs; the superseded version is retired with its own row, never overwritten in place.

## 7.2 Null best-of-N arithmetic (Bailey–López de Prado) — v2 windows

σ of an annualized Sharpe estimate ≈ 1/√years. On the owned window (**13.7 y**, 2012-06 → 2026-02), σ_SR ≈ **0.27**; with the ORATS extension (19.5 y), σ_SR ≈ 0.23. Expected maximum in-sample Sharpe under zero edge:

| Stage | N (this slate) | E[max Sharpe | null], 13.7 y | with ORATS ext. (19.5 y) |
|---|---|---|---|
| Wave 1 | 9 | ≈ **0.41** | ≈ 0.34 |
| + Wave 2 | ≈ 19 | ≈ **0.51** | ≈ 0.42 |
| + Waves 3–4 | ≈ 31–38 | ≈ **0.56–0.59** | ≈ 0.47–0.49 |

v1's separate intraday-window penalty is **retired** — intraday and EOD candidates now share the 13.7-y window. Its replacements: **(i) the missing-2008 caveat** — the owned window contains no GFC-class regime, right-censoring the loss distribution of the short-vol core; curable at EOD granularity only by the ORATS extension, permanent for intraday-native candidates; **(ii) per-root windows** — later listings shrink further (PLTR ≈ 5.7 y → σ ≈ 0.42; COIN ≈ 4.7 y → σ ≈ 0.46); each candidate's DSR uses *its own* window, and mixed-window baskets (020) report the binding shortest root.

**Lifetime reconciliation (CC task, Phase 4):** true N adds Homeguard's prior ledger history (RAMP Phase-3/4 variants and everything else) **against overlapping windows** — the equities-daily trials overlap the 2015+ portion of the options window, and **the shelved OpEx pinning strategy's 2025 backtest (18 trades, estimated gamma/OI, −$415) is a prior tested trial to be recorded**, per A1. Honest DSR counts shared-window trials, not calendar-disjoint ones. Illustrative only: lifetime N = 150 on the long window pushes the null max toward ≈ 0.7 at σ ≈ 0.27. **The real count comes from the repo ledger, not from this document and not from memory.**

**This slate's delta:** +50 emitted; ≤ 9 tested at Wave 1; 1 NO-GO and 3 DROP recorded as emitted-not-tested.

---

# 8. Kill switches and the reiterate decision

Registered now, before results exist:

1. **Wave-1 anchor failure.** If OPT-015 and OPT-019 jointly establish that index VRP is dead net of costs in the current regime, **cancel Wave 2's short-vol block** (018, 036, 005, 003, and by extension much of F3/F6). Do not generate more short-vol variants — that is searching, not researching. Reiteration means *new mechanism families* (financing/rates-adjacent structures, cross-asset vol) or new data.
2. **PR6 gate failure.** 007/008/009 drop silently, zero trials spent. Already wired.
3. **Diagnostic failures** (D-003, D-005, D-029, D-033, D-037, D-040b, D-043, D-048, D-049, D-050b) drop or reroute their dependents pre-trial. This is the cheapest budget protection in the chain.
4. **Deep-OTM integrity failure** (D-047/030): if smoothed marks diverge materially from owned raw quotes (cross-vendor if ORATS is purchased) at 0.05Δ, both 047 and 030 need a different mark source before running — a data problem, not a strategy verdict.
5. **Zero survivors after CPCV/PBO/DSR.** A valid, pre-registered outcome. It does **not** trigger re-running anything with adjusted parameters. STOP remains Shuyang's call, not the analysis's.

**Prohibited responses to disappointing results** (stated explicitly because they are the natural next move and they are the overfit): re-running a candidate with adjusted parameters; adding "just one more" variant of a failed mechanism; loosening a pre-registered criterion after seeing the number; reclassifying a tested candidate as a diagnostic to avoid the trial count; extending a window to capture a better period.

---

# 9. CC implementation phases

**Phase 0 — Repo reconciliation.** §0 table (eleven items, incl. 0.9–0.11 per A1). Blocking. Output: written report with paths, line numbers, SHAs. **Do not proceed without review.**

**Phase 1 — Data canonicalization + Group C verification (~$0).** No purchase. Build `options_chain_1m` (canonicalized) and the derived `options_chain_eod` at the registered 15:45 snapshot (§1.2), partitioning per 0.6, DuckDB access, derived dailies. Run the **V-battery (V1–V12)** and publish the data-quality report — V6 (OI join), V1–V3 (quote validity), and V7/V8 (corporate actions, root continuity) are release gates for the candidates that depend on them. Execute the 2026-03 → present refresh if 0.10 finds the subscription usable. Implement the snapshot-symmetry guard and the PIT rules (§1.3) **centrally** — not in strategies. If the ORATS purchase is taken, ingest it in parallel into the same tables (`source='orats'`); nothing waits on it. Output: canonicalization spec + V-battery report.

**Phase 2 — Primitives and harness.** P1–P10, M1–M7 (M7 per the build-vs-buy fork), `cost_model_v1 → v2` (v2 parameterized from V11's census). Unit-test M1 (early assignment) against hand-worked dividend cases and M2 (marks) against known wide-spread days pulled from the owned data. **The hedge ledger's tiered slippage schedule must be built from own Alpaca/IBKR execution data, not assumed.**

**Phase 3 — Wave 0 diagnostics.** Group A today on owned underlying data (it need not wait for Phase 1); Group B after Phase 1 on the derived tables. Report every gate outcome explicitly, including the ones that drop candidates — a diagnostic that kills a candidate is the highest-value output in this chain, not a failure.

**Phase 4 — Ledger + validation wrapper.** Ledger per §7 at the location Phase 0.3 identifies; reconcile lifetime N against the repo's prior history **including the OpEx pinning strategy's 2025 backtest** (A1). Wire CPCV embargo ≥ max holding period, PBO, DSR.

**Phase 5 — Wave 1 execution.** Nine candidates, in the §6.1 order (015 and 019 first — they anchor everything). Standard output per candidate: regime-sliced attribution, cost sensitivity at 1.0×/±50% on `cost_model_v2`, DSR at lifetime N, PBO, and the candidate's own pre-registered criterion evaluated as stated. **Stop at Wave 1.** Later waves require their gates and a review.

**Reporting discipline throughout:** report what the code does, not what the spec says it should do; if an implementation diverges from this document for any reason, surface the divergence in the results header rather than in a footnote. If a parameter proves unimplementable as written, stop and report — do not substitute.

---

# Appendix A — Candidate index by verdict

**GO (21):** 001, 002, 005, 006, 011, 013, 015, 016, 018, 019, 027, 028, 030, 032, 036, 038, 039, 044, 047, 048, 050
**CONDITIONAL (18):** 003, 007, 008, 009, 014, 020, 021, 024, 025, 026, 031, 033, 040, 041, 042, 043, 045, 049
**DEFER (7):** 004, 017, 029, 034, 035, 037, 046
**Specified here: 46.** DROP (3): 010, 012, 022 — Appendix C. NO-GO (1): 023 — excluded.

# Appendix B — Data-requirement matrix (condensed, v2)

| Requirement | Candidates |
|---|---|
| Owned derived EOD snapshot (chains + greeks) | 001, 002, 003, 005, 006, 011, 013, 015, 016, 017, 018, 027, 028, 029, 030, 032, 033, 034, 035, 037, 038, 039, 044, 046, 047, 048, 050, 004 |
| + own 1m underlying bars (signal/hedge critical) | 005, 013, 019, 036, 039, 048, 049 |
| + free calendars (VIX term / macro) | 013, 025, 036 |
| + classifier state log (Phase 0.8) | 001, 002, 005, 011, 013, 015, 036, 050 (+ all P5 gates) |
| Owned 1m option quotes — **pending V1–V3** | 014, 041, 042 (+ 040's Monday-open exit, + 021's T+1-open v1.0) |
| Rung 3 (earnings dates) | 020, 021, 024, 026, 045 |
| OI — **on disk, V6-gated** | 043 |
| Smoothed surface (`iv_smooth`: ORATS SMV or M7) | 006, 027, 030, 047 (+ P1 low-delta rule generally) |
| Live IBKR fill data | 031 |
| External workstream gate (PR6) | 007, 008, 009 |
| **Universe decision** (breadth beyond 31 roots) | 021, 026, 031, 045 (+ 001/002/020 beyond the on-disk megacaps) |

# Appendix C — DROP record (do not resurrect without a new mechanism argument)

- **OPT-010 Call Ratio Backspread on Breakout** — no evidence breakout right-tails are fat enough to pay the valley-of-death; the edge claim was payoff-shape cleverness, not mechanism.
- **OPT-012 Bearish Risk Reversal** — dominated ex ante by OPT-011's defined-risk form; bear-market rallies (the most violent) are precisely this structure's short leg.
- **OPT-022 Pre-Earnings IV Run-Up (long)** — run-up exists, capture net of theta and single-name spreads is unreliable; bounding the family is not worth a draw from the finite sample.

Ledger status for all three: `dropped`, `n_trials_contribution = 0`. If a genuinely new mechanism argument appears, that is a **new** candidate with a new ID, not a resurrection.

---

# Honesty block

Nothing in this chain has observed a backtest return. Every verdict, wave assignment, and prior tier is a feasibility-and-mechanism judgment made blind. "GO" means *worth one draw from a finite sample*, nothing more, and it is not a prediction. **No prior was upgraded by the A1 data correction — better feasibility is not better edge.**

The **cost model remains the weakest quantitative input**, now with a repair path: V11's empirical spread census (owned quotes) parameterizes `cost_model_v2` before Wave 1; fill-fraction conventions stay assumptions until own IBKR fill statistics replace them the moment any options strategy trades live — at which point every prior result is re-run. **Data-quality trust is conditional on the V-battery:** shipped IV/greeks carry the registered caveats (Black-Scholes, dividends off, deep-ITM/OTM error; gamma absent at minute level); OTM quote coverage, OI-join semantics, provenance mix, and corporate-action handling are unverified until Group C reports, and any result predating a relevant V-item is provisional. Vendor facts still open: ThetaData subscription status/tier (Phase 0.10); ORATS pricing re-verified at checkout if purchased. Decay evidence is mixed in quality, unchanged from v1: correlation-premium, index-performance, and 0DTE-share figures trace to named studies and index providers; VRP-compression and weekend-effect readings lean on practitioner sources; thin-evidence verdicts (026, 040, 022) already discount conservatively.

**Amendment state:** OPT-021 carries a pending exit-version decision (one version only; recommendation v1.0-restore); OPT-025's amendment stands; the P6 hedge-time alignment (15:50→15:45) is logged per A1 — all registered before any data contact by the affected tests. **Eleven Phase-0 integration points remain unverified against the repo** and are the reason Phase 0 blocks everything.

Zero survivors after validation is a legitimate, pre-registered outcome of this entire slate.

*End of CC handoff spec v2 (v1 retired per Amendment A1). This document, plus slate v1.1, feasibility screen v2, and Amendment A1, is the pre-registration of record. Deviations require a logged amendment before the affected test runs.*
