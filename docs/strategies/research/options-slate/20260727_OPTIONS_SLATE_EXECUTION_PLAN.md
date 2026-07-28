# Options Slate — Execution Plan (Homeguard-side)

**Date:** 2026-07-27
**Author:** CC (Claude Code), main loop
**Status:** PLAN. No strategy backtest has been run. No P&L has been observed.
**Governing pre-registration:** the doc chain delivered 2026-07-24/25 (see §1), committed alongside this file.

This document is the Homeguard-side execution plan for the 50-candidate equity options
slate. It does three things the incoming doc chain could not do, because the chain was
written without repo access:

1. Answers **"can we even test this with our data?"** with direct measurement, not assertion.
2. Records **divergences** where the chain's assertions about the repo/disk are wrong. Per the
   chain's own ground rule ("code and disk are ground truth; reality wins and you report it"),
   the divergences below override the chain.
3. Maps the chain's five phases onto **Homeguard's actual rules** — `strategy-lead` routing,
   the fill-logging mandate, `RunStatus`, and the existing ledger — which the chain does not
   know about and which are non-negotiable here.

---

## 1. Which documents govern

Eight documents were delivered. Three are **retired**; five govern.

| Document | Date | Status |
|---|---|---|
| `options_strategy_slate_v1.md` | 07-24 | **RETIRED** -> superseded by v1.1 |
| `options_slate_feasibility_screen_v1.md` | 07-24 | **RETIRED** -> superseded by v2 |
| `options_slate_cc_handoff_spec_v1.md` | 07-24 | **RETIRED** -> superseded by v2 |
| `options_docchain_amendment_A1.md` | 07-25 | **GOVERNS** — correction record + V-battery definition |
| `options_strategy_slate_v1_1.md` | 07-25 | **GOVERNS** — 50 candidate definitions, fixed params, priors |
| `options_slate_feasibility_screen_v2.md` | 07-25 | **GOVERNS** — verdicts (21 GO / 18 COND / 7 DEFER / 3 DROP / 1 NO-GO), waves |
| `options_slate_cc_handoff_spec_v2.md` | 07-25 | **GOVERNS** — buildable spec, primitives P1-P10, modules M1-M7, phases |
| `options_phase01_cc_work_order.md` | 07-25 | **GOVERNS, and is the immediate instruction set** — self-contained Phase 0 + Phase 1 |

Reading order for anyone picking this up: work order -> A1 -> screen v2 -> spec v2 -> slate v1.1.

**The work order is the active task.** It explicitly supersedes any Phase-0/1 instruction in
spec v1 and is self-contained. Everything in §4 below is scoped to it.

---

## 2. Verdict on testability: YES, and the data is better than the chain assumed

The chain's central open question was whether the owned store is **quote-bearing** (rows exist
independent of trades) or merely trade-print bars. If trade bars, the chain's own prohibition
re-binds and roughly a dozen candidates revert to blocked.

I ran the V1/V2/V3/V5 measurements on a real sample. **All four pass their registered gates,
by wide margins.**

### Preliminary V-battery results — SPY, 2024-01, 614,400 sampled rows

| ID | Measurement | Registered gate | Measured | Result |
|---|---|---|---|---|
| **V1** quote population, \|delta\| [0.05,0.15] | index roots >=95%/yr | **100.0%** | **PASS** |
| V1 | \|delta\| [0.15,0.35] | >=95% | **100.0%** | **PASS** |
| V1 | \|delta\| [0.35,0.65] | >=95% | **100.0%** | **PASS** |
| **V2** rows with `volume=0` AND valid quote | >=60% PASS, <10% FAIL | **77.4%** | **PASS** |
| **V3** crossed quotes (`bid>ask`) | <0.1% | **0.000%** | **PASS** |
| V3 | zero-bid frequency | document, do not repair | 11.7% | documented |
| **V5** `implied_vol` non-null | >=90% | **100.0%** | **PASS** |
| V5 | `delta`/`theta`/`vega` non-null | >=90% | **100.0%** | **PASS** |
| V5 | `gamma_eod`/`open_interest_eod` non-null | — | **100.0%** | populated |
| snapshot | `15:45:00` bar exists | required by spec | **present** | **feasible** |

**Interpretation.** 88.5% of sampled rows have zero volume, and 77.4% have zero volume *with a
valid two-sided quote*. Rows therefore exist independent of trades: these are **quote-bearing
minute records**, not trade prints. The chain's trade-bar prohibition is satisfied by the disk
set itself. Perfect quote coverage in the 0.05-0.15 delta bucket is the specific result that
unblocks the OTM-dependent work.

**Scope caveat, stated plainly:** this is **one root, one month, one year**. It is a strong
positive signal, not the V-battery. The registered gates are *per root x year*. The full sweep
(31 roots x 15 years) is Phase 1b and is the thing that actually clears the gate. Early years
(2012-2015) and thin single names (MSTR, COIN, IBIT) are where I expect trouble, not SPY 2024.

### V6 — the hard gate — is RESOLVED, and it is a leak

The chain flagged V6 as a hard gate that blocks all OI/gamma work (OPT-043) until answered.
**Answered, from code plus data:**

- `scripts/data/combine_options_data.py:179` derives `_date` by slicing the intraday timestamp,
  then joins EOD gamma/OI on that same date (`intraday_df.join(eod_df, on=join_cols, how="left")`,
  line 239). **The join is same-day.**
- Confirmed on data: OI is **constant across all minutes** of a session (0 of 18,879
  contract-sessions show intraday variation) and **varies day to day** (SPY 2024-02-16 expiry:
  15529 -> 16070 -> 16417 -> 15413 -> 16289 -> 16671). So it is a genuine daily series joined
  cleanly onto every minute of its own session.

**Consequence:** session *t*'s rows carry session *t*'s end-of-day OI. Exchange OI for session
*t* is not published until the following morning. Reading `oi_eod` at the 15:45 snapshot on
session *t* is therefore **a hard lookahead leak**.

**Ruling (registered here, before any OI-conditioned test runs):** every use of `oi_eod` and
`gamma_eod` **must lag >= 1 session**, enforced in the primitive layer, not per-strategy. This
matches the chain's own T-1-known rule; the measurement confirms the rule is load-bearing
rather than precautionary. OPT-043 is unblocked *conditional on the lag being enforced in code*.

---

## 3. Divergences — where the chain is wrong about this repo

Per the work order's ground rule 1, these override the chain. Each is evidence-cited.

| # | Chain asserts | Reality | Consequence |
|---|---|---|---|
| D1 | `infra_patterns.md`'s `OptionsDataLoader` column list is "suspected fiction" | **Substantially correct.** `src/strategies/options/data_loader.py:20-26` renames `bid_close->bid`, `ask_close->ask`, `gamma_eod->gamma`, `open_interest_eod->open_interest`, `underlying_px->underlying_price`, and derives `mid_price`, `expiry`, `days_to_expiry`, `option_type`, `date`, `time` | Doc is fine. **But** the loader's rename of `gamma_eod->gamma` and `open_interest_eod->open_interest` **erases the `_eod` marker that signals the leak**. This is a live leak-enabling footgun — a strategy reading `open_interest` has no cue it is same-day EOD data. Must be addressed in canonicalization (keep `_eod` suffix, per spec §1.2). |
| D2 | Registered snapshot minute is 15:45:00 ET | `OptionsDataLoader.get_eod_chain()` hardcodes **`time(16, 0)`** (`data_loader.py:51`). The 16:00 bar exists on disk | Existing loader default contradicts the registered snapshot. Do not reuse `get_eod_chain()` for slate work; build the 15:45 snapshot as a guarded function per spec §1.3 rule 1. |
| D3 | Schema is "20 cols (CC) vs 21 with leading `symbol` (DATA_INVENTORY)" — reconcile | SPY 2024-01 has **20 columns, no `symbol`**. `docs/reference/DATA_INVENTORY.md:248` claims 21 with `symbol` | DATA_INVENTORY is stale *or* the layout varies by vintage. V4 sweep must check per root-year, not assume. |
| D4 | Cost model: build `cost_model_v1` fresh; "no strategy may define its own costs" | **An options cost model already exists**: `src/backtesting/costs/options.py`, per methodology §4.5, with an alpha table keyed by liquidity class (`very_liquid` 0.4, `liquid_etf` 0.6, `single_stock_atm` 0.85, `wings_illiquid` 1.1) and IBKR $0.58/contract | **Extend, do not fork** (Phase 0.7). **Unit trap:** repo alpha = fraction of the **half-spread**; the chain's convention = fraction of **full width**. 25% of width == alpha 0.50. Reconcile units explicitly or every cost number is off by 2x. |
| D5 | Yang-Zhang: reuse if it exists | Exists: `src/features/volatility.py` | P9 reuses it. Do not fork. |
| D6 | HAR-RV is a new build (P10) | **Already exists**: `src/backtesting/vol/har_rv.py`, plus `atm_iv.py` (ATM IV extraction) and `vrp_strategy.py` | See §3.1 — this is prior art, not just reuse. |
| D7 | Ledger "may not exist; create per §7" | **Exists**: `output/experiments.duckdb`, `runs` table (496 rows) + `return_streams` (300,288 rows), with `combinations_in_run`, `combinations_project`, `git_sha`, `config_sha`, `cost_sensitivity`, `regime_breakdown` | Use it. Do not create a parallel ledger. **But** see §3.2 — the cumulative counter is not populated. |
| D8 | `docs/research/` is the deliverable location | `docs/research/` is **gitignored** (`.gitignore:280-281` only un-ignores `docs/strategies/**`); chain deliverables relocated to `docs/strategies/research/options-slate/` | Cosmetic; noted for the record. |

### 3.1 Prior art the chain does not know about: an existing VRP strategy

`src/backtesting/vol/vrp_strategy.py` is documented as *"#28 VRP signal expressed as a
VRP-sized short-VX1 stream... VRP = ATM implied vol - HAR realized-vol forecast"*, with a
pre-registered `_TOP_BAND = 0.5` percentile gate and a causal percentile-rank helper.

This is **the same mechanism as OPT-015/019** — the Wave-1 anchors — expressed on VIX futures
rather than SPY options. It was built during the futures SP-D campaign. Implications:

- **Machinery reuse:** `atm_iv.py` + `har_rv.py` are exactly P10's inputs. Build on them.
- **Trial-count duty:** if that VRP stream was ever graded, it is a **prior tested trial on an
  overlapping window** and must enter the lifetime N. The chain's ledger reconciliation task
  (Phase 4) is larger than it knows.
- **Interpretation duty:** OPT-015/019 results must be read against whatever that VRP work
  already concluded. Re-discovering a known negative and reporting it as new is not a finding.

### 3.2 The honest-N problem is worse than the chain estimates

The chain computes its DSR hurdle from `N = 9` at Wave 1 (`E[max SR | null] ~ 0.41` on 13.7 y)
and hand-waves lifetime N as "illustratively 150."

Measured reality:

- `runs` holds **496 rows**: 298 futures, 144 equity, 50 fx.
- `max(combinations_project)` = **0** and `sum(combinations_in_run)` = **4**. The cumulative
  project trial counter **is not populated**. The field the chain assumed would supply honest N
  is empty.
- **No options runs are recorded at all** — the 2026 RAMP options campaign (whose own
  `20260402_statistical_validation.json` reports `n_trials = 18`) and the OpEx pinning backtest
  are absent from the ledger.
- Prior campaigns overlap the options window heavily: `CarverMomentum` 207 runs (2010-2026),
  `RAMP-V31` 47 runs (2017-2026), `FuturesCarry` 44 (2010-2026), `FxCarry`/`FxTSMOM` (2011-2026).

**Consequence:** lifetime N must be *reconstructed*, not read. Until it is, no DSR number
computed for any options candidate is trustworthy. This is a Phase-4 blocker on the *verdict*,
not on the build — but it must land before Wave 1 results are graded, or the gate is
meaningless. Reconstruction inputs: dedup the 496 ledger runs to distinct specs, plus the
RAMP-options 18, plus OpEx 1, plus the futures variation campaign (my session notes put honest
N at 94 there), plus this slate's tested count.

Directionally: honest N is in the **low hundreds**, not 9. At sigma_SR ~ 0.27, N ~ 200 puts the
null expected max Sharpe near **~0.75**. A Wave-1 candidate posting 0.6 is *below the null
max*, not a discovery. This should be settled before anyone sees a number.

---

## 4. The plan

Phases follow the chain (work order + spec v2 §9), amended for Homeguard rules.

### Phase 0 — Repo reconciliation · **COMPLETE 2026-07-27**

All eleven items resolved, none ABSENT. Full report:
`20260727_options_phase0_reconciliation.md`. Headlines:

- **0.8 classifier PIT — the existential item — passes.** No persisted state log, but
  `analyze_regime_history()` performs a **causal replay** (`index <= date`), and
  `_calculate_vix_percentile` uses a trailing 252-row window of an already-truncated frame,
  so there is **no full-sample lookahead**. Regime gates are backtestable; **Wave 1 does not
  shrink**. Requires: materialize `regime_state_daily` once (O(n^2) replay, 8 consumers), and
  state the data-vintage caveat on results.
- **New Phase-1 prerequisite: VIX spot is not in local storage.** `/h/Stock_Data/alt_data/vix/`
  holds only `vx_curve.parquet` (VIX *futures* term structure, 2013-05 -> 2026-07). The regime
  detector's VIX index input is *fetched* via `src/utils/vix_provider.py` / yfinance `^VIX`.
  A fetched series is not reproducible -> materialize VIX spot locally before replay.
- **0.1 — use the FX/futures runner pattern, not the equity registry.** `src/strategies/registry.py`
  serves config-driven single-instrument equity work; FX and futures use dedicated runners on
  `walkforward_common`. Options (multi-leg, chain data) belongs with the latter — and that path
  already carries the `FillSink` convention.
- **0.6 — DuckDB is single-writer.** Parallel per-root V-battery jobs must shard to parquet and
  do one serialized ledger append; concurrent writes to `experiments.duckdb` will collide.
- **0.9 — `OptionsDataStore` is confirmed dead** (`chains/`, `gex_daily/` empty). Deprecation
  proposed, **not executed** (work-order §4 forbids deletion this phase). Awaiting go-ahead.

### Phase 1a — Canonicalization (~1-2 days)

Build the dual tables over the owned store (spec v2 §1.2). **Read-layer + one materialized
derived table; no bulk copy of 233 GB.**

- `options_chain_1m` — canonicalized view. Column mapping per spec. **Keep `gamma_eod` /
  `oi_eod` suffixes** (D1 footgun).
- `options_chain_eod` — materialized, one row per contract per session at **15:45:00 ET**,
  with `snapshot_ts` + `snapshot_fallback` carried, never assumed.
- **Snapshot-symmetry guard** as a single guarded function. Strategies never slice bars.
- Derived dailies: `atm_iv_daily`, `skew_daily`, `term_slope_daily`, `iv_rank_daily`, `rv_daily`.

**Homeguard-specific additions:**
- Timestamps normalize to the repo `Datetime[us, UTC]` standard. Source tz semantics must be
  confirmed first — bars run 09:30-16:00, which reads as naive ET, but *confirm, do not assume*.
- Wrap the sweep in **`RunStatus`** (`src/utils/run_status.py`) — mandatory for long runs per
  `.claude/rules/strategy-pipeline.md`. A 233 GB scan that dies silently is exactly the failure
  mode that rule exists for.
- Respect the **thread cap**: `POLARS_MAX_THREADS=1` + `--jobs 8`, per prior findings.
- Background jobs are reaped at ~60 min — **split the sweep per-root**, never one 31-root job.

### Phase 1b — V-battery V1-V13 (~2-3 days)

Full sweep, gates verbatim from work order §3. Priorities by risk:

- **V1/V2/V3 across all 31 roots x 15 years** — the SPY-2024 pass does not generalize for free.
  Expect the fight in 2012-2015 and thin names.
- **V6 — already resolved** (§2). Remaining work: encode the >=1-session lag in the primitive
  layer and unit-test it.
- **V7 corporate actions** — high risk. If strikes are "as reported," an adjustment layer is
  required *before any single-name candidate runs*. Test on AAPL 2020-08-31, TSLA 2020/2022,
  NVDA 2021/2024, AMZN 2022, GOOGL 2022.
- **V13 PIT universe coverage** — likely to bite. `U_MEGA20`/`U_TIER1_100` are **not covered**
  by 31 roots. This forces the universe decision (§5).
- **V11 spread census** -> `spread_census.parquet`. Deliverable, no gate. Feeds `cost_model_v2`.

**Deliverable:** `docs/strategies/research/options-slate/20260727_options_vbattery_report.md` — measured value,
registered gate restated verbatim, PASS/PARTIAL/FAIL, evidence. **No viability conclusions.**

### Phase 2 — Primitives + harness (~1 week)

P1-P10, M1-M7 per spec v2 §3/§4. Reuse mandates: P9 <- `src/features/volatility.py`;
P10 <- `src/backtesting/vol/har_rv.py`; M3 <- `src/backtesting/costs/options.py` (reconcile
half-spread vs width units first).

**Homeguard addition, non-negotiable:** the options backtest runner **must wire fill logging at
build time** — run-scoped `FillSink` -> `output/backtests/<strategy>/runs/<run_id>/` with
`manifest.csv`, per-window gzip fills, and `trades_oos.csv.gz` for walk-forward. This is
`CLAUDE.md` + methodology §12 and is **not optional**. A runner that produces only metrics is
rejected in review. Mirror `scripts/backtest_scripts/run_fx_walkforward.py`.

**M7 (smoothed IV surface) build-vs-buy** — decide here (§5).

### Phase 3 — Wave 0 diagnostics (zero trials)

Group A (underlying only, runnable now, $0): D-014/042 gap + first-hour, D-040a weekend
variance, D-013/030 drawdown shapes, D-050a transition RV timing.
Group B (post-Phase-1): D-003, D-005, D-011, D-018, D-027, D-029, D-033, D-037, D-040b,
D-043 (after V6 lag), D-047/030, D-048, D-049, D-050b.

These consume **no trials** and are the cheapest budget protection in the chain — roughly a
dozen candidates die or reroute here before spending a draw. Report every gate outcome,
**including the kills**; a diagnostic that kills a candidate is the highest-value output.

### Phase 4 — Ledger + validation wrapper (BLOCKING for verdicts)

- Extend `output/experiments.duckdb` per spec §7.1 (do not fork). Populate
  `combinations_project` — the honest cumulative N.
- **Reconstruct lifetime N** per §3.2. This blocks grading, not building.
- Wire CPCV with **embargo >= max holding period** (45-60 d for most; longer for 003/006),
  PBO, DSR. Repo already has `src/backtesting/statistics/dsr.py`,
  `validation/deflated_sharpe.py`, `validation/combined_gate.py` — extend, do not fork.

### Phase 5 — Wave 1 (9 trials) — **MUST route through `strategy-lead`**

**This is the biggest deviation from the chain, and it is mandatory.**

`CLAUDE.md` and `.claude/rules/strategy-pipeline.md` require that *any* phase producing a
strategy verdict — backtest, walk-forward, statistical gate, smoke — is delegated to the
`strategy-lead` agent. It is hard-enforced by a `PreToolUse` hook
(`.claude/hooks/strategy_lead_gate.py`) that **denies** backtest commands unless
`strategy-lead` has set the `.claude/.strategy-lead-active` sentinel.

Split on the build/verdict line:

- **Phases 0-4 = build + measurement.** Run directly. Data-property measurement is not a
  strategy verdict. (If a V-battery command trips the hook pattern, that is the guard doing its
  job — reroute rather than bypass.)
- **Phase 5 Wave 1 = 9 strategy verdicts.** Delegate to `strategy-lead`, which dispatches
  backtest-driver and enforces integrity gates and the fills artifact.

Wave 1 order (spec §6.1): **015, 019** first (they anchor everything), then 016, 001, 002, 027,
032, 047, 050. **Stop at Wave 1.** Waves 2-4 require their named gates plus review.

---

## 5. Decisions — LOGGED 2026-07-27

Registered before any affected test runs, per the chain's amendment rule.

| # | Decision | Resolution (2026-07-27) |
|---|---|---|
| 3 | ORATS ~$399 | **DEFERRED.** Not a gate for any wave (A1 demoted it to non-blocking). Consequence: build **M7** instead of buying SMV, and the **missing-2008 caveat attaches permanently to every short-vol result** and must be stated on each, not omitted |
| 4 | M7 build vs buy | **BUILD**, follows from (3). Dividend input from `src/data/yfinance/fundamentals.py` (already in repo), not ORATS `div_assumption`. Needed by Wave 1 (027, 047), so it is Phase-2 critical path, not deferrable |
| 5 | ThetaData subscription (Phase 0.10) | **ANSWERED: cancelled.** Phase 0.10 closed. Consequences: (a) the 2026-03 -> present refresh (V9) is **not executable** — data edge is 2026-02 across all 31 roots, confirmed; a **live-edge caveat** stands for current-premium sizing (015/019); (b) universe top-up download is **dead** as a route |
| 6 | Legacy `options_1min/` (V12) | **MOOT — the store does not exist on this machine.** `H:\Stock_Data\options\` contains only `_logs`, `chains` (empty), `gex_daily` (empty), `options_combined`. V12 closes with no action |
| 1 | OPT-021 exit version | **OPEN.** Not needed until Wave 2. Recommendation stands: restore **v1.0** (T+1 open) — the amendment to v1.1 was forced by an unmeasurable open mark, and V1/V3 show it is measurable |
| 2 | Universe breadth | **OPEN — but constrained.** Route (b) ThetaData top-up is dead (5); route (c) ORATS is deferred (3). Only **(a) narrow to on-disk roots** remains available without new spend. **Wave 1 is unaffected either way** (runs on `U_INDEX` + on-disk megacaps). Bites only at: 021 (still ~240-400 usable events on ~12 on-disk singles — workable, not crippled), and 026/031/045 (Wave-4 shelf, C+/B- priors, expected screen casualties) |

### Corrected ORATS dependency map (supersedes any earlier framing)

ORATS gates **nothing** in Waves 2/4. Its four roles and their substitutes:

| Role | Substitute | Status |
|---|---|---|
| Smoothed IV surface (P1 low-delta; 006/027/030/047) | **M7** — spec says "ORATS SMV **or** M7" at every call site | substitutable (build) |
| Dividends for M7 + M1 | `src/data/yfinance/fundamentals.py` | substitutable (free) |
| Breadth beyond 31 roots (021/026/031/045) | narrowing | a route, not a requirement |
| Cross-vendor quote/IV validation | D-047/030 already runs vs owned raw deep-OTM quotes; ORATS only upgrades it to a true cross-check | nice-to-have |
| **2007-2012 history (missing-2008)** | **none** | **genuinely unique** |

The unique role bites **Wave 1** (015, 016, 027, 047 — the short-vol/tail core), not Waves 2/4.
So ORATS is a **Wave-1 credibility purchase**, not a Waves-2/4 unlock. The smoothed-surface
need is likewise Wave 1 (027, 047), which is why M7 is Phase-2 critical path.

## 5b. Decisions still open

Items 1 (OPT-021 exit version) and 2 (universe breadth) above. Neither blocks Phase 0,
Phase 1, or Wave 1. Item 1 is needed before Wave 2 touches earnings data; item 2 before
Wave 2 (021) and Wave 4 (026/031/045).

## 6. Honest assessment

**What is genuinely good here.** The pre-registration discipline is real: fixed parameters,
falsifiers stated before contact, diagnostics separated from trials, kill switches registered
in advance, an explicit prohibited-responses list, and "zero survivors is a valid outcome"
stated up front. The wave structure spends ~12 candidates' worth of kills at zero trial cost
before Wave 1. That is the correct shape and it matches this repo's north star.

**What I would not take at face value.**

1. **The DSR arithmetic is the weak point, and it is not a detail.** The chain's Wave-1 hurdle
   (N=9 -> 0.41) is not the honest hurdle. Real N is in the low hundreds and unreconstructed
   (§3.2). If Wave 1 is graded against 0.41 when the true null max is ~0.75, the gate passes
   noise. **Fix N before anyone sees a Sharpe.**
2. **Missing-2008 on a short-vol-centred slate.** The owned window starts 2012-06. The slate's
   centre of mass (F1/F3/F6/F7) is short volatility, whose entire risk is the regime the window
   excludes. Every short-vol result carries this caveat permanently unless ORATS is bought, and
   even then only at EOD granularity.
3. **Prior probability of survival is low, and that is fine.** This repo's honest record is 0
   survivors across RAMP options (16 strategies), futures (26 combos + 28 variations, N 66->94),
   and FX. A 50-candidate slate that follows the same gates will most likely produce another
   well-documented zero. That is a legitimate, pre-registered outcome — but it should set
   expectations about how much build effort to sink before Wave 1 reports.
4. **Cost model remains the weakest quantitative input**, as the chain itself says. V11 improves
   widths from measurement; fill fractions stay assumptions until live IBKR fills exist.

**Recommended stopping rule for the build itself** (not in the chain, added here): if Phase 1b
shows V1/V2 failing on anything beyond the newest thin names, or V7 returns as-reported strikes
requiring a full adjustment layer, **stop and re-scope** rather than building the adjustment
layer on spec. The build cost of this chain is roughly 2-3 weeks before the first verdict; that
is worth it only if the data clears cleanly.

---

## 7. Immediate next actions

1. Commit the 8 chain documents + this plan (pre-registration of record).
2. Finish Phase 0 — the five open items, notably **0.10 ThetaData status** and the
   **classifier PIT state log** (existential for 8 candidates).
3. Answer decisions 1, 2, 5 (§5).
4. Phase 1a canonicalization + Phase 1b V-battery full sweep, under `RunStatus`, per-root jobs.
5. Report V-battery results. **No strategy backtest before then**, per work order §4.
