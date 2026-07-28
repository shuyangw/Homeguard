# Options Slate — Phase 0 Repo Reconciliation Report

**Date:** 2026-07-27
**Scope:** the eleven integration points asserted-and-unverified in
`2026-07-25_options_phase01_cc_work_order.md` §1, plus the classifier point-in-time check.
**Method:** direct inspection (`cat`, `grep`, `ls`, live queries). Every row cites evidence.
**Status:** Phase 0 **COMPLETE**. Phase 1 is unblocked.

Per work-order ground rule 1: where this report and the chain disagree, **this report wins**
and the divergence is stated rather than silently reconciled.

---

## Summary

| # | Item | Status |
|---|---|---|
| 0.1 | Strategy registry format | **CONFIRMED — but wrong seam for options** |
| 0.2 | `OptionsDataLoader` output schema | **CONFIRMED (divergent from chain)** |
| 0.3 | Multiple-testing ledger | **CONFIRMED — exists, richer than assumed** |
| 0.4 | `strategy_toggle.yaml` | **CONFIRMED — not needed this phase** |
| 0.5 | Yang-Zhang estimator | **CONFIRMED — reuse, do not fork** |
| 0.6 | Storage / DuckDB conventions | **CONFIRMED — single-writer constraint found** |
| 0.7 | Cost model | **CONFIRMED — exists; unit trap found** |
| 0.8 | Classifier point-in-time state | **ANSWERED — causal replay available; gates ARE backtestable** |
| 0.9 | `OptionsDataStore` divergence | **CONFIRMED DEAD — deprecate (do not delete yet)** |
| 0.10 | ThetaData subscription | **ANSWERED — cancelled** |
| 0.11 | Download/combine join semantics | **ANSWERED — same-day join = leak; lag rule registered** |

**No item is ABSENT.** Nothing downstream is blocked by a missing integration point.

---

## 0.1 — Strategy registry format · CONFIRMED, but it is the wrong seam

**Actual:** `src/strategies/registry.py`. `_STRATEGY_REGISTRY: Dict[str, Tuple[str, str]]` maps a
strategy name to `(module_path, class_name)` with lazy imports "to avoid import chain issues
between backtesting and live trading modules." Public API: `get_strategy_class(name)`,
`list_strategies()`. Registered classes subclass `BaseStrategy` (`src.backtesting.base.strategy`).

**Divergence / judgment call.** This registry serves **config-driven single-instrument equity**
backtesting. The FX and futures workstreams do **not** use it — they run through dedicated
runner scripts (`scripts/backtest_scripts/run_fx_walkforward.py`,
`run_carver_walkforward.py`) sharing `src/backtesting/walkforward_common.py` helpers
(`_build_windows`, `_oos_returns_dated`, `_stitch_oos_dedup`).

Options are structurally closer to FX/futures than to equity: multi-leg positions, a
strike x expiry x time chain rather than a bar series, and per-leg cost accounting.

**Recommendation:** follow the **FX/futures runner pattern**, not the equity registry. Reuse
`walkforward_common`. This also inherits the mandated `FillSink` wiring for free, since that is
where the fill-logging convention already lives.

## 0.2 — `OptionsDataLoader` output schema · CONFIRMED (chain was wrong)

`src/strategies/options/data_loader.py:20-26` — `COLUMN_RENAME_MAP` emits `bid_close->bid`,
`ask_close->ask`, `gamma_eod->gamma`, `open_interest_eod->open_interest`,
`underlying_px->underlying_price`; `_transform` (line 173) derives `option_type`, `datetime`,
`date`, `time`, `expiry`, `days_to_expiry`, `mid_price`.

The chain called `infra_patterns.md`'s column list "suspected fiction." **It is substantially
correct.** Two real problems instead:

1. **Leak-enabling rename.** `gamma_eod->gamma` and `open_interest_eod->open_interest` strip the
   `_eod` marker. A strategy reading `open_interest` gets no cue it is same-day EOD data (see
   0.11). **Canonicalization must preserve the `_eod` suffix.**
2. **Wrong snapshot minute.** `get_eod_chain()` (line 51) hardcodes `time(16, 0)`; the
   registered snapshot is **15:45**. Both bars exist on disk. **Do not reuse `get_eod_chain()`**
   for slate work.

## 0.3 — Multiple-testing ledger · CONFIRMED, richer than the chain assumed

**Actual:** `src/experiments/registry.py`, `DEFAULT_DB_PATH = output/experiments.duckdb`, schema
in `src/experiments/schema.sql`. Append-only; the module docstring is explicit: *"If the append
fails, the run fails. No silent success"* — it raises on every failure and callers must not
catch. Has a 5-attempt retry for Windows/Dropbox file-lock contention.

Tables: `runs` (496 rows; 30 columns incl. `params`, `metrics`, `regime_breakdown`,
`fold_metrics`, `cost_tier_used`, `cost_sensitivity`, `combinations_in_run`,
`combinations_project`, `git_sha`, `config_sha`, `data_snapshot_date`, `python_env_hash`,
`random_seeds`) and `return_streams` (300,288 rows).

**Do not create the §7 ledger from scratch** — extend this. But see the honest-N problem
(execution plan §3.2): `combinations_project` is unpopulated and **no options runs are
recorded**, so lifetime N must be reconstructed before any Wave-1 Sharpe is graded.

## 0.4 — `strategy_toggle.yaml` · CONFIRMED, not needed this phase

**Actual format** (`config/trading/strategy_toggle.yaml`):

```yaml
last_modified: '2026-05-23T00:00:00-04:00'
modified_by: claude-code
strategies:
  ramp: {enabled: true, shutdown_requested: false, variant: v11}
  mp:   {enabled: true, ...}
  omr:  {enabled: false, ...}
  cscm: {enabled: false, ...}
```

The `ramp.variant: v11` key the chain flagged as "known dead" is present. Note that production
deploys the V11 stable branch, so treating it as dead is questionable — **not resolved here, and
not needed**: this file governs *live* trading only. No options strategy goes live for many
phases. **Add nothing now.**

## 0.5 — Yang-Zhang · CONFIRMED

`src/features/volatility.py` (exported via `src/features/__init__.py`). **P9 reuses it. Do not
fork a second implementation** (work-order acceptance criterion).

## 0.6 — Storage / DuckDB conventions · CONFIRMED, with a constraint that bites

Parquet read via polars (`pl.read_parquet`) and pandas across
`src/backtesting/{benchmark,engine/streaming_data_loader,optimization/data_loader,session/session_bars,vol/atm_iv}.py`.
Hive partitioning is the norm. Canonical/derived options tables should follow the same shape.

**Constraint found — affects Phase 1 directly.** `src/experiments/registry.py` docstring:
*"DuckDB does not support concurrent writes from multiple processes on the same file. Callers
must serialize their backtests."* The V-battery sweep is planned as **parallel per-root jobs**
(to stay under the ~60-min background reap). Those jobs must **not** write to
`experiments.duckdb` concurrently — write per-root results to separate parquet shards and do a
single serialized ledger append at the end.

## 0.7 — Cost model · CONFIRMED, unit trap

`src/backtesting/costs/options.py`, per methodology §4.5. Model:
`fill_buy = mid + alpha*(1/2)(ask-bid)`, with `OPTIONS_ALPHA_TABLE = {very_liquid: 0.4,
liquid_etf: 0.6, single_stock_atm: 0.85, wings_illiquid: 1.1}` and
`DEFAULT_IBKR_PER_CONTRACT_USD = 0.58`.

**Extend, do not fork.** **Unit trap:** repo `alpha` = fraction of the **half-spread**; the
chain's convention = fraction of the **full width**. 25% of width == alpha 0.50. Reconcile
explicitly in M3 or every cost figure is wrong by 2x.

## 0.8 — Classifier point-in-time state · ANSWERED: gates ARE backtestable

This was the existential item — eight candidates depend on P5 regime gates, three of them in
Wave 1 (015, 001, 002).

**Finding: there is no persisted historical state log, but causal replay exists and is verified
causal.**

- `MarketRegimeDetector.classify_regime(spy_data, vix_data, timestamp)`
  (`market_regime_detector.py:111`) classifies from `.iloc[-1]` of whatever frame it is handed —
  causal **iff** the caller truncates.
- `analyze_regime_history()` (line 377) does exactly that:
  `spy_subset = spy_data[spy_data.index <= date]`, `vix_subset = vix_data[vix_data.index <= date]`,
  looped per date. **This is a causal replay.**
- `_calculate_vix_percentile()` (line 245) uses
  `vix_data['close'].iloc[-self.lookback_window:]` — a trailing 252-row window of an
  already-truncated frame, i.e. strictly backward-looking. **No full-sample percentile
  lookahead.** This was the one place lookahead could plausibly have hidden.
- No persistence in the detector (no `to_parquet` / `to_csv` / `json.dump`).
- `scripts/ops/backfill_regime_state.py` replays the detector and POSTs
  `hg_regime_state_code` to **VictoriaMetrics** — that is a Grafana/monitoring series, **not a
  research artifact**. Do not use it as the PIT source.

**Consequence:** we land in spec v2 §1.3 rule 6's second branch — *"regime gates are recomputed
causally and that recomputation is reported as a build artifact, not silently assumed correct."*
**Wave 1 does not shrink.**

**Three requirements this creates:**

1. **Materialize once.** Build `regime_state_daily(date, regime, confidence, scores...)` as a
   single Phase-1 artifact. Eight candidates recomputing it independently is both wasteful
   (`analyze_regime_history` is O(n^2) in dates) and a silent-divergence risk.
2. **Vintage caveat.** Replay uses *today's* SPY/VIX series. If that data was revised or
   backfilled since, recomputed state differs from what was known then. Low risk for SPY/VIX
   OHLC, but it must be **stated on results**, not assumed away.
3. **VIX spot must be materialized — it is not in local storage.** See below.

### New finding: VIX index levels are fetched, not stored

`/h/Stock_Data/alt_data/vix/` holds only `vx_curve.parquet` — VIX **futures** term structure
(`date, vx1_settle, vx2_settle, vx1_dte`), 3,304 rows, **2013-05-20 -> 2026-07-06**. That is the
OPT-013 trigger input, not the regime detector's input.

The regime detector needs **VIX index spot**, which comes from `src/utils/vix_provider.py` /
yfinance `^VIX` at call time (`backfill_regime_state.py:32`). A fetched series is not
reproducible: it can change, rate-limit, or fail.

**Action (Phase 1):** materialize VIX spot 2012-06 -> 2026-02 to local storage as a
first-class dataset, with a snapshot date, and drive replay from it.

**Also note:** `vx_curve` starts 2013-05, so OPT-013's VIX-term trigger has a **13.2 y** window
against the slate's 13.7 y. Its DSR uses its own window (per A1 §8's per-root rule).
Conversely `vx_curve` runs to 2026-07 — *fresher* than the options data's 2026-02 edge.

## 0.9 — `OptionsDataStore` divergence · CONFIRMED DEAD

`src/data/options/options_store.py` targets `<storage>/options/chains/` and
`.../gex_daily/`. Both exist on disk and are **empty** (0 bytes, confirmed by `du`). The live
reader is `OptionsDataLoader` against `options_combined/`.

**Proposal:** deprecate `OptionsDataStore` (raise on init with a pointer to `OptionsDataLoader`),
and remove the two empty directories. **Not executed** — work-order §4 prohibits deletion this
phase. Needs your go-ahead.

## 0.10 — ThetaData subscription · ANSWERED: cancelled

Confirmed by you, 2026-07-27. No Theta config present in `.env`. Consequences already logged in
the execution plan §5: the 2026-03 -> present refresh (V9) is **not executable** (edge confirmed
at 2026-02 across all 31 roots), and universe top-up download is dead as a route.

## 0.11 — Download/combine join semantics · ANSWERED: same-day join, a leak

`scripts/data/combine_options_data.py`: `_date` derived by slicing the intraday timestamp
(line 179), EOD frame joined on it (line 239, `how="left"`). Confirmed on data: OI constant
across all minutes of a session (0 of 18,879 contract-sessions vary), varying day to day.

Session *t*'s rows carry *t*'s end-of-day OI, published the following morning. **Reading
`oi_eod` at the 15:45 snapshot on session *t* is a hard lookahead leak.**

**Registered ruling:** all `oi_eod` / `gamma_eod` uses **lag >= 1 session**, enforced in the
primitive layer, not per-strategy. Combined with 0.2's rename problem, the canonical layer must
keep the `_eod` suffix so the constraint is visible at every call site.

Refresh procedure (V9 input) is moot — subscription cancelled (0.10).

---

## Consequences for Phase 1

1. **Build the options runner on the FX/futures pattern** (0.1), not the equity registry — and
   wire `FillSink` at build time, per the standing mandate.
2. **Canonicalization must preserve `_eod` suffixes** (0.2 + 0.11) and must not reuse
   `get_eod_chain()`'s 16:00 default.
3. **Materialize `regime_state_daily` once** via causal replay, and **materialize VIX spot**
   locally first (0.8). VIX spot is a Phase-1 prerequisite, not an afterthought.
4. **Parallel per-root V-battery jobs must not touch `experiments.duckdb` concurrently** (0.6) —
   shard to parquet, append once, serialized.
5. **Reconcile cost units** (half-spread vs width) before M3 is used anywhere (0.7).
6. **Reconstruct lifetime N** before any Wave-1 result is graded (0.3).

## Open, needing your decision

- **0.9:** approve deprecating `OptionsDataStore` + removing the two empty dead dirs?
- Still open from the execution plan: OPT-021 exit version (before Wave 2), universe breadth
  (before Wave 2's 021 / Wave 4's 026/031/045).

*No strategy backtest has been run. No P&L has been observed.*
