# Session Handoff: Spot FX backtesting platform (build -> merge -> real-data validation)

**Date:** 2026-07-06 · **Working dir:** /Users/shuyangw/Library/CloudStorage/Dropbox/cs/github/Homeguard · **Repo:** github.com/shuyangw/Homeguard (branch `main`)

## Resume Here (read this first)
- **Goal:** Build a reusable spot-FX backtesting vertical (asset_class fx) parallel to the futures one, then prove it works end-to-end on real data. Not to find a tradeable FX strategy; the two reference strategies were PoCs of the SYSTEM.
- **Status:** DONE. Vertical built via subagent-driven development (10 code tasks), reviewed, final whole-branch review + fix wave, merged to `main` (fast-forward), pushed to origin. Then VALIDATED on real data locally: trend + value walk-forwards ran, both correctly FAILED the statistical gate (expected for a thin book). System works; that was the objective. `main` == `origin/main` == `033a57c`.
- **Next steps (all OPTIONAL, none in progress):**
  1. (If FX revisited) Run the kurtosis-437 spike diagnostic -- find the bad daily-close bar(s) polluting trend's tail stats. It is a DATA/pipeline property, not a strategy issue, so any future strategy inherits it.
  2. (If breadth wanted) Expand to full G10: pull GBP/CAD/AUD/NZD short rates (keyless FRED, same method used this session), spot already on disk, re-run.
  3. v1.1 code fix: gap-spanning MTM move is dropped on gap reopen (see Gotchas).
  4. Do NOT build any execution/live layer -- nothing validated, and it was never in scope.
- **Blockers / open questions:** None blocking. Open: is trend's marginally-positive OOS Sharpe purely a data-spike artifact (kurtosis 437)? Unresolved, low priority.
- **To resume, you need:** conda env `fintech` (`conda run -n fintech ...`). For scripts under `scripts/`, prepend `PYTHONPATH=$(pwd)`. Local data + config are already wired on THIS Mac (see Infra/Data State). `git status`/`git diff`(no args)/`git checkout <branch>` are BROKEN here (see Gotchas) -- use targeted git only.

## Original Task
Verbatim arc of asks:
1. "Explore our repo... How much framework do we have to start testing general FX related strategies... Look into what we have for stocks, futures and options currently."
2. "Scope the spot FX build" -> then invoked `/brainstorming` to plan all requirements.
3. Chose subagent-driven execution -> built it.
4. "we don't have to do the execution layer if we don't have a working strategy" / "are we ready to start testing strategies".
5. "we do have data, look in our Dropbox for stock_data" -> "See if we can pull the necessary FRED data via API" -> "Yes" (proceed to run).
6. Clarification: "we tested these as PoCs of our system, not to seriously consider these strats."
7. Load the newly-pushed session-handoff skill; then `/session-handoff`.

## Subtasks & Progress
- [x] 4-agent parallel exploration of FX / stocks / futures / options framework readiness -- FX was data-rich but had zero backtest wiring; futures the most mature vertical; equities the most reusable general framework; options = CSP-only.
- [x] `/brainstorming` -> design spec approved. Decisions locked (see below).
- [x] `/writing-plans` -> 11-task implementation plan.
- [x] `/subagent-driven-development` -> Tasks 1-10 implemented, each with a fresh implementer + task reviewer (spec + quality), fix waves where needed. All Approved.
- [x] Final whole-branch review (opus): 0 Critical, 4 Important cross-cutting issues -> one fix wave (commit 8b82489) -> re-review "Ready to merge: Yes".
- [x] Merged FF to `main`, pushed to origin, deleted feature branch `feat/spot-fx-backtest`.
- [x] Discovered FX spot data IS local; wired storage; pulled FRED; ran real backtest + walk-forward.
- [x] Fixed a live carry bug found during the run (invalid CHF FRED series) -- commit 52e2065.
- [x] Recorded outcome to memory; loaded session-handoff skill; wrote this handoff.
- [ ] Task 11 "acceptance run" as originally written (EC2) -- SUPERSEDED; the acceptance run was done locally instead.

## Key Decisions & Tradeoffs
- **Own vertical, futures untouched.** Dedicated `FxSpotPortfolioSimulator` + `asset_class: fx` routing, mirroring the futures pattern. Why: spot FX needs notional/carry/cross-rate math that neither equity nor futures sim fits. Tradeoff: some deliberate structural parallelism (not shared code).
- **Model carry accrual now (not deferred).** Carry = daily interest-rate differential on held USD notional. Why: without it, trend/value equity curves are systematically biased; carry is first-order for FX. It is a SIM feature, required even though no carry STRATEGY ships.
- **Full cross-rate USD conversion up front.** quote-currency -> USD per pair per day, derived from in-universe USD legs. Missing leg raises (no silent mis-conversion).
- **v1 universe NARROWED to carry-covered currencies.** Only USD/EUR/CHF/JPY have FRED short rates readily available, so universe = EURUSD, USDJPY, USDCHF, EURJPY, EURCHF, CHFJPY, XAUUSD, XAGUSD (8 instruments). GBP/CAD/AUD/NZD deferred. Tradeoff: thin, EUR/CHF/JPY-correlated book with low breadth -- accepted for a system PoC.
- **Shared walk-forward helpers (user override of "import" -> "extract to module").** Pure helpers moved to `src/backtesting/walkforward_common.py`, both futures and FX WF import it. Regression gate = existing futures WF test.
- **Empty subclasses for FX strategies (user-approved).** `FxTrendStrategy(CarverMomentumStrategy)` / `FxValueStrategy(FuturesValueStrategy)` reuse price-only forecast_panel; distinct registry names + a divergence seam for future PPP value.
- **Calendar-day carry (final-review fix).** Accrue `rate_diff * (d - prev_d).days / 365`, not per-trading-day-bar `/365` (which understated carry ~31%).
- **Per-pair cost tier (final-review fix).** Crosses (EURJPY/EURCHF/CHFJPY) = minor tier, USD-leg pairs = major, metals = bps. A single global "major" under-costed the crosses ~2.5x.
- **Merge directly to main, push (user explicit).** No PR. Fast-forward.
- **Ran the validation LOCALLY, not on EC2.** Data was found on this Mac. Faster feedback (~15 min, no EC2).

## Discussion Summary
Explored (4 parallel Explore agents): FX had extensive data acquisition + an orphaned cost model but no backtest/loader/calendar/sizing wiring; futures was the most mature vertical (the template we mirrored); equities the most reusable config-driven framework; options = one live strategy (CSP). Scoped spot FX two ways (FX-futures runnable-today vs spot-FX new build); user chose to scope the spot build. Brainstormed to a design; wrote a bite-sized TDD plan; executed via subagent-driven development with per-task spec+quality reviews. Per-task reviews caught real bugs (Task 1 FX-date filter clip + DST docstring; Task 2 unhardened loader; Task 7 NaN-gap force-flatten -> NaN cost). The opus final review caught 4 cross-cutting Important issues the per-task reviews missed (FxTrend partial-cache KeyError, trading-day vs calendar-day carry, under-costed crosses, dead guard); fixed in one wave, re-reviewed clean, merged.

Then the pivotal turn: user said data is local ("Dropbox stock_data"). Found FX spot minute data at `Dropbox/Stock_Data/fx/massive/1min` (non-canonical path, but exact 8-col schema, 80 pairs, 2011+). FRED rate data was NOT local but pulls keyless via pandas-datareader. Probing FRED surfaced that the CHF series in the merged code (`IRSTCB01CHM156N`) is INVALID (would have silently zeroed CHF carry across half the book); found valid replacements. Wired storage (settings [macos] + symlink), pulled 4 FRED series, built daily cache, ran trend backtest + trend/value walk-forwards.

Result: both PoC strategies fail the gate. Trend WEAK (fails PBO), Value REJECT (negative OOS). User clarified these were PoCs of the SYSTEM, not serious strategy candidates -- so the win is that the platform runs end-to-end and the gate correctly discriminates (rejects weak strategies). The one finding that outlives the PoCs: trend's tail stats are pathological (skew 12.6, kurtosis 437), almost certainly a data-quality spike, a pipeline property future strategies inherit.

## Commands & Outputs
```
# FRED probe -- CHF series in merged code was invalid; found replacements
$ python -c "DataReader(s,'fred',...) for s in [DFF,ECBDFR,IRSTCB01CHM156N,IRLTLT01JPM156N]"
OK DFF, OK ECBDFR, FAIL IRSTCB01CHM156N (RemoteDataError, HTML error page), OK IRLTLT01JPM156N (monthly)
# valid short rates found:
OK USD DFF (daily 6026 rows) · EUR ECBDFR (daily 6026) · CHF IRSTCI01CHM156N (monthly, ENDS 2024-03) · JPY IRSTCI01JPM156N (monthly, current 2026-05)

# pull the 4 corrected FRED series -> Stock_Data/alt_data/fred/<id>/daily.parquet
$ python -c "FREDRatesPlugin().fetch_series(sid, 2010-01-01, 2026-07-01)"  # keyless, no API key needed
DFF 6026 · ECBDFR 6026 · IRSTCI01CHM156N 171 · IRSTCI01JPM156N 197

# build 17:00-ET daily cache (8 pairs) -- ran in 11s (files locally materialized)
$ PYTHONPATH=$(pwd) python scripts/data/build_fx_daily_cache.py --csv config/universes/fx_spot-2026.csv --start 2010-01-01 --end 2026-05-31
wrote EURUSD 4210 · USDJPY 4208 · USDCHF 4222 · EURJPY 4280 · EURCHF 4235 · CHFJPY 4276 · XAUUSD 3609 · XAGUSD 3524  (8 pairs)

# representative trend backtest
$ PYTHONPATH=$(pwd) python -m src.backtest_runner --config config/backtesting/fx_trend.yaml
FX backtest complete: n_days=4214, sharpe_ratio=0.198, trade_log=output/backtests/fx/FxTrend/2011-01-01_to_2025-12-31

# walk-forward gates (the actual verdict)
$ PYTHONPATH=$(pwd) python scripts/backtest_scripts/run_fx_walkforward.py --config config/backtesting/fx_trend.yaml --json output/fx_trend_gate.json
TREND: oos_sharpe=0.1346 (1.5x=0.0789) psr=1.0 dsr=1.0 pbo=0.2963 n_windows=12 n_oos_days=3436 skew=12.55 kurtosis=437.7 -> WEAK (fails PBO<0.25); registry run_id d43071d3-4b3c-4123-ba2f-a601c1709c47
$ PYTHONPATH=$(pwd) python scripts/backtest_scripts/run_fx_walkforward.py --config config/backtesting/fx_value.yaml --train-months 72 --json output/fx_value_gate.json
VALUE: oos_sharpe=-0.2921 (1.5x=-0.3340) psr=0.0 dsr=0.0 pbo=0.7759 -> REJECT (negative OOS)

# merge to main (normal checkout FATALs on the broken gitlink -> used ref-update FF instead)
$ git branch -f main b5d3b36 && git symbolic-ref HEAD refs/heads/main   # FF without working-tree checkout
$ git checkout HEAD -- src scripts config tests docs .superpowers        # restore tree clobbered by the aborted `git checkout main`
$ git push origin main   # 60fb125..b5d3b36
$ git branch -d feat/spot-fx-backtest
```

## PRs
- None. Work merged directly to `main` (fast-forward) and pushed to origin per user's explicit request. No PR opened.

## Linear / Jira
- None touched.

## Files Touched (all on `main`, merged)
New FX vertical (src):
- `src/backtesting/engine/fx_spot_portfolio_simulator.py` -- FxSpotPortfolioSimulator (MTM + calendar-day carry + leverage cap + bankruptcy floor + NaN-gap forward-hold)
- `src/backtesting/engine/fx_backtest.py` -- run_fx_backtest orchestration
- `src/backtesting/data/fx_backtest_loader.py` -- load_fx_daily_panel + build_quote_usd_panel
- `src/data/fx_rates.py` -- CURRENCY_FRED_SERIES + load_fx_rate_panel + build_rate_diff_panel (CHF/JPY ids fixed in 52e2065)
- `src/data/fx/clusters.py` -- fx_cluster_for; `src/data/fx/__init__.py`
- `src/backtesting/utils/position_sizer_fx.py` -- size_from_forecast_fx
- `src/backtesting/costs/fx.py` -- appended fx_round_trip_usd (pip + metals bps)
- `src/backtesting/utils/idm_weights.py` -- added backward-compatible cluster_fn param
- `src/strategies/advanced/fx_strategies.py` + `src/strategies/registry.py` -- FxTrend/FxValue
- `src/backtest_runner.py` -- additive asset_class==fx routing branch
- `src/backtesting/walkforward_common.py` -- extracted shared WF helpers
- `scripts/data/build_fx_daily_cache.py`, `scripts/backtest_scripts/run_fx_walkforward.py`
- `config/universes/fx_spot-2026.csv`, `config/backtesting/fx_trend.yaml`, `config/backtesting/fx_value.yaml`
- Tests under `tests/backtesting/**`, `tests/data/**`, `tests/strategies/test_fx_strategies.py`
Docs:
- `docs/plans/2026-07-05-spot-fx-backtest-design.md`, `docs/plans/2026-07-05-spot-fx-backtest-implementation-plan.md`
- `docs/reports/fx/FX_WALK_FORWARD.md` (trend), `docs/reports/fx/FX_VALUE_WALK_FORWARD.md` (value) -- generated by the runs, not committed unless asked
Local-only (NOT committed, gitignored):
- `settings.ini` -- `[macos] local_storage_dir` repointed to `/Users/shuyangw/Library/CloudStorage/Dropbox/Stock_Data`
Memory (outside repo):
- `~/.claude/projects/-Users-shuyangw-...-Homeguard/memory/{spot-fx-platform,macos-git-and-data-env-gotchas,MEMORY}.md`

## Key commits (on main)
- `033a57c` chore(skills): add session-handoff skill (from origin, this session pulled it)
- `52e2065` fix(fx): correct CHF/JPY carry rate series ids (CHF IRSTCB01CHM156N invalid -> IRSTCI01CHM156N; JPY -> IRSTCI01JPM156N)
- `b5d3b36` merge tip of feat/spot-fx-backtest (the whole vertical, FF-merged)
- `8b82489` fix(fx): final-review wave (present-pair strategy, calendar-day carry, per-pair cost tier, drop dead guard)
- `8423f18` Task10 walk-forward + gate; `3cc6696` Task9 runner; `78d76c2`/`365e147` Task7 sim + NaN-gap fix; earlier tasks defa58b..48a3e8b

## Infra / Data State (this Mac)
- `get_local_storage_dir()` -> `/Users/shuyangw/Library/CloudStorage/Dropbox/Stock_Data` (after the settings.ini edit).
- FX spot minute data: `Stock_Data/fx/massive/1min/symbol=<PAIR>/year=/month=/data.parquet` (canonical 8-col schema). Bridged with symlink `Stock_Data/fx_1min -> fx/massive/1min` so the loader finds the canonical path.
- FX daily cache built: `Stock_Data/fx_daily/` (8 pairs).
- FRED rates: `Stock_Data/alt_data/fred/{DFF,ECBDFR,IRSTCI01CHM156N,IRSTCI01JPM156N}/daily.parquet` (pulled keyless this session).
- FRED API key exists in `.env` (`FRED_API_KEY`) but was NOT needed; keyless pandas-datareader worked.
- Equities/futures/options data: presence on this Mac NOT verified; check before assuming.

## Key Takeaways & Gotchas
- **Broken gitlinks break git here.** `.claude/worktrees/ramp-equity-fix` and `.claude/worktrees/sip-validation` are tracked as gitlinks pointing at `C:/Users/qwqw1/...` Windows paths. Any working-tree-updating git command (`git status`/`git diff` no-args, `git checkout <branch>`, `git pull`, `git reset --hard`) FATALs, and a failed `git checkout` can PARTIALLY clobber the tree (it deleted the FX files mid-merge this session). Safe ops: `git add <paths>`, `git commit`, `git log`, `git push`, `git branch -d`, targeted `git checkout HEAD -- <paths>`, `git reset --soft`, and FF-by-ref-update (`git branch -f` + `git symbolic-ref HEAD`). A real fix would `git rm --cached` those two gitlinks (separate opt-in cleanup, offered, not done).
- **settings.ini `[macos]` originally pointed at a missing dir** (`cs/stonk/data`); repointed to `Stock_Data`. settings.ini is gitignored (local only).
- **The invalid-CHF-series bug** would never have been caught by unit tests (nothing pulls real FRED) -- only the real-data PoC surfaced it. Lesson: run real data before trusting a data-mapping config.
- **kurtosis 437 / skew 12.6 on trend** = data-quality spike, not a strategy trait. PSR/DSR saturate to 1.0 and are UNTRUSTWORTHY at that kurtosis; the honest read is PBO (0.296, fails) + the tail. Diagnose the offending daily bar(s) before running any strategy you care about.
- **v1.1 code limitation:** a position forward-held across a MULTI-day data gap does NOT realize the gap-spanning price move on reopen (`prev_close` holds the NaN row). Material for the 2020-10/11 EURUSD outage. Fix candidate: forward-fill prev_close to realize the accumulated move.
- **CHF rate ffill-stale after 2024-03** (series discontinued); JPY/CHF are monthly (ffilled to daily). Minor.
- **Scripts under `scripts/` need `PYTHONPATH=$(pwd)`** (they import `src`); `-m src.*` modules do not.
- **The gate correctly REJECTING weak strategies is the PoC success criterion**, not a disappointment. A gate that only says PASS is the dangerous one.

## References
- Design: `docs/plans/2026-07-05-spot-fx-backtest-design.md`
- Plan: `docs/plans/2026-07-05-spot-fx-backtest-implementation-plan.md`
- Reports: `docs/reports/fx/FX_WALK_FORWARD.md`, `docs/reports/fx/FX_VALUE_WALK_FORWARD.md`
- SDD ledger (all task-by-task detail + commit map): `.superpowers/sdd/progress.md` (FX section at the end)
- Experiment registry run: `output/experiments.duckdb` run_id `d43071d3-4b3c-4123-ba2f-a601c1709c47`
- Repo: github.com/shuyangw/Homeguard (main @ 033a57c)
