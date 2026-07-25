# Session Handoff: FX 60-Strategy Catalog Campaign (data layer + validation start)

**Date:** 2026-07-06 · **Working dir:** `/Users/shuyangw/Library/CloudStorage/Dropbox/cs/github/Homeguard` · **Model:** Opus 4.8

## Resume Here (read this first)

- **Goal:** Work the 60-strategy FX catalog (research in `~/Downloads/compass_artifact_wf-4265ee05-d103-499d-8315-20a63cdd6e8f_text_markdown.md` + `~/Downloads/fx_strategy_deep_dive.md`). Build the data/engine, then implement + statistically gate each strategy; track progress in a living file.
- **Status:** FX data+compute layer BUILT + merged to main (18-task subagent build). All data gaps recovered + cleaned + cross-vendor-validated (22-pair G10, gap-free). 60-strategy tracker created. Started strategy validation: 4 READY strategies implemented + gated; ALL FAIL the walk-forward gate in their NAIVE form (not the enhanced forms).
- **Next steps (decision pending, user leaning toward option 1):**
  1. Build + gate the ENHANCED carry variants **#16 carry-momentum** and **#19 carry-unwind** ("carry with a seatbelt") -- directly fixes why naive carry failed. Highest-value daily test. Plus a parameter sweep on the naive forms to confirm they are weak across the plateau, not one config.
  2. OR accept daily G10 is efficient and commit to the INTRADAY engine (the ~24 session/microstructure strategies the research ranks highest; large build).
  3. OR finish screening the remaining ~11 READY strategies for completeness.
- **Blockers / open questions:** None technical. Open strategic question above (which direction). Also: the 4 strategy commits are on an UNMERGED branch (see below) -- decide whether to merge them.
- **To resume, you need:**
  - Env: `source /opt/miniconda3/etc/profile.d/conda.sh && conda activate fintech` then `PYTHONPATH=$(pwd)`. Do NOT use plain `python` (base miniconda lacks pandas_datareader). `holidays`, `boto3`, `dukascopy-python`, `statsmodels` are installed in `fintech`.
  - Branch: currently on `feat/fx-strategy-validation` @ e15a31f (4 commits ahead of main). `main` = `origin/main` = 3a7db91.
  - Git HAZARD (critical): broken Windows worktree gitlinks make `git checkout <branch>`, bare `git status`/`git diff`, `git reset --hard` FATAL and can clobber the tree. Use targeted `git add <paths>` + `git commit` only. Merge via ref-update FF: `git branch -f main <tip> && git symbolic-ref HEAD refs/heads/main && git branch -d <feat>`. CLAUDE.md now grants STANDING merge-and-push authorization when a feature is complete.

## Original Task (evolution)

Started: "how much framework do we have to test FX strategies?" -> mapped the 60 -> "which pairs do we need" (G10) -> realized it is all a compute problem (data on disk) -> `/brainstorming` the full data+compute layer -> `/writing-plans` -> `/subagent-driven-development` (built 18 tasks) -> ran real builds -> found + fixed data gaps + quality issues -> made the tracker -> validate the 15 READY strategies. User repeatedly pushed for rigor (e.g. "we shouldn't outright reject those" -> forced proper walk-forward + gate).

## Subtasks & Progress

- [x] Map 60 strategies to buckets. Result: 15 READY, 8 OHLC-iface, 4 SPREAD, 3 BRACKET, 6 ML, 2 DATA, 22 INTRADAY. In `docs/strategies/FX_60_CATALOG_TRACKER.md`.
- [x] FX data+compute layer: spec `docs/superpowers/specs/2026-07-06-fx-data-compute-layer-design.md`, plan `docs/superpowers/plans/2026-07-06-fx-data-compute-layer.md`, built via 18 TDD tasks (subagent-driven), merged to main (commits through a3ef3bc). Two families: acquisition plugins + ArtifactBuilder registry. Artifacts: spread_model, vol_surface, currency_strength, pca_dollar, cointegration, regime, event_registries. CPCV + combined DSR/PBO gate.
- [x] Real data builds: rebuilt daily_ohlc_cache (22 pairs, OHLC), built the artifacts. `python -m src.data.fx_pipeline build daily_ohlc_cache` (fast).
- [x] Gap recovery: found empty vendor month-shards. Massive/Polygon flat-file re-pull recovered the majors (AUDUSD 2013, GBPUSD 2013, USDCAD 2014, EURUSD 2014-Q3, USDJPY 2013). REST API probe confirmed the rest genuinely absent at Polygon. yfinance PoC FAILED (no historical minute, no XAUUSD spot). Dukascopy (keyless) PoC PASSED + backfilled all 48 remaining pair-months. Result: 0 gaps across 22 pairs.
- [x] Data cleaning/validation: spike_clean (reverting bad prints), WEEKDAY FILTER (the big fix -- root cause of "artifacts"), FRED rate fix (IRSTCI01 discontinued -> IR3TIB01). Cross-vendor validated: every pair now normal FX kurtosis; only large moves are real events (SNB 2015, Brexit 2016, COVID 2020, silver squeeze 2026).
- [x] 60-strategy tracker created + committed (main 3a7db91).
- [~] Strategy validation: 4 of 15 READY implemented + walk-forward-gated. ALL FAIL-naive. On branch `feat/fx-strategy-validation` (unmerged).
- [ ] Enhanced carry variants (#16, #19), parameter sweeps, or intraday engine. NOT STARTED.

## Key Decisions & Tradeoffs

- **Dukascopy for backfill.** Why: only free source with historical MINUTE + XAUUSD spot (yfinance failed both). Keyless. Cross-validated sub-pip vs Polygon. Tradeoff: 3 exotic crosses (NOKSEK/NOKJPY/SEKJPY) derived by triangulation from USD legs.
- **Weekday filter over a bad-bars manifest.** Root cause of "non-reverting artifacts" was thin Sat/Sun stray-edge bars (~10-100 min vs 1439). Filtering Mon-Fri fixed everything (EURSEK kurt 180->5, USDCAD 322->1.8). REMOVED the cross-vendor bad-bars mechanism I first built (it chased a boundary artifact the weekday filter already fixed).
- **FRED IR3TIB01 (3M interbank) for all non-USD/EUR.** IRSTCI01 (call-money) was DISCONTINUED: SEK ended 2020-10 (5.7yr stale @ 0.10%), CHF 2024-03, NZD 2024-12 -> wrong carry. IR3TIB01 current for all (ends 2026-04/05). Tradeoff: ~10-30bp term premium over overnight; acceptable.
- **vol_target 0.03 for 22 pairs** (not 0.20). The 8-pair configs' 0.20 over-leverages the correlated FX book -> -99% blowup. 0.03 -> ~17% realized book vol. FX pairs highly correlated; a portfolio-level vol cap is a candidate infra improvement.
- **Verdicts require walk-forward + gate, not in-sample screen.** User corrected my premature REJECT labels. Re-ran with IDM on + walk-forward + PSR/DSR/PBO. Distinction: `FAIL-naive` (canonical form fails gate) is NOT `REJECT` (idea killed across forms + params).

## Discussion Summary

Naive daily FX factors do not work on clean G10 (gated, OOS): TSMOM -0.02, XS-mom -0.05, carry -0.33, gold/silver -0.31; DSR ~0, PBO 0.49-0.85. Matches research (FX trend decayed post-2010; G10 carry thin without EM/crash-mgmt; simple RV reversion fails). BUT these are the NAIVE forms; the deep-dive says edge lives in the enhancements (ranked carry basket + crash filter; gold/silver momentum-brake + regime stop) and in the intraday half. So the honest state: naive daily is weak, enhanced forms + intraday untested. Data is now clean/trustworthy (the real bottleneck, now cleared).

## Commands & Outputs (load-bearing)

```
# Gated OOS walk-forward verdicts (IDM on, 36/12/12, both cost legs)
$ run_fx_walkforward.py --config config/backtesting/fx_{tsmom,carry,goldsilver,xsectmom}.yaml
  fx_tsmom:      oos_sharpe=-0.016 psr=0.205 dsr=0.205 pbo=0.852 1.5x=-0.107 n=13
  fx_carry:      oos_sharpe=-0.327 psr=0.000 dsr=0.000 pbo=0.727 1.5x=-0.360 n=13
  fx_goldsilver: oos_sharpe=-0.313 psr=0.000 dsr=0.000 pbo=0.489 1.5x=-0.327 n=13
  fx_xsectmom:   oos_sharpe=-0.051 psr=0.006 dsr=0.006 pbo=0.655 1.5x=-0.159 n=13

# Data cleanliness proof (post weekday-filter, spike_clean on) -- every pair normal, max moves REAL:
  EURSEK kurt 5.3 max 3.5%; USDCAD 1.8 max 2.5%; XAUUSD 6.8 max 9.6% (2013 gold crash);
  USDCHF/EURCHF max 2015-01-15 (SNB); GBPUSD 2016-06-24 (Brexit); XAGUSD 2026-01-30 (silver squeeze).

# Backfill: 87 missing pair-months -> 48 (Massive) -> 0 (Dukascopy). XAUUSD kurt 296->7.9.
# Run a backtest: python -m src.backtest_runner --config config/backtesting/fx_tsmom.yaml
```

## Files Touched (key)

- `src/strategies/advanced/fx_strategies.py` -- ADDED FxTSMOMStrategy (#3), FxCarryStrategy (#15), FxGoldSilverStrategy (#43), FxXSectMomStrategy (#4). All `forecast_panel(close)` subclasses, Carver scale (10 = 1x). FxCarry loads its own FRED rates (forecast_panel only gets close). ON BRANCH.
- `config/backtesting/fx_{tsmom,carry,goldsilver,xsectmom}.yaml` -- vol_target 0.03, weekly, leverage_cap 4, idm true. ON BRANCH.
- `src/strategies/registry.py` -- registered FxTSMOM/FxCarry/FxGoldSilver/FxXSectMom. ON BRANCH.
- `docs/strategies/FX_60_CATALOG_TRACKER.md` -- the living tracker (base version on main 3a7db91; branch has the 4 gated verdicts).
- MERGED TO MAIN (data layer + cleaning): `src/data/artifacts/*` (8 builders), `src/data/acquisition/plugins/{fred_rates,oil_yfinance,equity_index_yfinance,dukascopy_fx}.py`, `src/data/feeds/holidays_calendar.py`, `src/data/macro_calendar.py`, `src/data/fx_pipeline/`, `src/data/artifacts/spike_clean.py`, `src/data/fx_rates.py` (IR3TIB01), `scripts/data/build_fx_daily_cache.py` (OHLC + weekday filter), `src/backtesting/data/fx_backtest_loader.py` (spike_clean on load), `src/backtesting/validation/{cpcv,combined_gate}.py`, `requirements.txt` (+dukascopy-python, holidays).
- Scratch (gitignored, in `scripts/scratch/`): fx_data_quality_diagnostic.py, fx_internal_validation.py, fx_cross_vendor_validate.py, fx_exhaustive_xval.py, fx_reconfirm_boundary.py, fx_backfill_majors.csv.

## Infra / Data State

- 22-pair G10 daily cache at `Stock_Data/fx_daily/` (gap-free, Mon-Fri, spike+weekday cleaned, cross-vendor validated). Pairs: EURUSD USDJPY USDCHF EURJPY EURCHF CHFJPY XAUUSD XAGUSD GBPUSD USDCAD AUDUSD NZDUSD AUDNZD AUDJPY NZDJPY EURNOK EURSEK USDNOK USDSEK NOKSEK NOKJPY SEKJPY.
- Minute source: all 80 pairs at `Stock_Data/fx/massive/1min/` (Polygon flatfiles S3, MASSIVE_S3_* in .env). `fx_1min` symlink -> `fx/massive/1min`.
- FRED rates: `Stock_Data/alt_data/fred/IR3TIB01*M156N` (+ DFF, ECBDFR). Keyless via pandas_datareader.
- Dukascopy: keyless public feed via `dukascopy_python`. `src/data/acquisition/plugins/dukascopy_fx.py` has DUKA_MAP (direct + derive) + `fetch_pair_month`/`backfill_gaps`.
- Artifacts at `Stock_Data/artifacts/fx/{spread_model,vol_surface,currency_strength,pca_dollar,cointegration,regime,event_registries}`.
- SDD ledger: `.superpowers/sdd/progress.md` (full campaign history).

## Key Takeaways & Gotchas

- `forecast_panel(close)` receives ONLY close (not OHLC, not rates). OHLC is in the cache + `load_fx_daily_panel` panel but NOT passed to the strategy -> the 8 OHLC-bucket strategies need a trivial interface change. Carry-type strategies load their own rates (FxCarry pattern).
- Weekday filter is THE data fix. Do not re-introduce Sat/Sun bars.
- Cross-vendor validation trap: Dukascopy INTERVAL_DAY_1 close vs our 17:00-ET close has a boundary mismatch that FALSE-POSITIVES real volatile days (it flagged SNB 2015). To confirm bad bars, aggregate Dukascopy MINUTE to fx_date (via `resample_fx_minute_to_daily`) OR just check cleaned max|ret| per pair (real events only). The bad-bars mechanism was removed for this reason.
- vol_target 0.03 (not 0.20) for the 22-pair book.
- The 4 strategy commits are on `feat/fx-strategy-validation` (a3d7b5b, 17b3bbc, e7277b2, e15a31f), UNMERGED to main. main=origin=3a7db91.
- Git hazard + standing merge-push authorization (see Resume Here).

## References

- Research: `~/Downloads/compass_artifact_wf-4265ee05-d103-499d-8315-20a63cdd6e8f_text_markdown.md`, `~/Downloads/fx_strategy_deep_dive.md` (strategies referenceable as "deep-dive #N").
- Tracker: `docs/strategies/FX_60_CATALOG_TRACKER.md`.
- Spec/plan: `docs/superpowers/specs/2026-07-06-fx-data-compute-layer-design.md`, `docs/superpowers/plans/2026-07-06-fx-data-compute-layer.md`.
- Walk-forward reports: `docs/reports/fx/FX_WALK_FORWARD.md` (each run overwrites; use the `.superpowers/sdd/wf_*.json` for per-strategy numbers).
- Prior handoff (spot-fx build): `docs/progress/SESSION-HANDOFF-spot-fx-2026-07-06.md`.
