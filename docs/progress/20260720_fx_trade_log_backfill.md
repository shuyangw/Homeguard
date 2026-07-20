# FX Fills-Level Trade-Log Backfill - 2026-07-20

## Summary
Backfilled the 8 missing fills-level `trades.csv` artifacts for the closed FX
catalog campaign (the 6 Wave-1-enhanced + Wave-2 strategies plus #20 that
persisted only daily return-streams). Artifact-only remediation: no verdict was
touched, no gate / walk-forward / PSR-DSR-PBO was re-run, no registry trial was
appended (`register=False` throughout). All 12 gated strategies remain FAIL; the
catalog stays CLOSED. This closes the trade-log integrity gap flagged at the end
of the Wave-2 session (only 4 of 12 strategies had fills logs).

## What happened
1. Routed through `strategy-lead` (backtests are gated by the
   `strategy_lead_gate` hook; only strategy-lead sets the sentinel). This is the
   first exercise of the strengthened trade-log verification rule added to the
   agent def in `3106b9c`.
2. Diagnosed the gap per path before dispatch: the daily strategies
   (#16/#19, #33, #39, #42) already had a `log_trades` path in
   `run_fx_backtest` that the gate runners never enabled; the spread simulator
   already returned a fills DataFrame the runner discarded; the intraday engine
   already collected `self.fills` the breakout runner discarded.
3. strategy-lead ran one full-window logged backtest per strategy, wired
   persistence into the two runners that lacked it, and verified each artifact
   by direct file read (row count > 1, fills not return-stream) before marking
   done.

## Artifacts produced (output/backtests/fx/<Strategy>/2011-01-01_to_2026-04-01/trades.csv)
| Strategy | Rows |
|---|---:|
| FxCarrySeatbelt | 5177 |
| LondonBreakout | 6162 (entry fills only, see limitation) |
| FxTurnOfMonth | 1679 |
| FxPcaDollarResidual | 4296 |
| FxRoroRegimeSpread | 1410 |
| AudNzdPairs | 360 |
| CointScanner | 1076 |
| VolRatioPair | 706 |

## Changes Made
- **scripts/backtest_scripts/run_fx_spread_backtest.py**: added `_write_trade_log`
  helper + optional `log_trades` param (default False, no behavior change for
  existing callers) + `--log-trades` CLI flag. Mirrors `fx_backtest.py`'s
  `_write_trade_log` convention. Persists `trades.csv` + `equity.csv`.
- **scripts/backtest_scripts/run_fx_london_breakout_walkforward.py**: added
  `build_trade_log` / `_run_trade_log_backfill` + `--trade-log` flag collecting
  the intraday engine's entry fills.
- **docs/strategies/research/20260720_fx_trade_log_backfill.md**: results doc
  (strategy-lead).
- 8 `trades.csv` artifacts under `output/backtests/fx/` (tracked normally; that
  path is not gitignored).

## Commits (all pushed to origin/main)
- `f3adc78` backfill 4 daily strategies' trades.csv
- `f95029f` wire trade-log persistence into run_spread_backtest
- `18a18aa` backfill 3 spread strategies' trades.csv
- `5345f03` add trade-log backfill mode to London Breakout runner
- `b411aa0` backfill LondonBreakout trades.csv
- `58ab626` results doc

## Known Issues / Remaining Work
- **LondonBreakout captures entry fills only.** Exit fills (stop/target/trail/EOD)
  are not persisted anywhere in the shared `OrderEngine`; reconstructing them was
  out of scope for an artifact-only backfill on a FAILED strategy. Documented in
  the runner docstring and results doc. If exit-level fidelity is ever needed,
  the fix belongs in `OrderEngine`, not the runner.
- **FxRoroRegimeSpread red flag (non-blocking):** only 2 of 4 universe pairs ever
  trade and the last fill is 2024-09-10. Consistent with the strategy's REJECT
  verdict; noted for the record, not investigated (campaign closed).

## Validation
- All 8 artifacts verified on disk by direct file read (row count, fills-not-
  return-stream) by strategy-lead before close-out.
- Both runner diffs reviewed line-by-line: minimal, backwards-compatible,
  ASCII-clean, no `print()`.
- Governance held: commits by explicit path only, no `settings.ini` / sentinel /
  `.tmp` committed, sentinel confirmed removed. Orchestrator (this loop) owned
  the push.
