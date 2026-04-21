# Data Directory Consolidation - 2026-04-20

## Summary

Consolidated Homeguard's local data stores from 4 roots across 3 drives (H:, E:, C:\Dropbox) down to 2 roots on H: — `H:\Stock_Data` for all market data and `H:\Homeguard_Output` for logs and reports. Freed ~294 GB (234 GB from E:\OptionsData + ~60 GB of Dropbox cloud quota). EC2 / `[linux]` configuration unaffected.

## Motivation

Pre-migration layout sprawled across 4 roots:

| Root | Size | Role |
|---|---|---|
| `H:\Stock_Data` | 111 GB | Equities/crypto/futures OHLCV (canonical) |
| `E:\OptionsData` | 234 GB | ThetaData options (canonical) |
| `C:\...\Dropbox\cs\stonk\data` | 16 GB | Stale partial duplicate from macOS era |
| `C:\...\Dropbox\cs\stonk\{logs,output,homeguard_gui_logs}` | 44 GB | Backtest logs + reports |

The Dropbox copy existed because `[macos]` in `settings.ini` pointed there, but macOS has been retired for 30+ commits. Having 4 roots also burned ~60 GB of Dropbox quota on throwaway logs, conflated data with backup policies, and required 3 drives for any new setup.

## Target Layout

```
H:\Stock_Data\                  # market data (~345 GB)
├── equities_1min\                (unchanged)
├── equities_1min_by_date\        (unchanged)
├── sp500_minute_cache*.parquet   (unchanged)
├── crypto_1min\, crypto_1hour\, crypto_1day\
├── futures_1min\, futures_trades\
├── news\, sentiment\, metadata\, cache\
├── metrics_snapshots\
└── options\                      # absorbed E:\OptionsData
    ├── options_combined\           (233 GB MOVED from E:)
    └── _logs\                      (208 MB MOVED from E:)

H:\Homeguard_Output\            # logs + reports (~45 GB)
├── logs\                         (38 GB MOVED from Dropbox\cs\stonk\logs)
├── output\                       (6 GB MOVED from Dropbox\cs\stonk\output)
├── discord_bot\                  (MOVED from Dropbox\cs\stonk\logs\discord_bot)
└── homeguard_gui_logs\           (MOVED from Dropbox\cs\stonk\homeguard_gui_logs)
```

Deleted after byte-level verification:
- `E:\OptionsData\` (234 GB)
- `C:\Users\qwqw1\Dropbox\cs\stonk\{data,logs,output,homeguard_gui_logs}` (~60 GB)

## Changes Made

### Config
- **`settings.ini`**: New `[windows]` section with H: paths. `[macos]` section removed. `[linux]` preserved with original EC2 paths. Backup saved at `settings.ini.bak`.
- **`settings.ini.example`**: Mirrored new layout.

### Code (4 hardcoded paths replaced)
- **`tools/options-downloader/config.go:6`**: `E:\OptionsData` → `H:\Stock_Data\options`
- **`src/gui/utils/error_logger.py`**: Hardcoded Dropbox path → `get_output_dir() / "homeguard_gui_logs"`. Added `get_log_directory()` accessor.
- **`src/gui/app.py:629`**: Fatal error message now builds log path from `get_log_directory()`
- **`scripts/test_sweep_tearsheet.py:85`**: Hardcoded Dropbox path → `get_output_dir() / "test_sweep_tearsheet"`
- **`scripts/data/check_sp500_coverage.py:6`**: Hardcoded `F:\Stock_Data` → `get_local_storage_dir()`

### Documentation (7 files refreshed)
- `.claude/data_handling.md` — platform path table
- `src/settings/CONFIGURATION_SYSTEM.md` — platform table + settings.ini example
- `src/data/DATA_PROVIDERS.md` — settings.ini example
- `src/visualization/VISUALIZATION.md` — settings.ini example
- `src/gui/docs/USER_GUIDE.md` — OS enumeration
- `SETUP.md` — Windows/Linux setup examples
- `docs/strategies/20251230_OPEX_PINNING_STRATEGY_STATUS.md:54` — options data path

### Copy operations
5 parallel robocopies using `/MIR /J /R:2 /W:5 /NFL /NDL` options. Options copy (233 GB) completed in 18m50s at 222 MB/s. All 6 source/destination pairs validated byte-for-byte via `Get-ChildItem -Recurse | Measure-Object Length -Sum`.

## Commits

To be added after commit.

## Known Issues / Remaining Work

- Historical progress doc `docs/progress/20251129_LIGHTGBM_MOMENTUM_STRATEGY_STATUS.md` retains `F:\Stock_Data` references — intentionally preserved as point-in-time record.
- `settings.ini.bak` kept for emergency rollback. Safe to delete after 1-2 weeks of stable operation.
- Dropbox will propagate cloud deletes (intentional — 30-day version history available if needed).

## Validation

1. **Byte-level integrity**: All 6 source/destination pairs compared; sizes matched. Spot-checked SPY options parquet byte-size (21,540,295 bytes == 21,540,295 bytes).
2. **Settings resolution**: `get_local_storage_dir()`, `get_options_data_dir()`, `get_output_dir()`, `get_discord_bot_log_dir()` all return correct new paths.
3. **Live imports**:
   - `ThetaDataAdapter` initializes with `options_combined_dir = H:\Stock_Data\options\options_combined`
   - `error_logger.LOG_DIR` resolves to `H:\Homeguard_Output\homeguard_gui_logs`; import-time logger creates new log file
   - SPY equities 1min parquet loads with canonical 8-column schema (7269 rows spot-checked)
4. **EC2 unaffected**: `[linux]` section untouched; bot status should remain green.

## Rollback

If issues arise: `copy /Y settings.ini.bak settings.ini`. Sources are deleted, so rollback restores config only — data is only in H: now.
