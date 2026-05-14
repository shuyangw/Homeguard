# FX Data Expansion - 2026-05-13

## Summary

Built a Polygon/Massive S3 flat-file FX downloader and used it to grow the `fx_1min/` store from 50 to 55 symbols (+10.5% rows, +27M minute bars). Net additions: 5 truly new pairs (NZDCHF, CADCHF, GBPNZD, EURSGD, SGDJPY) plus a full 2010-2026 history extension for 2 pre-existing pairs (USDNOK, USDSEK) that previously had only partial coverage. The work was driven by a CC-authored plan at `C:\Users\qwqw1\Downloads\20260513_fx_data_expansion_and_probing_plan.md`, adapted during execution to match the actual Massive S3 endpoint and CSV schema.

## Changes Made

### New code
- **`src/data/acquisition/plugins/massive_fx_flat.py`** (~250 LOC): standalone single-pass downloader. Not a `BaseDownloader` subclass because Massive's flat files are per-day-all-pairs, not per-symbol-per-day; symbol-iteration would re-download each 15 MB file once per ticker. Iterates days, fans out to per-symbol monthly buffers, atomic parquet write at month boundary. `ThreadPoolExecutor` parallelizes per-day S3 fetches within a month. Respects `skip_existing` for resumability.
- **`scripts/data/download_fx.py`** (~100 LOC): thin CLI wrapper reading the universe CSV and dispatching to `download_pairs`. Args: `--tier`, `--symbol`, `--start`, `--end`, `--no-skip-existing`, `--dry-run`.
- **`config/universes/fx-2026.csv`**: 7-row universe definition with `symbol, tier, added_date, effective_start_date, massive_ticker, notes` columns. Follows existing `config/universes/sp500-2025.csv` / `russell1000-2025.csv` pattern.
- **`tests/data/test_acquisition/test_massive_fx_flat.py`**: 12 unit tests covering CSV parsing, schema match, dtype enforcement, malformed-line resilience, sort + dedup, atomic write, frozen dataclass, month-grouping helper.

### Documentation
- **`docs/reference/DATA_INVENTORY.md`**: updated FX section with new totals (55 symbols, 9,903 partitions, 284.6M rows, 2010-01 → 2026-05). Recorded:
  - Source: Polygon/Massive flat-files (not "mixed" anymore)
  - Auth pattern (separate `MASSIVE_S3_*` env vars from REST `MASSIVE_API_KEY`)
  - Coverage notes (USDCNH pre-2014, XAGUSD pre-2013 archive limits; 2019-09 and 2020-10/11 Polygon-specific outages)
  - `volume == trade_count` FX convention (OTC market has no real volume)
  - `vwap = close` approximation for new pairs (Polygon flat-file omits vwap)

### Credentials
- **`.env`** (gitignored): added `MASSIVE_S3_ACCESS_KEY`, `MASSIVE_S3_SECRET_KEY`, `MASSIVE_S3_ENDPOINT=https://files.massive.com`, `MASSIVE_S3_BUCKET=flatfiles`.

### Dependencies
- `boto3` installed in `fintech` conda env (no pip requirement added to repo — it's an env-level dep).

## Commits

- `9288f21` feat(data): Massive flat-file FX downloader for universe expansion
- `9614f4b` docs(data): record FX universe expansion (50->55 pairs, 257.6M->284.6M rows)

Both pushed to `origin/main`.

## Pairs Acquired

| Pair | Type | Rows | Coverage | Notes |
|---|---|---|---|---|
| USDNOK | history extension | 5,863,904 | 2010-2026 | was partial; now full |
| USDSEK | history extension | 5,889,486 | 2010-2026 | was partial; now full |
| NZDCHF | new | 6,012,970 | 2010-2026 | Antipodean-haven cross |
| CADCHF | new | 6,000,937 | 2010-2026 | Commodity-haven cross |
| GBPNZD | new | 6,016,659 | 2010-2026 | GBP-antipodean cross |
| EURSGD | new | 6,033,520 | 2010-2026 | EUR-Asia cross |
| SGDJPY | new | 2,192,628 | **2020-2026** | Polygon archive starts 2020 |

## Validation

### Schema parity (vs existing EURUSD partition)
All 7 pairs exact match: columns `[timestamp, open, high, low, close, volume, trade_count, vwap]`, dtypes `[Datetime[ns, UTC], Float64×5, Int64×2, Float64]`. Verified via `tests/data/test_acquisition/test_massive_fx_flat.py` (12/12 passing) plus a side-by-side polars `read_parquet` comparison.

### Cross-rate triangulation (Dec 2025 overlap, ~30k bars each)
| Triangulation | Mean bps | Std bps | Max\|bps\| | Outliers >50bps |
|---|---|---|---|---|
| USDNOK ≈ EURNOK/EURUSD | 0.63 | 1.87 | 43.5 | 0.00% |
| USDSEK ≈ EURSEK/EURUSD | 0.69 | 2.03 | 28.6 | 0.00% |
| GBPNZD × NZDUSD ≈ GBPUSD | -2.93 | 3.56 | 118.9 | 0.08% |
| EURUSD × USDSGD ≈ EURSGD | 0.36 | 1.64 | 15.1 | 0.00% |
| NZDUSD × USDCHF ≈ NZDCHF | 2.89 | 11.33 | 158.4 | 0.99% |

All means under 3 bps. Strong evidence the new pairs are internally consistent with the existing 50.

### Bulk pull stats
- Elapsed: 23.3 min for 5,977 days attempted (5,277 present, 700 weekend/missing)
- Months written: 879
- Months skipped (pre-existing): 369 — confirming `--skip-existing` correctly preserved USDNOK/USDSEK partial pre-existing coverage
- 60 GB of S3 bandwidth, 6.76 GB final on-disk

### Validation script
`scripts/scratch/check_new_fx_data.py` runs all of the above checks. Re-runnable any time to re-validate.

## Process Notes

### Plan corrections during execution
The source plan had four factual errors caught by an Explore agent verification pass:
1. Assumed `s3://files.massive.com/currencies/minute_aggs_v1/...` URI (wrong — actual is `s3://flatfiles/global_forex/minute_aggs_v1/`)
2. Referenced `config/data/fx_universe.yaml` (doesn't exist; created `config/universes/fx-2026.csv` matching existing universe-CSV convention)
3. Referenced `chains.parquet` filename (canonical is `data.parquet`)
4. Required `dukascopy-python` dep (dropped; Probes 2/3 deferred per user decision)

### Schema mismatch caught during S3 probe
The plan assumed Massive flat-files were accessible with the existing `MASSIVE_API_KEY` (REST key). Empirical probe of 9 URL+auth combos returned all 403s — confirmed flat-files need separate S3 access-key + secret pair. User retrieved these from the Massive dashboard, plan resumed.

### XPTUSD/XPDUSD dropped from Tier 1
Source plan included platinum and palladium as Tier 1 metals. Probe confirmed these aren't in Massive's `global_forex` bucket — only XAU/XAG. They exist on Massive only as NYMEX/COMEX futures (PL, PA), a different product class. Dropped from Tier 1; deferred to a futures-data plan if needed.

### USDNOK/USDSEK already existed (caught mid-pull)
Initial framing called these "Tier 1 unconditional additions". Mid-pull verification revealed they were already in the 50-pair store with partial coverage (timestamps from 2025-04). The bulk pull's `--skip-existing` mode preserved the pre-existing months and added missing historical depth (2010-2024). Net effect: 5 truly new pairs + 2 history-extended pairs.

### SGDJPY effective_start corrected
Plan initially set SGDJPY effective_start_date to 2015-01-01 based on a hypothesis that "Polygon's SGD coverage thickens around 2015". Empirical bulk pull found Polygon has zero SGDJPY data before 2020. CSV updated to `effective_start_date=2020-01-01` post-acquisition.

## Known Issues / Remaining Work

### Deferred to separate plans
- **Dukascopy gap-fill** for 2019-09 multi-pair outage and EURUSD 2020-10/11. Need `dukascopy-python` dep and a new plugin. Source-of-truth comparison from `scripts/scratch/check_fx_anomalies.py` already identifies the gap windows.
- **FX validation domain** (`src/data/validation/fx/`). Currently a placeholder docstring. Mirror `src/data/validation/futures/` structure. Cross-rate triangulation logic from `check_new_fx_data.py` is ready to productionize.
- **Strategy universe pruning** (`active:bool` per pair in strategy configs). 11 pairs flagged for deactivation in CC's prior analysis (NDF-only, sanctions, pegged, etc.). Data stays on disk; only strategy config changes.
- **Dtype canonicalization** (`[ns, UTC]` → `[us, UTC]` for all 55 pairs). Documented inventory drift; doesn't break consumers because polars normalizes transparently.

### Subtle items worth noting
- New pairs have `vwap = close` (Polygon flat-file omits vwap); existing 50 pairs have a separate provider-computed vwap. Consumers using vwap should be aware of this dual convention.
- SGDJPY has only 6 years of history vs 16 for the others. Strategies should honor `effective_start_date` from `config/universes/fx-2026.csv` to avoid backtest bias.
- The `_SUCCESS` marker file under some symbol/year directories is from an earlier Hive write (unclear which); not produced by our plugin. Harmless but worth investigating in a cleanup pass.

## File References

- Plugin: `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\src\data\acquisition\plugins\massive_fx_flat.py`
- CLI: `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\scripts\data\download_fx.py`
- Universe: `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\config\universes\fx-2026.csv`
- Tests: `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\tests\data\test_acquisition\test_massive_fx_flat.py`
- Inventory: `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\docs\reference\DATA_INVENTORY.md`
- Validation: `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\scripts\scratch\check_new_fx_data.py` (gitignored scratch; re-runnable)
- Plan: `C:\Users\qwqw1\.claude\plans\approved-continue-on-for-encapsulated-hopcroft.md`
- Source plan: `C:\Users\qwqw1\Downloads\20260513_fx_data_expansion_and_probing_plan.md`
