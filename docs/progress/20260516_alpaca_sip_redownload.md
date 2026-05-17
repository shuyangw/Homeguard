# Alpaca SIP Equities Redownload - 2026-05-16

## Summary

Built infrastructure to redownload the full Alpaca-tradable US equity universe
(~10k symbols) at 1-min resolution using the SIP feed, producing two parallel
datasets (raw + split-adjusted) under H:\Stock_Data\ that mirror the existing
equities_1min folder structure and 8-column canonical schema.

## Changes Made

- **src/data/acquisition/plugins/alpaca_equities.py**: added `feed`,
  `adjustment`, `storage_subdir_override` ctor params; passes them through to
  StockBarsRequest.
- **src/data/acquisition/base.py**: atomic parquet writes (.tmp + os.replace),
  periodic manifest flush (every 25 completions or 60s), targeted 10s backoff
  on 429/5xx Alpaca API errors, JSONL progress event log per pass.
- **src/data/acquisition/manifest.py**: added `reap_in_progress()` for crash
  recovery (downgrades in_progress -> pending).
- **src/data/acquisition/alpaca_universe.py** (new): snapshots Alpaca's active
  tradable US equity list to a dated CSV.
- **src/data/acquisition/status_tracker.py** (new): joins manifest + on-disk
  parquet stats into per-pass status CSV with good/broken/incomplete/failed.
- **scripts/data/redownload_sip_equities.py** (new): two-pass orchestrator.
- **scripts/data/validate_sip_dataset.py** (new): OHLCV invariant + bar count
  + monotonicity checks per pass.
- **scripts/data/compare_raw_vs_split.py** (new): cross-feed consistency.
- Tests: extended `test_equities_plugin.py`, `test_base.py`, `test_manifest.py`;
  new `test_alpaca_universe.py`, `test_status_tracker.py`,
  `integration/test_sip_e2e.py` (gated `-m integration`).

## Usage

Full universe pull (raw + split, 12 threads):

    conda run -n fintech python scripts/data/redownload_sip_equities.py

Single feed, custom date range:

    conda run -n fintech python scripts/data/redownload_sip_equities.py \
        --feeds raw --start 2020-01-01 --threads 24

Resume after crash: rerun the same command. The orchestrator reaps in-progress
entries and skips symbols already marked complete.

Re-pull symbols that failed previously:

    conda run -n fintech python scripts/data/redownload_sip_equities.py --retry-failed

Validate a completed pass:

    conda run -n fintech python scripts/data/validate_sip_dataset.py --pass raw
    conda run -n fintech python scripts/data/validate_sip_dataset.py --pass split
    conda run -n fintech python scripts/data/compare_raw_vs_split.py

## Outputs

| Path | Description |
|---|---|
| `H:\Stock_Data\equities_1min_sip_raw\` | SIP raw bars (same partition layout as existing folder) |
| `H:\Stock_Data\equities_1min_sip_split\` | SIP split-adjusted bars |
| `H:\Stock_Data\_manifests\equities_1min_sip_raw.status.csv` | Per-symbol status tracker (raw) |
| `H:\Stock_Data\_manifests\equities_1min_sip_split.status.csv` | Per-symbol status tracker (split) |
| `H:\Stock_Data\_manifests\equities_1min_sip_raw.progress.jsonl` | Append-only event log (raw) |
| `H:\Stock_Data\_manifests\equities_1min_sip_split.progress.jsonl` | Append-only event log (split) |
| `output\data_acquisition\sip_redownload-YYYYMMDD-HHMMSS.log` | Per-run human log |
| `output\data_acquisition\sip_redownload-YYYYMMDD-HHMMSS-summary.md` | Final markdown summary |
| `output\data_validation\<subdir>_coverage.csv` | Per-symbol coverage |
| `output\data_validation\<subdir>_low_bar_days.csv` | Low-bar-day flags |
| `output\data_validation\raw_vs_split_diff.csv` | Cross-feed diffs |
| `config\universes\alpaca_active-YYYYMMDD.csv` | Frozen universe snapshot |

## Commits

- `06de33b` feat(data): AlpacaEquitiesPlugin accepts feed, adjustment, storage_subdir_override
- `ec17931` test(data): verify storage_subdir_override routes correctly
- `02e437e` feat(data): atomic parquet writes via .tmp + os.replace
- `a5fc959` feat(data): periodic manifest flush during download (every 25 completions or 60s)
- `477756f` feat(data): extra 10s backoff on 429/5xx Alpaca API errors
- `4ee893c` feat(data): JSONL progress event log per pass
- `9b18dea` feat(data): DownloadManifest.reap_in_progress for crash recovery
- `91b5e24` feat(data): alpaca_universe.list_active_us_equities snapshot helper
- `9d8dce7` feat(data): status_tracker module (compute_status, rebuild_tracker, atomic write)
- `e21e6d3` feat(scripts): SIP redownload orchestrator skeleton (CLI + logger)
- `e3680bc` feat(scripts): orchestrator resolves universe from --symbols-from
- `62cc362` feat(scripts): orchestrator runs raw+split passes with reap/resume + tracker regen
- `fd74e4d` feat(scripts): orchestrator writes final session summary markdown
- `84d898a` feat(scripts): validate_sip_dataset.py (OHLCV invariants + coverage + low-bar)
- `5aa83ce` feat(scripts): compare_raw_vs_split.py cross-feed consistency check
- `188a59d` test(data): SIP redownload end-to-end integration test (gated)
- `32fcfba` fix(scripts): use paper trading endpoint for universe snapshot (env has paper keys)
- `3176afd` perf(scripts): skip rebuild_tracker when pass had no work (saves ~50min on resume)

## Full-universe pull results (2026-05-16/17)

Executed the full ~10k-symbol pull. Pipeline ran successfully across two real-world
crashes (terminal closed mid-run, deliberate kill mid-split to bump from 12 to 24
threads). Resume worked correctly in both cases via the manifest reap + skip-existing
mechanism — zero data lost, zero parquet corruption.

| Metric | Raw pass | Split pass |
|---|---|---|
| Universe size | 12,345 | 12,345 |
| `status=good` | 12,043 (97.6%) | 12,043 (97.6%) |
| `status=incomplete` (zero-row, no SIP data) | 302 (2.4%) | 302 (2.4%) |
| `status=failed` | **0** | **0** |
| Total bars on disk | 3,022,010,033 | 3,022,010,033 |
| Median bars per symbol | 74,484 | 74,484 |
| Earliest first_date | 2016-01-01 | 2016-01-01 |
| Latest last_date | 2026-05-16 | 2026-05-16 |
| On-disk size | 94 GB | 90 GB |

Combined ~6 billion 1-min SIP bars, ~184 GB total on H:. Identical bar counts and
date ranges across passes (by design — adjustment changes prices but not timestamps
or trade counts). 302 "incomplete" symbols are tickers where Alpaca had no SIP data
at all (likely delisted pre-2016 or class-share variants).

**Wall-clock summary**:
- Raw pass: ~13h (including 1 crash + resume gap)
- Inter-pass tracker rebuild: ~52 min
- Split pass: ~8h (post-restart with 24 threads; ~3h with 12 threads pre-restart)
- Total: ~24h with planned 12->24 thread bump mid-run

**Frozen universe snapshot**: `config/universes/alpaca_active-20260516.csv` (12,346 symbols).

## Known Issues / Remaining Work

- Cutover plan: which strategies should switch from IEX-raw to SIP-raw or
  SIP-split. Requires per-strategy backtest comparison before any production
  change.
- Existing `H:\Stock_Data\equities_1min` folder remains untouched (IEX-raw).
  Decision on archive/delete deferred until cutover lands.
- Nightly incremental updates not yet implemented; the orchestrator is built
  for one-shot or resumable bulk pulls. Add an `--incremental` mode later if
  needed.
- Dividend-adjusted ("all" adjustment) feed not produced. Add later if any
  strategy explicitly needs total-return prices.

## Validation

- Unit tests: `conda run -n fintech pytest tests/data/test_acquisition/ -v`
- Integration tests (real API): `conda run -n fintech pytest -m integration tests/data/test_acquisition/integration/`
- Manual: 5-symbol e2e in the integration test covers reap-and-resume,
  folder/schema parity, and raw/split timestamp+trade_count consistency.
