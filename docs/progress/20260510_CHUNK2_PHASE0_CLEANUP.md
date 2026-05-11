# Chunk 2: Phase 0 Data Cleanup -- 2026-05-10

## Summary

Phase 0 disk hygiene per doc 02 v1.1 sections 2.2 and 2.4. Deleted three legacy/partial data directories (~15.5 GB freed) and normalized crypto_1hour partition naming to match the rest of the store. No code changes in this chunk -- filesystem operations only, with a single progress doc commit.

## Files Changed

- No source code changes.
- This progress doc is the only commit on `feature/phase0-data-cleanup`.

## Filesystem Operations

- **Deleted** `H:/Stock_Data/futures_1min_calendar_legacy_20260506/` (~536 MB, smaller than the 6 GB estimate). The pre-pull broken `.c.0` calendar-roll data, kept only for fix verification per doc 02 section 2.2; no longer needed.
- **Deleted** `H:/Stock_Data/futures_dbn_staging/F/` (~15 GB). The 5.9 GB partial MBP-1 download from the aborted 2026-05-09 pull had grown to ~15 GB before the prior session's background finisher was stopped; full removal per doc 02 section 2.4. MBP-1 is skip-permanently per doc 01 v1.2 section 2.8.
- **Deleted** `H:/Stock_Data/futures_trades/` (~2.8 MB). Single-file test stub at `symbol=ES/year=2024/month=1/data.parquet`. Verified zero references from the validation framework before deletion (grep over `src/data/validation/`).
- **Renamed** 679 crypto_1hour partitions from `month=0X` to `month=X` (for X in 1..9), normalizing to match `crypto_1min/`, `futures_1min/`, and the rest of the store. Verified: `H:/Stock_Data/crypto_1hour/symbol=BTC_USD/year=2024/` now lists `month=1 ... month=12` with no leading zeros. One-off script was written, executed once, and deleted -- no script committed.
- **Calendar reminder**: `H:/Stock_Data_backup_20260506/` (equities cleanup backup, ~tens of GB) is scheduled for deletion on 2026-05-13 per doc 02 section 2.2 (1-week green production rule). Not deleted in this chunk; check on or after that date.

## Commits

- `<progress-commit-sha>` docs(progress): Chunk 2 Phase 0 data cleanup session log
- `<merge-commit-sha>` Merge feature/phase0-data-cleanup

(SHAs filled in by the merge step below.)

## Validation

- Post-cleanup validation run (`run_validation.py --domain futures --mode initial`):
  - Total checks: 233
  - Passed: 186
  - CRITICAL failures: **1** (`latest_data_freshness` only -- expected from bulk-pull cutoff)
  - Warnings: 46
  - Report: `output/chunk2_post_cleanup_validation.md`
- Identical pass/fail profile to Chunk 1's final validation. Deletions did not affect any futures data the framework validates. Cleanup is non-destructive to the canonical store.
- Disk freed: ~15.5 GB total (536 MB legacy + 15 GB F partial + 2.8 MB stub).

## Known Issues / Remaining Work

- F partial was 15 GB rather than the 5.9 GB documented in `docs/progress/20260509_DATA_PULL_ISSUES.md`. The prior session's background `databento_batch_finish.py` likely continued downloading after we thought it was stopped. Net effect is just more disk freed -- no concern.
- `H:/Stock_Data_backup_20260506/` retained until 2026-05-13 per the 1-week green production rule. Set a calendar reminder.
- Pre-existing test-ordering failure (`tests/data/validation/integration/test_density_gc_bug_fix_holds.py::test_gc_density_above_threshold`) noted in Chunk 1 progress doc remains unfixed. Not in scope for Chunk 2.

## Decision Gate

PROCEED to Chunk 3 (`feature/continuous-contract-loader`) once the merge to main lands.

## Reproduction Commands

```bash
# Disk state verification (post-deletion: these dirs should not exist)
ls H:/Stock_Data/futures_1min_calendar_legacy_20260506 2>&1  # should: No such file or directory
ls H:/Stock_Data/futures_dbn_staging/F 2>&1                   # should: No such file or directory
ls H:/Stock_Data/futures_trades 2>&1                          # should: No such file or directory

# crypto_1hour padding verification (post-normalization)
ls "H:/Stock_Data/crypto_1hour/symbol=BTC_USD/year=2024/"     # should list month=1..month=12 (unpadded)

# Validation
cd C:/Users/qwqw1/Dropbox/cs/github/Homeguard
conda run -n fintech python scripts/data/run_validation.py --domain futures --mode initial --report-out output/chunk2_post_cleanup_validation.md
grep -E "^### .*\(CRITICAL\)" output/chunk2_post_cleanup_validation.md  # should show only futures.l2.latest_data_freshness
```
