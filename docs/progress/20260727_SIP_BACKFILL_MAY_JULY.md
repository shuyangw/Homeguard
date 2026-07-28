# SIP Equities Backfill: May-July 2026 Gap - 2026-07-27

## Summary

Closed a ~50-trading-day gap in the SIP equity datasets (data ended 2026-05-15;
today 2026-07-27). Both feeds now extend to 2026-07-27 23:59 UTC with 12,345
symbols complete and zero failures. Along the way, found and repaired 209
symbols whose split-adjusted series had been corrupted by the partial re-pull,
and fixed two real robustness bugs in the acquisition framework.

## What was done

### 1. Month-aligned backfill (not date-aligned)

`_save_partitioned` overwrites each monthly partition wholesale -- it does not
merge. Starting the re-pull at 2026-05-15 (where data ended) would have
rewritten the May partition with only May 15-31 and **silently destroyed May
1-14**. The pull was therefore aligned to the month boundary (`--start
2026-05-01`), so May was rewritten complete and June/July landed as new
partitions.

Verified against a pre-run baseline: May went from 11 trading days / 9,563 rows
(AAPL) to 20 days / 16,787 rows, first bar still 2026-05-01. No data lost.

### 2. Split-adjustment discontinuity: found and repaired

Back-adjustment is retroactive and therefore **not point-in-time stable**:

    P_adj(t | as_of=T) = P_raw(t) / prod(f_s for t < s <= T)

Pre-May partitions carried adjustment factors as of the 2026-05-16 download;
freshly-written May+ partitions carried factors as of 2026-07-28. Any symbol
that split in between had a spurious jump at the April/May partition boundary.

Detection: `r(t) = raw_close(t) / split_close(t)` equals the cumulative split
factor from `t` forward. Compared `r` on the last April bar vs the first May
bar across the full universe.

- **209 symbols flagged** (173 reverse splits, 36 forward)
- **8 leveraged/inverse ETFs affected**: BOIL, BZQ, DRIP, EFZ, EMTY, SCO,
  SOXS, TZA -- directly in the overnight-reversion universe. SOXS showed a
  10x apparent overnight move.
- Large caps affected too (KLAC 10-for-1), so the momentum sleeve was exposed.
- Reverse-split dominance is structural: leveraged/inverse ETFs decay by
  construction and reverse-split periodically to stay in a tradable range.

Repair: full-history re-pull of the 209 on the split feed only (44.4M rows,
~14 min), so every partition carries current factors. Re-scan: **209 -> 5**.

The 5 residuals (GMEX, MRAL, MSTP, SMCL, TLIH) are **genuine splits at the
boundary**, not corruption -- confirmed with a second test: raw jumps
7.6x-21.2x while the split series stays continuous (0.97-1.09). The
boundary-ratio test alone cannot distinguish a real boundary split from
corruption; the continuity check is the discriminator.

**Raw was unaffected throughout** -- raw prices are immutable, which is exactly
why they are the correct ground truth.

### 3. Bug: telemetry could kill an entire download

`_emit_event` appends to the progress JSONL. On Windows, a concurrent reader
(tail/grep/antivirus) can hold that file without write-sharing, making the
append fail with PermissionError. That exception escaped the worker thread,
propagated through `future.result()`, and **aborted a 12,345-symbol pass at
~85% completion**. Triggered in this session by a monitoring loop grepping the
log, but antivirus would do the same on an unattended run.

Fix: retry briefly (3x, 50ms), then drop the event with a logged warning.
Telemetry can no longer take down the payload it describes.

Note the failure mode it exposed: workers write parquet and update the
in-memory manifest, but `manifest.save()` runs only in the main loop. When the
main loop died, ~3,300 symbols' completed work was on disk but unrecorded.
Data was never at risk; bookkeeping lagged.

### 4. Bug: unflattened tracker CSV path

The asset-class migration flattened nested subdirs for the manifest JSON and
event log but missed `run_pass` / `validate_sip_dataset`, which were still
writing `_manifests/equities/sip_split/1min.status.csv` instead of
`_manifests/equities_sip_split_1min.status.csv`. Fixed; stray nested directory
removed.

## Commits

- `c3c6ac4` fix(data): telemetry must not kill downloads; flatten tracker CSV path

## Final state

| | value |
|---|---|
| Feeds | sip_raw, sip_split |
| Symbols complete | 12,345 each, 0 failed |
| Coverage | 2016-01-01 -> 2026-07-27 23:59 UTC |
| July 2026 trading days | 18 |
| Split-adjustment discontinuities | 0 (5 genuine splits, correctly adjusted) |

## Known Issues / Remaining Work

- **Tracker status CSVs are stale** (May 17/18). Purely informational -- the
  manifest JSON drives all resume/retry logic. Regenerate when convenient;
  the full scan takes ~50 min per feed.
- **`--incremental` mode still unbuilt.** Without it the gap simply reopens.
  Critical constraint for whoever builds it: **incremental updates are sound on
  raw only.** The split feed must be re-pulled in full, or have affected
  symbols detected and repaired each time -- roughly 200 symbols per quarter
  based on this run. The boundary-ratio scan should become a standing
  post-update step.
- **Background jobs are reaped at ~60 min wall-clock.** The full two-feed run
  (~2h, dominated by tracker rebuilds rather than network) exceeds this and was
  killed mid-rebuild. Split long runs into per-feed jobs.
- **Tracker rebuild costs more wall-clock than the data acquisition it
  describes**, and re-scans every symbol even when only three months changed.
- **New listings since 2026-05-16 are not included** -- the frozen universe was
  used deliberately so existing symbols got consistent treatment. Genuinely-new
  symbols need a separate full-history pull.
- **Total-return (dividend-adjusted) series still absent.** Price-return
  ranking systematically penalizes high-payout names in cross-sectional
  momentum.

## Validation

- `pytest tests/data/test_acquisition/` -> 122 passed
- Baseline comparison on 8 liquid names confirmed May 1-14 survived
- Full-universe split-consistency scan: 11,813 symbols checked, 0 corrupt
- Manifest completeness: 12,345 complete / 0 failed on both feeds
