# Options Phase-1 Readiness -- Four Gaps Closed - 2026-07-30

## Summary

Closed the four Phase-1 readiness gaps in the options data layer so Phase 2 (primitives +
harness) can build on a stable, production-located foundation: materialized IWM end to end,
relocated the derived daily tables out of the gitignored Wave-0 scratch path into the
conventional storage location with provenance sidecars, re-scored the V5 gate on usable-value
rate, and wrote the readiness statement. Data/infrastructure only -- no backtest, no P&L, no
strategy verdict.

## Changes Made

- **Gap 1 -- IWM materialized** (`H:\Stock_Data\options\...`): built `options_chain_eod`
  (108 partitions, 2,686,042 rows, 2017-01 .. 2025-12), then `options_iv_surface` (48,128
  slices) and `options_iv_smooth` (2,686,042 rows, exactly 1:1 with the chain). Used the
  existing CLIs unchanged -- `scripts/data/build_options_chain_eod.py` via
  `build_eod_chain_parallel.sh --jobs 8`, then `scripts/data/build_m7_surface.py --roots IWM`.
  **No IWM-specific code path was needed.** Pre-2017 rows deliberately not emitted, matching
  SPY's behaviour at the ThetaData greeks boundary.
- **`src/data/options/derived_store.py`** (new): the materialized store for the derived daily
  tables -- `<storage>/options/derived/<table>/<table>.parquet` plus a `<table>.meta.json`
  provenance sidecar (source, snapshot timestamp, build git SHA, columns, row counts, overall
  and per-root coverage span, build census). Follows the convention `spread_census` already
  uses and the sidecar shape `regime_state_daily` / `vix_spot` already emit.
- **`scripts/backtest_scripts/wave0_b1_build_derived.py`**: writes through the store instead of
  `output/wave0/derived/`, and now includes IWM (2017-2025). Rebuilt all 7 tables for SPY + QQQ
  + IWM.
- **`scripts/backtest_scripts/wave0_b1_diagnostics.py`**: reads through
  `derived_store.load_derived_table` rather than the scratch path.
- **`scripts/data/vbattery/rescore_v5_usable.py`** (new): re-scores the V5 gate on usable-value
  rate, applying the registered 90% threshold verbatim to the corrected metric. Writes
  `output/vbattery/sweep/v5_gate_usable_by_root_year.csv`.
- **`docs/strategies/research/options-slate/20260731_phase1_readiness.md`** (new): the Gap-4
  readiness statement -- every artifact with path/coverage/rows/provenance, what Phase 2 can and
  cannot assume, the corrected V5 table, known limitations carried forward, and a divergence
  table.
- **`src/data/options/canonical.py`** + `tests/.../test_canonical.py`: corrected the
  enumeration of date32 `expiration` partitions (IWM 2017-01..04 also ship the variant).
- Tests: `tests/data/test_options/test_derived_store.py` (8), `test_v5_rescore.py` (9), both
  written before the implementation.

## Commits

- `f1a6555` feat(options): re-score the V5 gate on usable-value rate (Phase-1 Gap 3)
- `fa541cc` feat(options): relocate derived dailies to the materialized store (Gap 2)
- `de32617` docs(options): record the IWM date32 expiration partitions
- `5d36dd1` docs(options): Phase-1 readiness statement (Gap 4, draft pending IWM M7)
- `7dbc69e` docs(options): finalize the Phase-1 readiness statement (Gap 4)

Branch: `feat/options-phase1-readiness`. **Not pushed** (per the task brief).

## Key Findings / Divergences

1. **IWM's 2017 is clean; SPY's is not.** IWM 2017 carries 19-23 sessions/month, essentially
   complete, whereas SPY 2017 ramps from 2 sessions in January to full only by October. Verified
   raw-side (the source SPY 2017-01 partition itself holds 2 sessions), so it is not a build
   artifact. Over the identical 2017-2025 window `atm_iv_daily` holds **2,250 IWM sessions vs
   2,155 SPY sessions**. The registered "SPY 2017 H1 near-absent" limitation does NOT generalize
   to the other roots.
2. **CORRECTION ADDENDUM C1's stated cause for the V5 defect is wrong.** C1 says the gate was
   computed on `null_count` and therefore passed 100%-NaN root-years. Re-reading
   `scripts/data/vbattery/sweep_v1_v2_v3_v5_v9_v11.py`, every V5 accumulator uses `np.isfinite()`
   on values -- the `nonnull_*` columns are finite-value rates, already NaN-aware -- and the
   shipped CSV correctly scores every ALL_NAN root-year 0.0 / UNTRUSTED. **The real defect is
   different:** the gate bound on the finite rate alone and never applied the registered SANITY
   bounds (`IV in (0.01, 5.0)`, `|delta| <= 1`, delta sign), even though the report's own V5
   section calls that screen load-bearing. That is what was corrected.
3. **The corrected V5 costs SPY its first year.** 312 PASS / 103 UNTRUSTED of 415 root-years
   (was 331 / 84). 19 flips, **all PASS -> UNTRUSTED, nothing promoted**, binding screen
   `plausible_iv_frac` in every case. Notably **SPY 2017 (0.8960) and QQQ 2013 (0.8910) fail**.
   Combined with finding 1, SPY's honest window reads 2018-2025 while IWM's is the full
   2017-2025. IWM passes every year of its window, 2017 at 0.9449.
4. **Gap 1 needed no schema fix.** IWM built cleanly through the unmodified canonical layer.
   The 4 date32 `expiration` partitions it does have (2017-01..04) were already covered by the
   existing regression test -- only the docstring enumeration was stale.
5. **The relocation is a pure move.** SPY/QQQ rows in the relocated tables are bit-identical to
   the old scratch copies (33,270 rows matched, `atm_iv` all-close).

## Known Issues / Remaining Work

- `spread_census` is the one table under `<storage>/options/derived/` without a provenance
  sidecar. Pre-existing, out of Gap-2 scope, worth closing.
- **OPT-006 remains BLOCKED** by M7's registered 400-day DTE cap -- its 12-18 month long leg
  exceeds 400 days at entry. Needs an explicit amendment, not a workaround.
- **Single-name roots remain BLOCKED** on the corporate-action adjustment layer (V7 / ESC-1:
  7 of 7 checked actions are as-reported). Materializing more roots does not unblock this.
- Whether Addendum C1 in `20260727_options_vbattery_report.md` should be amended to record the
  corrected cause (finding 2) is a doc-chain decision -- the pre-registration documents were not
  modified, per the brief.
- Branch is not merged or pushed.

## Validation

- `pytest tests/data/test_options/ tests/backtesting/test_diagnostics/` -> **193 passed**.
- IWM builds verified on disk: 108/108 `options_chain_eod`, 108/108 `options_iv_surface`
  partitions; `options_iv_smooth` row count exactly equals `options_chain_eod` (2,686,042).
- Greek usability spot-checked with `np.isnan` on values (IWM 2017-01: 0.0 NaN fraction for both
  `implied_vol` and `delta`) -- not via `null_count`.
- Derived store round-tripped through `load_derived_table` / `load_derived_meta`; the updated
  `wave0_b1_diagnostics._load()` loads all 7 tables with all 3 roots.
- V5 re-score cross-checked against the independent per-partition greek census: **zero
  contradictions** (no root-year passes the corrected gate with zero OK partitions).
- All new files verified ASCII-only.
- All runs wrapped in `RunStatus`; both long builds exited 0 with DONE sentinels.
- Nothing deleted -- `output/wave0/derived/*.parquet` left in place.
