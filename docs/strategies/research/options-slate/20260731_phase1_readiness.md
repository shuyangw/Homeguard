# Options Slate -- Phase-1 Readiness Statement (2026-07-31)

**Scope:** what the Phase-1 options data layer actually contains, what Phase 2 (primitives +
harness) may assume from it, and what it must NOT assume.

**This is a data/infrastructure statement.** No strategy backtest was run. No P&L, no positions,
no strategy verdict of any kind appears here.

**Ground truth is code and disk.** Every number below was measured on the materialized stores at
the stated snapshot, not copied from an earlier document. Where an earlier document and disk
disagree, disk wins and the divergence is called out in Section 6.

---

## 1. Phase-1 artifacts

Paths resolve via `from src.settings import get_local_storage_dir` (`H:\Stock_Data` on this host).
Never hardcode.

### 1.1 `options_chain_eod` -- the canonical EOD chain

`<storage>/options/options_chain_eod/root=<R>/year=<Y>/month=<M>/data.parquet`

| root | partitions | rows | span | size |
|---|---|---|---|---|
| SPY | 108 | 6,801,800 | 2017-01 .. 2025-12 | 247.7 MB |
| QQQ | 162 | 5,260,977 | 2012-06 .. 2025-12 | 195.0 MB |
| **IWM** | **108** | **2,686,042** | **2017-01 .. 2025-12** | **100.3 MB** |
| **total** | **378** | **14,748,819** | | **543.0 MB** |

IWM was materialized for this readiness pass (Gap 1) using the existing CLI
`scripts/data/build_options_chain_eod.py` -> `src/data/options/canonical.py`, with no
IWM-specific code path. Provenance: source store `options_combined`, snapshot minute
15:45:00 ET clamped to `min(15:45, real session close)`, one row per (contract, session).

### 1.2 `options_iv_surface` / `options_iv_smooth` -- the M7 smoothed surface

`<storage>/options/options_iv_surface/...` and `<storage>/options/options_iv_smooth/...`

| root | surface slices (session x expiry) | smooth rows | span |
|---|---|---|---|
| SPY | 59,148 | 6,801,800 | 2017-01 .. 2025-12 |
| QQQ | 67,629 | 5,260,977 | 2012-06 .. 2025-12 |
| **IWM** | **IWM_SURFACE_ROWS** | **IWM_SMOOTH_ROWS** | **2017-01 .. 2025-12** |
| **total** | **TOTAL_SURFACE** | **TOTAL_SMOOTH** | |

`options_iv_smooth` is 1:1 with `options_chain_eod` by construction (one smoothed row per
contract-session, NULL `iv_smooth` with a `surface_reason` where the slice was refused).
Builder: `src/data/options/iv_surface.py` + `iv_surface_build.py`, `FIT_VERSION = m7_svi_v1`.

### 1.3 Derived daily tables -- RELOCATED OUT OF SCRATCH (Gap 2)

Previously `output/wave0/derived/*.parquet` -- a Wave-0 scratch path inside the gitignored
`output/` tree. Now materialized at the conventional location, following the convention
`spread_census` already uses:

`<storage>/options/derived/<table>/<table>.parquet` + `<table>.meta.json`

DERIVED_TABLE_BLOCK

Each table carries a **provenance sidecar** (`<table>.meta.json`) with `dataset`, `source`,
`snapshot_timestamp`, `git_sha`, `columns`, `rows`, `date_min`/`date_max`, `roots`,
`rows_by_root`, `coverage_by_root`, and the build census -- matching the shape
`regime_state_daily.meta.json` and `vix_spot.meta.json` already emit.

Reader: `src/data/options/derived_store.py::load_derived_table(name)`. Writer:
`write_derived_table(...)`. The only in-repo consumer of the old path
(`scripts/backtest_scripts/wave0_b1_diagnostics.py`) now reads through the store.

**Nothing was deleted.** The original `output/wave0/derived/*.parquet` files are left in place.

### 1.4 Supporting tables (unchanged by this pass)

| artifact | path | contents |
|---|---|---|
| `spread_census` | `<storage>/options/derived/spread_census/spread_census.parquet` | 31 roots, 181,761 rows, 33 cols |
| `regime_state_daily` | `<storage>/alt_data/regime/regime_state_daily.parquet` | 3,556 rows, 2012-06-01 .. 2026-07-24, sidecar present |
| VIX spot | `<storage>/alt_data/vix/vix_spot.parquet` | 3,913 rows, 2011-01-03 .. 2026-07-24, sidecar present |
| greek coverage census | `docs/strategies/research/options-slate/20260728_options_greek_coverage_census.csv` | 4,510 partitions classified OK / ALL_NAN / NO_COLUMN / PARTIAL_NAN |

---

## 2. The corrected V5 gate (Gap 3)

### 2.1 The registered gate, restated verbatim -- threshold UNCHANGED

> "Null rate for `implied_vol`, `delta`, `theta`, `vega` per root x year. Sanity:
> `IV in (0.01, 5.0)`, `|delta| <= 1`, delta sign correct per `right`. **>=90%** non-null per
> root-year = PASS. Below => that root-year is flagged **untrusted**, routed to later
> recomputation. Do **not** recompute now."

The **90% threshold is not changed**. What changes is the metric it is applied to.

### 2.2 What was actually wrong -- a correction to CORRECTION ADDENDUM C1

Addendum C1 states that V5's gate "was computed on `null_count` and therefore passed root-years
that are 100% NaN". **That is not what the shipped artifact does, and it is not what the code
does.** Re-reading `scripts/data/vbattery/sweep_v1_v2_v3_v5_v9_v11.py` (the `---- V5 ----`
block), every V5 accumulator is built with `np.isfinite()` on the VALUES:

```
a5[1] += int(np.isfinite(iv).sum())
a5[5] += int((np.isfinite(iv) & (iv > 0.01) & (iv < 5.0)).sum())
```

The columns named `nonnull_*` in `v5_gate_by_root_year.csv` are therefore **finite-value rates,
already NaN-aware**. Consistent with that, every ALL_NAN and NO_COLUMN root-year -- SPY and IWM
2012-2016 among them -- already scores `0.0` and `UNTRUSTED` in the shipped CSV. The
NaN-vs-NULL trap caught the analyses around V5; it did not catch V5 itself.

**The real defect is different, and it is real.** The gate bound on `min_nonnull` -- the finite
rate alone. It **never applied the registered sanity bounds**, even though the report's own V5
section says "the registered plausibility screen is load-bearing; the non-null rate alone must
not be used to establish trust." Correcting that is the substantive re-score.

### 2.3 The corrected metric

```
usable_rate = min(
    rate(implied_vol finite AND 0.01 < iv < 5.0),   # plausible_iv_frac
    rate(delta finite AND |delta| <= 1),            # abs_delta_le1_frac
    rate(delta finite AND sign correct per right),  # delta_sign_ok_frac
    rate(theta finite),                             # nonnull_theta
    rate(vega finite),                              # nonnull_vega
)
gate_usable = PASS if usable_rate >= 0.90 else UNTRUSTED
```

### 2.4 What was recomputed vs reused -- stated explicitly

**No rescan of 24bn rows was performed.** Two existing measurements were reused:

1. `output/vbattery/sweep/v5_gate_by_root_year.csv` -- the shipped V5 sweep, verified at code
   level to be `np.isfinite`-based (Section 2.2). This supplies all five component rates.
2. The per-partition greek census -- collapsed to per-root-year OK/ALL_NAN/NO_COLUMN/PARTIAL_NAN
   counts and used as an **independent cross-check**: any root-year with zero OK partitions that
   passed the corrected gate would be logged as a contradiction. **Zero contradictions were
   found.**

This is sound because (1) is a full-store measurement at value level and (2) is a full-store
partition classification derived independently; they agree everywhere.

Code: `scripts/data/vbattery/rescore_v5_usable.py` (9 unit tests in
`tests/data/test_options/test_v5_rescore.py`). Output:
`output/vbattery/sweep/v5_gate_usable_by_root_year.csv`.

### 2.5 Result

**312 PASS / 103 UNTRUSTED of 415 root-years** (was 331 / 84). **19 root-years flip, all in the
same direction: PASS -> UNTRUSTED. Nothing is promoted.** In every flip the binding screen is
`plausible_iv_frac` -- the registered `IV in (0.01, 5.0)` bound.

| root | year | shipped non-null rate | corrected usable rate | binding screen | shipped gate | corrected gate |
|---|---|---|---|---|---|---|
| AAPL | 2014 | 1.0000 | **0.8749** | plausible_iv_frac | PASS | **UNTRUSTED** |
| AAPL | 2015 | 0.9068 | **0.8211** | plausible_iv_frac | PASS | **UNTRUSTED** |
| AAPL | 2017 | 1.0000 | **0.8998** | plausible_iv_frac | PASS | **UNTRUSTED** |
| DIA | 2017 | 1.0000 | **0.8987** | plausible_iv_frac | PASS | **UNTRUSTED** |
| MSFT | 2012 | 1.0000 | **0.8861** | plausible_iv_frac | PASS | **UNTRUSTED** |
| MSFT | 2013 | 1.0000 | **0.8621** | plausible_iv_frac | PASS | **UNTRUSTED** |
| MSFT | 2014 | 1.0000 | **0.8845** | plausible_iv_frac | PASS | **UNTRUSTED** |
| **QQQ** | **2013** | 1.0000 | **0.8910** | plausible_iv_frac | PASS | **UNTRUSTED** |
| **SPY** | **2017** | 1.0000 | **0.8960** | plausible_iv_frac | PASS | **UNTRUSTED** |
| TLT | 2016 | 0.9310 | **0.8687** | plausible_iv_frac | PASS | **UNTRUSTED** |
| VIX | 2017 | 1.0000 | **0.7771** | plausible_iv_frac | PASS | **UNTRUSTED** |
| VIX | 2018 | 1.0000 | **0.8713** | plausible_iv_frac | PASS | **UNTRUSTED** |
| VIX | 2019 | 1.0000 | **0.8802** | plausible_iv_frac | PASS | **UNTRUSTED** |
| VIX | 2021 | 1.0000 | **0.8963** | plausible_iv_frac | PASS | **UNTRUSTED** |
| VIX | 2023 | 1.0000 | **0.8958** | plausible_iv_frac | PASS | **UNTRUSTED** |
| VIX | 2024 | 0.9999 | **0.8973** | plausible_iv_frac | PASS | **UNTRUSTED** |
| VIX | 2025 | 0.9999 | **0.8815** | plausible_iv_frac | PASS | **UNTRUSTED** |
| VIX | 2026 | 1.0000 | **0.8949** | plausible_iv_frac | PASS | **UNTRUSTED** |
| XLF | 2017 | 1.0000 | **0.8937** | plausible_iv_frac | PASS | **UNTRUSTED** |

### 2.6 The corrected table for the materialized universe (SPY, QQQ, IWM)

| root | year | usable rate | corrected gate | census OK frac |
|---|---|---|---|---|
| IWM | 2012 | 0.0000 | UNTRUSTED | 0.00 |
| IWM | 2013 | 0.0000 | UNTRUSTED | 0.00 |
| IWM | 2014 | 0.0000 | UNTRUSTED | 0.00 |
| IWM | 2015 | 0.0000 | UNTRUSTED | 0.00 |
| IWM | 2016 | 0.0000 | UNTRUSTED | 0.00 |
| IWM | 2017 | 0.9449 | PASS | 1.00 |
| IWM | 2018 | 0.9827 | PASS | 1.00 |
| IWM | 2019 | 0.9856 | PASS | 1.00 |
| IWM | 2020 | 0.9897 | PASS | 1.00 |
| IWM | 2021 | 0.9911 | PASS | 1.00 |
| IWM | 2022 | 0.9862 | PASS | 1.00 |
| IWM | 2023 | 0.9805 | PASS | 1.00 |
| IWM | 2024 | 0.9766 | PASS | 1.00 |
| IWM | 2025 | 0.9771 | PASS | 1.00 |
| IWM | 2026 | 0.9719 | PASS | 1.00 |
| QQQ | 2012 | 0.9073 | PASS | 1.00 |
| **QQQ** | **2013** | **0.8910** | **UNTRUSTED** | 1.00 |
| QQQ | 2014 | 0.9010 | PASS | 1.00 |
| QQQ | 2015 | 0.9330 | PASS | 1.00 |
| QQQ | 2016 | 0.9403 | PASS | 1.00 |
| QQQ | 2017 | 0.9145 | PASS | 1.00 |
| QQQ | 2018 | 0.9800 | PASS | 1.00 |
| QQQ | 2019 | 0.9841 | PASS | 1.00 |
| QQQ | 2020 | 0.9890 | PASS | 1.00 |
| QQQ | 2021 | 0.9893 | PASS | 1.00 |
| QQQ | 2022 | 0.9854 | PASS | 1.00 |
| QQQ | 2023 | 0.9704 | PASS | 1.00 |
| QQQ | 2024 | 0.9737 | PASS | 1.00 |
| QQQ | 2025 | 0.9740 | PASS | 1.00 |
| QQQ | 2026 | 0.9850 | PASS | 1.00 |
| SPY | 2012 | 0.0000 | UNTRUSTED | 0.00 |
| SPY | 2013 | 0.0000 | UNTRUSTED | 0.00 |
| SPY | 2014 | 0.0000 | UNTRUSTED | 0.00 |
| SPY | 2015 | 0.0000 | UNTRUSTED | 0.00 |
| SPY | 2016 | 0.0000 | UNTRUSTED | 0.00 |
| **SPY** | **2017** | **0.8960** | **UNTRUSTED** | 1.00 |
| SPY | 2018 | 0.9754 | PASS | 1.00 |
| SPY | 2019 | 0.9800 | PASS | 1.00 |
| SPY | 2020 | 0.9883 | PASS | 1.00 |
| SPY | 2021 | 0.9860 | PASS | 1.00 |
| SPY | 2022 | 0.9855 | PASS | 1.00 |
| SPY | 2023 | 0.9658 | PASS | 1.00 |
| SPY | 2024 | 0.9703 | PASS | 1.00 |
| SPY | 2025 | 0.9755 | PASS | 1.00 |
| SPY | 2026 | 0.9846 | PASS | 1.00 |

**IWM is the cleanest of the three roots**: every year in its usable window passes, and 2017 --
the first usable year -- passes at 0.9449, comfortably clear, whereas SPY 2017 fails at 0.8960.

Full 415-row table: `output/vbattery/sweep/v5_gate_usable_by_root_year.csv`.

---

## 3. What Phase 2 CAN assume

1. **Three roots are materialized end to end: SPY, QQQ, IWM.** Each has
   `options_chain_eod` -> `options_iv_surface` -> `options_iv_smooth` -> the derived dailies,
   built by the same code with no per-root special-casing.
2. **The derived dailies are at a stable, non-scratch, non-gitignored storage path** with a
   provenance sidecar per table. Load them via
   `src.data.options.derived_store.load_derived_table`.
3. **Greek usability is tested on values.** `greek_usable_mask` in
   `src/backtesting/diagnostics/options_iv_state.py` uses `np.isfinite`, never `null_count` or
   column presence, and the derived tables inherit that. SPY simply has no pre-2017 rows; this
   behaviour is preserved for IWM (no pre-2017 IWM rows are emitted).
4. **The snapshot minute is 15:45:00 ET**, clamped to `min(15:45, real session close)`, and it
   lives in exactly one place: `canonical.SNAPSHOT_TIME_ET`. Do not slice bars anywhere else.
   `src/strategies/options/data_loader.py::get_eod_chain()` hardcodes 16:00 -- it is WRONG for
   this work and must not be reused.
5. **`oi_eod` / `gamma_eod` lag >= 1 session.** Use `canonical.add_eod_lag()` and consume
   `oi_eod_lag1` / `gamma_eod_lag1`. The unlagged columns keep their `_eod` suffix deliberately.
6. **`mid = (bid + ask) / 2` on `quote_valid` rows is the only sanctioned mark.**
   `open`/`high`/`low`/`close`/`volume`/`vwap` are trade-derived and are never a mark.
7. **Index-root quote population is excellent.** V1 `[0.05, 0.15]` delta bucket: 99.86-100.00%
   for SPY/QQQ/IWM on every gradeable year. This gate is not a constraint on Phase 2.
8. **Regime state and VIX spot are built with provenance sidecars** and cover 2012-06 .. 2026-07.

---

## 4. What Phase 2 must NOT assume

1. **Single-name roots are NOT materialized.** `options_chain_eod` contains SPY, QQQ and IWM
   only. The 28 other roots exist in the raw `options_combined` store but have no canonical EOD
   chain, no surface, and no derived dailies.
2. **Single-name work is separately BLOCKED on the corporate-action adjustment layer.** V7
   (ESC-1): **7 of 7 checked corporate actions are AS-REPORTED, zero back-adjusted** (AAPL
   2020-08-31 4:1; TSLA 2020-08-31 5:1 and 2022-08-25 3:1; NVDA 2021-07-20 4:1 and 2024-06-10
   10:1; AMZN 2022-06-06 20:1; GOOGL 2022-07-18 20:1), with non-standard deliverables present.
   The registered ruling is unambiguous: **as-reported => an adjustment layer is required before
   ANY single-name work.** That layer does not exist. Materializing more roots does not unblock
   this; it is a separate build.
3. **No pre-2017 SPY or IWM.** ThetaData does not serve IV/greeks for the ETF/index roots before
   2017-01. This is a vendor boundary, not a bad download (Addendum C2) -- a re-pull will not
   recover it. QQQ alone runs 2012-2026.
4. **The store ends before "today".** Last session on disk is **2026-02-04**; last complete month
   is **2025-12**. The materialized tables stop at 2025-12 by construction.
5. **`iv_smooth` is not a price mark.** See Section 5.
6. **A refused surface slice is a positive record, not a missing row.** `options_iv_surface` rows
   with `reason != OK` carry NULL params by design; `options_iv_smooth` carries NULL `iv_smooth`
   plus `surface_reason`. Do not treat these as data gaps to be filled.
7. **`options_chain_eod` month partitions compute `oi_eod_lag1` WITHIN the month**, so each
   month's first session carries NULL lags. Consumers spanning a month boundary must re-run
   `add_eod_lag()` over the concatenated frame.

---

## 5. Known limitations carried forward -- stated, not buried

These are live constraints on Phase 2, not historical notes.

### 5.1 SPY 2017 H1 is near-absent -- and it is SPY-specific

SPY's 2017 monthly session counts in `options_chain_eod`: **2, 3, 6, 7, 10, 16, 16, 17, 19, 21,
21, 20** against ~20-21 trading days per month. 2017-01 holds **2 of 20 sessions**. Verified
raw-side: the source `options_combined` SPY 2017-01 partition itself contains only 2 sessions,
so this is not a build artifact and not repairable downstream.

**New finding from this pass:** this does **not** generalize to the other roots. IWM 2017 runs
**20, 19, 23, 19, 22, 22, 20, 23, 20, 22, 21, 20** sessions and QQQ 2017 is the same -- both
essentially complete. Any Phase-2 study that needs a clean 2017 has it in IWM and QQQ but not in
SPY. Combined with Section 2.5 (SPY 2017 now UNTRUSTED at 0.8960), **SPY 2017 should be treated
as unusable and SPY's honest window read as 2018-2025**, while IWM's is the full 2017-2025.

### 5.2 M7's far wing is heavy-tailed -- OPT-047 / OPT-030 mark off RAW quotes

In the `|delta| < 0.05` bucket the M7 fit residual p99 is **17.1 vol points (SPY) / 23.7 vol
points (QQQ)**, with the max approaching 100, and roughly **31% of contracts in that bucket are
extrapolated** beyond the fitted strike range. The median is fine; the tail is not. This is the
region OPT-047 reads.

**Amendment A3 is binding:** OPT-047 and OPT-030 take their price **MARKS from the raw quote mid**
(`options_chain_eod.mid`, `quote_valid` rows only, crossed/zero-bid excluded per V3) -- returned
to M2's general rule. Strike **selection** from the surface was validated separately and is
acceptable. Trade prints (`close`/`vwap`) remain prohibited as marks.

### 5.3 OPT-006 is BLOCKED by M7's 400-day cap

M7 refuses expiries beyond a registered **400-day** cap (`DTE_OUT_OF_RANGE`; 1,873 QQQ slices).
OPT-006's 12-18 month long leg **exceeds 400 days at entry**, so OPT-006 cannot be run off M7 as
registered. This requires an **explicit amendment** before OPT-006 runs. It is flagged, not
worked around. Separately, the ITM 0.80-delta leg is where the shipped vendor greeks are worst
(median delta error 0.0187).

### 5.4 The NaN-vs-NULL trap

Greek columns are **NaN-valued float64, not SQL NULL**. Arrow reports `null_count = 0` for a
100%-NaN column. **Column presence and `null_count` are both invalid tests for greek
usability**; only `np.isnan()` / `np.isfinite()` on the values is valid. This has defeated
multiple analyses. Any new Phase-2 code that judges greek usability must use the value test --
`greek_usable_mask` already does.

### 5.5 `spread_census` has no provenance sidecar

Pre-existing and out of scope for this pass (Gap 2 covered the derived dailies). It is the one
table under `<storage>/options/derived/` without a `.meta.json`. Worth closing when convenient.

---

## 6. Divergences between the Phase-1 brief and disk

Reported because reality wins.

| # | Brief / prior doc said | Disk / code says | Consequence |
|---|---|---|---|
| 1 | IWM shares SPY's coverage profile in 2017 | IWM 2017 is **essentially complete** (19-23 sessions/month); SPY 2017 ramps from 2 | Positive for OPT-016 -- IWM contributes a clean 2017 that SPY cannot |
| 2 | Addendum C1: V5's gate "was computed on `null_count`" and passed 100%-NaN root-years | The shipped sweep uses `np.isfinite` on values throughout; ALL_NAN root-years already scored 0.0 / UNTRUSTED | C1's stated cause is wrong. The real defect -- the gate ignoring the registered SANITY bounds -- is corrected here and flips 19 root-years including SPY 2017 |
| 3 | `expiration` ships as date32 in "100 partitions (SPY 2017-2018, QQQ 2017/2018/2023)" | **IWM 2017-01..04 also do**, for 104 in the materialized universe | None -- the existing regression test already covers the variant; enumeration corrected |
| 4 | Gap 1 might need an IWM schema fix | IWM built cleanly through the unmodified canonical layer, 108/108 partitions | No IWM-specific code was written, as required |
| 5 | -- | `spread_census` lacks a provenance sidecar | Noted in 5.5 |

---

## 7. Reproduction

```bash
# Gap 1 -- IWM chain + surface
bash scripts/data/build_eod_chain_parallel.sh output/wave0/jobs_iwm.txt 8
PYTHONPATH=. POLARS_MAX_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/data/build_m7_surface.py --roots IWM --jobs 8

# Gap 2 -- derived dailies to the materialized store (SPY + QQQ + IWM)
python scripts/backtest_scripts/wave0_b1_build_derived.py

# Gap 3 -- corrected V5
python scripts/data/vbattery/rescore_v5_usable.py

# Tests
python -m pytest tests/data/test_options/ -q
```

*No strategy backtest has been run. No P&L has been observed.*
