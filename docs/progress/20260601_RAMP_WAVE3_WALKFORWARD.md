# RAMP Wave-3 Walk-Forward OOS Validation -- 2026-06-01

## Summary

Ran the mandatory walk-forward OOS robustness validation for V28 (multi-horizon ensemble,
full-window Sharpe 0.811) and V31 (beta-residual, 0.769) against V11 incumbent (0.528).
7 sequential calendar-year OOS windows (2019 through 2025+). Both candidates REJECT:
neither beats V11 in all 7 windows, and both show negative Sharpe in 2022. The full-window
edge is real but temporally concentrated (driven by 2023-2025+ windows). Null option applies:
V11 remains the deployed paper incumbent. The Wave-3 signal-construction campaign is complete.

## Changes Made

- **`scripts/backtest_scripts/ramp_phase4_wave3_walkforward.py`** (NEW): Walk-forward
  orchestrator. Loads full-window return streams from the experiment registry (6 variants),
  slices by 7 calendar-year OOS windows, computes per-window Sharpe/PSR/rank, pools returns
  (Section 3.4), runs rank-stability analysis, and verifies sliced-stream equivalence via
  direct run_variant on the 2022 window. Appends aggregate results to registry under
  phase='wave3_walkforward'. Writes atomic .md + .json reports.

- **`tests/research/ramp_phase4/test_wave3_walkforward.py`** (NEW): 42 TDD tests covering
  _slice_window, _window_sharpe, _window_psr, _pool_returns, _verdict, _rank_stability,
  WindowResult, WalkForwardResult, _compute_per_window_results (end-to-end on synthetic
  streams), and structural constants. Suite: 201 passing (was 159).

- **`docs/reports/ramp/20260601_wave3_walkforward.md`** (NEW): Full validation report with
  methodology design, per-window Sharpe table (all 6 variants), pooled OOS Sharpe, rank
  stability, slice verification, 7.5bps cost gate, verdicts.

- **`docs/reports/ramp/20260601_wave3_walkforward.json`** (NEW): Machine-readable artifact
  with per-window raw data, verification result, run_id prefixes.

## Key Findings

### Per-window Sharpe (5 bps near_close)

| Window | V11 | V28 | V31 | V28>V11 | V31>V11 |
|---|---:|---:|---:|:---:|:---:|
| 2019 | 1.483 | 0.946 | 1.535 | N | Y |
| 2020 | 0.838 | 0.825 | 0.813 | N | N |
| 2021 | 1.181 | 0.916 | 1.528 | N | Y |
| 2022 | -0.266 | -0.496 | -0.745 | N | N |
| 2023 | 1.170 | 1.485 | 1.864 | Y | Y |
| 2024 | 0.829 | 1.020 | 1.295 | Y | Y |
| 2025+ | 0.527 | 1.685 | 1.341 | Y | Y |

### Summary stats

| Metric | V28 | V31 |
|---|---:|---:|
| Win rate vs V11 | 43% (3/7) | 71% (5/7) |
| Worst OOS Sharpe | -0.496 | -0.745 |
| Sharpe dispersion (std) | 0.699 | 0.870 |
| Pooled OOS Sharpe | 0.889 | 0.910 |
| Verdict | **REJECT** | **REJECT** |

### Methodological notes

- Purging and embargo are NOT applicable (no IS fitting step -- parameters are a-priori fixed).
- Slicing the full-window return stream by calendar year is valid because signals are purely causal.
- Verified: sliced Sharpe for 2022 V28 = -0.4964, direct run_variant Sharpe = -0.4964 (exact match).
- 2022 is the deciding window: V11 at -0.266 outperforms both V28 (-0.496) and V31 (-0.745).
  V11's lower turnover and regime-aware cash mode appear more resilient in the 2022 bear.
- Rank stability: V28 mean rank 3.7/6 (only #1 in 2025+); V31 mean rank 2.6/6 (more stable).
- The PBO 0.503 finding at the family gate was correct -- the selection instability is confirmed
  temporally: the family cross-section ranking is unstable across years.

### Null option

V11 remains the deployed paper incumbent. The Wave-3 signal-construction campaign is closed
with the conclusion that no Wave-3 variant improves on V11 sufficiently to warrant deployment.
The pooled OOS edge (V28 0.889, V31 0.910 > V11 0.647) is real but concentrated in 2023-2025+.

## Commits

- `9a62a62` feat(wave3): walk-forward OOS validation -- V28/V31 REJECT (2022 bear period not isolated)

## Known Issues / Remaining Work

- The 2022 underperformance of V28/V31 vs V11 is mechanistically interesting: V11 goes to cash
  in BEAR regime; V28/V31 (regime-free) stay invested. A targeted BEAR-protection layer on V28
  might recover the 2022 gap without losing the 2023-2025+ edge -- but this would be a NEW
  experimental branch, not a continuation of Wave-3.
- V11 paper validation (A7 monitoring) continues as the ground truth.
- Wave-3 campaign is closed. Next steps per docs/progress/20260525_ramp_next_steps.md.

## Validation

- 201 ramp_phase4 unit tests pass (42 new for walk-forward harness).
- Slice-vs-direct verification: exact match (abs diff 0.0000) on 2022 V28 window.
- Registry run_ids written for V28 (f22d2bd4...) and V31 (d59a2954...) walk-forward aggregates.
  Verification run_id: 73215dc5... (phase=wave3_walkforward, 2022 window, V28 direct run).
