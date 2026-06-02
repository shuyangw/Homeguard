# RAMP Wave-3 Walk-Forward OOS Robustness Validation -- 2026-06-01

**Branch**: archive/regime-detector-campaign-2026-05
**Code commit**: 075754bfc9127df3b467009140bfbff83bacbda8
**Data snapshot**: 2026-05-16 (Alpaca SIP daily, split-adjusted)
**Universe**: sp500-2025 (494 symbols)
**Primary cost tier**: 5.0 bps per side (near_close timing)
**Incumbent**: V11 (Sharpe 0.528 full-window at 5 bps)

## Section 1: Methodology Design

### What binds from Section 3 (Walk-Forward), given no fitting

V28 and V31 have FIXED, a-priori parameters:
- V28: blend weights 0.5/0.3/0.2 (horizons 21/63/126d) + 0.1 reversal weight
- V31: 90-day beta window, 21-day residual momentum horizon

**Section 3.1 (window structure)**: APPLIES. We use 7 sequential
calendar-year OOS windows to assess temporal Sharpe stability.

**Section 3.2 (purging)**: NOT APPLICABLE. Purging removes training
observations whose label overlaps the test set. Since there is no
fitting step, there is no training set from which to purge. Stated
explicitly: zero purging is applied, and this is correct.

**Section 3.3 (embargo)**: NOT APPLICABLE. Embargo isolates the next
training window from the prior test window. With no training step,
embargo has no meaning. The only leakage vector is signal lookback
(126 trading days for V28, 90 for V31); both use only past data
(Section 1.1 causal constraint), making this safe.

**Slicing validity**: Slicing the full-window return stream by calendar
year produces the correct OOS Sharpe for that year. This is valid
because: (a) all signals are purely causal -- return at day D used
only data through D-1; (b) there is no fitting boundary that would
make later windows IS-contaminated. Sliced approach verified against
direct run_variant on the 2022 window (see Section 4).

**The real overfitting vector**: Variant SELECTION (we picked V28 as
best-of-6; PBO = 0.503 at the family gate). This is addressed by the
per-window rank stability analysis (Section 3).

### OOS Window Design

| Window | Start | End | Trading days | Warmup before start |
|---|---|---|---:|---|
| 2019 | 2019-01-01 | 2019-12-31 | 252 | 503d (>= 252 req) |
| 2020 | 2020-01-01 | 2020-12-31 | 253 | all prior years |
| 2021 | 2021-01-01 | 2021-12-31 | 252 | all prior years |
| 2022 | 2022-01-01 | 2022-12-31 | 251 | all prior years |
| 2023 | 2023-01-01 | 2023-12-31 | 250 | all prior years |
| 2024 | 2024-01-01 | 2024-12-31 | 252 | all prior years |
| 2025+ | 2025-01-01 | 2026-05-15 | 343 | all prior years |

Minimum folds required: 5 (Section 3.1). Implemented: 7.

### Acceptance Criteria

GRADUATE: Beats V11 in ALL 7 OOS windows AND worst-window Sharpe > 0
HOLD: Mixed (beats V11 in >= 5/7 windows AND worst >= -0.10)
REJECT: Fails OOS (concentration in one period, or < 5/7 windows)

## Section 2: Per-Window Sharpe Table (5 bps near_close)

Per-window Sharpe for all 6 family variants. V11 is the benchmark.

| Window | n | V11 | V28 | V31 | V02+V05 | V26 | V33-core | V28>V11 | V31>V11 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| 2019 | 252 | 1.483 | 0.946 | 1.535 | 1.377 | 1.056 | 1.009 | N | Y |
| 2020 | 253 | 0.838 | 0.825 | 0.813 | 1.083 | 1.181 | 0.871 | N | N |
| 2021 | 252 | 1.181 | 0.916 | 1.528 | 0.972 | 0.740 | 0.797 | N | Y |
| 2022 | 251 | -0.266 | -0.496 | -0.745 | -0.120 | -0.589 | -1.543 | N | N |
| 2023 | 250 | 1.170 | 1.485 | 1.864 | 1.403 | 1.577 | 1.355 | Y | Y |
| 2024 | 252 | 0.829 | 1.020 | 1.295 | 1.192 | 0.866 | 1.494 | Y | Y |
| 2025+ | 343 | 0.527 | 1.685 | 1.341 | 0.248 | 0.458 | 0.536 | Y | Y |

### Pooled OOS Sharpe (Section 3.4: pool returns, then compute Sharpe)

| Variant | Pooled OOS Sharpe | n_days pooled |
|---|---:|---:|
| V11 | 0.647 | 1853 |
| V28 | 0.889 | 1853 |
| V31 | 0.910 | 1853 |
| V02+V05 | 0.730 | 1853 |
| V26 | 0.637 | 1853 |
| V33-core | 0.432 | 1853 |

### V28 OOS Summary

| Metric | Value |
|---|---:|
| Win rate vs V11 | 43% (3/7) |
| Worst OOS Sharpe | -0.496 |
| Best OOS Sharpe | 1.685 |
| Sharpe dispersion (std) | 0.699 |
| Pooled OOS Sharpe | 0.889 |

### V31 OOS Summary

| Metric | Value |
|---|---:|
| Win rate vs V11 | 71% (5/7) |
| Worst OOS Sharpe | -0.745 |
| Best OOS Sharpe | 1.864 |
| Sharpe dispersion (std) | 0.870 |
| Pooled OOS Sharpe | 0.910 |

## Section 3: Selection Rank Stability

Rank-stability of V28 and V31 across OOS windows (1 = best of 6 variants).
Ties the PBO 0.503 finding to temporal evidence: does the family ranking
agree across time periods?

### Per-Window Family Ranking

| Window | Rank(V28) | Rank(V31) | Rank(V02+V05) | Rank(V26) | Rank(V11) | Rank(V33-core) |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| 2019 | 6 | 1 | 3 | 4 | 2 | 5 |
| 2020 | 5 | 6 | 2 | 1 | 4 | 3 |
| 2021 | 4 | 1 | 3 | 6 | 2 | 5 |
| 2022 | 3 | 5 | 1 | 4 | 2 | 6 |
| 2023 | 3 | 1 | 4 | 2 | 6 | 5 |
| 2024 | 4 | 2 | 3 | 5 | 6 | 1 |
| 2025+ | 1 | 2 | 6 | 5 | 4 | 3 |

### Rank Stability Summary

| Variant | Mean rank | Median | Best | Worst | % top-2 |
|---|---:|---:|---:|---:|---:|
| V28 | 3.7 | 4.0 | 1 | 6 | 14% |
| V31 | 2.6 | 2.0 | 1 | 6 | 71% |

### Connection to PBO 0.503

The PBO of 0.503 (family gate) means that in a majority of CSCV folds,
the IS-best variant underperforms the OOS median. The per-window rank
table above shows whether this is driven by temporal concentration.
If V28 ranks #1 in 5+ of 7 windows, the PBO concern is quantifiable:
the selection bias is real but V28's dominance is not purely IS-period artefact.

## Section 4: Slice vs Direct Run_Variant Verification

Verification window: **2022**

Run with full history from 2017-01-01 to 2022-12-31 to provide correct warmup.
OOS returns filtered to 2022-01-01 through 2022-12-31 (251 trading days).

| Metric | Value |
|---|---:|
| Slice Sharpe | -0.4964 |
| Direct run_variant Sharpe | -0.4964 |
| Absolute difference | 0.0000 |
| Tolerance | 0.05 |
| n OOS days | 251 |
| Elapsed | 38.0s |
| Status | **PASS** |

Slicing and direct run produce exactly equivalent OOS Sharpe (abs diff = 0.0000 < 0.05 tolerance).
The sliced-stream approach is validated for all other windows.
Registry run_id for verification run: `73215dc5...` (phase=wave3_walkforward, variant=V28, 2022 window).

## Section 5: Cost Gate at 7.5 bps (1.5x Cost Sensitivity)

Per-window Sharpe at 7.5 bps for V28, V31, and V11.

| Window | V11 (7.5bps) | V28 (7.5bps) | V31 (7.5bps) | V28>V11 | V31>V11 |
|---|---:|---:|---:|:---:|:---:|
| 2019 | 1.391 | 0.842 | 1.421 | N | Y |
| 2020 | 0.767 | 0.785 | 0.763 | Y | N |
| 2021 | 1.095 | 0.853 | 1.431 | N | Y |
| 2022 | -0.331 | -0.550 | -0.797 | N | N |
| 2023 | 1.045 | 1.442 | 1.793 | Y | Y |
| 2024 | 0.771 | 0.981 | 1.190 | Y | Y |
| 2025+ | 0.419 | 1.650 | 1.277 | Y | Y |

## Section 6: Verdicts

### V28: **REJECT**

| Criterion | Value | Required | Result |
|---|---:|---|---|
| Win rate vs V11 (5bps) | 43% (3/7) | 7/7 for GRADUATE | FAIL |
| Worst OOS window Sharpe | -0.496 | > 0 for GRADUATE | FAIL |
| Selection rank stability (mean) | 3.71 | <= 2 preferred | INFO |
| Sharpe dispersion (std) | 0.699 | < 0.5 preferred | INFO |

**V28 RATIONALE**: OOS performance insufficient. Full-window edge was concentrated in specific periods; walk-forward confirms REJECT.

### V31: **REJECT**

| Criterion | Value | Required | Result |
|---|---:|---|---|
| Win rate vs V11 (5bps) | 71% (5/7) | 7/7 for GRADUATE | FAIL |
| Worst OOS window Sharpe | -0.745 | > 0 for GRADUATE | FAIL |
| Selection rank stability (mean) | 2.57 | <= 2 preferred | INFO |
| Sharpe dispersion (std) | 0.870 | < 0.5 preferred | INFO |

**V31 RATIONALE**: OOS performance insufficient. Full-window edge was concentrated in specific periods; walk-forward confirms REJECT.

### Null Option

V11 remains the deployed paper incumbent regardless of the above verdicts.
No candidate can be deployed until walk-forward graduation AND separate
approval for paper promotion. If both V28 and V31 are REJECT, the null
option (keep V11 as permanent incumbent) is the correct call.

## Section 7: Registry Run IDs

Full-window return streams used for slicing (5 bps near_close):

- V11: `3cb8b7be...`
- V28: `6a6eced1...`
- V31: `9b29f3f1...`
- V02+V05: `5612e64c...`
- V26: `0ca42590...`
- V33-core: `2703750a...`

Full-window return streams used for 7.5 bps slicing:

- V11: `49ebc8e1...`
- V28: `d2bd80a6...`
- V31: `5cbc40b0...`

Walk-forward aggregate run_ids are recorded in the registry
under phase='wave3_walkforward'.
