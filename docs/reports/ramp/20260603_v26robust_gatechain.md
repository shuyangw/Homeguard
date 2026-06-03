# RAMP V26-robust Gate Chain Report -- 2026-06-03

**Branch**: archive/regime-detector-campaign-2026-05
**Code commit**: 833e57e (feat: add V26-robust make_v26_robust_plan_fn factory + gate chain harness updates)
**Data snapshot**: 2026-05-16 (Alpaca SIP daily, split-adjusted)
**Universe**: sp500-2025 (494 symbols)
**Primary cost tier**: 5.0 bps per side (near_close timing)
**Cost sensitivity gate**: 7.5 bps (1.5x)
**Incumbent**: V11 (Sharpe 0.528 full-window, 5 bps, run_id=3cb8b7be)
**Candidate**: V26-robust (Sharpe 0.635 full-window, 5 bps, run_id=8d287db0)
**Edge over V11**: +0.107 Sharpe

---

## Section 1: Methodology Design

### Candidate description

V26-robust is V26 with the normalization method swapped from inline sigma z-score to the
canonical src.features toolbelt primitives:
  - winsorize (quantile 0.01/0.99) via src.features.winsorize
  - robust_zscore_cross_sectional (MAD-based) via src.features

Score: robust_z(winsorize(21d_return)) - 1.0 * robust_z(winsorize(5d_return))

lambda=1.0, winsor_lo=0.01, winsor_hi=0.99, h_long=21d, h_short=5d -- all FIXED,
a-priori. NOT grid-searched (single selection trial).

Everything else is identical to V11 (rank_buffer + min_hold + delta via cfg).
This is a canonical method A/B vs V26, not a new strategy.

### Section 3 applicability (no fitting step)

V26-robust has FIXED a-priori parameters -- nothing is fitted in-sample.

Section 3.2 (purging): NOT APPLICABLE. No training set from which to purge.
Section 3.3 (embargo): NOT APPLICABLE. No training/test fitting boundary.
Section 3.1 (window structure): APPLIES. 7 sequential calendar-year OOS windows.

Slicing validity: V26-robust's full-window return stream was computed causally
(all signals use .shift(1)/lag equivalent). Slicing by calendar year produces
the correct OOS Sharpe for that year. This was pre-verified for V28 on the
2022 window in the earlier Wave-3 run.

### OOS window design

7 calendar-year windows: 2019, 2020, 2021, 2022, 2023, 2024, 2025+
Warmup before 2019: 503 trading days (>= 252 required by detector).
Minimum folds required: 5 (Section 3.1). Implemented: 7.

### Robustness grid design (Gate 2)

Grid structure: 25 points total (GOOD tier, candidate beats V11 by +0.107)
Center: lambda=1.0, winsor_lo=0.01, winsor_hi=0.99, h_short=5d, h_long=21d
Perturbations:
  - lambda at {0.8, 0.9, 1.1, 1.2} (+/-10%/+/-20%)
  - winsor_lo at {0.005, 0.008, 0.012, 0.015} (approx +/-20%/+/-50%)
  - winsor_hi at {0.985, 0.988, 0.992, 0.995}
  - h_short at {3, 4, 6, 7}d (approx +/-20%/+/-40%)
  - h_long at {14, 17, 25, 28}d (approx +/-19%/+/-33%)
  - combo: (lambda, h_short, h_long) moved together at +/-10%/+/-20%

Canonical primitives used in ALL variations: _CANONICAL_ZSCORE and
_CANONICAL_WINSORIZE (module-level aliases to src.features toolbelt). No
inline math in any variation.

### Acceptance criteria

GATE 1 (walk-forward): GRADUATE iff V26-robust beats V11 in ALL 7 OOS windows
AND worst-window Sharpe > 0. Otherwise REJECT (or HOLD if >= 5/7 AND worst >= -0.1).

GATE 2 (robustness): STABLE if all 24 neighbor variations hold >= 0.9 of center
Sharpe (0.9 x 0.635 = 0.572). BRITTLE if any neighbor drops below 0.572.

COMBINED VERDICT: GRADUATE iff GATE 1 GRADUATE AND GATE 2 STABLE.

### Honest prior

V28 (+0.283 vs V11) and V31 (+0.241 vs V11) -- larger edges -- both rejected on the
every-window bar because they lost 2020/2022 BEAR years. V26-robust has a smaller
edge (+0.107) and a similar normalization-improvement rationale. The prior is that it
also rejects on the BEAR-year test.

---

## Section 2: GATE 1 -- Walk-Forward Results

**Registry run_id**: cb71e4f2 (phase=wave3_walkforward, strategy=RAMP-V26-robust)
**Full report**: docs/reports/ramp/20260601_wave3_walkforward.md
**Full JSON**: docs/reports/ramp/20260601_wave3_walkforward.json

### Per-window Sharpe table (5 bps near_close)

| Window | n_days | V11 | V26-robust | V26r > V11 |
|--------|-------:|----:|----------:|:----------:|
| 2019   | 251    | 1.483 | 1.086 | N |
| 2020   | 253    | 0.838 | 0.966 | Y |
| 2021   | 251    | 1.181 | 0.731 | N |
| 2022   | 250    | -0.266 | -0.461 | N |
| 2023   | 250    | 1.170 | 1.621 | Y |
| 2024   | 251    | 0.829 | 0.959 | Y |
| 2025+  | 349    | 0.527 | 1.029 | Y |

### Per-window at 7.5 bps (1.5x cost gate)

| Window | V11 (7.5bps) | V26-robust (7.5bps) | V26r > V11 |
|--------|------------:|-------------------:|:----------:|
| 2019   | 1.391 | 0.924 | N |
| 2020   | 0.767 | 0.909 | Y |
| 2021   | 1.095 | 0.604 | N |
| 2022   | -0.331 | -0.558 | N |
| 2023   | 1.045 | 1.490 | Y |
| 2024   | 0.771 | 0.825 | Y |
| 2025+  | 0.419 | 0.935 | Y |

### OOS summary

| Metric | Value |
|--------|------:|
| Win rate vs V11 (5bps) | 57% (4/7) |
| Worst OOS window Sharpe | -0.461 (2022) |
| Best OOS window Sharpe | 1.621 (2023) |
| Sharpe dispersion (std) | 0.638 |
| Pooled OOS Sharpe | 0.742 |

### Gate 1 verdict table

| Criterion | Value | Required | Result |
|-----------|------:|----------|--------|
| Win rate vs V11 (5bps) | 57% (4/7) | 7/7 for GRADUATE | FAIL |
| Worst OOS window Sharpe | -0.461 | > 0 for GRADUATE | FAIL |
| Win rate for HOLD | 57% | >= 5/7 | FAIL |
| Worst for HOLD | -0.461 | >= -0.10 | FAIL |

**GATE 1 VERDICT: REJECT**

V26-robust loses in 3 of 7 OOS windows: 2019 (1.086 vs V11 1.483), 2021 (0.731 vs
1.181), and 2022 (-0.461 vs -0.266). The 2022 failure is the most diagnostic: V26-robust
loses more severely in BEAR than V11, exactly the same pattern as V28 and V31.

This confirms the prior: the normalization improvement (MAD vs sigma z-score) does not
fix the fundamental BEAR-year selection problem. The issue is not the normalization
method -- it is the stock-selection signal itself under BEAR regime conditions.

Note: V26-robust's pooled OOS Sharpe (0.742) looks decent but the every-window test is
the correct gate -- pooled Sharpe can be dominated by strong non-BEAR years masking
regime-specific failure.

---

## Section 3: GATE 2 -- Robustness Sweep Results

**Registry runs**: 25 runs logged with phase=robustness_sensitivity, strategy=RAMP-V26-robust
**Report**: docs/reports/ramp/20260603_robustness_v26-robust.md
**JSON**: docs/reports/ramp/20260603_robustness_v26-robust.json
**Total elapsed**: 28.1 minutes

### Summary statistics

| Metric | Value |
|--------|------:|
| Center Sharpe (measured) | 0.6350 |
| Center Sharpe (ref from full-window run) | 0.6350 |
| Stability threshold | 90% x 0.6350 = 0.5715 |
| Worst neighbor Sharpe | 0.4704 (lam_m20_0.8) |
| min(neighbor) / center | 0.741 |
| Red-flag variations (beat center) | 6 |

### Key per-variation results

| Label | Sharpe | Ratio/Center | Note |
|-------|-------:|-------------:|------|
| CENTER_lam1.0_wlo0.01_whi0.99_hs5_hl21 | 0.6350 | 1.000 | CENTER |
| lam_m20_0.8 | 0.4704 | 0.741 | WORST |
| lam_m10_0.9 | 0.5651 | 0.890 | below threshold |
| lam_p10_1.1 | 0.6659 | 1.049 | RED FLAG |
| lam_p20_1.2 | 0.6763 | 1.065 | RED FLAG |
| wlo_m50_0.005 | 0.7153 | 1.127 | RED FLAG (largest) |
| wlo_m20_0.008 | 0.6458 | 1.017 | RED FLAG |
| wlo_p20_0.012 | 0.5975 | 0.941 | near threshold |
| wlo_p50_0.015 | 0.5700 | 0.898 | below threshold |
| whi_m15_0.985 | 0.6387 | 1.006 | RED FLAG |
| whi_m12_0.988 | 0.6296 | 0.992 | ok |
| whi_p2_0.992 | 0.6525 | 1.027 | RED FLAG |
| whi_p5_0.995 | 0.5904 | 0.930 | ok |
| hs_m40_3d | 0.6549 | 1.031 | RED FLAG |
| hs_m20_4d | 0.5067 | 0.798 | below threshold |
| hs_p20_6d | 0.6246 | 0.984 | ok |
| hs_p40_7d | 0.6220 | 0.980 | ok |
| hl_m33_14d | 0.5740 | 0.904 | ok |
| hl_m19_17d | 0.5708 | 0.899 | below threshold |
| hl_p19_25d | 0.5019 | 0.790 | below threshold |
| hl_p33_28d | 0.4964 | 0.782 | below threshold |
| combo_m10_lam0.9_hs4d_hl19d | 0.5975 | 0.941 | ok |
| combo_m20_lam0.8_hs3d_hl17d | 0.4900 | 0.772 | below threshold |
| combo_p10_lam1.1_hs6d_hl23d | 0.6246 | 0.984 | ok |
| combo_p20_lam1.2_hs7d_hl25d | 0.5958 | 0.938 | ok |

### Gate 2 verdict table

| Criterion | Value | Required | Result |
|-----------|------:|----------|--------|
| min(neighbor)/center | 0.741 | >= 0.90 | FAIL |
| Worst neighbor | 0.4704 (lam_m20) | >= 0.572 | FAIL |
| Red-flag variations (beat center) | 6 of 24 | 0 preferred | WARN |

**GATE 2 VERDICT: BRITTLE**

The surface is clearly not a stable plateau:
- 7 of 24 neighbors fall below 0.9 x center (threshold = 0.572)
- The worst neighbor (lambda=0.8, -20%) gives only 0.741 of center Sharpe
- 6 neighbors beat the center, indicating the center is not at a local optimum
- The h_long dimension is particularly sensitive: lengthening to 25d or 28d drops
  Sharpe to 0.502/0.496 (0.79/0.78 of center)

The 6 red-flag variations (lam=1.1, lam=1.2, wlo=0.005, wlo=0.008, whi=0.992, hs=3d)
all beat the center. Per Phase 6.5 rules, none are adopted. The presence of multiple
superior neighbors suggests the a-priori center (lambda=1.0, h_long=21d, h_short=5d)
is not at the plateau -- these parameters could have been placed differently, and the
FULL-WINDOW result partially reflects parameter luck rather than a stable edge.

This is independent evidence that V26-robust should not be graduated, consistent with
GATE 1's rejection.

---

## Section 4: Combined Verdict

**GATE 1**: REJECT (win rate 4/7, worst -0.461 in 2022)
**GATE 2**: BRITTLE (min_ratio=0.741, 7 of 24 neighbors below threshold, 6 red flags)

**COMBINED VERDICT: REJECT**

Both gates fail independently. The failure pattern is coherent:
1. The normalization change (MAD vs sigma z-score) did not fix the BEAR-year selection
   problem that caused V28 and V31 to fail. The candidate retains the fundamental
   momentum-signal behavior of V26 under BEAR conditions.
2. The parameter surface around the a-priori center is lumpy (BRITTLE), not a stable
   plateau. The full-window Sharpe of 0.635 is inflated by period-specific performance
   rather than a robust, regime-stable edge.

### Null option

V11 remains the deployed paper incumbent. No candidate from the V26/V28/V31/V26-robust
family beats V11 on the every-window bar. The Wave-3 campaign is closed with V11 as
the permanent incumbent.

### Comparison to prior rejections

| Candidate | Full-window edge vs V11 | Gate 1 win rate | BEAR year | Combined |
|-----------|------------------------:|:--------------:|----------:|---------|
| V28       | +0.283                  | 3/7 (43%)      | -0.496    | REJECT  |
| V31       | +0.241                  | 5/7 (71%)      | -0.745    | REJECT  |
| V26-robust| +0.107                  | 4/7 (57%)      | -0.461    | REJECT  |

V26-robust has the smallest edge and fails both gates. The pattern is consistent across
all three candidates: BEAR years (especially 2022) expose the fundamental fragility of
this signal class.

---

## Section 5: Registry Run IDs

| Phase | Run ID | Strategy | Verdict |
|-------|--------|----------|---------|
| Full-window (5bps) | 8d287db0 | RAMP-V26-robust | wave3_readiness |
| Full-window (7.5bps) | 86079262 | RAMP-V26-robust | wave3_readiness |
| Gate 1 walk-forward | cb71e4f2 | RAMP-V26-robust | REJECT |
| Gate 2 center | 0b483b42 | RAMP-V26-robust | robustness_sensitivity |
| Gate 2 worst (lam_m20) | 20fbeca8 | RAMP-V26-robust | robustness_sensitivity |
| Gate 2 red flag (wlo_m50) | fa323245 | RAMP-V26-robust | robustness_sensitivity |
| Gate 2 all 25 variations | see JSON | RAMP-V26-robust | robustness_sensitivity |

Full variation run_id list: docs/reports/ramp/20260603_robustness_v26-robust.json

---

## Section 6: Tests

**Tests written**: 6 new tests added to tests/research/ramp_phase4/test_variants.py

| Test | What it asserts | Result |
|------|-----------------|--------|
| test_make_v26_robust_plan_fn_exists | make_v26_robust_plan_fn importable and callable | PASS |
| test_make_v26_robust_default_matches_registered_variant | factory-with-defaults == _variant_v26_robust on synthetic panel (symbols AND weights) | PASS |
| test_make_v26_robust_perturbed_lambda_runs | lambda=0.8 variation runs without error | PASS |
| test_make_v26_robust_perturbed_horizons_runs | h_short=4, h_long=17 variation runs without error | PASS |
| test_make_v26_robust_perturbed_winsor_runs | winsor_lo=0.02, winsor_hi=0.98 runs without error | PASS |
| test_make_v26_robust_uses_canonical_primitives_via_factory | factory routes through _CANONICAL_ZSCORE and _CANONICAL_WINSORIZE (monkeypatch verification) | PASS |

**Total test suite**: 219 passed, 0 failed (was 213; +6 new tests)

---

## Section 7: Modifications

| File | Change |
|------|--------|
| src/research/ramp_phase4/variants.py | Added _V26_ROBUST_CENTER_* constants; parametrized _compute_v26_robust_scores; added make_v26_robust_plan_fn factory |
| tests/research/ramp_phase4/test_variants.py | Added 6 new factory tests (lines 2064-2167) |
| scripts/backtest_scripts/ramp_phase4_wave3_walkforward.py | Added V26-robust to FULL_WINDOW_RUN_IDS and FULL_WINDOW_RUN_IDS_75; updated tables, rank stability, verdicts, registry loop |
| scripts/backtest_scripts/ramp_phase4_robustness.py | Added _V26_ROBUST_CENTER_* imports, make_v26_robust_plan_fn import, _build_v26_robust_grid, V26-robust branch in _run_one_variation and _parse_args |

**Commit**: 833e57e (feat: add V26-robust make_v26_robust_plan_fn factory + gate chain harness updates)
