# RAMP Phase 6.5 Robustness Report: V26-robust -- 2026-06-03

## Methodology

**Gate applied**: Phase 6.5 parameter robustness (strategy-lead Section 2)
**Stability metric**: full-window net Sharpe at near_close, 5 bps, 2017-01-01 to 2026-05-16
**Universe**: sp500-2025
**Data snapshot**: 2026-05-16
**Code commit**: 833e57ee451126ea21329bc0e35edc9241740e97

**This is a robustness MAP, NOT an optimization search.**
The center config is the a-priori candidate. Neighborhood variations test whether the edge is a stable plateau or a cliff-edge. No variation is adopted regardless of result -- promotion requires re-entering as its own a-priori candidate with its own gate + walk-forward + trial count.

**Stability criterion (Section 5.5):**
STABLE if all neighbors hold >= 90% of center Sharpe (90% x 0.6350 = 0.5715); else BRITTLE.

**Trial-count treatment (Section 9.4):**
All variations logged with phase=robustness_sensitivity. None are promoted, so none add to the project SELECTION-trial count. If a variation were later promoted as its own a-priori candidate, it would receive its own selection-trial entry at that time.

## Exact grid used

**V26-robust grid**: 1 center + 4x5 one-at-a-time + 4 combos = 25 points
Center: lambda=1.0, winsor_lo=0.01, winsor_hi=0.99, h_short=5d, h_long=21d
Perturbations: lambda at {0.8, 0.9, 1.1, 1.2} (+/-10%/+/-20%); winsor_lo at {0.005, 0.008, 0.012, 0.015} (approx +/-20%/+/-50%); winsor_hi at {0.985, 0.988, 0.992, 0.995}; h_short at {3, 4, 6, 7}d (approx +/-20%/+/-40%); h_long at {14, 17, 25, 28}d (approx +/-19%/+/-33%); combo: (lambda, h_short, h_long) moved together at +/-10%/+/-20%.
Canonical primitives: robust_zscore_cross_sectional (MAD-based) and winsorize (quantile) from src.features -- used in ALL variations.

## Results summary

**Center Sharpe measured** : 0.6350
**Center Sharpe (ref)**    : 0.6350
**Delta vs ref**           : +0.0000
**Stability threshold**    : 90% x center = 0.5715
**Worst neighbor Sharpe**  : 0.4704 (lam_m20_0.8)
**min(neighbor)/center**   : 0.7408
**Verdict**                : **BRITTLE**

## RED FLAGS: variations that beat the center

> These are reported as flags only. Per Phase 6.5 rules, a variation that materially beats the center is evidence of an arbitrary center or lumpy surface. It is NOT adopted here.

| Label | Sharpe | Delta vs center | Params |
|---|---:|---:|---|
| lam_p10_1.1 | 0.6659 | +0.0309 | lambda_=1.1, winsor_lo=0.01, winsor_hi=0.99, h_short=5, h_long=21 |
| lam_p20_1.2 | 0.6763 | +0.0413 | lambda_=1.2, winsor_lo=0.01, winsor_hi=0.99, h_short=5, h_long=21 |
| wlo_m50_0.005 | 0.7153 | +0.0803 | lambda_=1.0, winsor_lo=0.005, winsor_hi=0.99, h_short=5, h_long=21 |
| wlo_m20_0.008 | 0.6458 | +0.0108 | lambda_=1.0, winsor_lo=0.008, winsor_hi=0.99, h_short=5, h_long=21 |
| whi_p2_0.992 | 0.6525 | +0.0175 | lambda_=1.0, winsor_lo=0.01, winsor_hi=0.992, h_short=5, h_long=21 |
| hs_m40_3d | 0.6549 | +0.0199 | lambda_=1.0, winsor_lo=0.01, winsor_hi=0.99, h_short=3, h_long=21 |

## Full grid results

| Label | Sharpe | CAGR | Max DD | Ratio/Center | is_center |
|---|---:|---:|---:|---:|:---:|
| wlo_m50_0.005 | 0.7153 | 15.70% | -38.61% | 1.1264 |  |
| lam_p20_1.2 | 0.6763 | 14.71% | -43.87% | 1.0651 |  |
| lam_p10_1.1 | 0.6659 | 14.49% | -43.26% | 1.0487 |  |
| hs_m40_3d | 0.6549 | 13.80% | -41.83% | 1.0314 |  |
| whi_p2_0.992 | 0.6525 | 13.91% | -41.20% | 1.0275 |  |
| wlo_m20_0.008 | 0.6458 | 13.76% | -41.77% | 1.0170 |  |
| whi_m15_0.985 | 0.6387 | 13.45% | -41.62% | 1.0057 |  |
| CENTER_lam1.0_wlo0.01_whi0.99_hs5_hl21 | 0.6350 | 13.38% | -41.64% | 1.0000 | YES |
| whi_m12_0.988 | 0.6296 | 13.21% | -42.27% | 0.9916 |  |
| hs_p20_6d | 0.6246 | 13.16% | -43.82% | 0.9837 |  |
| combo_p10_lam1.1_hs6d_hl23d | 0.6246 | 13.21% | -46.91% | 0.9837 |  |
| hs_p40_7d | 0.6220 | 13.16% | -46.36% | 0.9796 |  |
| combo_m10_lam0.9_hs4d_hl19d | 0.5975 | 12.11% | -40.40% | 0.9409 |  |
| wlo_p20_0.012 | 0.5975 | 12.36% | -41.99% | 0.9409 |  |
| combo_p20_lam1.2_hs7d_hl25d | 0.5958 | 12.85% | -48.16% | 0.9383 |  |
| whi_p5_0.995 | 0.5904 | 12.26% | -42.93% | 0.9297 |  |
| hl_m33_14d | 0.5740 | 11.85% | -43.77% | 0.9040 |  |
| hl_m19_17d | 0.5708 | 11.67% | -39.98% | 0.8989 |  |
| wlo_p50_0.015 | 0.5700 | 11.60% | -44.02% | 0.8976 |  |
| lam_m10_0.9 | 0.5651 | 11.31% | -42.30% | 0.8899 |  |
| hs_m20_4d | 0.5067 | 9.86% | -45.40% | 0.7979 |  |
| hl_p19_25d | 0.5019 | 9.77% | -45.54% | 0.7904 |  |
| hl_p33_28d | 0.4964 | 9.70% | -48.32% | 0.7818 |  |
| combo_m20_lam0.8_hs3d_hl17d | 0.4900 | 9.21% | -37.86% | 0.7717 |  |
| lam_m20_0.8 | 0.4704 | 8.74% | -41.35% | 0.7408 |  |

## Registry run IDs

Total variations logged: 25
- CENTER_lam1.0_wlo0.01_whi0.99_hs5_hl21: run_id=0b483b42-947e-43f9-beb0-f3cf53799270
- lam_m20_0.8: run_id=20fbeca8-35db-4fa1-9bfb-91a62e3a0820
- lam_m10_0.9: run_id=b0ef165f-22aa-492b-84e5-bff9c69adc8f
- lam_p10_1.1: run_id=55ad536c-faf6-4505-b751-a9a12a4693c2
- lam_p20_1.2: run_id=162e66bc-9c2a-4c5f-8cd1-606416012df2
- wlo_m50_0.005: run_id=fa323245-17f5-4dde-a473-3fe9e37f3883
- wlo_m20_0.008: run_id=f3a733fc-5e7f-4943-b4d3-3d49e69d7189
- wlo_p20_0.012: run_id=a231875f-f243-4450-bd1c-187ab0d454d7
- wlo_p50_0.015: run_id=1577c91c-7e9d-4194-8e43-e9e0b3094bfa
- whi_m15_0.985: run_id=7f88872e-cc5f-471a-b2c3-e862f2364738
- whi_m12_0.988: run_id=7355849a-ab9d-4c92-8b33-2b4a8ff6b363
- whi_p2_0.992: run_id=b52d6dac-13b5-40b5-b86f-c189ea48268a
- whi_p5_0.995: run_id=c8169324-d156-49b8-8953-f004aa0da7f3
- hs_m40_3d: run_id=584ebc88-4a34-4f58-81c3-f60e3473bd75
- hs_m20_4d: run_id=518c55ee-911d-4d79-836e-775aeafb2b65
- hs_p20_6d: run_id=a1f51909-3f6b-4e23-bbc0-c568134216e9
- hs_p40_7d: run_id=73b245fe-7830-41bb-b1ae-4be6b4044163
- hl_m33_14d: run_id=2845949c-beed-44de-9121-80ff881808b1
- hl_m19_17d: run_id=da18accf-de8f-4435-9919-1ec7cc017ecb
- hl_p19_25d: run_id=fb94c118-d452-44ba-9769-6fcca36665a7
- hl_p33_28d: run_id=f93a15c9-1dea-49d6-955f-0d648ba1cea8
- combo_m10_lam0.9_hs4d_hl19d: run_id=09fd8a35-241b-44e3-99b7-305d26eb59e5
- combo_m20_lam0.8_hs3d_hl17d: run_id=ae47842e-a5f5-4493-8d81-e286841cd20f
- combo_p10_lam1.1_hs6d_hl23d: run_id=ca116c3b-15ce-4da5-b2a7-eceecfa7e3b6
- combo_p20_lam1.2_hs7d_hl25d: run_id=7f34d9b6-9617-486c-8cac-0834ffbfcdc6

## Interpretation

V26-robust is **BRITTLE**: at least one neighbor variation drops below 90% of the center Sharpe (min ratio = 0.7408, worst = lam_m20_0.8 at Sharpe 0.4704). Cliff-edge sensitivity indicates the parameter surface is lumpy, which is consistent with overfitting. The HYBRID build decision should treat this as a negative signal for V26-robust.

**Note**: this robustness verdict does NOT change V28/V31's existing walk-forward REJECT verdict (they remain OOS-rejected). It only informs whether the HYBRID lead (V28/V31 signal + V11 regime overlay) is worth building.

## G0.5 durability status

- Total variations: 25
- Total elapsed: 27.2 min
- Every variation written to registry BEFORE continuing (per G0.5 protocol).
