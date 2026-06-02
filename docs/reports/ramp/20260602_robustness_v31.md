# RAMP Phase 6.5 Robustness Report: V31 -- 2026-06-02

## Methodology

**Gate applied**: Phase 6.5 parameter robustness (strategy-lead Section 2)
**Stability metric**: full-window net Sharpe at near_close, 5 bps, 2017-01-01 to 2026-05-16
**Universe**: sp500-2025
**Data snapshot**: 2026-05-16
**Code commit**: a2b4b9a4fb8b0406be45dcb6dc2f4363713ebc9e

**This is a robustness MAP, NOT an optimization search.**
The center config is the a-priori candidate. Neighborhood variations test whether the edge is a stable plateau or a cliff-edge. No variation is adopted regardless of result -- promotion requires re-entering as its own a-priori candidate with its own gate + walk-forward + trial count.

**Stability criterion (Section 5.5):**
STABLE if all neighbors hold >= 90% of center Sharpe (90% x 0.7685 = 0.6916); else BRITTLE.

**Trial-count treatment (Section 9.4):**
All variations logged with phase=robustness_sensitivity. None are promoted, so none add to the project SELECTION-trial count. If a variation were later promoted as its own a-priori candidate, it would receive its own selection-trial entry at that time.

## Exact grid used

**V31 grid**: 5 x 5 = 25 points
beta_window in {72, 81, 90, 99, 108} (center=90, approx -20%/-10%/center/+10%/+20%)
residual_horizon in {17, 19, 21, 23, 25} (center=21, approx -19%/-10%/center/+10%/+19%)

## Results summary

**Center Sharpe measured** : 0.7685
**Center Sharpe (ref)**    : 0.7690
**Delta vs ref**           : -0.0005
**Stability threshold**    : 90% x center = 0.6916
**Worst neighbor Sharpe**  : 0.5598 (bw72_rh25)
**min(neighbor)/center**   : 0.7285
**Verdict**                : **BRITTLE**

## RED FLAGS: variations that beat the center

> These are reported as flags only. Per Phase 6.5 rules, a variation that materially beats the center is evidence of an arbitrary center or lumpy surface. It is NOT adopted here.

| Label | Sharpe | Delta vs center | Params |
|---|---:|---:|---|
| bw99_rh21 | 0.8455 | +0.0770 | beta_window=99, residual_horizon=21 |
| bw108_rh21 | 0.8061 | +0.0376 | beta_window=108, residual_horizon=21 |

## Full grid results

| Label | Sharpe | CAGR | Max DD | Ratio/Center | is_center |
|---|---:|---:|---:|---:|:---:|
| bw99_rh21 | 0.8455 | 19.75% | -31.04% | 1.1002 |  |
| bw108_rh21 | 0.8061 | 18.54% | -31.06% | 1.0489 |  |
| CENTER_bw90_rh21 | 0.7685 | 17.40% | -33.51% | 1.0000 | YES |
| bw99_rh17 | 0.7556 | 17.55% | -40.17% | 0.9833 |  |
| bw81_rh17 | 0.7404 | 17.02% | -38.83% | 0.9634 |  |
| bw90_rh17 | 0.7357 | 16.85% | -40.67% | 0.9574 |  |
| bw72_rh21 | 0.7293 | 16.29% | -33.97% | 0.9490 |  |
| bw81_rh21 | 0.7227 | 16.11% | -32.77% | 0.9405 |  |
| bw72_rh23 | 0.7033 | 15.63% | -33.57% | 0.9151 |  |
| bw72_rh17 | 0.6997 | 15.76% | -39.08% | 0.9104 |  |
| bw99_rh23 | 0.6979 | 15.60% | -33.27% | 0.9082 |  |
| bw90_rh23 | 0.6880 | 15.20% | -33.05% | 0.8953 |  |
| bw72_rh19 | 0.6854 | 15.18% | -37.70% | 0.8919 |  |
| bw108_rh17 | 0.6820 | 15.29% | -40.16% | 0.8875 |  |
| bw81_rh23 | 0.6725 | 14.79% | -34.06% | 0.8751 |  |
| bw99_rh25 | 0.6555 | 14.16% | -38.95% | 0.8529 |  |
| bw108_rh23 | 0.6481 | 14.07% | -34.93% | 0.8434 |  |
| bw108_rh25 | 0.6440 | 13.82% | -40.55% | 0.8380 |  |
| bw81_rh19 | 0.6250 | 13.46% | -36.61% | 0.8133 |  |
| bw99_rh19 | 0.6071 | 12.91% | -37.26% | 0.7899 |  |
| bw108_rh19 | 0.6048 | 12.79% | -37.38% | 0.7870 |  |
| bw90_rh25 | 0.6027 | 12.63% | -42.27% | 0.7843 |  |
| bw90_rh19 | 0.5966 | 12.58% | -36.60% | 0.7764 |  |
| bw81_rh25 | 0.5866 | 12.21% | -43.28% | 0.7633 |  |
| bw72_rh25 | 0.5598 | 11.42% | -42.34% | 0.7285 |  |

## Registry run IDs

Total variations logged: 25
- bw72_rh17: run_id=2c589bb3-06a9-4dbe-a713-563194b98c1e
- bw72_rh19: run_id=7baa104c-3591-46bc-9420-1ebd1946c663
- bw72_rh21: run_id=171f4fc3-7760-4c9b-8ae6-9131585df688
- bw72_rh23: run_id=d1520be9-c331-4296-b697-fb1acdc16baa
- bw72_rh25: run_id=ce4ef496-ffdb-4264-b094-511812150e6a
- bw81_rh17: run_id=17907b1d-f58c-4f17-96eb-f3ec5bf7416e
- bw81_rh19: run_id=95954ca7-c4bc-4d06-aa00-4bec5e5d8237
- bw81_rh21: run_id=aa32962d-8ede-4255-9873-a4951e7eb71c
- bw81_rh23: run_id=02355f46-0c36-4e15-9316-792d8a87c8bf
- bw81_rh25: run_id=61815ce7-ba25-4759-92c4-37bb67f52ee0
- bw90_rh17: run_id=fee7967f-7936-4092-8e15-335ec40ffa6e
- bw90_rh19: run_id=1db03fef-4135-4101-bb10-b225a285f5ac
- CENTER_bw90_rh21: run_id=a7e1b49c-6624-4306-a233-c85926522fe2
- bw90_rh23: run_id=95ac85ab-a165-45d8-96f8-c43917d59a5e
- bw90_rh25: run_id=a2023e45-92b4-42fd-af55-2a5bf4f315a1
- bw99_rh17: run_id=5fcba597-018c-4990-a777-b5ce0a6cb099
- bw99_rh19: run_id=3162b35d-f8a9-4549-914b-20a0b7efdcc2
- bw99_rh21: run_id=7070d38e-8fda-4764-8ead-5305e786ecab
- bw99_rh23: run_id=94a9c96c-1a40-4ef0-be57-f422571239b2
- bw99_rh25: run_id=0a133b8e-0425-45bd-b6aa-401d939235fa
- bw108_rh17: run_id=0c950a51-5dbe-4fb8-9de9-55231124d458
- bw108_rh19: run_id=06f95836-82b8-4d52-8244-ce60bca6cd79
- bw108_rh21: run_id=6b58d1f7-bd78-40a1-9c15-8308f2c80f9e
- bw108_rh23: run_id=7155c40c-def0-4f14-960a-03232330902c
- bw108_rh25: run_id=4c07e9fe-77d1-432f-a295-42482308c3a8

## Interpretation

V31 is **BRITTLE**: at least one neighbor variation drops below 90% of the center Sharpe (min ratio = 0.7285, worst = bw72_rh25 at Sharpe 0.5598). Cliff-edge sensitivity indicates the parameter surface is lumpy, which is consistent with overfitting. The HYBRID build decision should treat this as a negative signal for {variant_id}.

**Note**: this robustness verdict does NOT change V28/V31's existing walk-forward REJECT verdict (they remain OOS-rejected). It only informs whether the HYBRID lead (V28/V31 signal + V11 regime overlay) is worth building.

## G0.5 durability status

- Total variations: 25
- Total elapsed: 26.6 min
- Every variation written to registry BEFORE continuing (per G0.5 protocol).
