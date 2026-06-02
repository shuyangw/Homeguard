# RAMP Phase 6.5 Robustness Report: V28 -- 2026-06-02

## Methodology

**Gate applied**: Phase 6.5 parameter robustness (strategy-lead Section 2)
**Stability metric**: full-window net Sharpe at near_close, 5 bps, 2017-01-01 to 2026-05-16
**Universe**: sp500-2025
**Data snapshot**: 2026-05-16
**Code commit**: a2b4b9a4fb8b0406be45dcb6dc2f4363713ebc9e

**This is a robustness MAP, NOT an optimization search.**
The center config is the a-priori candidate. Neighborhood variations test whether the edge is a stable plateau or a cliff-edge. No variation is adopted regardless of result -- promotion requires re-entering as its own a-priori candidate with its own gate + walk-forward + trial count.

**Stability criterion (Section 5.5):**
STABLE if all neighbors hold >= 90% of center Sharpe (90% x 0.8112 = 0.7301); else BRITTLE.

**Trial-count treatment (Section 9.4):**
All variations logged with phase=robustness_sensitivity. None are promoted, so none add to the project SELECTION-trial count. If a variation were later promoted as its own a-priori candidate, it would receive its own selection-trial entry at that time.

## Exact grid used

**V28 grid**: 1 center + 4x6 one-at-a-time = 25 points
Center: w21=0.5, w63=0.3, w126=0.2, lam=0.1, h21=21, h63=63, h126=126
Perturbations: each of {w21, w63, w126} varied at +/-10%/+/-20% (remaining weights renormalized proportionally); lambda_rev at +/-10%/+/-20%; h21 at {19,17,23,25}; h63 at {57,50,69,76}.

## Results summary

**Center Sharpe measured** : 0.8112
**Center Sharpe (ref)**    : 0.8110
**Delta vs ref**           : +0.0002
**Stability threshold**    : 90% x center = 0.7301
**Worst neighbor Sharpe**  : 0.6750 (h21_m20_17d)
**min(neighbor)/center**   : 0.8321
**Verdict**                : **BRITTLE**

## RED FLAGS: variations that beat the center

> These are reported as flags only. Per Phase 6.5 rules, a variation that materially beats the center is evidence of an arbitrary center or lumpy surface. It is NOT adopted here.

| Label | Sharpe | Delta vs center | Params |
|---|---:|---:|---|
| w21_p10_0.550_w63=0.270_w126=0.180 | 0.8387 | +0.0275 | blend_21d=0.55, blend_63d=0.26999999999999996, blend_126d=0.18, lambda_rev=0.1, h_21=21, h_63=63, h_126=126 |
| w63_m20_0.240_w21=0.543_w126=0.217 | 0.8241 | +0.0129 | blend_21d=0.5428571428571429, blend_63d=0.24, blend_126d=0.2171428571428572, lambda_rev=0.1, h_21=21, h_63=63, h_126=126 |
| h63_m10_57d | 0.8483 | +0.0371 | blend_21d=0.5, blend_63d=0.3, blend_126d=0.2, lambda_rev=0.1, h_21=21, h_63=57, h_126=126 |
| h63_m20_50d | 0.8711 | +0.0599 | blend_21d=0.5, blend_63d=0.3, blend_126d=0.2, lambda_rev=0.1, h_21=21, h_63=50, h_126=126 |
| h63_p20_76d | 0.8948 | +0.0836 | blend_21d=0.5, blend_63d=0.3, blend_126d=0.2, lambda_rev=0.1, h_21=21, h_63=76, h_126=126 |

## Full grid results

| Label | Sharpe | CAGR | Max DD | Ratio/Center | is_center |
|---|---:|---:|---:|---:|:---:|
| h63_p20_76d | 0.8948 | 22.89% | -37.33% | 1.1031 |  |
| h63_m20_50d | 0.8711 | 21.91% | -36.20% | 1.0738 |  |
| h63_m10_57d | 0.8483 | 21.19% | -37.87% | 1.0458 |  |
| w21_p10_0.550_w63=0.270_w126=0.180 | 0.8387 | 20.75% | -42.56% | 1.0340 |  |
| w63_m20_0.240_w21=0.543_w126=0.217 | 0.8241 | 20.41% | -41.47% | 1.0159 |  |
| lam_m20_0.0800 | 0.8176 | 20.19% | -42.02% | 1.0079 |  |
| w63_m10_0.270_w21=0.521_w126=0.209 | 0.8117 | 19.89% | -41.46% | 1.0006 |  |
| CENTER_w21=0.5_w63=0.3_w126=0.2_lam=0.1 | 0.8112 | 20.03% | -42.00% | 1.0000 | YES |
| w126_m10_0.180_w21=0.512_w63=0.307 | 0.8110 | 19.82% | -42.20% | 0.9998 |  |
| lam_m10_0.0900 | 0.8094 | 19.93% | -41.88% | 0.9978 |  |
| w126_m20_0.160_w21=0.525_w63=0.315 | 0.8079 | 19.70% | -42.33% | 0.9959 |  |
| w21_m10_0.450_w63=0.330_w126=0.220 | 0.8077 | 19.89% | -38.92% | 0.9956 |  |
| w21_p20_0.600_w63=0.240_w126=0.160 | 0.8077 | 19.83% | -38.99% | 0.9956 |  |
| w126_p10_0.220_w21=0.487_w63=0.292 | 0.8063 | 19.80% | -39.25% | 0.9940 |  |
| lam_p20_0.1200 | 0.8017 | 19.70% | -42.14% | 0.9883 |  |
| lam_p10_0.1100 | 0.7986 | 19.65% | -42.72% | 0.9844 |  |
| w126_p20_0.240_w21=0.475_w63=0.285 | 0.7935 | 19.50% | -42.64% | 0.9782 |  |
| h21_p20_25d | 0.7924 | 19.33% | -45.00% | 0.9769 |  |
| w63_p20_0.360_w21=0.457_w126=0.183 | 0.7916 | 19.42% | -40.83% | 0.9758 |  |
| h63_p10_69d | 0.7902 | 19.47% | -41.82% | 0.9741 |  |
| w63_p10_0.330_w21=0.479_w126=0.191 | 0.7710 | 18.63% | -43.52% | 0.9504 |  |
| w21_m20_0.400_w63=0.360_w126=0.240 | 0.7622 | 18.52% | -41.25% | 0.9396 |  |
| h21_m10_19d | 0.7550 | 18.17% | -42.08% | 0.9307 |  |
| h21_p10_23d | 0.7055 | 16.60% | -45.37% | 0.8697 |  |
| h21_m20_17d | 0.6750 | 15.73% | -43.37% | 0.8321 |  |

## Registry run IDs

Total variations logged: 25
- CENTER_w21=0.5_w63=0.3_w126=0.2_lam=0.1: run_id=d56491a7-203d-4af8-b2b3-658886dc2d93
- w21_m10_0.450_w63=0.330_w126=0.220: run_id=e9138143-1296-4a23-85d7-6a31320662bb
- w21_m20_0.400_w63=0.360_w126=0.240: run_id=27b74780-1aca-4fc2-bf04-6dd7aec8ec19
- w21_p10_0.550_w63=0.270_w126=0.180: run_id=6a935ef2-348b-4828-96dc-b964f1381117
- w21_p20_0.600_w63=0.240_w126=0.160: run_id=c6338ee3-1a67-4d2d-a60d-a75081559be9
- w63_m10_0.270_w21=0.521_w126=0.209: run_id=0fd837ad-0357-46f6-8044-01ce64118a45
- w63_m20_0.240_w21=0.543_w126=0.217: run_id=f3717c60-a9ce-4409-8360-fa98854d8154
- w63_p10_0.330_w21=0.479_w126=0.191: run_id=bf2b2095-2d29-4f26-877a-0b0390a502d8
- w63_p20_0.360_w21=0.457_w126=0.183: run_id=a4185bbb-16b6-46e4-96da-29f51198e228
- w126_m10_0.180_w21=0.512_w63=0.307: run_id=bc8db9ba-f3fc-4fdc-ab67-5a7ea7ae3db5
- w126_m20_0.160_w21=0.525_w63=0.315: run_id=4601aa2e-2bb5-4922-87fd-7a35a3546462
- w126_p10_0.220_w21=0.487_w63=0.292: run_id=87c4ed1e-056a-484d-8219-8b2f6c482b89
- w126_p20_0.240_w21=0.475_w63=0.285: run_id=ea881789-a371-44e7-bcd7-18754135b61c
- lam_m10_0.0900: run_id=de21ae55-a34b-47fa-a5d8-72ef8d82418c
- lam_m20_0.0800: run_id=b766adad-c239-4ea5-ae7f-05c83dd9b02a
- lam_p10_0.1100: run_id=3e61205c-bdd4-4046-999c-44e8f9ed7440
- lam_p20_0.1200: run_id=3c73ec35-2322-48ee-a3ed-65053ad05d83
- h21_m10_19d: run_id=e6281d4e-49bc-4982-83d2-1dffd1e8d45b
- h21_m20_17d: run_id=c31151ae-338e-4d5e-8bc2-508aca36d0fe
- h21_p10_23d: run_id=7692c121-4b4f-419c-9fb3-9430b4ab97af
- h21_p20_25d: run_id=2bc5ca19-3cd3-467d-b036-ce85544c0c36
- h63_m10_57d: run_id=59ffadc2-319d-4210-8393-8fafcf958152
- h63_m20_50d: run_id=bfc037b0-aef2-412c-ba6a-a754e6739c72
- h63_p10_69d: run_id=8539c6b8-dcb0-4c9c-8447-f341a7db5a1f
- h63_p20_76d: run_id=83892c34-bbb7-499a-ae2a-682dabb28018

## Interpretation

V28 is **BRITTLE**: at least one neighbor variation drops below 90% of the center Sharpe (min ratio = 0.8321, worst = h21_m20_17d at Sharpe 0.6750). Cliff-edge sensitivity indicates the parameter surface is lumpy, which is consistent with overfitting. The HYBRID build decision should treat this as a negative signal for {variant_id}.

**Note**: this robustness verdict does NOT change V28/V31's existing walk-forward REJECT verdict (they remain OOS-rejected). It only informs whether the HYBRID lead (V28/V31 signal + V11 regime overlay) is worth building.

## G0.5 durability status

- Total variations: 25
- Total elapsed: 26.6 min
- Every variation written to registry BEFORE continuing (per G0.5 protocol).
