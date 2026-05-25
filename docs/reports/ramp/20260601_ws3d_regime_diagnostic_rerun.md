# WS-3d Diagnostic Rerun -- H1-H5 on the v1 LightGBM Detector

**Date**: 2026-06-01
**Branch**: v12-bear-to-cash
**Spec**: docs/superpowers/specs/2026-05-25-ws3d-detector-replacement-design.md
**Gate**: Gate 1 (H5 lag reduction, GATING)
**Status**: Gate 1 FAIL

## Headline

Two H5 measurement bases are reported. The G4-event basis matches
the methodology of the 20260523 v0 baseline-of-record (14d) and is
the apples-to-apples comparison the Gate 1 verdict uses. The
G1_BEAR-onset basis (spec methodology) is also reported but
typically saturates to 0d because G1_BEAR is a drawdown-confirmed
label that fires AFTER the price weakness both detectors react to.

- v0 H5 median lag, G4-event basis (this run): 14.0 days
- v0 H5 baseline of record (20260523, G4-event basis): 14.0 days
- v1 H5 median lag, G4-event basis (this run): 21.0 days
- Reduction vs v0 G4-basis same run: -50.0%
- Reduction vs 14d baseline of record: -50.0%

- v0 H5 median lag, G1_BEAR-onset basis (this run): 0.0 days
- v1 H5 median lag, G1_BEAR-onset basis (this run): 0.0 days
- Reduction vs v0 G1-basis same run: nan%

- Pre-commitment 5 threshold: >= 30% reduction (v1 median <= 10d)

**Verdict: Gate 1 FAIL**

## H5 -- G4 same-basis (apples-to-apples)

### H5 (G4-event basis): lag from drawdown event start
to first detector-BEAR label within event window

| Metric | v0 (baseline) | v1 (WS-3d) |
|---|---|---|
| n events | 5 | 5 |
| n fired | 5 | 4 |
| capture rate | 100.0% | 80.0% |
| median lag | 14.0 | 21.0 |
| P25 lag | 14.0 | 17.0 |
| P75 lag | 14.0 | 23.0 |
| mean lag | 16.6 | 19.0 |

Per-event:

| event | start | v0 lag | v0 fired | v1 lag | v1 fired |
|---|---|---|---|---|---|
| Q4_2018_selloff | 2018-10-03 | 14 | True | 26 | True |
| COVID_crash | 2020-02-19 | 14 | True | 8 | True |
| 2022_bear_market | 2022-01-04 | 14 | True | 20 | True |
| 2025_tariff_drawdown | 2025-02-19 | 5 | True | 22 | True |
| 2025_dec_drawdown | 2025-12-15 | 36 | True | n/a | False |


## H5 -- G1_BEAR onset basis (spec methodology)

### H5: Median lag from G1_BEAR onset to first detector-BEAR label

Methodology: for each G1_BEAR onset (False -> True transition),
measure the lag in calendar days from the onset to the first
detector-BEAR label within a forward 60-day window.
Onsets with no BEAR fire in window are reported separately and
excluded from the lag distribution.

NOTE: G1_BEAR is a drawdown-confirmation label. It fires AFTER
SPY has already declined >= 10% from its trailing 252-day peak.
Both v0 and v1 typically already have BEAR active by the time
G1_BEAR turns True, so the G1-basis lag often saturates to 0d.
The Gate 1 verdict is decided on the G4-event basis above, which
measures from drawdown START rather than from drawdown CONFIRMATION.

| Metric | v0 (baseline) | v1 (WS-3d) |
|---|---|---|
| n onsets | 22 | 22 |
| n fired in window | 21 | 22 |
| capture rate | 95.5% | 100.0% |
| median lag (days) | 0.0 | 0.0 |
| P25 lag (days) | 0.0 | 0.0 |
| P75 lag (days) | 3.0 | 0.0 |
| mean lag (days) | 3.6 | 2.6 |


## Diagnosis and recommendation

Gate 1 FAIL. WS-3d is BLOCKED at the diagnostic-rerun gate.

### Root cause

The v1 detector was trained to predict G1_BEAR (drawdown >= 10%
from trailing 252-day high), and consumed via an argmax-flip-on-0.5
mapping per the spec (BEAR_PROB_THRESHOLD = 0.5 in
src/strategies/advanced/market_regime_detector_v1.py). By
construction, P(G1_BEAR | indicators) only crosses 0.5 around the
same time G1_BEAR itself fires -- which is AFTER the drawdown is
confirmed at ~10%. The argmax label therefore tracks confirmation
rather than precedes it. v1 H2 recall (96.5%) vs v0 (46.1%) is the
other side of this: v1 is dominant on confirmed G1_BEAR days but
does not flip BEAR earlier than v0 on the GATE-relevant G4-event
basis.

Per-event detail (G4 basis):

- Q4_2018_selloff (2018-10-03): v0=14, v1=26  ->  WORSE by 12d
- COVID_crash (2020-02-19): v0=14, v1=8  ->  BETTER by 6d
- 2022_bear_market (2022-01-04): v0=14, v1=20  ->  WORSE by 6d
- 2025_tariff_drawdown (2025-02-19): v0=5, v1=22  ->  WORSE by 17d
- 2025_dec_drawdown (2025-12-15): v0=36, v1=DID NOT FIRE  ->  WORSE (v1 missed)

### Recommended spec revisions

1. **Lower BEAR_PROB_THRESHOLD or move to a Schmitt-trigger consumer.**
   The raw P(BEAR) trace shows v1 crosses 0.25-0.30 days before the
   argmax fires at 0.5. The spec already plans Gate 2 (pre-spec tau
   from G1_BEAR median on v1 outputs); that tau will likely be in
   the 0.10-0.30 band and would make V14-style Schmitt-trigger
   variants (V20-rd-bear-cash, etc.) fire earlier than v0.
   However, this defers the test to Gate 3 (readiness) instead of
   the diagnostic gate. The spec needs amendment to either (a) move
   Gate 1 to evaluate on the Schmitt-fired label rather than
   argmax, or (b) explicitly accept that Gate 1 measures argmax
   lag and is informational only when the consumer is Schmitt-based.

2. **Train on a LEADING target instead of G1_BEAR.**
   G1_BEAR is a CONFIRMATION label by construction. Train on G2_BEAR
   (forward 30-day return < -5% AND forward vol > 25%) instead.
   G2 is forward-looking but in-sample-only is acceptable since
   training data is by definition historical. The trade-off is that
   G2 has more class imbalance and harder to learn.

3. **Alternative: train on G1_BEAR shifted backward by k days.**
   Use label = G1_BEAR.shift(-k) for k in {5, 10, 15}, picking k
   that maximizes recall on G4 events at the target lag. This is
   methodologically cleanest -- still supervised, but on a leading
   target rather than a coincident one.

4. **Consider the alternative architectures in the spec Appendix.**
   HMM or threshold-ensemble may have different lag characteristics.
   But none of them address the underlying issue that a confirmation
   label cannot be predicted ahead of itself by a supervised model
   with a 0.5 decision threshold.

5. **Escalate to halt-or-redirect per parent WS-3 spec Appendix.**
   Three independent measurements of structural detector lag (V12
   gap_days=-3.42, v0 H5=14d, E8 exit-to-low=-8d) led to this spec.
   If WS-3d cannot reduce H5 lag with a fresh architecture AND a
   fresh input set, the regime-aware approach may be at its useful
   limit for RAMP regardless of detector iteration.

### Stop here per Pre-commitment 5

Per spec: "the diagnostic rerun is a gating check before the
readiness orchestrator runs: if H5 lag is not reduced by 30%, the
leading indicator set OR the architecture is wrong and we don't
proceed to readiness gating."

Gates 2-6 are NOT run. Spec revision is required before continuing.


## H1 -- Regime distribution

### H1: Regime distribution parity

| Regime | v0 % | v1 % |
|---|---|---|
| STRONG_BULL | 27.33 | 20.64 |
| WEAK_BULL | 36.57 | 31.19 |
| SIDEWAYS | 18.18 | 13.64 |
| UNPREDICTABLE | 1.74 | 16.31 |
| BEAR | 16.19 | 16.14 |


## H2 -- BEAR vs G1_BEAR precision/recall

### H2: BEAR label vs G1_BEAR ground truth (precision/recall)

| Metric | v0 | v1 |
|---|---|---|
| Total G1_BEAR days | 371 | 371 |
| Total detector-BEAR days | 382 | 381 |
| Recall (BEAR | G1_BEAR) | 46.1% | 96.5% |
| Precision (G1_BEAR | BEAR) | 44.8% | 94.0% |


## H3 -- BEAR vs G3_vol_spike

### H3: BEAR co-occurrence with G3_vol_spike

| Metric | v0 | v1 |
|---|---|---|
| % of BEAR days with G3_vol_spike | 24.3% | 30.4% |


## H4 -- Run lengths and flicker

### H4: Run-length / flicker (transitions and persistence)

| Regime | v0 n_runs | v0 median | v0 max | v1 n_runs | v1 median | v1 max |
|---|---|---|---|---|---|---|
| STRONG_BULL | 104 | 4.0 | 46 | 94 | 3.0 | 46 |
| WEAK_BULL | 165 | 2.0 | 33 | 142 | 2.0 | 32 |
| SIDEWAYS | 144 | 2.0 | 21 | 127 | 2.0 | 21 |
| UNPREDICTABLE | 14 | 1.5 | 11 | 89 | 2.0 | 38 |
| BEAR | 63 | 2.0 | 53 | 31 | 2.0 | 204 |

Transition-matrix mean diagonal mass: v0=0.761, v1=0.815


## Per-onset lag detail

### v0 lag per G1_BEAR onset

| onset_date | lag_days | fired_within_window |
|---|---|---|
| 2018-02-08 | 4 | True |
| 2018-04-02 | 4 | True |
| 2018-10-29 | 0 | True |
| 2018-11-20 | 0 | True |
| 2018-11-23 | 0 | True |
| 2018-12-07 | 3 | True |
| 2018-12-14 | 0 | True |
| 2019-01-22 | 0 | True |
| 2019-01-28 | n/a | False |
| 2020-02-27 | 6 | True |
| 2020-03-03 | 1 | True |
| 2020-03-05 | 15 | True |
| 2020-06-11 | 0 | True |
| 2020-06-24 | 2 | True |
| 2020-06-26 | 0 | True |
| 2022-02-22 | 0 | True |
| 2022-03-07 | 0 | True |
| 2022-04-22 | 0 | True |
| 2023-02-03 | 40 | True |
| 2023-10-27 | 0 | True |
| 2025-03-13 | 0 | True |
| 2025-04-03 | 0 | True |

### v1 lag per G1_BEAR onset

| onset_date | lag_days | fired_within_window |
|---|---|---|
| 2018-02-08 | 43 | True |
| 2018-04-02 | 0 | True |
| 2018-10-29 | 0 | True |
| 2018-11-20 | 0 | True |
| 2018-11-23 | 13 | True |
| 2018-12-07 | 0 | True |
| 2018-12-14 | 0 | True |
| 2019-01-22 | 0 | True |
| 2019-01-28 | 0 | True |
| 2020-02-27 | 0 | True |
| 2020-03-03 | 0 | True |
| 2020-03-05 | 0 | True |
| 2020-06-11 | 0 | True |
| 2020-06-24 | 2 | True |
| 2020-06-26 | 0 | True |
| 2022-02-22 | 0 | True |
| 2022-03-07 | 0 | True |
| 2022-04-22 | 0 | True |
| 2023-02-03 | 0 | True |
| 2023-10-27 | 0 | True |
| 2025-03-13 | 0 | True |
| 2025-04-03 | 0 | True |
