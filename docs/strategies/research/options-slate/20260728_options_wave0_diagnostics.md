# Equity Options Slate -- Wave 0 Diagnostics (Groups A and B)

**Date:** 2026-07-28
**Orchestrator:** strategy-lead
**Status:** MEASUREMENT COMPLETE. **No strategy backtest has been run. No P&L has been observed. Zero trials consumed.**
**Governing spec:** `2026-07-25_options_slate_cc_handoff_spec_v2.md` **Section 5** (gates restated verbatim below).

---

## 0. Scope statement

Every number in this document is a property of the data -- an effect size, a frequency, an
episode count, a percentile, a distance distribution. No position was simulated, no fill was
generated, no equity curve or Sharpe ratio was computed. **These measurements therefore do NOT
increment the project-wide multiple-testing trial count `N`.** That is the entire point of
running them before Wave 1.

**No conclusion is drawn here about whether any strategy would be profitable.** That is Wave 1
and it is not authorized by this document.

Gates were pre-registered in Section 5 and are restated verbatim. **No threshold was adjusted
after seeing a measurement.** Where a registered gate was qualitative or underdetermined, the
operationalization was fixed and written down *before* measuring; each such case is flagged
explicitly below as a researcher degree of freedom.

---

## 1. Headline scoreboard

| ID | Gates | Verdict | Consequence |
|---|---|---|---|
| **D-014** | OPT-014 | **INCONCLUSIVE (leaning FAIL)** | Split by root; fragile. Do not clear |
| **D-042** | OPT-042 | **FAIL** | **DROP OPT-042** |
| **D-040a** | feeds D-040b | DESCRIPTIVE | delivered |
| **D-013/030** | OPT-013, OPT-030 | DESCRIPTIVE | valley-of-death falsifier is **live** |
| **D-050a** | OPT-050 | **PASS** | OPT-050 not dead pre-trial |
| **D-003** | OPT-003 | **PASS** | cleared |
| **D-005** | OPT-005 | **FAIL** | **DROP OPT-005** |
| **D-011** | OPT-011 | **BLOCKED** | universe not on disk |
| **D-018** | OPT-018 | DESCRIPTIVE | trap is a **tail** effect |
| **D-027** | OPT-027 | DESCRIPTIVE | reads as regime label |
| **D-029** | OPT-029 | **PASS (a fortiori)** | cleared, with decay flag |
| **D-033** | OPT-033 | **PASS** | Wave 3 eligible |
| **D-037** | OPT-037 | **FAIL** | **DROP OPT-037** |
| **D-040b** | OPT-040 | **FAIL** | **DROP OPT-040** |
| **D-043** | OPT-043 | **FAIL (wrong direction)** | **DROP OPT-043** |
| **D-047/030** | OPT-047, OPT-030 | **BLOCKED** | M7 does not exist |
| **D-048** | OPT-048 | **SPLIT** | SPY KEEP / QQQ DROP; threshold-fragile |
| **D-049** | OPT-049 | **SPLIT** | 22 PASS vs 12 FAIL by horizon |
| **D-050b** | OPT-050 | **KEEP** | agrees with D-050a |

**Net: 4 candidates killed outright (042, 005, 037, 040, 043 -- five), 2 blocked, 3 cleared,
2 split, 4 descriptive.** Wave 0 cost zero trials and removed five candidates from the queue.

---

## 2. Windows actually used (the binding constraints)

Three boundaries bind everything, and two of them were **discovered during this run**:

1. **Greeks boundary (known):** ThetaData serves no IV/greeks for ETF/index roots before
   **2017-01**. SPY/IWM/SPX all start 2017. **QQQ is the sole exception** (2012-2026).
2. **Underlying 1m bars start 2016** -- so anything needing realized vol on QQQ is 2016+,
   *not* 2012+. This is a **harder** boundary than the greeks one for QQQ.
3. **NEW -- undocumented SPY 2017 source-store gap.** SPY has **107 of 2,262 calendar sessions
   missing** from `options_combined`, ~90 of them in 2017 H1 (2017-01 holds 2 of 20 sessions;
   2017-06 holds 16 of 22). The partitions exist and their greeks are `OK` -- the sessions are
   simply absent. **SPY effectively has no usable 2017 H1.** This pushes SPY's first valid 1-year
   IV rank to **2018-06-04** and first valid 2-year percentile to **2019-06-05**, and it cost
   D-050b 62 of 167 trigger events.
4. **NEW -- QQQ 2015-11 is absent** from the source store entirely (a genuine hole, checked
   twice ~20 min apart). QQQ has 162 of 163 possible partitions, not the 168 assumed.

### `options_chain_eod` materialized for this run

Built as part of this task (it previously held a single proof-of-concept partition, SPY 2024-01):

| root | window | built | source-present | unbuildable |
|---|---|---|---|---|
| SPY | 2017-01 .. 2025-12 | **108 / 108** | 108 | 0 |
| QQQ | 2012-06 .. 2025-12 | **162 / 162** | 162 | QQQ 2012-01..05 (predate store), **QQQ 2015-11 (real gap)** |

**270 partitions, 100% of what is buildable.** Zero partitions were excluded by the greek census
(everything on disk for these two roots is `OK`) -- the boundary is enforced upstream by the
absence of pre-2017 SPY data.

---

## 3. Group A -- underlying data only

Window **2016-01-04 .. 2025-12-31, 2,514 sessions** (exact match to the market calendar).
Store: `sip_split` 1-minute (split-adjusted). Quarantined and counted, never imputed: SPY 1
session, QQQ 3 sessions.

**Registered before measuring:** bar timestamps are **bar-START** (established, not assumed:
first bar of each day is 04:00 ET, last is 19:59 ET, so pre/post-market bars are present);
session mark is the registered **15:45** snapshot, clamped to the last bar at/before the real
close on the 21 early closes (all 13:00 ET); decide on bar t, enter at bar t+1; 2 bp haircut
total per round trip; "post-2023" computed as **both** `>=2023` and `>=2024` with the gate
applied to each, chosen before seeing numbers.

### D-014 / D-042 -- gap-continuation and first-hour-trend effect size

> **Registered gate (verbatim):** "Proceed toward the options backtest (V1-V3 permitting) only if
> underlying effect >= **15 bps/trade** net of a 2 bp slippage haircut **in the post-2023 slice**.
> Else DROP both"

| Arm | Sym | Slice | n | ev/yr | gross bps | **net bps** | t | hit | boot 95% CI (net) | GATE |
|---|---|---|--:|--:|--:|--:|--:|--:|---|---|
| gap | SPY | 2016+ | 85 | 8.5 | -6.88 | **-8.88** | -0.51 | .518 | [-33.2, +18.9] | FAIL |
| gap | SPY | 2023+ | 27 | 9.0 | 27.74 | **25.74** | 0.88 | .630 | [-25.7, +94.2] | **PASS** |
| gap | SPY | 2024+ | 21 | 10.5 | 27.27 | **25.27** | 0.67 | .571 | [-40.6, +114.1] | **PASS** |
| gap | QQQ | 2016+ | 103 | 10.3 | -3.21 | **-5.21** | -0.29 | .544 | [-27.9, +15.9] | FAIL |
| gap | QQQ | 2023+ | 35 | 11.7 | -25.01 | **-27.01** | -1.04 | .486 | [-81.6, +13.0] | FAIL |
| gap | QQQ | 2024+ | 27 | 13.5 | -19.32 | **-21.32** | -0.66 | .519 | [-87.8, +24.6] | FAIL |
| 1st-hr | SPY | 2016+ | 695 | 69.5 | 1.77 | **-0.23** | 0.49 | .548 | [-7.4, +6.9] | FAIL |
| 1st-hr | SPY | 2023+ | 194 | 64.7 | 9.14 | **7.14** | 1.53 | .567 | [-3.6, +20.1] | FAIL |
| 1st-hr | SPY | 2024+ | 128 | 64.0 | 6.59 | **4.59** | 0.79 | .531 | [-10.2, +22.0] | FAIL |
| 1st-hr | QQQ | 2016+ | 1147 | 114.7 | 5.54 | **3.54** | 1.85 | .548 | [-2.3, +9.4] | FAIL |
| 1st-hr | QQQ | 2023+ | 358 | 119.3 | 9.47 | **7.47** | 1.85 | .570 | [-2.5, +17.6] | FAIL |
| 1st-hr | QQQ | 2024+ | 235 | 117.5 | 6.49 | **4.49** | 0.94 | .557 | [-8.7, +18.4] | FAIL |

**D-014 (gap) -- INCONCLUSIVE, leaning FAIL.** The gate is met by **SPY only**, under both
post-2023 definitions. **QQQ fails both.** No tie-break rule was invented; both are reported. The
SPY pass rests on n=27 (n=21 for 2024+), t=0.88, and a bootstrap CI spanning
[-25.7, +94.2] that comfortably contains zero. **Critically, it is not robust to definition:**
re-anchoring the gap arm entirely on the real close (prior-close reference *and* 15:59 exit)
**flips the sign of every gap cell** -- SPY 2023+ becomes **-24.36**, SPY 2024+ **-24.35**,
QQQ 2023+ **-42.12**. The full-sample effect is negative on both symbols. **Consequence: OPT-014
is NOT cleared.** A one-symbol, wide-CI, definition-fragile pass is not a pass.

**D-042 (first-hour) -- FAIL. DROP OPT-042.** Every symbol, both post-2023 definitions; best net
is 7.47 bps against a 15 bp bar, with CIs containing zero. Samples are large (358-695 events), so
this is a **well-measured near-zero, not a power problem.** The gate's own instruction is "Else
DROP both."

### D-040a -- weekend realized-variance share

> **Registered gate (verbatim):** "Feeds D-040b" -- descriptive, no pass/fail.

| Sym | Slice | weekend var | ctrl overnight var | mean intraday RV | wknd/(3x c2c) | **wknd:weekday per cal-day** |
|---|---|--:|--:|--:|--:|--:|
| SPY | 2016+ | 8.31e-5 | 4.67e-5 | 7.22e-5 | 0.230 | **0.562** |
| SPY | 2023+ | 6.05e-5 | 3.23e-5 | 5.55e-5 | 0.219 | **0.601** |
| SPY | 2024+ | 8.09e-5 | 3.71e-5 | 5.95e-5 | 0.257 | **0.703** |
| QQQ | 2016+ | 1.015e-4 | 6.70e-5 | 1.129e-4 | 0.185 | **0.483** |
| QQQ | 2023+ | 9.83e-5 | 5.43e-5 | 9.29e-5 | 0.206 | **0.580** |
| QQQ | 2024+ | 1.310e-4 | 6.09e-5 | 9.66e-5 | 0.250 | **0.693** |

**Headline delivered to D-040b:** a weekend carries only **~1.78x** a single weekday overnight's
variance despite spanning ~3.13 calendar days. Per calendar day, weekend variance is **48-70% of
a weekday's**, and that discount has **narrowed** in recent slices (0.56 -> 0.70 for SPY).
So the mechanism OPT-040 relies on is real in the underlying -- which makes D-040b's finding
below the decisive one.

### D-013/030 -- drawdown-shape census

> **Registered gate (verbatim):** "Descriptive; frames both candidates' valley-of-death falsifier"

245 regime-downgrade triggers; SPY; 60-session forward window; marks at 15:45.
Registered before measuring: "meaningful" DD = **> 3%**.

| Subset | n | GAP | GRIND | mean gap share | mean maxDD | median maxDD | DD > 3% |
|---|--:|--:|--:|--:|--:|--:|--:|
| All triggers | 245 | 79 | **166** | 0.4284 | 7.46% | 5.66% | 187 |
| Non-overlapping (greedy 60-sess) | 35 | 9 | **26** | 0.4301 | 6.87% | 5.00% | 25 |

Named episodes: **2018-Q4 GRIND** (16.5% DD, gap share 0.323); **2020-Q1 GAP** (34.6% DD, 0.680);
**2022 GRIND** (20.7% DD, 0.420 and 16.8%, 0.458); **2025 GRIND** (19.8% DD, 0.467).

**Consequence: the valley-of-death falsifier for OPT-013 and OPT-030 is LIVE.** GRIND dominates
GAP 2:1, the median episode puts only 43% of its decline in overnight gaps, and **2020-Q1 is the
only cited era that is gap-dominated.** The gap share is also falling over time (mean 0.29-0.34
across 2023-2025 vs 0.44-0.54 in 2016-2021). Both candidates depend on declines being gappy;
they mostly are not. 58 of 245 triggers (24%) were followed by no decline exceeding 3%.

### D-050a -- realized-vol timing around classifier transitions

> **Registered gate (verbatim):** "If RV peaks **before** transitions, the classifier lags and
> OPT-050 is dead pre-trial"

| RV source | Trigger set | n | **peak offset** | post/pre mean | Wilcoxon p | paired-t(log) p |
|---|---|--:|--:|--:|--:|--:|
| SPY intraday RV | all transitions | 490 | **+10** | 1.412 | 0.859 | 0.187 |
| SPY intraday RV | into UNPREDICTABLE | 14 | **+1** | 4.720 | 0.0245 | 0.0090 |
| SPY intraday RV | out of STRONG_BULL | 101 | **+10** | 1.643 | 0.0257 | 0.0024 |
| SPY intraday RV | **OPT-050 pooled** | 110 | **+10** | **1.927** | **0.0068** | **0.0005** |
| detector `realized_vol` | OPT-050 pooled | 110 | +10 | 1.157 | 0.0003 | 0.0000 |

Event-study shape, OPT-050 pooled (normalized): t-10 **0.71**, t-5 0.82, t-1 1.16,
**t=0 1.72**, t+1 2.42, t+10 2.99.

**PASS. The classifier does NOT lag. OPT-050 is not dead pre-trial.** RV peaks at a **positive**
offset in every set and both RV sources, never negative; pre-transition RV sits *below* its
trailing mean. Two honest caveats: (1) the peak sits at **+10, the edge of the measurement
window**, so the true peak is at or beyond +10 -- more favorable to the gate, but unmeasured;
(2) the regime series is a **causal replay on today's SPY/VIX vintage**, not an archived
point-in-time log. A PASS means "the classifier is not provably late", not "OPT-050 has edge."

---

## 4. Group B -- on the derived chain tables

### D-003 -- financing ratio -- **PASS**

> **Registered gate (verbatim):** "Proceed only if ratio >= **0.7 in >= 60% of months**; else DROP 003"

Window **SPY 2017-01 .. 2025-12** (100 of 108 months measured). The registered "full owned window"
is **NOT achievable for SPY** -- the greeks boundary makes 2017 the first usable year.

| | SPY | QQQ (different underlying, not a substitute) |
|---|---|---|
| months measured | 100 / 108 | 154 / 162 |
| dropped: missing leg / delta tolerance / date substituted | 1 / 7 / 8 | 4 / 4 / 4 |
| **fraction of months ratio >= 0.7** | **0.990** | **1.000** |
| **gate (>= 60%)** | **PASS** | **PASS** |
| mean / median | 1.271 / 1.311 | 1.416 / 1.420 |

Per-year `frac>=0.7` is 1.00 in every SPY year except 2017 (0.833, n=6). By regime: BEAR 1.00
(n=20), SIDEWAYS 1.00, STRONG_BULL 1.00, WEAK_BULL 0.971. **Consequence: OPT-003 cleared** --
short-dated call richness comfortably finances the longer-dated put, with margin (median 1.31,
not 0.7).

### D-005 -- post-pullback vs unconditional put VRP -- **FAIL**

> **Registered gate (verbatim):** "Proceed only if conditional > unconditional; else DROP 005"

Windows SPY 2017-2025 (2,105 sessions), QQQ **2016**-2025 (2,505 -- bars, not greeks, bind).
Registered primary trigger: `close(t) <= 0.98*close(t-1)`.

| | SPY primary | SPY secondary | QQQ primary | QQQ secondary |
|---|---|---|---|---|
| n conditional / all | 81 / 2105 | 668 / 2105 | 159 / 2505 | 902 / 2505 |
| **mean cond / mean all** | **0.0229 / 0.0290** | 0.0338 / 0.0290 | **0.0362 / 0.0377** | 0.0438 / 0.0377 |
| diff of means | **-0.0061** | +0.0048 | **-0.0015** | +0.0061 |
| iid bootstrap 95% CI | [-0.046, +0.031] | [-0.001, +0.011] | [-0.022, +0.017] | [+0.001, +0.011] |
| **block bootstrap (block=DTE)** | **[-0.073, +0.047]** | [-0.009, +0.018] | **[-0.031, +0.023]** | [-0.005, +0.017] |
| **gate on means** | **FAIL** | pass | **FAIL** | pass |

**FAIL on the pre-registered primary trigger. DROP OPT-005.** The mean-based reading is negative
on both roots. Every **block**-bootstrap CI straddles zero -- forward windows of adjacent sessions
overlap by ~27 of 28 days, so the iid CI and Welch t are badly anticonservative, and the one
"significant" cell (QQQ secondary, Welch p=0.0087) **loses significance under the block
bootstrap.** Reaching a pass would require switching to the secondary trigger AND the iid
statistic -- two degrees of freedom, which is exactly the p-hacking move the methodology forbids.

### D-011 -- entry-day put-IV percentile census -- **BLOCKED**

> **Registered gate (verbatim):** "Descriptive; if entries systematically land at IV pctile > 80,
> expect the drift edge to be consumed"

**CANNOT RUN AS SPECIFIED.** OPT-011's registered universe is *"bottom-decile momentum S&P 500
names, max 10"*. Those single-name chains are **not on disk** (31 roots only); the ThetaData
top-up route is dead (subscription cancelled, logged 2026-07-27) and ORATS is deferred.
**No proxy was substituted for the gate.**

A **scope-reduced supplementary** index-level analogue was measured and is explicitly **not** the
D-011 census: at SPY/QQQ BEAR-regime 20-day-low breakdowns, put IV lands at mean percentile
**0.79-0.87** vs a ~0.51 unconditional baseline, with 59-78% of triggers above 0.80. Suggestive of
the registered concern, but **single-name idiosyncratic put IV at momentum breakdowns can behave
very differently from index put IV, and nothing here bounds that.**

### D-018 -- IV-rank state-transition matrix -- DESCRIPTIVE

> **Registered gate (verbatim):** "Descriptive; frames the conditioning trap"

Windows SPY 2018-06-04..2025-12-26 (n=1,898), QQQ 2013-06-06..2025-12-26 (n=3,136).

| | SPY | QQQ |
|---|---|---|
| unconditional P(IVR>50) | 0.1596 | 0.1738 |
| P(IVR>50 at t+21 given IVR>50 at t) | 0.5050 | 0.4147 |
| **lift** | **3.16x** | **2.39x** |
| mean / max run (sessions) | 7.97 / 66 | 6.06 / 71 |
| **mean dRV ratio yz20(t+21)/yz20(t), conditional** | **1.166** | **1.142** |
| same, unconditional | 1.108 | 1.076 |
| **median** ratio, conditional / unconditional | 0.949 / 0.979 | 0.933 / 0.970 |
| P(RV rising), conditional / unconditional | 0.442 / 0.470 | 0.428 / 0.455 |
| conditional sigma of dRV / unconditional (SPY) | 0.187 / 0.110 | -- |

**The conditioning trap has a specific shape, and it is a TAIL effect, not a mean effect.** After
IVR>50 the *median* path is falling realized vol (0.95 vs 0.98 unconditional) and the probability
of rising vol barely improves (0.442 vs 0.470). What the conditioner actually buys is a
**materially fatter right tail** -- conditional sigma 0.187 vs 0.110, p90 +0.128 vs +0.078, mean
ratio 1.166 vs 1.108. **Consequence for OPT-018:** selling vol on IVR>50 is not selling into a
calmer regime; it is selling into a regime whose typical outcome is slightly calmer and whose
tail outcome is considerably worse. The pre-registered 018-vs-015 ablation should be read with
tail statistics, not Sharpe alone.

### D-027 -- skew-percentile persistence + conditional crash frequency -- DESCRIPTIVE

> **Registered gate (verbatim):** "Descriptive"

Windows SPY 2019-06-13..2025-12-26 (n=1,623), QQQ 2014-12-08..2025-12-26 (n=2,690).

| | SPY | QQQ |
|---|---|---|
| unconditional P(above 70th pctile) | 0.2945 | 0.3335 |
| P(above at t+21 given above at t) | 0.5814 | 0.5856 |
| lift / mean run / max run | 1.97x / 6.46 / 132 | 1.76x / 7.60 / 122 |

Forward-45-session drawdown conditional on steep skew vs unconditional:

| threshold | SPY steep | SPY uncond | QQQ steep | QQQ uncond |
|---|--:|--:|--:|--:|
| <= -5% | 0.4561 | 0.4716 | 0.6221 | 0.5827 |
| <= -10% | 0.1423 | 0.1480 | **0.3233** | 0.2555 |
| <= -20% | 0.0377 | 0.0271 | 0.0346 | 0.0376 |

**Reads as a regime LABEL, not a mispricing -- and the two roots disagree in a way that must not
be averaged.** On SPY steep skew carries essentially **zero** information about forward drawdown
at -5% and -10% (conditional is marginally *below* unconditional); only a thin excess at -20%
(18 events vs 44). On QQQ steep skew is followed by materially **more** moderate damage (-10%:
1.27x lift). High persistence (1.8-2.0x at 21 sessions) is the signature of a slow-moving regime
variable. **Consequence for OPT-027:** the conditioning trap named in its own falsifier is
visible in the data. A QQQ put-spread seller conditioning on steep skew is measurably selling
into a higher -10% frequency.

### D-029 -- jade-lizard starvation census -- **PASS (a fortiori)**

> **Registered gate (verbatim):** "Proceed only if >= **6 qualifying entries/yr** average; else DROP 029"

| | SPY | QQQ |
|---|---|---|
| sessions measured | 1,681 | 2,344 |
| no complete 3-leg selection / delta tol / degenerate width | 452 / 22 / 20 | 893 / 153 / 91 |
| **raw qualifying fraction** | **0.357** | 0.250 |
| non-overlapping entries (45d skip) | 58 | 80 |
| **entries/yr** | **6.44** | 5.71 |
| entries/yr excluding partial first year | 6.375 | 6.00 |
| **gate (>= 6/yr)** | **PASS** | pass ex-partial |

**PASS.** OPT-029's registered universe is "10 megacaps + SPY"; only SPY/QQQ are materialized.
**SPY alone clears 6/yr, so the gate passes a fortiori** -- adding megacaps can only add entries.
The 0.15-delta call sits above the 0.10 floor, so P1's smoothed-surface rule does not bind.

**Decay flag (not part of the gate, but do not read this as a green light without it):** the raw
qualifying fraction collapses post-2021 -- SPY 0.66 (2021) -> 0.28, 0.16, 0.14, 0.28 (2022-2025);
QQQ 0.14 -> 0.036-0.078. The entry *count* is held up only because the non-overlapping cadence
leaves ~8 slots/yr. SPY entries by year: 7, 7, 7, 7, 8, 4, 6, 6, 6.

### D-033 -- backwardation episode count -- **PASS**

> **Registered gate (verbatim):** ">= 12 episodes -> Wave 3 backtest; < 12 -> route to forward
> paper validation, do not spend the historical sample"

Registered before counting: episode starts on first `slope < 0.97`, ends on last session before
`slope >= 0.97`; **episodes separated by < 30 sessions are MERGED** (the candidate holds 60 DTE).
Sensitivities at raw-no-merge and 60-session merge. **ORATS deferred**, so the window is the
owned window.

| root | window | sessions backwardated | raw | **merge-30 (PRIMARY)** | merge-60 | **gate** |
|---|---|--:|--:|--:|--:|---|
| **QQQ** (gate read) | 2012-06 .. 2025-12 | 640 / 3,386 (18.9%) | 147 | **36** | 10 | **PASS** |
| SPY | 2017-01 .. 2025-12 | 587 / 2,150 (27.3%) | 100 | **20** | 4 | **PASS** |

Gate read on **QQQ** because it is the only root with full-history usable greeks, which is the
window the gate text names ("2012-2026"). SPY's 20 is a strict **undercount** -- bounded to 2017+
by the greeks boundary and further eroded by the 2017 H1 source gap. Both roots clear the 12 bar;
QQQ passes under all three merge rules (147/36/10). The only failing cell is SPY at 60-session
merge (4), which is **not** the registered rule. **Consequence: OPT-033 is Wave 3 eligible** --
the sample-starvation falsifier does not fire.

### D-037 -- broken-wing-fly starvation census -- **FAIL**

> **Registered gate (verbatim):** "Same 6/yr rule as D-029" (>= 6 qualifying entries/yr)

| | SPY | QQQ (supplementary) |
|---|---|---|
| sessions measured | 1,957 | 2,162 |
| degenerate strikes quarantined | 191 | 324 |
| **raw qualifying fraction (net cost <= 0)** | **0.0593** | 0.0444 |
| non-overlapping entries (38d skip) | 28 | 33 |
| **entries/yr** | **3.11** | 3.30 |
| **gate (>= 6/yr)** | **FAIL** | FAIL |
| net entry cost mean / median | +$0.831 / **+$0.795** | +$0.847 / +$0.650 |

**FAIL. DROP OPT-037.** Entries by year (SPY 2017-2025): 3, 4, 4, 3, 3, 2, 3, 3, 3 -- **no single
year reaches 6.** The structure is a debit in ~94% of sessions; the no-cost constraint starves it
by roughly a factor of two on the gate.

**Two disclosed caveats.** (1) **OUT OF SPEC:** the 2.5-sigma far wing has median |delta| 0.0467,
and 93.5% of far-wing selections sit **below the 0.10 delta floor** where P1 mandates a smoothed
surface -- **M7 does not exist.** Mitigating: this is a *price-availability* census, not an IV
question, and the far wing has a valid two-sided quote in **100.0%** of sessions (median
`spread_rel` 0.0206), so the marks are real quotes. The census should be re-run once M7 lands.
(2) The registered geometry was **underdetermined** (two sigma levels named for three strikes);
the body offset is an operationalization and a genuine researcher degree of freedom. A 13x
shortfall in qualifying rate is unlikely to be a geometry artifact, but this FAIL is conditional
on that reading.

### D-040b -- Friday vs Thursday term-adjusted ATM IV discount -- **FAIL**

> **Registered gate (verbatim):** "Proceed only if measured Friday discount < **50%** of
> calendar-day theta differential; else DROP 040 (expected)"

Registered before measuring: pairs are consecutive Thu/Fri sessions on the **same listed expiry**
(under the calendar-uniform null, quoted annualized IV is invariant to the 1-day elapse, so the
raw same-expiry difference **is** the term-adjusted difference); expiry chosen with Friday DTE in
[7,10]; both a theoretical and an empirical (D-040a-based) denominator computed and **both
reported, neither chosen after seeing which passes.**

| | n pairs | mean Fri-Thu | t | frac Fri<Thu | **ratio (a) theoretical** | ratio (b) empirical |
|---|--:|--:|--:|--:|--:|--:|
| **SPY core 7-10 (PRIMARY)** | 376 | **-0.006152** | -4.054 | 0.713 | **1.416** | 1.735 |
| SPY long-weekend subset | 42 | -0.009865 | -4.570 | 0.738 | 1.559 | 1.833 |
| SPY core 12-16 | 377 | -0.003702 | -2.960 | 0.684 | 1.560 | 1.909 |
| QQQ core 7-10 | 580 | -0.007481 | -6.465 | 0.716 | 1.488 | 1.818 |
| QQQ long-weekend | 66 | -0.011712 | -4.832 | 0.788 | 1.501 | 1.764 |
| QQQ core 12-16 | 579 | -0.004001 | -4.301 | 0.686 | 1.434 | 1.751 |

**FAIL. DROP OPT-040.** The measured Friday discount is **141.6%** of the theoretical calendar-day
theta differential against a **< 50%** bar, and 173.5% of the empirical D-040a benchmark. **Every
cell -- both roots, both DTE ranges, core and long-weekend, both denominators -- lands between
1.43x and 2.45x, i.e. 3x to 5x above the gate.** This is not a near-miss and does not depend on
any chosen degree of freedom. The market **over-discounts** Friday relative to a pure trading-time
model: Friday IV is below Thursday's on 71% of pairs by a mean 0.62 vol points (t = -4.05). The
weekend theta edge OPT-040 seeks **has already been taken out of the quote.** D-040a confirmed the
underlying mechanism is real; D-040b shows it is already priced. Registered expectation was DROP;
the measurement independently produces DROP.

*Disclosed:* at **matched constant maturity** this gate is **degenerate** -- a Thursday 7-calendar-day
window and a Friday 7-calendar-day window both span exactly 5 sessions, so the theoretical
differential is exactly zero and the ratio has a zero denominator. The same-expiry framing was
registered instead, with the reasoning recorded rather than a denominator quietly chosen.

### D-043 -- pin distance, OpEx week vs control -- **FAIL (wrong direction)**

> **Registered gate (verbatim):** "Proceed only if OpEx-week distance is materially smaller;
> else DROP 043"

**T-1-lagged OI only**, restitched across month seams (the within-file `oi_eod_lag1` nulls the
first session of every month). SPY: 6,801,800 contract-session rows, 1,690,843 (24.9%) still NULL
after stitching (never forward-filled), **230,958 rows recovered by the cross-month stitch**.
Registered operationalization of "materially smaller": OpEx median at least **20% below** control
**and** Mann-Whitney p < 0.05. **PRIMARY metric = vol-normalized** distance (raw dollars are
confounded by SPY's price tripling over the window).

| group | n opex / control | median opex | median control | ratio | MW p | gate |
|---|--:|--:|--:|--:|--:|---|
| SPY all | 449 / 1462 | **3.558** | **1.413** | **2.52** | <1e-4 | **FAIL** |
| SPY **Wednesday** (entry day) | 92 / 300 | 4.255 | 1.397 | 3.05 | <1e-4 | FAIL |
| SPY Friday (exit day) | 88 / 298 | 2.483 | 1.621 | 1.53 | 0.023 | FAIL |
| SPY pre-2021 | 162 / 518 | 3.277 | 1.079 | 3.04 | <1e-4 | FAIL |
| SPY post-2020 | 287 / 944 | 3.785 | 1.780 | 2.13 | <1e-4 | FAIL |
| QQQ all | 494 / 1626 | 2.396 | 0.895 | 2.68 | <1e-4 | FAIL |

**FAIL. DROP OPT-043.** The effect is significant and **in the wrong direction**: OpEx-week pin
distance is **2.5-3x LARGER** in vol units, not smaller. Secondary readings agree: spot-fraction
medians 0.0434 (opex) vs 0.0530 (control) is 18.2% below -- short of the 20% pre-commitment -- and
on **Wednesday, the day the candidate actually enters, there is no difference at all** (0.0534 vs
0.0537). No post-2020 decay story either; the wrong-direction effect is present in both halves.

**Structural confound, disclosed:** the comparison as registered is **DTE-confounded and cannot be
de-confounded.** OpEx-week DTE mean is 1.97 (max 4); control is 17.25 (min 6). A DTE-matched
control at DTE<=4 has **n=0 by construction**, because control weeks reference the *next* monthly.
Both the dollar and sigma readings are therefore partly artifacts of the expiry rule. **The
registered diagnostic has no design that avoids this.** The verdict stands (nothing supports the
pinning claim, and the entry-day reading is a flat null), but it is a weaker instrument than
Section 5 assumed.

### D-047/030 -- deep-OTM integrity -- **BLOCKED**

> **Registered gate (verbatim):** "Integrity gate -- if the surface diverges materially at 0.05D,
> both candidates need a different mark source"

**NOT RUN.** The registered diagnostic compares a **smoothed surface** against raw quotes.
Module **M7 (`iv_smooth`) does not exist anywhere in this repo** -- verified by direct search --
and the ORATS purchase that would supply the alternative was **DEFERRED by logged decision
2026-07-27**. There is nothing to cross-check against. **No proxy was substituted.**

**Consequence: OPT-047 and OPT-030 cannot be cleared, and this blocks Wave 1**, since OPT-047 is
a Wave-1 candidate and P1's low-delta rule binds at its 0.05-delta strikes. **M7 is on the Wave-1
critical path.**

A **supplementary raw-quote census** (explicitly **not** the gate; an input for when M7 lands):
at ~0.05 delta, DTE 21-45, SPY 2017-2025 / QQQ 2012-2025 -- **valid two-sided quote in 100.0%** of
sessions on both roots and both rights; median `spread_rel` SPY calls 0.0500 / puts 0.0230, QQQ
calls 0.1053 / puts 0.0513; **zero-bid frequency 0.0%**; static monotonicity violations
0.064-0.131% of pairs. Deep-OTM raw *availability* is excellent. The concern that motivates M7 is
IV/greek *reliability*, which this census cannot address.

### D-048 -- IV percentile in compressed-RV states -- **SPLIT**

> **Registered gate (verbatim):** "If IV sits at its floor, sellers are not extrapolating ->
> DROP 048 pre-trial"

Registered **before measuring** (the gate text is qualitative): IV is "at its floor" iff mean
`iv_pctile_2y` of 30-DTE ATM IV across compressed sessions **< 0.20** (the same 20th-percentile
bar OPT-048 applies to RV); the thesis survives if mean IV percentile exceeds mean RV percentile
by **> 0.10**. Windows SPY/QQQ 2018-01-03 .. 2026-07-27.

| | SPY | QQQ |
|---|---|---|
| n compressed sessions / episodes | 100 (4.65%) / **29** | 97 (4.51%) / **22** |
| **mean `iv_pctile_2y` (30 DTE)** | **0.2337** | **0.1936** |
| median | 0.1548 | 0.1690 |
| frac below 0.10 / below 0.20 | 0.308 / 0.560 | 0.378 / 0.622 |
| unconditional mean | 0.5187 | 0.5139 |
| mean RV pctile in compressed | 0.1051 | 0.1044 |
| **IV pctile - RV pctile** | **+0.1286** | **+0.0892** |
| **verdict** | **KEEP** | **DROP** |

**SPLIT, reported rather than resolved by picking a side.** SPY clears the pre-registered 0.20 bar
by 0.034 and the +0.10 gap test; QQQ misses both, by 0.006 and 0.011 respectively. **This is a
genuine straddle of a threshold fixed in advance and the call is threshold-fragile on QQQ and only
modestly robust on SPY.** Both roots share the same qualitative shape: in compressed states IV sits
at ~0.19-0.23 of its own 2-year range while RV sits at ~0.10 -- IV is compressed but **less**
compressed than RV, by 9-13 percentile points. Event count is thin (29 / 22 episodes).
**Consequence: OPT-048 is not killed, but it is not cleanly cleared either.**

### D-049 -- HAR-RV forecast > IV_30 episode count -- **SPLIT**

> **Registered gate (verbatim):** ">= 15 episodes -> Wave 2; fewer -> forward paper route"

Frozen (1,5,22) `har_forecast` reused unmodified. **Registered horizon-mismatch caveat:** HAR
forecasts *next-day* RV while IV_30 prices 30 calendar days; **both** readings computed and
reported. Aligned window first usable date **2017-01-03** (1m bars 2016 + 252-session warmup +
SPY greeks boundary). 2,155 sessions.

| | (i) next-day HAR | (ii) horizon-matched trailing-22 HAR |
|---|---|---|
| frac sessions forecast > IV_30 | 0.0484 | 0.0596 |
| raw / **merge-30** / merge-60 episodes | 65 / **22** / 13 | 36 / **12** / 8 |
| **gate** | **PASS (22 >= 15)** | **FAIL (12 < 15)** |
| **corr(gap, VIX level)** | **-0.165** | **-0.552** |

**SPLIT, reported honestly rather than resolved by taking the passing read.** The gate as literally
written ("HAR-RV forecast > IV_30") is the next-day comparison and gives **22**; the read that
fixes the acknowledged horizon mismatch -- which is the economically correct one, since IV_30 is a
30-day price -- gives **12** and **fails**. Neither read passes at the 60-session merge (13 / 8).

**The registered spurious-path check (spec P10) materially weakens the mechanism regardless of
which count is read:** correlation of (forecast - IV_30) with the VIX level is **-0.552** on the
horizon-matched read. The "HAR beats IV" signal fires **disproportionately when VIX is LOW** -- it
is substantially a low-VIX artifact, not an independent vol-mispricing signal.

### D-050b -- entry-day IV percentile at classifier transitions -- **KEEP**

> **Registered gate (verbatim):** "If mean pctile > **80**, classifier lags the vol -> DROP 050"

167 trigger events (into UNPREDICTABLE + out of STRONG_BULL) across 2012-06..2026-07.

| | SPY (2017-01..2025-12) | QQQ (2012-06..2025-12) |
|---|---|---|
| trigger events **lost to chain coverage** | **62 of 167** | 14 of 167 |
| t0-only mean `iv_pctile_2y` | 0.5515 | 0.5421 |
| **t0..t0+5 mean `iv_pctile_2y`** | **0.5209 -> 52.09/100** | **0.5238 -> 52.38/100** |
| ALL transitions, t0..t0+5 mean | 0.5382 | 0.5397 |
| **unconditional baseline** | **0.5187** | 0.5211 |
| **GATE (>80 -> DROP)** | **52.09 -> KEEP** | **52.38 -> KEEP** |

**KEEP. The gate is not remotely close to tripping** (52 vs the 80 bar), on both roots and both
readings. SPY loses 62 of 167 events to the coverage boundary; QQQ, with full 2012+ greeks, loses
only 14 and gives an **essentially identical answer** -- the pre-2017 events do not change the
picture. (QQQ is a different underlying, reported as corroboration, not a substitute.)

**Reconciliation with D-050a -- the two diagnostics AGREE.** D-050a: realized vol around these
same transitions peaks at a *positive* offset (post/pre 1.93, Wilcoxon p=0.0068) -- the classifier
does not lag the RV. D-050b: on entry days IV sits at the **52nd percentile of its own 2-year
range, statistically indistinguishable from the 51.9 unconditional baseline.** So realized vol
rises sharply *after* the trigger while implied vol at the trigger is priced at an unremarkable
middle-of-range level. **Neither diagnostic finds the classifier lagging, and D-050b specifically
finds that IV has NOT already repriced by entry.** That is the coherent, mutually supporting, and
favorable result for OPT-050 on this axis.

---

## 5. Consequences by candidate

| Candidate | Diagnostic | Outcome | Action |
|---|---|---|---|
| **OPT-005** | D-005 | conditional put VRP <= unconditional on the pre-registered trigger | **DROP** |
| **OPT-037** | D-037 | 3.11 entries/yr vs 6/yr bar; no year reaches 6 | **DROP** |
| **OPT-040** | D-040b | Friday discount 142% of theta differential vs <50% bar | **DROP** |
| **OPT-042** | D-042 | 7.47 bps vs 15 bps bar, large n, CI contains zero | **DROP** |
| **OPT-043** | D-043 | OpEx pin distance 2.5-3x LARGER, not smaller; flat on entry day | **DROP** |
| OPT-014 | D-014 | SPY passes, QQQ fails, sign flips on definition change | **NOT CLEARED** -- inconclusive |
| OPT-003 | D-003 | ratio >= 0.7 in 99.0% of months | **CLEARED** |
| OPT-029 | D-029 | 6.44 entries/yr on SPY alone (a fortiori) | **CLEARED** (decay flag) |
| OPT-033 | D-033 | 36 episodes (QQQ), 20 (SPY) vs 12 bar | **CLEARED** -- Wave 3 eligible |
| OPT-050 | D-050a + D-050b | classifier does not lag; IV not pre-repriced | **CLEARED** on both axes |
| OPT-048 | D-048 | SPY 0.2337 KEEP / QQQ 0.1936 DROP | **SPLIT** -- threshold-fragile |
| OPT-049 | D-049 | 22 PASS (next-day) / 12 FAIL (horizon-matched); VIX corr -0.55 | **SPLIT** -- weakened |
| OPT-011 | D-011 | single-name universe not on disk | **BLOCKED** |
| **OPT-047, OPT-030** | D-047/030 | M7 smoothed surface does not exist | **BLOCKED** -- Wave 1 impact |
| OPT-018 | D-018 | conditioning trap is a tail effect | proceed; report tail stats |
| OPT-027 | D-027 | steep skew reads as regime label | proceed with the trap in view |
| OPT-013, OPT-030 | D-013/030 | GRIND dominates GAP 2:1 | falsifier **live** |

---

## 6. Apparatus corrections made during this run (NOT specification changes)

Per methodology, fixing a bug or mis-specification in the apparatus and re-running the same
pre-registered measurement is an **apparatus correction, not a trial**.

1. **`canonicalize_frame` rejected the `date32` `expiration` variant.** 100 of 4,510 partitions
   (including **SPY 2017-2018**, exactly the window the slate is bounded to) ship `expiration` as
   date32 rather than string; the canonical layer assumed String and raised `SchemaError`,
   blocking materialization. Fixed with a regression test. Commit `ebb5d06`.
2. **`RunStatus` destination paths collided across parallel workers.** The path was
   `<name>_<YYYYmmdd_HHMMSS>.json` -- second resolution only -- so workers launched in the same
   second shared one destination and their atomic `replace()` calls fought (WinError 5). **This
   killed 9 of 276 parallel build jobs.** The 2026-07-27 fix had made only the *tmp* path unique
   and left the destination shared. Fixed with a regression test. Commit `a54a281`.
3. **A strict all-finite trailing window is not neutral** (B1, disclosed): requiring all 252/504
   values finite let 5 unbracketed SPY sessions void ~1,260 downstream sessions. Replaced with a
   registered **95% coverage floor** -- missing sessions are simply absent from the comparison set
   and are counted; **this is not imputation.** Recovered 519 SPY sessions in D-018 alone.
   **Disclosed because it was made after a first set of numbers had been seen.**
4. **Pre-2015 monthly expiries are dated SATURDAY**, not Friday (OCC change, Feb 2015). A
   Friday-only rule silently discarded all QQQ 2012-2013 term-slope rows (364 sessions). Both
   conventions now accepted and unit-tested; QQQ term-slope rows 2,980 -> 3,386.

---

## 7. Divergences from the brief (reality wins)

1. **SPY 2017 H1 is effectively absent** from the source store (107 missing sessions) -- not a
   greeks-boundary effect and not previously documented. Delays SPY's first valid 2y IV percentile
   to 2019-06-05.
2. **QQQ 2015-11 partition does not exist**; QQQ has 162 of 163 possible partitions, not 168.
3. **Underlying 1m bars (2016+), not options greeks, are the binding boundary for QQQ** on any
   diagnostic needing realized vol (D-005, D-037, D-043).
4. **Mixed float dtypes across `options_chain_eod` partitions** (Float32 in some, Float64 in
   others) break a naive multi-month `pl.concat`. Worked around with a widening cast (no value
   altered). **Should be normalized in the canonical writer** -- open follow-up.
5. **`quote_valid == False` rows: zero** across all 270 partitions -- the canonical builder drops
   them upstream, so the M2 valid-quote filter is a no-op on this store (still implemented/tested).
6. **D-040b's gate is degenerate at matched constant maturity** (zero denominator); the same-expiry
   framing was registered instead, with reasoning recorded.
7. **D-043 cannot be DTE-matched** (n=0 in the matched cell) -- the registered design is confounded.
8. **D-037's geometry was underdetermined** by the registered text; the body offset is an
   operationalization and a genuine researcher degree of freedom.
9. `scripts/backtest_scripts/` is gitignored; drivers were force-added so the deliverable is
   actually committed.

---

## 8. Open follow-ups

- **M7 (smoothed IV surface) is on the Wave-1 critical path.** It blocks D-047/030 outright, and
  P1's low-delta rule binds at OPT-047's 0.05-delta strikes and OPT-037's 2.5-sigma wing. Decision
  logged 2026-07-27 was **BUILD** (ORATS deferred).
- **D-037 must be re-run once M7 lands** (93.5% of far-wing selections are below the 0.10 floor).
- **Normalize float dtypes** in the `options_chain_eod` writer.
- **Honest lifetime `N` remains unreconstructed** (execution plan Section 3.2: the
  `combinations_project` counter in `output/experiments.duckdb` is empty; directionally N is in the
  low hundreds, not 9). **This blocks Wave 1 grading, not building.** No trials were spent here, so
  Wave 0 does not change N.
- **Megacap EOD chains are unbuilt.** D-011 is blocked and D-029/D-043 ran on a narrowed universe.

---

*No strategy backtest has been run. No P&L has been observed. No trials were consumed.*
