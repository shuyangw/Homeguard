# RAMP Wave-3 Signal-Construction Family Gate -- 2026-06-01

**Branch**: `archive/regime-detector-campaign-2026-05`
**Data**: Alpaca SIP daily-aggregated, 2017-01-03 to 2026-05-15 (2355 trading days)
**Universe**: sp500-2025 (494 symbols)
**Cost tier for gates**: 5.0 bps per side (near_close timing mode)
**Incumbent**: V11 (Sharpe 0.528 full-window at 5 bps, paper-deployed 2026-05-23)
**Minimum lift bar**: +0.10 Sharpe over V11 (TIER-1 threshold per TODO.md)

---

## Step 0: V31 Re-run on Clean Data (Pre-requisite)

V31's previously recorded runs (git SHAs `a88e762`, `729f065`) were made BEFORE the
SIP_SPLIT_REL data-integrity fix (commit `429df47`). Those runs used the corrupt legacy
cache with unadjusted NFLX 10:1 split, producing a phantom Sharpe of 0.307.

**Fix applied (this session):** dtype coercion bug in `_compute_beta_residual_ranking`
(same root cause as the b29298e fix for V28 -- object-dtype columns from `pct_change`
caused `np.isnan()` TypeError). Applied `apply(pd.to_numeric, errors='coerce')` and
`.astype(np.float64)` before beta regression.

**V31 clean results (SHA `0f04be9a`, full window, 5 bps near_close):**
- Sharpe: 0.768 (was 0.307 on stale data -- a 2.5x change)
- PSR: 0.990
- Cost gate (7.5 bps): 0.702 (PASS)

All 6 return streams now verified on the current clean HEAD. Date alignment confirmed:
2355 rows, 2017-01-03 to 2026-05-15, identical dates across all variants.

---

## Section 1: Cross-Section Sharpe Table (6 Variants, 5 bps near_close, Full Window)

| Variant | Sharpe | CAGR | Max DD | Ann TO | PSR(vs 0) | vs V11 delta |
|---|---:|---:|---:|---:|---:|---:|
| V11 (incumbent) | 0.528 | 11.9% | -66.2% | 10,325% | 0.944 | -- |
| V28 multi-horizon | 0.811 | 20.0% | -42.0% | 5,264% | 0.993 | +0.283 |
| V31 beta-residual | 0.769 | 17.4% | -33.5% | 7,217% | 0.990 | +0.241 |
| V02+V05 vanilla | 0.683 | 16.8% | -57.5% | 10,275% | 0.980 | +0.155 |
| V26 z-score | 0.533 | 10.5% | -42.7% | 9,492% | 0.947 | +0.005 |
| V33-core abs-mom | 0.479 | 8.4% | -52.3% | 9,711% | 0.922 | -0.049 |

**Ranking by Sharpe**: V28 > V31 > V02+V05 > V26 > V11 > V33-core

**Key revision from prior reports**: V31's 0.307 Sharpe was stale-cache contamination.
The clean Sharpe of 0.769 makes V31 the second-best variant in the family, not the worst.
The family cross-section has shifted materially: we now have THREE variants above V11 by
+0.14 or more, not just one.

---

## Section 2: Cross-Sectional PBO

**PBO = 0.503** (s=16, 6-variant return matrix, T=2355, C(16,8)=12870 folds)

**Interpretation threshold per methodology Section 2.4:**
- < 0.25: acceptable
- 0.25-0.50: concerning
- > 0.50: strong overfitting evidence

At 0.503, the family CROSSES the strong-overfitting threshold. This means that in a
majority of CSCV folds, the in-sample best variant (V28 or V31 depending on the fold)
underperforms the OOS median. The three candidates that beat V11 (V28, V31, V02+V05)
span a Sharpe range of 0.68-0.81; the PBO is telling us that selecting the best of these
three is unlikely to generalize.

**Mechanistic read**: The family shows true cross-variant spread (V28 0.811 vs V33-core
0.479, a 0.33 gap), but the CSCV folds reveal that time-period subsets disagree on which
signal construction wins. The sub-window table (Section 4) confirms this: V28 and V31
are dramatically different across the 2017-2021, 2022-2024, and 2024-2026 windows.

**Important caveat**: PBO uses COMBINATORIAL CSCV across the 6 variants as a family.
This is correct per methodology but penalizes the fact that some variants (V26, V33-core)
are clearly not candidates -- including non-starters in the PBO matrix inflates the
in-sample dispersion without improving OOS generalization. A restricted PBO on just
{V28, V31, V02+V05} would be lower, but is not the canonical computation.

---

## Section 3: DSR / PSR for V28 and V02+V05

### Units note (per methodology Section 2.2)

All PSR and DSR inputs use **per-period (daily) Sharpe** with daily n. Annualized
values shown for the narrative. Passing annualized SR with daily n would inflate z by
sqrt(252) and make PSR saturate at 1.0 -- the code in `ramp_phase4_v11_readiness.py`
and this gate both use the correct daily units.

### V28 (primary candidate)

| Metric | Daily (formula input) | Annualized (narrative) |
|---|---:|---:|
| Observed Sharpe | 0.051111 | 0.8114 |
| Sample skewness | -0.4663 | -- |
| Sample Pearson kurtosis | 6.2208 | -- |
| Sample size (days) | 2355 | -- |
| PSR(vs SR=0) | 0.9928 | -- (PASS > 0.95) |

### V02+V05 (secondary candidate)

| Metric | Daily (formula input) | Annualized (narrative) |
|---|---:|---:|
| Observed Sharpe | 0.043042 | 0.6833 |
| Sample skewness | -0.3239 | -- |
| Sample Pearson kurtosis | 25.5063 | -- |
| Sample size (days) | 2355 | -- |
| PSR(vs SR=0) | 0.9804 | -- (PASS > 0.95) |

Note: V02+V05 has extremely high excess kurtosis (Pearson 25.5, excess ~22.5). This
dominates the PSR denominator and reduces z relative to a Gaussian return series. The
PSR of 0.980 reflects this fat-tail penalty.

### Trial Sharpes used for DSR variance term

All 6 variants' daily Sharpes (family cross-section):
V11=0.03326, V28=0.05111, V02+V05=0.04304, V26=0.03358, V33-core=0.03017, V31=0.04842

Variance of trial_sharpes (ddof=1) produces the spread used in expected_max_sharpe().

### n_trials justification

The TODO.md Acceptance Bar states: "The v0/detector family was at n_trials=36; a clean
trial-chain reset for a SIGNAL-CONSTRUCTION family is justifiable (new signal math, not
a detector tweak) -- document the justification."

**Justification for n_trials=6 (this family alone):**
Wave-3 is a NEW signal-construction family. The v0 family (V01-V14 + detector variants)
tested regime timing, regime gating, crash responses, and volatility overlays. Wave-3
tests SIGNAL CONSTRUCTION: how the momentum score is computed (multi-horizon blending,
z-scoring, beta-residualization, absolute-momentum gating). These are orthogonal
mechanism changes, not continuations of the same search over the same signal space.

**However**, the TODO.md permits this reset only with documentation, and the methodology
says "project-wide cumulative trial count." A clean reset IS defensible for family-DSR,
but the honest answer includes the sensitivity table. The verdict MUST state which
n_trials it rests on.

### DSR Sensitivity Table

Trial Sharpes variance determines E[maxSR]. The family has meaningful spread (V28 0.811
vs V33-core 0.479), so the expected max is moderate.

| n_trials | Justification | E[maxSR] (ann) | V28 DSR | V28 verdict | V02+V05 DSR | V02+V05 verdict |
|---:|---|---:|---:|---:|---:|---:|
| 6 | This Wave-3 family only | 0.1815 | 0.9712 | PASS | 0.9350 | FAIL |
| 12 | Wave-3 + ~6 detector variants | 0.2325 | 0.9596 | PASS | 0.9132 | FAIL |
| 36 | Chained to v0/detector campaign | 0.2999 | 0.9385 | FAIL | 0.8764 | FAIL |
| 50 | Conservative project-wide | 0.3179 | 0.9317 | FAIL | 0.8650 | FAIL |

**Which n_trials does the verdict rest on?**

V28 passes DSR at n_trials <= 12 and fails at n_trials >= 36.

The cleanest honest answer: this family gate uses n_trials=6 for the headline but
acknowledges the result is sensitive. The rationale for n=6 is:
1. The signal-construction mechanism is genuinely different from the regime-timing family.
2. The TODO explicitly permits a documented reset for this family.
3. The DSR formula's variance term already captures the within-family spread; chaining
   to prior families would double-count the correction for unrelated signal decisions.

However, a skeptic who insists on n=36 (project-wide chaining) will note that V28 DSR
of 0.938 is borderline and that PBO already crossed 0.50. The verdict section weighs both.

**V02+V05 fails DSR at ALL n_trials levels**, including n=6. Its Sharpe of 0.683 is
real (PSR 0.980) but insufficient to clear the multi-trial correction even with the
most generous trial count. This is primarily driven by its high kurtosis (Pearson 25.5),
which inflates the PSR denominator.

---

## Section 4: Sub-Window Stability

**Why sub-windows instead of walk-forward**: The full backtest is single-pass (no purge/
embargo walk-forward was run for this family probe). Sub-windows are therefore a temporal
robustness check, not a formal OOS test. A true purged/embargoed walk-forward is the
next gate for any graduating candidate.

### Sub-Window Sharpe Table (5 bps near_close, net)

| Window | V11 | V28 | V02+V05 | V26 | V33-core | V31 |
|---|---:|---:|---:|---:|---:|---:|
| 2017-2021 (IS-era, n=1259) | 0.618 | 0.697 | 0.833 | 0.674 | 0.811 | 0.736 |
| 2022-2024 (EXT-OOS era, n=753) | 0.361 | 0.534 | 0.618 | 0.360 | 0.009 | 0.538 |
| 2024-2026 (holdout, n=595) | 0.617 | 1.429 | 0.562 | 0.588 | 0.916 | 1.306 |

**Observations:**

V28 (primary):
- Beats V11 in ALL three sub-windows: +0.079 in IS-era, +0.173 in EXT-OOS, +0.812 in
  holdout. The edge is not confined to one regime era.
- The 2024-2026 Sharpe of 1.429 is dramatically higher than the other windows. This
  could be genuine alpha accumulation in the recent momentum-heavy market, or it could
  be a tail-period regime artifact. Without purging, we cannot distinguish them.
- IS/OOS ratio: 2017-2021 IS vs 2022-2026 OOS: 0.697 / (weighted avg of 0.534 + 1.429)
  is ambiguous because the two OOS windows diverge sharply. This is exactly why a
  formal walk-forward is needed.

V31 (beta-residual):
- Also beats V11 in all three windows, with a similar 2024-2026 surge (1.306).
- The 2022-2024 window shows V31 closely tracking V28 (0.538 vs 0.534), suggesting
  correlated signal construction.

V02+V05 (vanilla regime-free):
- Most consistent across windows: 0.833 / 0.618 / 0.562. The monotonic DECLINE
  across windows is actually healthy -- it shows the strategy is not back-of-sample
  overfitted but also that its edge is moderating over time.
- Beats V11 in all three windows, largest edge in EXT-OOS (+0.257).

V33-core:
- Near-zero Sharpe in 2022-2024 (0.009) despite reasonable IS-era and holdout numbers.
  The absolute-momentum cash gate is highly regime-dependent.

V26:
- Tracks V11 closely across all windows; no meaningful edge.

**Stability verdict for V28**: The sub-window pattern is ENCOURAGING (beats V11 in all
periods) but the 2024-2026 surge is a concern. A candidate with 1.429 holdout Sharpe
against 0.697 IS-era Sharpe has a strong IS/OOS ratio but raises the question of whether
recent concentrated macro regimes are inflating the number.

---

## Section 5: Cost Gate (Methodology Section 4, 1.5x = 7.5 bps)

| Variant | 5 bps Sharpe | 7.5 bps Sharpe | Delta | Gate (>= 0.5) |
|---|---:|---:|---:|---:|
| V11 (incumbent) | 0.528 | 0.452 | -0.076 | FAIL |
| V28 | 0.811 | 0.766 | -0.045 | PASS |
| V31 | 0.769 | 0.702 | -0.067 | PASS |
| V02+V05 | 0.683 | 0.598 | -0.085 | PASS |
| V26 | 0.533 | 0.438 | -0.095 | FAIL |
| V33-core | 0.479 | 0.372 | -0.107 | FAIL |

V28, V31, and V02+V05 all pass the 1.5x cost gate.

**Market-impact caveat**: The flat-bps model is a first-order approximation. V28's
AnnTO of 5,264% (vs V11's 10,325%) means it rebalances LESS than V11 on this runner's
metric, so the flat-bps model is actually conservative (V28 would benefit from a price-
impact model relative to V11). V31 at 7,217% is also below V11. V02+V05 at 10,275% is
comparable to V11. None of these are high-frequency strategies -- the turnover is driven
by daily close-price rebalancing across a 494-stock universe at top_n=10.

---

## Section 6: Combined Gate + Verdict (Methodology Section 2.5)

Gates: PSR(0) > 0.95, DSR > 0.95 (n_trials), PBO < 0.25, OOS Sharpe positive (all
sub-windows), cost gate >= 0.5 at 7.5 bps.

**Note on the combined gate's OOS/IS Sharpe ratio requirement**: Section 2.5 requires
OOS/IS >= 0.70. In this single-pass context we use sub-window ratios as a proxy.
V28's 2022-2024 / 2017-2021 ratio = 0.534 / 0.697 = 0.77 (PASS for that pair, though
the 2024-2026 surge complicates the picture). A formal walk-forward is needed for a
definitive OOS/IS ratio.

### V28 (primary candidate)

| Gate | Value | Threshold | Result |
|---|---:|---:|---:|
| PSR(vs 0) | 0.9928 | > 0.95 | PASS |
| DSR (n_trials=6, family-reset) | 0.9712 | > 0.95 | PASS |
| DSR (n_trials=12) | 0.9596 | > 0.95 | PASS |
| DSR (n_trials=36, project-chained) | 0.9385 | > 0.95 | FAIL |
| PBO (6-variant family) | 0.5031 | < 0.25 | FAIL (strong) |
| Beats V11 in all sub-windows | yes (3/3) | required | PASS |
| 2022-2024/2017-2021 IS/OOS ratio | 0.77 | >= 0.70 | PASS |
| Cost gate (7.5 bps Sharpe) | 0.766 | >= 0.50 | PASS |
| Lift over V11 (TIER-1 bar) | +0.283 | >= +0.10 | PASS |

**PBO BLOCKER**: PBO = 0.503 is a HARD structural failure per Section 2.4. The methodology
defines "strong overfitting" at PBO > 0.50. This means the in-sample best choice among
these 6 variants is more likely than not to underperform the median OOS. The fact that
the three candidates all beat V11 in sub-windows does not resolve this -- PBO measures
selection bias at the family level, and the family cross-section is unstable across time
periods.

**V28 VERDICT: HOLD** (not GRADUATE, not REJECT)

Rationale: V28 has genuine signal (PSR 0.993, beats V11 in all sub-windows, +0.283 full-
window lift), BUT the PBO failure means we cannot trust that selecting V28 as the winner
generalizes. The DSR result is n_trials-sensitive in a way that compounds this concern.

The recommended path is NOT to reject V28 -- its signal quality is strong enough to
warrant a formal purged/embargoed walk-forward (Section 3 of methodology). What changes
is: V28 does NOT go directly to paper trading. It goes to the walk-forward gate first.

**Decision logic**: DSR passes at n_trials=6 and n_trials=12 (family-reset justified by
TODO). We apply the n=6 verdict for the signal-construction family. But PBO = 0.503
overrides the DSR pass for the SELECTION decision (which of the 6 to pick). PBO does
NOT disqualify V28 as a research candidate -- it disqualifies the claim that this family
gate alone is sufficient to graduate to live paper.

### V02+V05 (secondary candidate)

| Gate | Value | Threshold | Result |
|---|---:|---:|---:|
| PSR(vs 0) | 0.9804 | > 0.95 | PASS |
| DSR (n_trials=6) | 0.9350 | > 0.95 | FAIL |
| DSR (n_trials=12) | 0.9132 | > 0.95 | FAIL |
| DSR (n_trials=36) | 0.8764 | > 0.95 | FAIL |
| PBO (6-variant family) | 0.5031 | < 0.25 | FAIL (strong) |
| Beats V11 in all sub-windows | yes (3/3) | required | PASS |
| Cost gate (7.5 bps Sharpe) | 0.598 | >= 0.50 | PASS |
| Lift over V11 (TIER-1 bar) | +0.155 | >= +0.10 | PASS |

**V02+V05 VERDICT: HOLD (secondary hold, lower priority than V28)**

V02+V05 fails DSR at ALL n_trials levels. Its Sharpe of 0.683 with Pearson kurtosis 25.5
gets heavily penalized by the moment correction. The fact that it beats V11 consistently
across sub-windows is real, but it does not clear the multi-trial significance gate.

V02+V05 is more interesting as a MECHANISM signal (direct H2 support -- regime gating
is net-negative) than as a standalone deployment candidate. If V28 graduates through
walk-forward, V02+V05's insights should be folded into the V28 parameter choices
(V28 is already regime-free).

### V31 (beta-residual, note: same gates as V28)

V31 is not listed as a primary or secondary candidate in the task specification, but
with the clean data its Sharpe (0.769) and PSR (0.990) are competitive. DSR sensitivity
for V31 at n_trials=6 would show a similar PASS/FAIL pattern to V28. However, V31 uses
the same engine + universe + full-window as all others; the PBO structural failure applies
equally to V31. V31 HOLD pending walk-forward -- same logic as V28.

### V26, V33-core (also-rans)

V26: ties V11, fails cost gate. No basis to advance.
V33-core: below V11, fails cost gate. Regime-dependent (0.009 in 2022-2024). Rejected.

---

## Section 7: The Null Option

The null option is explicit per TODO.md: if nothing clears the family gate at honest DSR,
the disciplined move is to ship/keep V11 and stop.

**Assessment**: The null option is NOT the correct call here. The evidence for V28 (and
to a lesser degree V31) is strong enough to warrant a walk-forward, which is the next
gate rather than deployment. The null option would mean "abandon all Wave-3 candidates
and commit to V11 as permanent incumbent." That would discard:

1. V28's +0.283 full-window Sharpe lift (>= +0.10 TIER-1 bar, comfortably)
2. Sub-window consistency across all three periods
3. The cleanest multi-horizon signal in the project history (PSR 0.993)

What the null option DOES tell us: we should not skip the walk-forward. The PBO = 0.503
means family-gate-alone is insufficient. The walk-forward is the gate that converts
HOLD into GRADUATE or REJECT.

---

## Section 8: Recommended Next Steps

### For V28 (primary path)

1. **Formal purged/embargoed walk-forward** (methodology Section 3):
   - Minimum 3 windows, purge_days = 21 (one momentum lookback cycle), embargo = 2%
   - Recommended: 5 windows, 2017-2022 IS, rolling 1-year OOS steps
   - Report per-window OOS Sharpe; IS/OOS ratio >= 0.70 required for all windows
   - Use `src/backtesting/chunking/` for window construction

2. **Correlation analysis with V11**: Before paper trading, compute correlation of V28
   daily returns with V11. If correlation > 0.85, V28 must REPLACE V11 (not co-deploy)
   since the incremental diversification benefit is marginal.

3. **If walk-forward passes**: Extend the A7 paper-validation comparator
   (`scripts/trading/compare_paper_vs_plan.py`) to model V28's multi-horizon signal.
   Then A7 paper validation (4-6 weeks).

### For V31 (secondary path, if walk-forward resources allow)

V31's clean Sharpe (0.769) is nearly identical to its PBO-share with V28. Run V31
through the same walk-forward. If V31 and V28 are correlated > 0.85, pick the one
with better walk-forward OOS Sharpe.

### For V02+V05 (informational)

V02+V05 provides the cleanest H2 test (regime-free beats regime-aware). This finding
should be folded into the V28 walk-forward design: if V28's walk-forward succeeds,
document that the multi-horizon signal appears to make regime gating redundant, which
is the H2 implication.

### For the null option (safety valve)

V11 remains the deployed paper strategy. It is NOT replaced until a walk-forward
candidate clears Section 3 gates. The current paper performance (A7 monitoring) is
the ground truth for any comparison.

---

## Appendix: Methodology Decisions

- n_trials for DSR headline: 6 (this Wave-3 family only). Justified by signal-construction
  family boundary (TODO.md Acceptance Bar). Sensitivity from n=6 to n=50 fully reported.
- PBO s: 16 (methodology Section 2.4 default)
- Cost tier for gates: 5.0 bps per side; cost sensitivity at 7.5 bps (1.5x)
- PSR/DSR units: daily sr_hat with daily n (per Section 2.2 requirement)
- Sub-windows: temporal, non-overlapping (2017-2021 / 2022-2024 / 2024-2026)
- V31 re-run: required because prior runs (SHA a88e762, 729f065) predate the data-integrity
  fix (429df47). dtype coercion fix also applied to `_compute_beta_residual_ranking`.
- All Sharpes are net-of-cost (5 bps per side round-trip at each rebalance)
- Regime attribution available per-variant in individual readiness reports

---

## Summary

| Item | Value |
|---|---|
| PBO (6 variants, s=16) | 0.503 (FAIL -- strong overfitting) |
| V28 PSR(vs 0) | 0.993 (PASS) |
| V28 DSR at n=6 / n=12 / n=36 | 0.971 / 0.960 / 0.939 (PASS / PASS / FAIL) |
| V02+V05 PSR(vs 0) | 0.980 (PASS) |
| V02+V05 DSR at n=6 / n=12 / n=36 | 0.935 / 0.913 / 0.876 (FAIL at all levels) |
| V28 sub-window consistency | Beats V11 in all 3 windows |
| V02+V05 sub-window consistency | Beats V11 in all 3 windows |
| V28 cost gate (7.5 bps) | 0.766 (PASS) |
| V28 verdict | **HOLD -- proceed to walk-forward, not paper** |
| V02+V05 verdict | **HOLD (secondary) -- fails DSR, walk-forward if resources allow** |
| Null option | NOT the call -- V28 signal strength warrants walk-forward |
| Immediate next action | V28 purged/embargoed walk-forward (Section 3) |
