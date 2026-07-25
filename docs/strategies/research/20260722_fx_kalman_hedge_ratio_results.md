# Kalman Dynamic Hedge Ratio -- Scoping Diagnostic Results

**Date:** 2026-07-25 (apparatus correction appended same day) | **Status:** CLOSED -- pre-registered prediction confirmed on the CORRECTED apparatus | **Owner:** strategy-lead / backtest-driver
**Pre-registration:** `docs/strategies/research/20260722_fx_kalman_hedge_ratio_preregistration.md` (LOCKED 2026-07-22)

**CORRECTION NOTICE (read first):** the run originally reported here (OOS
Sharpe 1x = 0.4955, registry N=132) had a defect: a NaN-seeded observation-noise
variance (`R`) silently zeroed out windows 1 and 2026-01-01 (window 13) --
Arm A's two worst windows (-6.82% and -10.09% total return). This document has
been rewritten so the CORRECTED run (OOS Sharpe 1x = 0.4171, registry N=133,
all 13 windows trading or genuinely flat) is the headline Arm B result. The
defective run is retained in the experiment registry (never deleted -- per the
North Star, N is never reduced to help an arm pass) and is documented below as
SUPERSEDED, with the defect, the fix, and the quantified size of the artifact
reported in full. See Section 1a for the defect/fix and Section 2 for the
corrected-vs-defective-vs-Arm-A comparison. **The gate verdict does not
change: FAIL, both before and after the correction** (Section 8).

**ADVERSARIAL CODE REVIEW (2026-07-25, completed same day):** an independent
adversarial review of the corrected apparatus found NO LOOKAHEAD, with
proofs (Section 7 has the full detail; summary: `causal_dynamic_beta` has
exactly one forward loop and no smoother, `select_warmup_indices` is a
prefix-determined monotone selection consuming only rows `<= seed_end`, and
z-construction is provably identical between arms). The review also
explained the mechanism behind the Sharpe swing -- it is regularization of a
near-collinear rolling-OLS beta, not genuine time-variation (new Section
1b) -- and surfaced ten apparatus caveats, all either symmetric across arms
or optimistic, none of which change the FAIL verdict (new Section 13). The
gate-wording precision between the pre-registration's literal "DSR > 0" and
the authoritative methodology combined gate is now stated explicitly in
Section 8. The scoped conclusion (Section 10) has been sharpened
accordingly.

This is the VERDICT run of the locked scoping diagnostic. Two registered
trials total (see Section 11): the defective Arm B run (N: 132->133) and the
corrected re-run (N: 133->134) that supersedes it as the graded result. No
tuning was performed in either run; `delta=1e-4` and `R` (OLS residual
variance over the training segment, robustly seeded -- see Section 1a) are
exactly as pre-registered.

## 1. Beta-path pre-check (reported first, per Section 7 of the pre-registration)

Computed on real AUDUSD/NZDUSD daily data, full available history
(2011-06-01 to 2026-04-01, 3838 raw observations, 3719 with both a static-OLS
and a causal-Kalman beta defined -- both require the 120-day warmup).

| Metric | Value |
|---|---|
| Correlation (OLS beta vs Kalman beta) | 0.2291 |
| Mean absolute difference | 0.2208 |
| OLS beta range | [0.0130, 1.6562] |
| OLS beta std | 0.3003 |
| Kalman beta range | [0.4985, 0.9252] |
| Kalman beta std | 0.0786 |
| Mean daily \|delta beta\|, OLS | 0.00899 |
| Mean daily \|delta beta\|, Kalman | 0.00137 |
| Kalman / OLS daily \|delta beta\| ratio | 0.152 (Kalman ~6.6x smoother) |

This independently confirms the prior orchestrator pre-check (corr 0.229,
mean\|diff\| 0.221, OLS range/std and Kalman range/std matching to within
rounding, ratio ~0.15 / ~6.5x). **The two beta paths are NOT near-identical**
-- the estimator was a live candidate explanation for Arm A's deficit, so the
scoping question was not answered trivially by this pre-check alone and the
full walk-forward gate below was required.

## 1a. Apparatus defect and fix (added 2026-07-25)

**The defect.** `AudNzdPairsKalman._beta_path` (`src/strategies/advanced/fx_audnzd_pairs.py`)
seeded the Kalman filter's fixed observation-noise variance `R` from the
LITERAL first `lookback=120` rows of each window's training-start panel
(`ln_a[:120], ln_b[:120]`), and `causal_dynamic_beta`
(`src/backtesting/signals/kalman_beta.py`) seeded `theta_0` the same way. If
ANY of those 120 literal rows contained a NaN -- e.g. window 1's `train_start`
(2011-01-01) predates real AUDUSD price history (starts 2011-06-01), or a
data-quality spike-revert null landed in the slice (window 13) -- the OLS fit
over that slice produced `r_var = NaN`, which poisoned the entire causal
recursion for the WHOLE window: `causal_dynamic_beta` returned all-NaN paths,
`_regression_z` returned `None` at every rebalance date, and the strategy
silently took zero positions for the whole window. This is a robustness bug
in the R-seed/warmup implementation, not a property of the estimator or a
change to the pre-registered R definition.

**Why it matters.** Windows 1 and 13 were Arm A's two worst OOS windows
(-6.82% and -10.09% total return, Sharpe -0.955 and -4.139). The defect meant
Arm B silently sat out both without economic reasoning -- a confound that
could inflate Arm B's apparent improvement over Arm A for reasons having
nothing to do with the hedge-ratio estimator.

**The fix.** `select_warmup_indices(y, x, size)` (new function,
`src/backtesting/signals/kalman_beta.py`) returns the indices of the first
`size` FINITE, ALIGNED `(y, x)` observations -- skipping any leading or
interior non-finite rows -- rather than a literal `[:size]` slice. Both the
`R`-seed OLS in `_beta_path` and the `theta_0`-seed OLS inside
`causal_dynamic_beta` now call this helper with the same `(y, x, lookback)`
so they agree on the same seed rows. `causal_dynamic_beta`'s forward
recursion now starts the day after the LAST selected seed row (`seed_end + 1`)
instead of the literal `warmup`-th row -- strictly later when rows were
skipped, never earlier -- so the returned path is still NaN until the warmup
genuinely completes, and every state remains conditioned on `0..t` only (no
smoother, no backward pass, no use of `t > current` anywhere). `delta=1e-4`,
the training-only scope of the R estimate, `lookback`, `entry_z`, `target_z`,
`stop_z`, `max_days`, and the z-construction are all unchanged.

**Behavior-neutrality check (windows with no leading/interior NaN in the
literal first-120-row slice).** Directly verified for window 6 (2019 OOS,
train_start 2016-01-01, the cleanest case): `select_warmup_indices` returns
`idx = [0, ..., 119]` (identical to the old literal slice) whenever all 120
rows are already finite, so `r_var`, `theta_0`, and the entire downstream
recursion are bit-identical to before the fix -- confirmed: first finite beta
index = 119 (= `lookback - 1`, exactly as before), `spread_book` produces the
same 29-entry book. More decisively, the corrected full walk-forward run's
per-window Sharpe/return for every window that was already active under the
defective run (2, 3, 4, 5, 6, 7, 8, 9, 11, 12) reproduces the defective run's
numbers to 3-4 decimal places (see Section 3a) -- direct empirical proof the
fix changed nothing where the seed was already valid, and only engaged its
new NaN-robust path for windows 1 and 13.

**Unit tests.** `tests/backtesting/signals/test_kalman_beta.py` gained six
tests: `select_warmup_indices` skips interior NaNs and returns `None` when
insufficient finite pairs exist anywhere in the input; a leading-NaN block
(reproducing the window-1 shape) still warms up and recovers the true beta
once enough finite history accumulates, with the path NaN exactly through the
new (later) warmup completion index; a handful of interior nulled rows
(reproducing the window-13 shape) shifts the warmup-completion index by
exactly the number of rows skipped and still recovers the true beta; the
insufficient-finite-anywhere case still fails closed (all-NaN, no exception);
and the causality (no-lookahead) guarantee is re-proven under the fix
specifically (perturbing future observations after the now-later warmup
point leaves every earlier filtered value bit-identical).
`tests/strategies/test_fx_audnzd_kalman_arm.py` gained a strategy-level
regression test reproducing the window-1 shape directly on `_beta_path` and
`spread_book`, asserting the book is no longer empty. All 17 tests in both
files pass (`conda run -n fintech python -m pytest
tests/backtesting/signals/test_kalman_beta.py
tests/strategies/test_fx_audnzd_kalman_arm.py
tests/strategies/test_fx_audnzd_pairs.py -v`).

## 1b. Mechanism behind the swing: regularization, not time-variation (added 2026-07-25, adversarial review)

The -0.2362 -> +0.4171 OOS Sharpe swing (Section 2) is large enough in
isolation that, before this review, it was reported without a mechanistic
explanation -- and "a large, unexplained improvement from an estimator swap"
is exactly the shape a lookahead leak would produce. The review both proved
no leak exists (Section 7) and supplied the missing mechanism, which
reframes the finding.

Arm A's rolling 120-day OLS hedge ratio ranges over `[0.0130, 1.6562]`
(std 0.30, Section 1). A fitted beta of 0.013 is not a hedge ratio in any
economically meaningful sense -- it is an outright long-AUDUSD position
wearing a pairs-trade label, because the OLS fit assigned essentially zero
weight to the NZDUSD leg. This is the classic symptom of a near-collinear
regression between two I(1) series that share a common dollar factor: over
a 120-observation rolling window, the AUDUSD-on-NZDUSD slope is only weakly
identified, and it is free to wander across a wide, economically
implausible range as the window rolls.

With `delta=1e-4` (the pre-registered Kalman process-noise scale, unchanged
by the fix), Arm B's per-step state-transition variance is tiny relative to
the observation noise `R`. In that regime the causal Kalman filter is not
primarily behaving as "a hedge ratio that tracks genuine time-variation in
the true relationship" -- it is behaving as a heavily SHRUNK / REGULARIZED
estimator, pulling beta toward a stable central value and confining it to
`[0.4985, 0.9252]` (std 0.079, Section 1), moving ~6.6x less per day than
Arm A. That is a shrinkage/regularization signature, not a time-varying-beta
signature.

**Consequence for how the swing should be read.** The pre-registration's
0.1-0.3 Sharpe prior for a "genuinely time-varying vs static" comparison
implicitly assumed a well-conditioned static baseline. That assumption did
not hold: Arm A's -0.2362 partly reflects a degenerate baseline
estimator -- an unstable, near-unidentified rolling-OLS beta -- rather than a
clean measurement of "does allowing the hedge ratio to move over time help."
**The swing should be reported as REGULARIZATION, not time-variation.** This
sharpens what the scoped negative (Section 10) actually eliminates: not "the
hedge-ratio estimator" generically, but specifically "an unstable,
unregularized rolling-OLS hedge ratio" as the explanation for the pairs
trade's failure -- a shrunk/regularized estimator was also tried, and it
also fails the gate, so the elimination still holds, but the reason the raw
Sharpe moved is now understood and is not evidence of a leak.

## 2. Both arms, side by side

Arm A is cited from the already-gated Wave 2 record (NOT re-graded here; a
diagnostic re-simulation with `--no-register` reproduced it exactly, see
Section 4). Arm B is the one new trial.

**This is the headline table -- Arm B column is the CORRECTED (apparatus-fixed)
run.** The previously-reported (defective, zero-fills-in-windows-1/13) Arm B
run is shown alongside for transparency, not as the graded result.

| Metric | Arm A (static 120d OLS) | Arm B CORRECTED (causal Kalman, fixed) | Arm B defective (SUPERSEDED) |
|---|---|---|---|
| Strategy class | `AudNzdPairs` | `AudNzdPairsKalman` | `AudNzdPairsKalman` |
| OOS Sharpe (1.0x cost) | -0.2362 | **0.4171** | 0.4955 |
| OOS Sharpe (1.5x cost) | -0.3016 | 0.3466 | 0.4300 |
| PSR (vs SR=0) | 0.0000 | 1.0000 | 1.0000 |
| DSR (deflated, project-wide N) | 0.0000 (at N=109) | **9.96e-186 (at N=133)** | 1.10e-112 (at N=132) |
| PBO | 0.8242 | **0.8931** | 0.9249 |
| OOS trade count (fills, 1.0x leg) | 286 | 260 | 208 |
| S&P 500 correlation (OOS) | 0.0436 | 0.0161 | 0.0261 |
| n_windows (all trading or genuine no-signal) | 13 | 13 | 13 (2 silently empty by defect) |
| n_oos_days | 3185 | 3185 | 3185 |
| window_start / window_end | 2014-01-01 / 2026-04-01 | 2014-01-01 / 2026-04-01 | 2014-01-01 / 2026-04-01 |
| skew | -0.5286 | 0.8894 | 1.7099 |
| kurtosis (Pearson) | 40.1163 | 30.5439 | 39.5705 |
| registry run_id | (original wave-2 run) | `08722e3d-9272-43f9-8da5-0f27ea380ebd` | `31a4cb58-193c-41c5-990e-902df81a0de4` (kept, not deleted) |

Windows match exactly between arms (36m/12m/12m, same 13 windows, same
3185 OOS days), as required so the comparison isolates the estimator.

**How much of the defective +0.4955 was the zero-fill artifact?** The fix
moved OOS Sharpe from 0.4955 to 0.4171, a drop of 0.0784. Framed against Arm
A: the defective run reported a swing of -0.2362 -> +0.4955 (+0.7317); the
corrected swing is -0.2362 -> +0.4171 (+0.6533). So the artifact accounts for
**0.0784 / 0.7317 = 10.7% of the originally-reported swing** -- real, and
directionally exactly the confound flagged before this correction was run
(Arm B silently sat out Arm A's two worst windows), but quantitatively
**smaller than "substantially"**: roughly 89% of Arm B's apparent edge over
Arm A survives the correction. The reason it is not larger: window 1 and
window 13, once tradable, do NOT both help Arm B's defective number in the
same direction -- window 1 turns out mildly POSITIVE for Arm B (+1.23%, a
real win Arm B was missing out on) while window 13 is mildly NEGATIVE
(-2.26%); the two partially offset rather than compounding (Section 3a has
the full per-window detail). What is unambiguously worse under correction:
DSR moved from numerically indistinguishable from zero (1.10e-112 at N=132)
to even more decisively zero (9.96e-186 at N=133) -- both fail the gate the
same way, but the corrected number leaves no room for any argument that the
defect was flattering DSR specifically.

**DSR precision note:** Arm B's (corrected) DSR is `9.9595e-186`, i.e.
numerically zero to any practical precision -- not "small but real," but a
probability indistinguishable from zero given float precision. It is
technically `> 0` in a strict literal sense, but treating that as satisfying
the pre-committed "DSR > 0" clause would be exactly the kind of
engineering-around-a-failure the North Star prohibits. The gate is failed
decisively by PBO regardless (Section 8).

## 3. Turnover / cost-drag comparison (the mechanism check, Section 7)

**Recomputed on the CORRECTED Arm B run** (defective-run numbers shown
alongside for reference, marked superseded). Computed from each arm's OOS
fills (1.0x cost leg), rejoined to daily close prices to get USD notional
(both AUDUSD and NZDUSD are already USD-quoted, so `notional_usd = |units| *
close`). Capital = $100,000; n_oos_days = 3185 (12.64 years).

| Metric | Arm A (static OLS) | Arm B CORRECTED | Change vs Arm A | Arm B defective (SUPERSEDED) |
|---|---|---|---|---|
| OOS fill count | 286 | 260 | -9.1% | 208 |
| Total notional traded (USD) | $20,771,334 | $20,382,767 | -1.9% | $16,452,724 |
| Turnover x capital, annualized | 16.43x/yr | 16.13x/yr | -1.9% | 13.02x/yr |
| Total cost (USD, OOS) | $7,026.44 | $6,942.16 | -1.2% | $5,729.61 |
| Cost drag, annualized (frac. of capital) | 0.556%/yr | 0.549%/yr | -1.2% (-0.01pp) | 0.453%/yr |

**Finding, corrected: turnover is essentially UNCHANGED vs Arm A, not
meaningfully lower.** The defective run's headline "-20.8% turnover" finding
was itself substantially a zero-fill artifact -- two whole windows of Arm A's
trading activity had no Arm B counterpart to compare against. With the fix,
Arm B's turnover (16.13x/yr) sits only 1.9% below Arm A's (16.43x/yr),
essentially within noise. The beta-path pre-check's finding that the Kalman
beta is ~6.6x smoother day-to-day (Section 1) is real, but its effect on
realized hedge-leg turnover is much smaller in practice than either the
pre-registration's primary rationale ("updating every period RAISES
turnover") or the defective run's secondary finding ("turnover fell 20.8%")
suggested. Both directional claims were, in different ways, overstated;
the honest corrected finding is "turnover is approximately a wash."

**Does the turnover/cost difference explain the Sharpe difference? No --
even more clearly than before.** The annualized cost-drag difference is now
~0.01 percentage points/year, three orders of magnitude too small to explain
the ~0.65 move in annualized Sharpe (-0.24 to +0.42, corrected). The Sharpe
change is dominated by different trade timing and selection (different
residual paths from the different beta estimates), not by trading costs.
This is corroborated by the shape of the return distribution: Arm B's skew
flips from -0.53 (Arm A) to +0.89 (corrected; was +1.71 defective),
consistent with the improvement being concentrated in a minority of
favorable trades/windows rather than a broad-based, stable edge -- exactly
the pattern PBO (0.89) and DSR (~0) are built to catch.

## 3a. Return concentration / per-window stability (recomputed on the corrected apparatus, 2026-07-25)

**Purpose:** the raw OOS Sharpe move (-0.2362 -> 0.4171, corrected) is large
enough in isolation to warrant checking whether it reflects a broad, stable
improvement or a concentration artifact in a handful of trades/days, before
the scoped conclusion (Section 10) is finalized. This section supersedes the
originally-published version, which used the defective (windows 1/13
zero-fill) run; the original per-window numbers for the previously-active
windows are retained below for the behavior-neutrality comparison, since they
reproduce the corrected numbers essentially exactly and that reproduction is
itself evidence the fix is behavior-neutral where the seed was already valid.

### Per-window OOS Sharpe, all 13 windows: Arm A, Arm B corrected, Arm B defective (superseded)

| Window | OOS period | n_days | Arm A Sharpe | Arm A ret | Arm B CORRECTED Sharpe | Arm B CORRECTED ret | Arm B CORRECTED fills | Arm B defective Sharpe (SUPERSEDED) |
|---|---|---|---|---|---|---|---|---|
| 1 | 2014-01-01 to 2015-01-01 | 259 | -0.955 | -6.82% | **0.246** | **+1.23%** | 36 | n/a (0 fills, defect) |
| 2 | 2015-01-01 to 2016-01-01 | 259 | 0.622 | +1.80% | 1.238 | +3.76% | 20 | 1.238 (matches) |
| 3 | 2016-01-01 to 2017-01-01 | 260 | 0.000 | -0.09% | 0.464 | +0.92% | 10 | 0.464 (matches) |
| 4 | 2017-01-01 to 2018-01-01 | 260 | 0.365 | +1.40% | 1.264 | +5.11% | 26 | 1.264 (matches) |
| 5 | 2018-01-01 to 2019-01-01 | 260 | -0.565 | -2.41% | 0.641 | +3.01% | 38 | 0.641 (matches) |
| 6 | 2019-01-01 to 2020-01-01 | 259 | 1.668 | +7.34% | 1.786 | +7.24% | 18 | 1.786 (matches) |
| 7 | 2020-01-01 to 2021-01-01 | 261 | 0.267 | +0.83% | 0.649 | +0.88% | 10 | 0.649 (matches) |
| 8 | 2021-01-01 to 2022-01-01 | 260 | -0.486 | -2.63% | -0.804 | -3.51% | 20 | -0.804 (matches) |
| 9 | 2022-01-01 to 2023-01-01 | 260 | -0.392 | -1.63% | 0.278 | +1.44% | 20 | 0.278 (matches) |
| 10 | 2023-01-01 to 2024-01-01 | 261 | 0.983 | +0.22% | n/a (genuine no-signal) | 0.00% | 0 | n/a (genuine no-signal, matches) |
| 11 | 2024-01-01 to 2025-01-01 | 263 | 0.534 | +2.67% | 1.418 | +4.97% | 16 | 1.418 (matches) |
| 12 | 2025-01-01 to 2026-01-01 | 262 | -1.730 | -3.12% | -0.574 | -2.52% | 30 | -0.574 (matches) |
| 13 | 2026-01-01 to 2026-04-01 | 65 | -4.139 | -10.09% | **-1.156** | **-2.26%** | 16 | n/a (0 fills, defect) |

**Behavior-neutrality, confirmed empirically:** every window that was already
active under the defective run (2,3,4,5,6,7,8,9,11,12) reproduces its
per-window Sharpe to 3 decimal places under the fix -- direct evidence the
fix changed nothing where the seed was already valid. Only windows 1 and 13
change, from "n/a, 0 fills" to genuinely-traded, genuinely-computed values.

**Windows positive:** Arm A 7/13 (windows 2,3,4,6,7,10,11). Arm B corrected
9/13 positive (windows 1,2,3,4,5,6,7,9,11), 3/13 negative (8,12,13), 1/13
genuinely flat/no-signal (10) -- now ALL 13 windows have a defined,
economically-meaningful outcome (trade or genuine no-signal), none are
silently empty by defect.

### Return concentration (Arm B corrected, Arm A and Arm B defective for reference)

Stitched OOS daily returns x $100,000 capital, 3185 days:

| Metric | Arm A (static OLS) | Arm B CORRECTED | Arm B defective (SUPERSEDED) |
|---|---|---|---|
| Total OOS $P&L | -$12,831.14 | **+$20,465.45** | +$21,307.68 |
| Best single day | +$2,876.00 | +$2,205.95 | +$2,205.95 |
| Best 5 days, sum | +$12,081.41 | +$10,405.38 | +$10,405.38 |
| Best 10 days, sum | +$20,792.55 | +$19,257.71 | +$18,977.56 |
| Best 5 days as % of total P&L | n/m (total negative) | **50.8%** | 48.8% |
| Best 10 days as % of total P&L | n/m | **94.1%** | 89.1% |
| Annualized Sharpe, full sample | -0.2362 | 0.4171 | 0.4955 |
| Annualized Sharpe, excluding best 5 days | -0.4918 | **0.2179** | 0.2749 |

The corrected run's total P&L, best-1/5/10-day dollar figures are close to
the defective run's (the best days themselves are drawn from the same
already-active windows, since neither window 1 nor 13 produced a single-day
move large enough to enter the top 10) -- but concentration as a SHARE of
total is slightly WORSE under correction (best-10-days 94.1% of total vs
89.1% defective) because the corrected total P&L is very slightly lower
($20,465 vs $21,308) while the top-10 dollar sum is nearly unchanged.
Excluding the best 5 days does NOT erase the edge (Sharpe 0.4171 -> 0.2179,
still clearly positive) -- this is not a single-trade fluke. But the
improvement remains materially concentrated: a small minority of windows and
days account for most of the P&L, rather than a uniform, broad-based edge
across the sample -- direct, day-level corroboration of the same instability
the corrected PBO computation (0.8931, Section 8) is designed to detect at
the window level.

### Why Arm B had zero fills in window 1 and window 13 (fixed 2026-07-25; kept as history)

Verified from the data, not inferred from the entry-threshold logic: window
1's `train_start` is 2011-01-01, but real AUDUSD price data does not begin
until 2011-06-01 (confirmed: `close_panel["AUDUSD"]` was NaN for the first
107 of the window's first 120 rows, 2011-01-03 through 2011-05-31). The
DEFECTIVE `AudNzdPairsKalman._beta_path` computed the Kalman filter's fixed
observation-noise variance `R` ONCE, from an OLS fit over the LITERAL first
`lookback=120` rows of the window's train-start panel. Because those 120 rows
were NaN-contaminated, `r_var` was NaN, and this single NaN seed poisoned the
entire causal Kalman recursion for every subsequent day: `alpha_path`/
`beta_path` were 0/1038 finite for the full window (train+test span).
`_regression_z` therefore returned `None` at all 163 weekly rebalance dates
that had a defined regression window, and the strategy took zero positions
for the whole window. This was a different mechanism than the
entry-threshold hypothesis originally floated in the diagnostic request --
it was not that `|z|` failed to cross `entry_z=2.0` (Arm A's rolling-OLS
`|z|` on the identical price data DOES cross 2.0 33 times in this window,
max `|z|`=3.87); Arm B never produced a defined `z` at all in window 1.

Window 13 (`train_start`=2023-01-01) showed the identical symptom via a
different upstream trigger: the data loader nulls 6 NZDUSD bars as
spike-revert artifacts project-wide (logged as `NZDUSD: nulled 6
spike-revert artifact bar(s)`), and at least one fell inside window 13's
literal first-120-row `R`-seed slice, again NaN-poisoning `r_var` and the
entire window's beta path.

**Both are now fixed** (Section 1a): `select_warmup_indices` seeds `R` and
`theta_0` from the first `lookback` FINITE, aligned observations wherever
they occur, so a leading NaN block or a handful of interior nulled rows no
longer poisons the whole window -- window 1 now warms up starting at its
226th row (107 leading NaNs + 120 finite seed rows - 1) and trades 36 times
in-OOS; window 13 warms up a few rows later than the literal slice and trades
16 times in-OOS.

Window 10 (`train_start`=2020-01-01) remains, under the corrected apparatus,
a genuine no-signal case (unaffected by the defect or the fix): its Kalman
beta path is valid (50 fills occurred, all dated 2021-07-19 through
2022-11-07, entirely inside the training segment, none in the 2023 OOS year)
-- the position was closed before OOS start and `|z|` never re-crossed
`entry_z` during the OOS year. This is confirmed unchanged by the corrected
run: window 10 still shows 0 OOS fills, and its 261-day OOS return series is
still all-zero (flat), not a NaN-driven absence.

**Takeaway for the estimator comparison:** Arm B's fixed, once-per-window
`R`-seeding (fit on the FIRST `lookback` FINITE rows of each window, per the
pre-registration, now correctly implemented) is no longer fragile to a
single NaN or data-quality null landing in the seed window -- both known
failure modes (leading NaN block, interior nulled bar) are fixed, verified
by direct reproduction (Section 1a) and by the fact that all 13 windows now
produce a defined, economically-meaningful outcome. This robustness
difference from Arm A's rolling 120-day OLS (which recomputes fresh at every
rebalance and was never affected by this class of defect) is worth recording
for any future dynamic hedge-ratio work, but no longer affects the estimator
comparison's validity -- the corrected run trades or genuinely abstains in
all 13 windows.

### Honest read: broad improvement or concentration artifact?

Neither pure, and the read is essentially unchanged by the correction. Arm
B's positive Sharpe is not a single-trade or single-window fluke: 9 of its
12 active windows (all but window 10, which is genuinely flat) are
individually positive, and excluding the single best 5 days leaves the
Sharpe still clearly positive (0.2179, down from 0.4171). But the
improvement is also not a broad, uniform edge -- 10 of 3185 days (0.31%)
produce 94.1% of total P&L. This concentration is consistent with, and helps
explain in day-level terms, the extreme corrected PBO value (0.8931): a
P&L profile this lumpy is exactly the pattern that makes cross-window (CSCV)
resampling unstable -- drop or reweight the handful of dominant windows and
the ranking of "this config beats the median" flips easily. Combined with
the DSR being numerically indistinguishable from zero at the honest N=133
project-wide trial count (even more decisively than the defective run's
N=132 figure -- 9.96e-186 vs 1.10e-112), this concentration evidence
supports, rather than merely coexists with, the gate's FAIL verdict: the raw
Sharpe improvement is real in the sense that it is not attributable to one
lucky trade, but it is not broad-based or resample-stable enough to
constitute credible evidence of skill.

## 4. Arm A diagnostic re-simulation (reproducibility check, not a new trial)

Re-ran Arm A (`AudNzdPairs`, `config/backtesting/fx_audnzd_pairs.yaml`) through
the (apparatus-fixed) walk-forward runner with `--no-register`, purely to
capture its OOS fills for the turnover comparison above. This did NOT append
to the experiment registry (verified: `register=False` -> "skipping
experiment-registry append_run" logged; registry run_id in its report is
`None`).

| Metric | Recorded (Wave 2, N=109) | Diagnostic re-sim (this run) | Match? |
|---|---|---|---|
| OOS Sharpe (1.0x) | -0.2362 | -0.2362 | Yes, exact |
| OOS Sharpe (1.5x) | -0.3016 | -0.3016 | Yes, exact |
| PSR | 0.0000 | 0.0000 | Yes |
| PBO | 0.8242 | 0.8242 | Yes |

Reproducibility PASSED. Its own report cites "DSR (N=133)" because it ran
after Arm B's registry append (N: 132 -> 133); that DSR is not used for
grading -- Arm A's official gate result remains the one already recorded at
N=109. This is called out explicitly to avoid any appearance of re-grading
an already-counted trial. Arm A's code (`AudNzdPairs`,
`src/strategies/advanced/fx_audnzd_pairs.py`) was not touched by the
apparatus fix in Section 1a (only `AudNzdPairsKalman._beta_path` and
`causal_dynamic_beta` changed), so this reproducibility check remains valid
unchanged after the correction.

## 5. Fills artifact verification (methodology Section 12.0 / CLAUDE.md mandatory)

| Arm | Fill-sink run_id | Run dir | trades_oos.csv.gz rows | manifest.csv rows | Status |
|---|---|---|---|---|---|
| Arm B CORRECTED (Kalman, gated, graded) | `20260725T044928Z_327859` | `output/backtests/AudNzdPairsKalman/runs/20260725T044928Z_327859/` | 260 (non-empty; all 13 windows present, incl. w01 and w13) | 54 (53 artifacts + header) | **Current headline result** |
| Arm B defective (Kalman, superseded) | `20260725T042937Z_327859` | `output/backtests/AudNzdPairsKalman/runs/20260725T042937Z_327859/` | 208 (non-empty; w01/w13 fill files 0-row, per the defect) | 54 (53 artifacts + header) | Superseded, kept for audit trail |
| Arm A (diagnostic re-sim) | `20260725T042943Z_914aa5` | `output/backtests/AudNzdPairs/runs/20260725T042943Z_914aa5/` | 286 (non-empty) | 54 (53 artifacts + header) | Unaffected by the fix |

The corrected `trades_oos.csv.gz` is non-empty and contains `date, pair,
units, cost` columns for the 1.0x-cost leg, sliced to each window's OOS
`[test_start, test_end)` segment (the final window is inclusive of
`test_end`) and concatenated across the 13 windows, per
`FillSink.finalize(oos_windows=..., oos_cfg_hash="c1x")`. Per-window fill
counts in the corrected run (grouped by OOS date range): w01=36, w02=20,
w03=10, w04=26, w05=38, w06=18, w07=10, w08=20, w09=20, w10=0 (genuine
no-signal), w11=16, w12=30, w13=16 -- sum 260, matching the registry row.

## 6. Exact N used for DSR deflation

**Corrected (graded) run:** `get_campaign_trial_distribution()` read **N=133**
immediately before the corrected Arm B run (queried directly, confirmed from
the registry row's own `trial_count` field: 133). The corrected run's DSR
(9.96e-186) was deflated at N=133. After this run's registry append, N
advances to **134** for any subsequent run.

**Defective (superseded) run, for the record:** used N=132, advancing to 133
after its append -- the same N=133 the corrected run then used, i.e. the
defective run's own registry row is itself one of the 133 trials the
corrected run's DSR was deflated against (never deleted, per the North
Star's "N is never reduced to help an arm pass").

**Trial-accounting honesty note:** this diagnostic now spans **two**
registered trials (132->133 for the defective run, 133->134 for the
corrected run), not the single trial the pre-registration's Section 8
originally committed to. The corrected re-run was executed because Task 1's
fix was an apparatus correction (implementing the pre-registered R
definition correctly, not a new specification) -- but the registry append is
honest and count-based regardless of intent, so N correctly reflects 2
trials from this diagnostic, and every subsequent DSR computation in this
repository is deflated against that correct, undercounted-nothing total.

## 7. Filter-only integrity check (automatic-reject condition)

`src/backtesting/signals/kalman_beta.py::causal_dynamic_beta` was read in
full. It implements only the forward filtering recursion (predict/update per
`t`, conditioned on observations `0..t`); there is no backward pass, no
RTS/fixed-interval smoother, and no revision of past `theta` values using
later observations. A repo-wide grep for smoother-related symbols
(`rts_smooth`, `fixed_interval_smooth`, `backward_pass`) returned no matches
outside the module's own docstring, which explicitly documents the
smoother as deliberately NOT implemented. **Integrity check PASSED** -- this
is a genuine causal filter, not a lookahead-contaminated smoother.

Additionally verified: R (observation-noise variance) is computed from the
first `lookback` FINITE, aligned rows of each window's `close` panel (per
the Section 1a fix -- previously a literal `[:lookback]` slice, now
`select_warmup_indices`), which is loaded from `train_start` (36 months
before `test_start`), so R is estimated entirely within the training segment
in every window -- including 1 and 13, where the finite-row search extends
somewhat past the literal first 120 rows but stays far short of
`test_start` (120 finite rows are found within ~230 raw rows even with a
107-row leading gap, versus a ~1095-day training segment) -- so this remains
entirely within-training, never touching test data. `causal_dynamic_beta`'s
forward recursion starts strictly after the last seed row and is
re-initialized fresh per walk-forward window (`AudNzdPairsKalman` is
re-instantiated per window by `_make_strategy`), so there is no cross-window
state leakage and no observation at `t' > t` is used to set state `t`
anywhere in the fix. The new unit tests in Section 1a explicitly re-prove
the no-lookahead guarantee under the fix (perturbing observations after the
now-later warmup point leaves all earlier filtered values bit-identical).
**Integrity check re-confirmed PASSED after the fix.**

## 8. Pre-committed gate evaluation (Section 5 of the pre-registration)

The verdict "changes" ONLY if Arm B clears ALL FIVE. **Evaluated on the
CORRECTED run** (defective-run values shown for reference):

| Condition | Threshold | Arm B CORRECTED | Pass? | Arm B defective (SUPERSEDED) |
|---|---|---|---|---|
| OOS Sharpe (1.0x) | > 0 | 0.4171 | PASS | 0.4955 |
| OOS Sharpe (1.5x) | > 0 | 0.3466 | PASS | 0.4300 |
| PSR | > 0.95 | 1.0000 | PASS | 1.0000 |
| DSR (at current N) | > 0 | 9.96e-186 (~0) | FAIL in substance (numerically zero; see Section 2 note) | 1.10e-112 (~0) |
| PBO | < 0.5 | 0.8931 | **FAIL** | 0.9249 |

**Result: FAIL, unchanged by the correction.** The gate requires ALL FIVE
conditions; PBO fails decisively both before and after the fix (0.89 vs the
0.5 bar -- worse than the 0.25/0.5 thresholds by a wide margin, meaning in
~89% of CSCV resamples the in-sample-best window underperformed the OOS
median), and DSR provides no real support even under a literal reading --
if anything, DSR is MORE decisively zero after correction (9.96e-186 at
N=133, vs 1.10e-112 at N=132) since the higher trial count and the slightly
lower Sharpe both push it further down. Per the pre-committed decision
rule, "any improvement short of that is NOT a pass and NOT a 'promising
lead.'" The apparatus fix changed the exact numbers but not the verdict.

**Gate wording precision (added 2026-07-25, adversarial review).** The
pre-registration's Section 5 wrote the DSR criterion literally as "DSR > 0."
Arm B's DSR of `9.9595e-186` is numerically indistinguishable from zero at
any practical precision, but it is LITERALLY greater than zero as a raw
float. Two readings therefore exist, and both are reported here for full
honesty:

- **Literal pre-registration wording ("DSR > 0"):** Arm B's DSR technically
  PASSES this clause (a positive float, however small, satisfies `> 0`).
  Under this literal reading, the ONLY failing condition among the five is
  PBO (0.8931 vs the `< 0.5` bar).
- **Authoritative methodology gate (`docs/methodology/backtesting.md`
  Section 2.5, which CLAUDE.md makes authoritative over any agent-prompt or
  pre-registration wording when the two disagree):** the combined gate
  requires `DSR >= 0.95`. Arm B's DSR of `9.96e-186` fails this threshold
  DECISIVELY and INDEPENDENTLY of PBO -- it is not a borderline miss, it is
  off by roughly 184 orders of magnitude. Under this reading PBO is not
  even load-bearing to the verdict; DSR alone fails the gate.

**Both readings reach the same verdict: FAIL.** But "DSR > 0" is recorded
here as a drafting weakness in the pre-registration -- a threshold that any
positive float passes by construction and therefore carries no real
statistical content -- and should not be relied on as the operative gate
condition in future pre-registrations. The authoritative Section 2.5
combined gate (PSR > 0.95, DSR >= 0.95, PBO < 0.25, trades >= 30 OOS,
OOS/IS Sharpe ratio >= 0.7) is what actually governs the verdict here, and
Arm B fails it on at least two independent legs (DSR and PBO) regardless of
which reading of the pre-registration's own wording is used.

## 9. Registered prediction check

Pre-registration Section 6 registered: **"Arm B fails."** CONFIRMED, on the
corrected apparatus.

The prediction's PRIMARY rationale (closing a ~1.3 Sharpe gap via an
estimator swap alone is not credible) also held up qualitatively: the raw
OOS Sharpe move (-0.24 to +0.42, corrected) looked large in isolation, but
once corrected for the honest, growing project-wide trial count (N=133 for
the graded run), the DSR shows the move carries no statistically credible
evidence of skill, and PBO shows severe cross-window instability. The
prediction's SECONDARY, turnover-based rationale ("a beta that updates
every period generally RAISES hedge-leg turnover") remains directionally
WRONG in sign, but its magnitude claim from the (defective) intermediate
finding does not hold up either: the corrected turnover comparison
(Section 3) shows turnover and cost drag are essentially UNCHANGED versus
Arm A (-1.9%/-1.2% respectively, not the defective run's -20.8%/-18.5%) --
the earlier "turnover fell substantially" finding was itself partly a
zero-fill artifact. Turnover was never the dominant driver of either arm's
Sharpe either way (Section 3), so this correction does not affect the
overall prediction's correctness, but it is recorded here for full honesty:
BOTH the pre-registration's original turnover rationale and the defective
run's turnover finding overstated the size of the turnover effect, in
opposite directions.

## 10. Scoped conclusion

**What this bounds.** The Wave 2 pairs-trading negative for AUDUSD/NZDUSD is
now scoped to cover **both** hedge-ratio estimators tested: the static
trailing-120d OLS (Arm A) and the causal, time-varying Kalman/regularized
filter (Arm B, `delta=1e-4`). It is no longer correct to say the negative
bounds "only static-OLS-hedged pairs trading" -- it now also bounds this
specific causal dynamic-beta (in substance, shrunk/regularized -- Section
1b) variant of the same spec (same universe, same entry/exit logic, same
costs, same walk-forward windows). AUDUSD/NZDUSD pairs trading now fails
under both a static rolling-OLS hedge ratio AND a causal Kalman/regularized
time-varying hedge ratio: the raw Sharpe moved meaningfully when the
estimator was swapped (corrected: -0.24 to +0.42, explained by
regularization of a near-collinear OLS beta rather than genuine
time-variation -- Section 1b) but did not produce a statistically robust,
deflation-surviving, resample-stable edge (DSR ~0 at N=133 under either
gate reading, PBO 0.89 -- Section 8). **The Wave 2 pairs negative for this
symbol pair is no longer explainable by hedge-ratio mis-specification** --
that candidate explanation is eliminated, confirmed on the corrected
apparatus with all 13 windows trading or genuinely abstaining, not just the
10 that traded under the defective run, and confirmed lookahead-free by
adversarial review (Section 7).

**What this does NOT establish -- the limits of the scope.** Per the North
Star, a negative bounds the specification tested, not the mechanism or
asset class. This diagnostic tested exactly: ONE pair (AUDUSD/NZDUSD), ONE
Kalman process-noise scale (`delta=1e-4`, not swept -- Section 13 item 10),
DAILY-bar data, SPREAD-TAKING execution costs (no liquidity-provision/maker
modeling), ZERO execution lag (fills at the same close the signal is built
from -- Section 13 item 2, optimistic and shared with the rest of the FX
spread vertical), and an entry blackout filter that is INERT in 12 of 13
windows because the underlying calendar has only one RBA and one RBNZ date
on file (Section 13 item 3) -- i.e. the spec that was gated is not fully
the spec that was described. None of the following are addressed by this
diagnostic and remain untested, live hypotheses: cointegration/
relative-value trading on other pairs, other Kalman process-noise scales or
other lookback windows, other z-thresholds, other execution styles
(specifically earning the spread as a liquidity provider/maker rather than
paying it as a taker), or a working event-blackout filter. This diagnostic
does NOT establish that relative-value/cointegration trading in FX
generally fails, and that claim should not be attributed to this result.
See Section 13 for the full list of apparatus caveats, all either symmetric
across arms or optimistic (i.e. none of them could be hiding a positive
result that a fix would reveal).

Per Section 5 of the pre-registration, no further iterations on this
strategy are planned -- this is one diagnostic, closed.

## 11. Trial accounting

Cumulative project-wide trial count advanced from 132 to 133 with the
defective Arm B run, then from 133 to 134 with the corrected Arm B re-run
(this wave used **two** registered trials total, not the single trial
originally committed to in the pre-registration's Section 8 -- see Section 6
for the full honesty note on why). Arm A's diagnostic re-simulation used
`--no-register` and did not increment N. The corrected run is the graded
result; the defective run's registry row is retained (not deleted) and
correctly counted in the corrected run's own DSR deflation (N=133 includes
it).

## 13. Apparatus caveats and known defects (added 2026-07-25, adversarial code review)

The reviewer verified and proved no lookahead exists in this apparatus
(Section 7). It separately surfaced ten caveats/defects, none of which
change the FAIL verdict -- each is either symmetric across both arms (so it
cannot explain the arm-to-arm swing) or optimistic (so it can only have
inflated reported performance, and the FAIL verdict already stands despite
that inflation). They are recorded here for completeness and for anyone
extending this apparatus in future work.

1. **PBO truncation (High).** `_compute_pbo` keeps any walk-forward window
   with `>= 32` rows, then truncates ALL columns to the shortest window's
   length before running CSCV. Window 13 is a 65-day stub (2026-01-01 to
   2026-04-01), so the PBO computation for both arms uses only the FIRST 65
   of each window's ~260 OOS days, seasonally aligned to Jan-Mar. Both
   arms' reported PBO (Arm A 0.8242, Arm B 0.8931) are therefore 65-day
   figures, not full-window figures. This does not change the verdict here
   -- both values would need to fall below the `< 0.5` gate threshold to
   matter, and 0.82/0.89 are far above it -- but it should be recorded as a
   known limitation of `_compute_pbo` for any future PBO computation over
   windows of uneven length.
2. **Zero execution lag (High).** The book built from data through close of
   day `i` is filled at close of day `i` -- there is no `shift()` anywhere
   in the fill chain. This is an optimistic convention shared by the ENTIRE
   FX spread-execution vertical (all Wave 2 Track B results use the same
   convention), and it is symmetric between Arm A and Arm B here. Because
   it is optimistic (real fills would be worse) and every Track B verdict
   under this convention was already FAIL, the negatives in this family
   stand *a fortiori* -- a more realistic (lagged) fill convention could
   only make the results worse, not better. Recorded so any reader of the
   reported Sharpes knows they carry a zero-lag assumption.
3. **Inert event filter (High).** The strategy docstring and the
   pre-registration's Section 3 describe a "+/-7d RBA/RBNZ entry blackout."
   In practice this filter is a no-op in 12 of the 13 walk-forward windows:
   `config/macro_calendar/cb_decisions.yaml` contains exactly one RBA date
   (2025-02-18) and one RBNZ date (2025-02-19), and the file's own header
   describes itself as a "2025-2026 starter set" with historical backfill
   explicitly deferred. The spec that was actually gated is therefore not
   fully the spec that was described in the pre-registration. Symmetric
   across arms (both use the same calendar file), so it cannot explain the
   Arm A/Arm B swing, but it means neither arm's result should be read as
   validating (or invalidating) an event-aware version of this strategy.
4. **No purge / no embargo (Medium).** `_build_windows` emits contiguous
   train/test windows with no purge gap and no embargo period, while the
   pre-registration's Section 4 constraint 5 describes "full walk-forward
   with purge/embargo." This is not a lookahead leak in THIS specific
   design -- the training segment is used only to seed `R`/`theta_0` (a
   warmup, not a fitted/selected parameter), and the one quantity carried
   forward into the OOS path is consumed strictly causally (Section 7) --
   but the pre-registration's documented claim does not match what the
   code does, and that wording should be corrected rather than repeated in
   future documents describing this apparatus.
5. **Vol-target leakage (Medium).** `_spread_sigma` includes bar `i`'s own
   return in the volatility estimate used to size the position taken at bar
   `i` (a Section 1.7-style same-bar volatility leak). This is a small
   effect (1 of 61 observations in the rolling vol window), symmetric
   across both arms, and optimistic (using same-day vol to size the same
   day's trade cannot be replicated live). Does not change the verdict.
6. **Future-conditioned spike cleaning (Medium).** `spike_clean` nulls a
   return at time `t` using `r[t] + r[t+1]` -- i.e. it uses the NEXT bar's
   return to decide whether to null the CURRENT bar. Affects roughly 6
   NZDUSD bars project-wide. This compounds with item 7 below (a nulled bar
   can silently disable the signal for an extended period). Symmetric
   across arms (the same data-cleaning step feeds both).
7. **Silent-skip defect, still live (High).** When `_regression_z` or
   `_spread_sigma` returns `None` on a rebalance date while a position is
   currently open, `spread_book` emits nothing for that date. The
   downstream simulator interprets the gap as a flat book, flattens the
   position, and charges a round-trip cost -- but the strategy's own
   internal state machine still believes it holds the original position, so
   it re-enters later (when the signal returns) and pays a SECOND round
   trip, with the holding period still measured from the ORIGINAL entry
   date. Because a single nulled input bar disables the causal Kalman
   signal for the next `lookback=120` trailing rows (roughly 24 weekly
   rebalance dates) via the same NaN-propagation mechanism documented in
   Section 1a/3a, this defect means every window's fill series is partly an
   artifact of data-quality nulls rather than pure signal logic. This is
   NOT fixed in this run (only the `R`/`theta_0` SEED robustness was fixed,
   per Section 1a) -- it affects both arms and is recorded as an open item
   for any future work in this strategy family, not as something that
   changes this diagnostic's verdict.
8. **Trade log lacks exit schema (Critical per methodology Section 11.9).**
   The persisted fills contain only `{date, pair, units, cost}` -- no
   `exit_reason`, MAE/MFE, `bars_held`, or round-trip pairing, even though
   the strategy computes a target/stop/time exit reason internally before
   discarding it. Consequently, methodology Section 12.1 trade-level
   metrics (win rate, profit factor, expectancy, avg winner/loser) and
   Section 11.6 MAE/MFE exit diagnostics could NOT be computed for either
   arm in this diagnostic, and the "OOS trade count" figures reported in
   Sections 2/3/3a are FILL counts (individual position changes), not
   round-trip trade counts. This is required infrastructure before any
   strategy in this family would be considered for live deployment; it is
   NOT blocking for a REJECT/FAIL verdict, which this diagnostic already
   reaches on statistical grounds (DSR, PBO) independent of trade-level
   detail.
9. **`trades_oos` boundary-day attribution.** On the ~4 dates that sit on a
   walk-forward window boundary, the persisted fill can come from window
   `N+1` while the gated OOS return for that date is attributed to window
   `N`, because the fill-sink's "keep first occurrence" convention and the
   return-stitcher's `d < hi` (half-open) convention differ slightly at the
   boundary. A minor bookkeeping mismatch between the fills artifact and
   the graded return series on a handful of dates; does not affect the
   gated Sharpe/PSR/DSR/PBO numbers (those are computed from the return
   series, not from the fills file).
10. **Parameter budget.** The strategy has 5 tunable parameters (lookback,
    entry_z, target_z, stop_z, max_days), 3 of which are exit-logic
    parameters, against the methodology's informal `<= 3` target. This is
    mitigated by the fact that NONE of the 5 were searched in this
    diagnostic -- each arm ran a single, fixed, a-priori configuration
    (Arm A's from the original Wave 2 registration, Arm B's `delta=1e-4`
    from the pre-registration) -- so the parameter count is a structural
    note about the strategy family, not evidence of search-driven
    overfitting in this specific result.

## 14. Deliverables and artifacts

- Adversarial code review (2026-07-25): documentation-only pass, no code or
  backtest changes; findings folded into Sections 1b, 7, 8, 10, 13 of this
  file and the corresponding sections of `docs/reports/fx/kalman_hedge_ratio_gate.md`.
- This file: `docs/strategies/research/20260722_fx_kalman_hedge_ratio_results.md`
- Working gate report: `docs/reports/fx/kalman_hedge_ratio_gate.md`
- Auto-generated gate report, CORRECTED run (current, overwrote the
  defective run's version at the same path since both used `--report`
  default): `docs/reports/fx/fx_audnzd_pairs_kalman_wave2_gate.md`
- Auto-generated diagnostic report (Arm A re-sim): `docs/reports/fx/fx_audnzd_pairs_armA_diagnostic_resim.md`
- **Arm B CORRECTED fills (graded):** `output/backtests/AudNzdPairsKalman/runs/20260725T044928Z_327859/`
- Arm B defective fills (superseded, kept for audit trail): `output/backtests/AudNzdPairsKalman/runs/20260725T042937Z_327859/`
- Arm A (diagnostic) fills: `output/backtests/AudNzdPairs/runs/20260725T042943Z_914aa5/`
- Return-concentration addendum (Section 3a, defective-run version) fills, `register=False` both arms:
  `output/backtests/AudNzdPairsKalman/runs/20260725T043758Z_327859/`,
  `output/backtests/AudNzdPairs/runs/20260725T043757Z_914aa5/`
- Return-concentration recomputation (Section 3a, CORRECTED, `register=False`,
  diagnostic-only, does not consume a trial): produced by inline analysis of
  the corrected run's per-window `_run_window_spread` output, not persisted
  as a separate FillSink run (the graded run's fills, above, are the
  source of record)
- Apparatus fix (bug fix, not a trial): `src/backtesting/signals/kalman_beta.py`
  (new `select_warmup_indices` helper; `causal_dynamic_beta` seeds `theta_0`
  from it instead of a literal `[:warmup]` slice) and
  `src/strategies/advanced/fx_audnzd_pairs.py` (`AudNzdPairsKalman._beta_path`
  seeds `R` from the same helper)
- Unit tests (fix coverage): `tests/backtesting/signals/test_kalman_beta.py`,
  `tests/strategies/test_fx_audnzd_kalman_arm.py`
- Prior apparatus fix (bug fix, not a trial, from the original defective
  run's session): `scripts/backtest_scripts/run_fx_spread_walkforward.py`
  (added run-scoped `FillSink` wiring + `register` parameter / `--no-register`
  CLI flag)
