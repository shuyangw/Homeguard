# Kalman Dynamic Hedge Ratio -- Scoping Diagnostic Pre-Registration

**Date:** 2026-07-22 | **Status:** LOCKED 2026-07-22 (approved; no post-hoc edits to arms/params/gate) | **Owner:** main-loop -> strategy-lead for the verdict

Pre-registration per the North Star. This is a **SCOPING DIAGNOSTIC, not a rescue
attempt.** Read Section 2 before interpreting any result.

## 1. The question this answers (and the question it does NOT answer)

Wave 2 rejected three pairs/relative-value strategies (#35 AudNzdPairs OOS -0.24,
#37 CointScanner -0.24, #30 VolRatioPair -0.48). All three estimated their hedge
ratio with a **static trailing-window OLS**. Per the North Star principle "a
negative bounds the SPECIFICATION you tested, not the asset class," those results
strictly bound *static-OLS-hedged* pairs trading -- they do NOT establish that the
cointegration/relative-value MECHANISM fails, because a structurally mis-specified
hedge ratio is a plausible alternative explanation for the deficit.

**Question:** does a causal, time-varying (Kalman-filtered) hedge ratio materially
change the pairs verdict?

**NOT the question:** "can we make #35 pass?" The expected answer is NO (Section 6
registers that prediction in advance). The VALUE here is closing a scoping gap so
the pairs negative can be stated honestly, not manufacturing a pass.

## 2. Why only ONE strategy, and why this one

Retrofitting Kalman across all five estimator-bearing strategies and reporting
whichever improves would be a specification search (Section 2.2), would add 5
trials, and would raise the DSR bar for everything. One representative test answers
the scoping question at a cost of one trial.

**#35 AudNzdPairs** is the fairest test: AUDUSD/NZDUSD is the most canonical
cointegrated pair in the book (two commodity currencies, shared risk factor), and
beta drift is *theoretically expected* there -- RBA/RBNZ policy divergence shifts
the relationship, which is exactly the condition a time-varying estimator is for.
(#42 RoroRegimeSpread has better surviving stats but its failure mode was COST, and
dynamic hedging tends to ADD turnover -- wrong tool for that failure.)

## 3. The two arms (identical except the hedge-ratio estimator)

Everything not listed as "differs" is byte-identical between arms: universe
(AUDUSD/NZDUSD), weekly rebalance, `entry_z=2.0`, `target_z=0.5`, `stop_z=3.25`,
`max_days=20`, RBA/RBNZ +-7d entry blackout, `_strength` scaling, spread-sigma
sizing, walk-forward windows, cost model, cost_mults (1.0, 1.5).

**ARM A (baseline, ALREADY RUN -- do NOT re-run).** Static trailing 120d OLS of
`ln(AUDUSD)` on `ln(NZDUSD)`; z = standardized residual over that window.
Recorded result: OOS Sharpe **-0.24** (1.5x: **-0.30**), PSR 0, DSR 0, PBO 0.82,
S&P corr 0.04 (N was 109 at its gate). Cite these; re-running is not a trial.

**ARM B (new, 1 trial).** Replace ONLY `_regression_z`'s beta estimation with a
causal dynamic linear regression:
- Observation: `ln_a[t] = alpha_t + beta_t * ln_b[t] + e_t`, `e_t ~ N(0, R)`
- State: `theta_t = [alpha_t, beta_t]' = theta_{t-1} + w_t`, `w_t ~ N(0, Q)`
- **Q fixed a priori:** `Q = (delta/(1-delta)) * I` with **`delta = 1e-4`** (the
  standard textbook default, Chan). NOT tuned. NOT fitted to the test set.
- **R fixed causally:** variance of the OLS residual computed on the TRAINING
  window only, per walk-forward window; held fixed within that window.
- **Init:** `theta_0` = OLS (alpha, beta) on the FIRST 120 days of the training
  window; `P_0 = 1e-3 * I`. Warmup entirely inside the training window.
- **z is constructed IDENTICALLY to Arm A** (residual standardized over the same
  120d trailing window, using the Kalman beta instead of the OLS beta), so the ONLY
  thing that varies is the hedge-ratio estimator and any difference is attributable
  to it.

**Explicitly NOT run** (would be separate trials; naming them here prevents them
being slipped in later as "we also tried"): the standardized-innovation z
(`e_t / sqrt(S_t)`), any other `delta`, any other pair, and any smoother.

## 4. Hard constraints (Section 2.1 / strategy-lead Phase 6.5 guardrails)

1. **FILTER ONLY -- the forward causal pass.** Any RTS / fixed-interval SMOOTHER
   revises past states using future observations and is an automatic REJECT.
2. **No Q/R tuning.** The values in Section 3 are final. If Arm B fails, we do NOT
   try another `delta` -- that is the specification search this document exists to
   prevent.
3. **BOTH arms reported**, always, including in the summary line. Reporting only the
   better arm is selective reporting.
4. **Mandatory fills** (run-scoped FillSink -> non-empty `trades_oos.csv.gz`).
5. Full walk-forward, same windows as Arm A. **CORRECTION (2026-07-25): this line
   originally read "with purge/embargo". `_build_windows` in fact emitted
   CONTIGUOUS train/test windows with no purge gap, so the claim was wrong. No
   result changes: for this design 0 purge is CORRECT, because the training
   segment is warmup-only (it seeds R/theta_0 and fills rolling windows, nothing
   is fitted or selected on it) and every feature is causal. `purge_days` now
   exists on `_build_windows` and MUST be set non-zero by any spec that fits,
   optimizes, or selects parameters on the training segment.**

## 5. Pre-committed decision rule

- The verdict "changes" ONLY if Arm B clears the **full combined gate**: OOS Sharpe
  > 0 AND positive at 1.5x cost AND PSR > 0.95 AND DSR > 0 (deflated at the current
  cumulative N) AND PBO < 0.5.
- **Any improvement short of that is NOT a pass and NOT a "promising lead."** If
  Arm B comes in at, say, -0.05 vs Arm A's -0.24, the finding is reported as
  "the static estimator explains part of the deficit; the mechanism still fails" --
  and the pairs negative stands, now correctly scoped.
- No further iterations on this strategy either way. This is one diagnostic.

## 6. Registered prediction (so we cannot claim we expected whatever happens)

**Arm B fails.** Rationale: closing a ~1.3 Sharpe gap (from -0.24 to the ~1.0
SR_zero bar at N~123) via an estimator swap is not credible; estimator upgrades
typically move Sharpe by 0.1-0.3. Additionally, a beta that updates every period
generally RAISES hedge-leg turnover, which is a headwind after costs -- though it
can also REDUCE turnover versus rolling OLS by removing window-edge beta jumps.
Direction of the turnover effect is genuinely uncertain and is a reported metric,
not an assumption.

## 7. Required diagnostics (beyond the standard gate metrics)

- **Beta-path comparison (the cheap pre-check, report FIRST):** correlation, mean
  absolute difference, and drift/volatility of the Kalman beta path vs the OLS beta
  path. **If the two paths are near-identical, the estimator was never the binding
  constraint** -- that alone answers the scoping question and should be stated
  plainly regardless of the Sharpe outcome.
- **Turnover and realized cost drag, both arms** (the mechanism check per Phase 6.5
  guardrail 5), not just Sharpe.
- Standard set: OOS Sharpe 1x/1.5x, PSR, DSR, PBO, trade count, S&P corr.

## 8. Trial accounting

Arm A is already counted. **This wave adds exactly 1 trial (Arm B).** Cumulative N
advances by 1. The DSR for Arm B is deflated at the updated cumulative N. N is
never reduced to help Arm B pass.

## 9. Deliverables

- `docs/strategies/research/20260722_fx_kalman_hedge_ratio_results.md` (durable)
- `docs/reports/fx/kalman_hedge_ratio_gate.md` (working)
- Tracker: update #35's row with the scoped result and, if the beta paths were
  near-identical, record that the static-estimator explanation is eliminated.
