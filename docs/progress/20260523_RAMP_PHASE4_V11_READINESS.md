# V11 Phase D Readiness - 2026-05-23

## Summary

V11 passed all four Phase D readiness gates: PSR, DSR (methodology-correct), PBO across the 5 Phase 4 variants, and one-day-lag Sharpe robustness. Two real bugs caught and fixed along the way -- a filter-state misalignment in the new `one_day_lag` engine path and the orchestrator wiring up the wrong DSR helper.

## Changes Made

### Engine: one_day_lag timing wired into the Phase 4 harness

- `HarnessState` gained `pending_targets` field.
- `run_variant` now branches on `cfg.timing_mode`. In `one_day_lag` mode, yesterday's pending targets execute FIRST at today's prices, then `plan_fn` is called against the post-execution state. The variant's filter logic (V11's rank_buffer + min_hold) now sees positions that match what is actually held when its plan eventually executes.
- 3 TDD tests in `tests/research/ramp_phase4/test_engine.py` validate the semantics: shifts execution by one bar, produces materially different P&L vs `near_close`, SAFE_MODE today clears tomorrow's pending plan.
- First implementation had a misalignment bug (caught in smoke phase, see "Bugs caught" below). The fix moves execution before plan_fn.

### Reports: PSR + DSR section in per-variant reports

- New `_format_statistical_gates(records, n_trials)` in `src/research/ramp_phase4/reports.py` emits a 4-row markdown table covering PSR (vs SR=0), DSR p-value, expected max Sharpe under null, observed annualized Sharpe.
- `build_variant_report` now accepts `n_trials: int = 20` and appends the section at the 5 bps pivot tier (same pivot as regime attribution).
- PSR uses Pearson kurtosis (normal = 3) computed independently; the DSR wrapper's excess kurtosis is NOT shared with PSR.
- 3 new tests in `test_reports.py`.

Note: the per-variant report's DSR still uses `src.backtesting.validation.deflated_sharpe.compute_deflated_sharpe`, which has the unscaled Euler-Mascheroni issue described below. The orchestrator was switched to the methodology-correct `src.backtesting.statistics.dsr.dsr`. The per-variant report should be migrated in a follow-up.

### Orchestrator: V11 Phase D readiness battery

- New `scripts/backtest_scripts/ramp_phase4_v11_readiness.py` runs 12 backtests:
  - V01 / V04 / V05 / V06 / V11 at 5 bps near_close (5 runs)
  - V11 at [0, 2.5, 7.5] bps near_close (3 runs)
  - V11 at [0, 2.5, 5, 7.5] bps one_day_lag (4 runs)
- Computes PSR vs SR=0, DSR via methodology Section 2.3 (n_trials=20, variance from the 5 Phase 4 trial Sharpes), and PBO with s=16 across the 5-variant returns matrix.
- Writes `docs/reports/ramp/20260523_phase4_v11_readiness.md` with the verdict table.
- Total wall-clock: 11.5 min for the full 9-year sweep on SP500.

## Headline result

| Gate | Result | Value | Threshold |
|---|:---:|---:|---:|
| PSR (vs SR=0) | PASS | 1.0000 | > 0.95 |
| DSR (n_trials=20) | PASS | 1.0000 | > 0.95 |
| PBO across {V01,V04,V05,V06,V11} | PASS | 0.1256 | < 0.5 |
| One-day-lag Sharpe robustness (5 bps) | PASS | nc=0.528 lag=0.580 delta=+9.79% | within 20% |

**Overall**: READY for Phase D paper deploy.

Per-variant Sharpes at 5 bps near_close exactly match Wave 1: V01 = 0.282, V04 = 0.313, V05 = 0.503, V06 = 0.278, V11 = 0.528. One-day-lag Sharpes across all four cost tiers are slightly BETTER than near_close (+6.71% / +9.23% / +9.79% / +17.51%), confirming V11 has no structural lookahead.

## Two real bugs caught this session

### 1. Filter-state misalignment in one_day_lag

The initial `one_day_lag` implementation called `plan_fn` BEFORE executing yesterday's pending plan. So `state.positions` seen by the variant on day T reflected day T-2's plan installed on day T-1 -- two execution cycles behind what would be held when the variant's new plan (P_T) actually executes on day T+1.

V11's `rank_buffer` and `min_hold` operate on `state.positions`. With the misalignment, they protected/retained names from P_{T-2} that were about to be sold by today's-pending execution of P_{T-1}, then bought them back on day T+1. The net effect was cost-amplified churn with no signal value.

Smoke evidence (3-year window 2017-2020 under the buggy implementation):

| Cost bps | near_close Sharpe | one_day_lag Sharpe | Delta % |
|---|---:|---:|---:|
| 0.0 | 0.640 | 0.688 | +7.5% |
| 2.5 | 0.524 | 0.459 | -12.3% |
| 5.0 | 0.411 | -0.029 | -106.9% |
| 7.5 | 0.318 | -0.469 | -247.5% |

At 0 bps the modes are similar (signal is fine). With cost, the churn destroys returns -- a turnover bug, not a lookahead bug.

Fix: in `one_day_lag`, execute pending FIRST, then call `plan_fn` against the post-execution state. The variant's filters now align with what's actually held at execution time. After the fix, the full-window sweep shows lag at all cost tiers SLIGHTLY BETTER than near_close (+6.71% to +17.51%).

### 2. Orchestrator used the wrong DSR helper

There are two DSR implementations in the codebase:

- `src/backtesting/statistics/dsr.py` -- methodology Section 2.3 compliant. Computes `expected_max_sharpe` as `sqrt(V[trial_sharpes]) * Euler-Mascheroni` and returns `psr(sr_hat, sr_benchmark=expected_max_sharpe, ...)`.
- `src/backtesting/validation/deflated_sharpe.py` -- simplified wrapper. Uses raw Euler-Mascheroni without scaling by V[trial_sharpes]. For n_trials=20, returns expected_max_sharpe ~= 2.02 (the expected max of 20 standard normals in Z-score units, treated AS IF it were a Sharpe).

The orchestrator initially imported the simplified wrapper. With the bug, expected max Sharpe = 2.02 vs V11's 0.528 -> DSR p_value = 1.0 -> FAIL.

After switching to the methodology-correct `dsr()`:
- V[trial_sharpes across 5 variants] = 0.0154
- sqrt(V) = 0.124
- expected_max_sharpe = 0.236
- DSR via PSR formula: sr_hat=0.528 vs sr_benchmark=0.236, n=2355, skew=-0.596, Pearson kurt=33.31 -> z=7.5 -> Phi(z)=1.000 -> PASS

The `compute_deflated_sharpe` wrapper at `src/backtesting/validation/deflated_sharpe.py` should be reviewed and either fixed or marked deprecated; the per-variant Phase 4 report still calls it. Not blocking V11 readiness but worth tracking.

## Commits

- `d2a3da5` feat(research): wire one_day_lag timing into Phase 4 engine
- `248b609` feat(research): add PSR + DSR statistical gates to Phase 4 variant reports
- `a374023` fix(research): execute pending before plan_fn in one_day_lag
- `65a25e1` report(ramp): V11 Phase D readiness -- PSR/DSR/PBO + one-day-lag sweep
- This session log

## Known Issues / Remaining Work

- **A7 paper-validation comparator must be extended for V11.** The comparator at `scripts/trading/compare_paper_vs_plan.py` models V01's filter-free plan; V11 adds `rank_buffer + min_hold + delta_threshold` whose state machine the comparator doesn't simulate. Phase D paper deploy of V11 needs the comparator extended first.
- **`compute_deflated_sharpe` wrapper at `src/backtesting/validation/deflated_sharpe.py` has the unscaled expected_max_sharpe issue.** The per-variant Phase 4 report's "Statistical gates" section still uses it. Either fix the wrapper to scale by `sqrt(V[trial_sharpes])` (requires passing trial_sharpes through, which the wrapper currently does not accept), or migrate the report to use `src.backtesting.statistics.dsr.dsr` directly. Existing tests for the wrapper would need to be updated.
- **No walk-forward purge/embargo for Phase 4 yet.** Phase 4 is single-pass over the full window. Methodology Section 3 requires walk-forward for production claims. Likely Wave 3 work.
- **V11 fails the strict 2022 OOS degradation gate** (-0.343 Sharpe vs V01). This is documented in the Wave 1 findings. EXT-OOS rescue (+0.527 vs -0.216) and overall readiness gate pass make this an acceptable trade for now, but Wave 2 V12 (BEAR-to-cash on V11 base) remains the natural follow-up.
- **Branch state**: local at `65a25e1`, divergent from `origin/ramp-phase4-turnover-regime-research` at `b9bde50` until pushed. The hard-reset earlier in the session restored Wave 1 work that was missing from local; that's now reflected in the linear ancestry.

## Validation

- `python -m pytest tests/research/ramp_phase4/ -v` -> 69 passed (no regressions; Wave 1 + 3 one_day_lag + 3 statistical gates).
- Orchestrator full-window run: 12 backtests in 11.5 min. Per-variant Sharpes match Wave 1 reports to 4 decimals.
- Lag at 0 bps very close to near_close at 0 bps (and a hair better, consistent with a clean signal under a 1-day delay), confirming the engine fix.
- Manual hand calc of expected_max_sharpe agrees with the script output (0.236), confirming the DSR formula application is methodology-correct.

## Addendum: PSR/DSR units correction (same day)

The "all four gates PASS" verdict reported above was wrong. Both PSR = 1.000 and DSR = 1.000 were spurious -- the orchestrator and reports.py passed **annualized** Sharpe into `psr()` / `dsr()` with daily `n`. The Bailey-Lopez de Prado formula's Mertens (2002) variance term applies to the **per-period** Sharpe estimator; passing annualized SR with daily n inflates the z-statistic by approximately sqrt(252), saturating PSR at 1.0 for any positive-Sharpe strategy on multi-year daily data.

The user caught it ("1.0000 seems too perfect"). Mathematical derivation in the conversation transcript confirms an ~8.5x z-inflation for V11.

### What changed in this session (units correction)

- `src/research/ramp_phase4/reports.py`: `_format_statistical_gates` renamed to `_format_psr_gate`. PSR uses `sr_daily` not `sr_annual`. DSR row removed from per-variant reports because DSR is inherently cross-variant (needs trial Sharpe distribution).
- `scripts/backtest_scripts/ramp_phase4_v11_readiness.py`: PSR + DSR call sites pass `sr_daily`. `trial_sharpes` converted to daily via `/sqrt(252)` before `expected_max_sharpe()`. Verdict logic distinguishes structural PASS vs significance PASS, producing a PARTIAL outcome instead of READY/BLOCKED binary. Sensitivity table for `n_trials ∈ {2, 3, 6, 20}` added.
- `src/backtesting/validation/deflated_sharpe.py`: replaced the previous body (which had two bugs: unscaled Euler-Mascheroni AND annualized-SR-with-daily-n) with a thin delegate to `src.backtesting.statistics.dsr.dsr` using per-period units. Falls back to a scale-aware trial-Sharpe spread when caller doesn't provide one.
- `docs/methodology/backtesting.md:185`: clarified the per-period requirement with the Mertens derivation citation and explicit warning about the saturation pitfall.
- `tests/backtesting/statistics/test_statistics.py`: added `test_psr_does_not_saturate_for_moderate_sharpe` and `test_dsr_calibrated_on_moderate_signal` as regression tests. Both use V11-realistic parameters (daily SR 0.0333, n=2355, skew -0.6, kurt 33) and assert the corrected ~0.94 / ~0.81 values. Prior tests used Sharpe ~2.4 which saturates under both conventions and could not catch this class of bug. Also fixed `test_combined_gate_acceptance_example` to pass `sr_daily` and Pearson kurtosis (not annualized + excess), modeling the corrected convention.
- `tests/research/ramp_phase4/test_reports.py`: tests renamed and assertions updated to reflect the per-variant PSR-only section.

### Corrected verdict

| Gate | Prior (inflated) | Corrected (per-period BLdP) |
|---|---:|---:|
| PSR vs SR=0 | 1.0000 PASS | **0.9442 FAIL** (delta -0.006) |
| DSR (n_trials=20) | 1.0000 PASS | **0.8108 FAIL** |
| PBO across 5 variants | 0.1256 PASS | 0.1256 PASS (unaffected) |
| One-day-lag delta at 5 bps | +9.79% PASS | +9.79% PASS (unaffected) |

DSR sensitivity table demonstrates V11 cannot pass at any plausible `n_trials`:

| n_trials | DSR |
|---:|---:|
| 2 | 0.9188 |
| 3 | 0.8984 |
| 6 | 0.8655 |
| 20 | 0.8108 |

The limit is V11's Sharpe magnitude (0.528 annualized over 9 years), not the multi-trial correction. The corrected readiness doc at `docs/reports/ramp/20260523_phase4_v11_readiness.md` reflects this.

### Revised verdict

V11 is **PARTIAL READINESS**: structurally sound (no overfitting per PBO, no lookahead per one-day-lag) but with weak absolute significance after multi-trial correction. Three paths forward documented in the readiness doc: advance V11 to paper with the caveat documented, fall back to V05 (similar significance situation), or pause Phase D for Wave 2 (V12 BEAR-to-cash on V11 base).

### What we learned about the codebase convention

The Homeguard PSR/DSR helpers (`src/backtesting/statistics/psr.py`, `dsr.py`) were always correct as written -- the bug was only at callsites. The existing test `test_combined_gate_acceptance_example` used a Sharpe ~2.4 strategy whose PSR saturates at 1.0 under either convention, so the test couldn't catch the units issue. The new regression tests use V11-realistic moderate-Sharpe parameters where the conventions produce meaningfully different results. The methodology spec line 185 had ambiguous language ("per-period OR annualized") that did not match the formula's actual derivation; that's now corrected.
