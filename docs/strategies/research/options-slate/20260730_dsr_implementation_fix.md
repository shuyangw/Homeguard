# DSR Implementation Fix -- Two Defects, Opposite Directions

**Date**: 2026-07-30
**Scope**: engineering + integrity only. **No backtest was run, no P&L computed,
no verdict re-graded.**
**Status**: both defects fixed and committed; blast radius enumerated below.

---

## 0. Headline

| | Bug 1 (`n_trials_project_wide`) | Bug 2 (`validation/deflated_sharpe.py`) |
|---|---|---|
| Nature | Returned 0 forever | Unit error: quantile minus Sharpe |
| Direction | **Under-deflated** (too generous) | **Over-deflated** (too harsh) |
| Magnitude | N = 0 vs 367; hurdle 0.0000 vs 1.0268 | Hurdle 1.97 vs 0.75 at N=18 (2.6x); 3.04 vs 1.03 at N=349 (3.0x) |
| Recorded verdicts affected | **3 artifacts** | **3 artifacts** |
| Both confirmed? | Yes | Yes |

**The most important correction to the brief**: both bugs are real, but their
recorded blast radius is *far* smaller than the brief assumed. The brief states
they "corrupt strategy verdicts across futures, FX and equities." They do not.
Essentially the whole futures/FX/equity corpus went through a **third** path
(`statistics/dsr.py`), which has neither defect. See Section 4 -- and Section 4.4,
which identifies the defect that *did* corrupt that corpus, and which is neither
of the two in this brief.

---

## 1. Bug 1 -- `n_trials_project_wide()` has always returned 0

### 1.1 The implementation defect (unambiguous)

`src/experiments/registry.py:309-321` (pre-fix):

```sql
SELECT COALESCE(SUM(combinations_in_run), 0) FROM runs
WHERE agent_name = 'backtest-optimizer'
```

Verified against the live `output/experiments.duckdb` (500 rows). Writers:

| agent_name | rows |
|---|---:|
| `futures-harness` | 241 |
| `robustness-runner` | 78 |
| `backtest-driver` | 66 |
| `futures-harness-walkforward` | 57 |
| `fx-harness-walkforward` | 39 |
| `fx-harness` | 6 |
| `fx-spread-walkforward` | 5 |
| `backtest-runner` | 4 |
| `trial-count-reconstruction` | 4 |

`backtest-optimizer`: **0 rows, ever.** The brief's figures match the database
exactly. The function returned 0 on every call in the registry's lifetime, and
`expected_max_sharpe()` returns `0.0` for `N < 2`, so any DSR taking N from this
path received **zero deflation**.

`combinations_project` was 0 everywhere for the same reason:
`src/backtest_runner.py:680` faithfully stored a function that always returned
zero. (That column has since been backfilled to `max = 367` by
`scripts/maintenance/backfill_trial_counts.py`; this fix is consistent with it
and does not undo it.)

### 1.2 The policy question -- NOT settled silently

The superseded test `tests/experiments/test_registry.py:108-135` contained:

```python
# A driver run doesn't count -- only optimizer runs.
append_run(..., agent_name="backtest-driver", combinations_in_run=5)
assert n_trials_project_wide(db_path) == 0
```

So the agent-name filter was **not a typo**. It encoded a deliberate policy:
only optimizer sweeps count toward the trial budget. That policy is a real
methodology judgment and is not mine to overturn quietly. What I did instead:

- Made the rule **explicit and selectable at the call site** rather than buried
  in a hardcoded agent-name filter.
- Implemented `TRIAL_RULE_OPTIMIZER_ONLY` **structurally**
  (`parent_run_id IS NOT NULL OR phase LIKE 'optimization%'`), never as an
  agent-name allowlist -- an allowlist is exactly the failure mode that caused
  this.
- Set `TRIAL_RULE_EVERY_SPEC` as the **default**, justified below.

**Quantification (the whole decision, measured on the live registry):**

| Rule | N | E[max Sharpe] hurdle at sigma=1/sqrt(8.3y) |
|---|---:|---:|
| Old agent-name filter | **0** | **0.0000** |
| `TRIAL_RULE_OPTIMIZER_ONLY` (structural) | **0** | **0.0000** |
| `TRIAL_RULE_EVERY_SPEC` (**default**) | **367** | **1.0268** |

**The decisive empirical fact: the optimizer-only policy yields 0 under *any*
marker, not merely the broken one.** No optimizer sweep has ever written a
registry row. `make_trial_callback` is wired at `src/backtest_runner.py:960`,
but every real campaign -- futures, FX, RAMP -- used its own harness with its
own `agent_name`. The policy was never in force; operationally it has always
meant "count nothing, deflate nothing."

So the default is not chosen merely on the Bailey & Lopez de Prado reading
(N is the number of configurations whose Sharpe was *observed* -- every draw the
maximum could have been taken over, not only grid-search draws). It is chosen
because **the alternative is not a stricter policy, it is no gate at all**.
Selecting `optimizer_only` today would return 0, which now raises rather than
silently disabling deflation.

Noted for Shuyang: this convention has now been challenged twice
independently -- here, and by the parallel task that overrode
`trial_count_treatment=NOT_PROMOTED_no_selection_trial_added` on 78 RAMP
robustness rows for the same reason. **Two independent challenges is a signal,
not a consensus.** The policy remains a live decision; the code now makes it
visible instead of accidental.

### 1.3 What changed

`src/experiments/registry.py`:

```python
N = max( SUM(COALESCE(combinations_in_run, 1)), MAX(combinations_project) )
```

- **No allowlist.** Any new writer is counted the day it appears.
- **NULL count -> 1.** A writer that omits `combinations_in_run` still ran a
  trial. Defaulting to 0 would silently under-deflate -- the original bug's
  direction.
- **Explicit 0 honored.** The backfill's "exact rerun of an already-counted
  spec" marker.
- **`MAX(combinations_project)` is a floor**, so the reconstruction is never
  discarded and N never shrinks. A floor exceeding the sum logs a warning.
- **Zero is loud**: `TrialCountUnavailableError`. `strict=False` is available
  for bookkeeping call sites, and `backtest_runner.py` uses it (plus adds the
  row's own contribution, keeping the stored column a running cumulative count).

---

## 2. Bug 2 -- unit error in `validation/deflated_sharpe.py`

Pre-fix, `src/backtesting/validation/deflated_sharpe.py:75-98`:

```python
expected_max_sr = sqrt(2*log_n) - (gamma_em + log(log(n_trials) + pi/2)) / (2*sqrt(2*log_n))
...
dsr_stat = (sr_annual - expected_max_sr) / se_sr
```

`expected_max_sr` is the expected maximum of a **standard normal** -- a pure
quantile in units of sigma_SR. `sr_annual` is an **annualized Sharpe**.
Subtracting them is dimensionally invalid. The correct benchmark is
`SR_0 = sigma_SR * E[max Z]`; the sigma_SR factor was **absent entirely**, so
the hurdle could not even vary with trial dispersion.

Measured magnitude:

| N | old hurdle | correct hurdle (sigma = 1/sqrt(years)) | ratio |
|---:|---:|---:|---:|
| 18 (6.5 y) | 1.9733 | 0.75 | 2.6x |
| 349 (8.0 y) | 3.0447 | 1.03 | 3.0x |

Confirmed in-repo: `docs/reports/ramp-long-calls/20260402_statistical_validation.json`
records `expected_max_sharpe: 1.9733` at `n_trials: 18`, `p_value: 1.0`.

### 2.1 Consolidation choice

`src/backtesting/statistics/__init__.py` already declares its functions "the
project's single source of truth ... Callers must use them rather than
reimplementing." I therefore made `validation/deflated_sharpe.py` a **thin
adapter that delegates** to `statistics/dsr.py`, rather than retiring it.

Reasons:

1. **It deletes the wrong formula outright.** Delegation removes the bad code;
   deprecation would leave it callable. The brief asked to make the loser
   "impossible to call by accident" -- there is now no loser to call.
2. **The adapter's signature has no equivalent in `statistics/`.**
   `compute_deflated_sharpe(returns, n_trials)` takes a raw return series and
   derives Sharpe/skew/kurtosis itself; `statistics.dsr.dsr()` requires
   pre-computed moments. Retiring the adapter would push that arithmetic into
   every future caller -- a new duplication risk.
3. **Migration cost is zero either way** (only `combined_gate` calls it), so the
   decision rests entirely on 1 and 2.

sigma_SR resolution: empirical `Var(trial_sharpes, ddof=1)` when the caller
supplies `trial_sharpes`; otherwise the prior `1/sqrt(years)`. Both branches call
the shared `expected_max_sharpe()`; the prior is injected via the two-point
`{0, s*sqrt(2)}` construction (whose ddof=1 variance is exactly `s^2`), so **no
formula is duplicated anywhere**.

`DSRResult` gains `dsr_probability` -- the actual DSR in [0,1], gated at 0.95 per
methodology Section 2.5. `p_value` is now its complement rather than a normal
tail on a mis-scaled z-statistic.

---

## 3. Third implementation found (not in the brief)

`docs/reports/general/20251205_BACKTEST_VALIDATION_ANALYSIS.md:311-340` contains
an inline Python snippet with a **third, hand-rolled DSR**:

```python
E_SR_max = np.sqrt(2 * np.log(k_trials))                      # no Euler-Mascheroni correction
SE_SR    = np.sqrt(1/n_years + SR_observed**2/(2*n_years))    # Lo (2002), no skew/kurt
DSR      = stats.norm.cdf(z_score)                            # 0.9293
```

It matches neither path. It proposed creating `scripts/analysis/calculate_dsr.py`
-- **that file and directory do not exist**, so the number was computed by hand
in the document. Its recorded **DSR 0.9293 for OMR/MP** is not reproducible from
any code in the repo and should be treated as unsourced.

---

## 4. Blast radius

Method: full sweep of `docs/reports/`, `docs/progress/`, `docs/strategies/`,
`docs/archive/`, and `output/` for recorded DSR values, classified by the
field-signature of each implementation. **No verdict was re-graded.**

### 4.1 Bug 2 (too STRICT) -- 3 artifacts, all one campaign

| Artifact | Recorded | Direction | Confidence |
|---|---|---|---|
| `docs/reports/ramp-long-calls/20260402_statistical_validation.json` | `expected_max_sharpe 1.9733`, `dsr_statistic -33.92`, `p_value 1.0`, N=18 | Over-deflated | **High** |
| `docs/reports/ramp-long-calls/20260402_statistical_validation.md` | same, rendered | Over-deflated | **High** |
| `docs/reports/ramp-long-calls/20260402_statistical_validation_analysis.md` | `p = 1.000`, `DSR statistic -33.92` | Over-deflated | **High** |

**Verdict impact: none.** Observed Sharpe was **-0.766**. A negative Sharpe fails
under any correct benchmark, so the REJECT stands. The *number* is wrong; the
*conclusion* is not. `combined_gate()` has **zero production callers** anywhere
in `src/` or `scripts/`, and the generating script
(`ramp_long_calls_validation.py`) has been deleted -- only a stale `.pyc`
remains. This path never touched futures, FX, or equities.

### 4.2 Bug 1 (too LENIENT) -- 3 artifacts

Only three recorded artifacts consumed `n_trials_project_wide()` as a gate input:

| Artifact | Recorded | Direction | Confidence |
|---|---|---|---|
| `docs/reports/fx/costsens/fx_carry_seatbelt_daily_0.5x.md:27` | "Trial count 2 (project-wide `n_trials_project_wide()` = 0, + 2 local configs)" | Under-deflated | **High** (states it openly) |
| `docs/reports/fx/costsens/fx_carry_seatbelt_weekly_0.5x.md:27` | same | Under-deflated | **High** |
| `docs/reports/fx/costsens/fx_london_breakout_0.5x.md:37-41` | notes `n_trials_project_wide()` returns 0 | Under-deflated | **High** |

**Verdict impact: none.** All three recorded DSR at or near **0.0000** -- they
FAILED even with zero deflation. Correcting N upward can only lower DSR further.
These verdicts are directionally safe.

A prior audit reached the same conclusion independently:
`docs/strategies/research/20260726_cross_estate_audit.md:21` -- "**Clean.** One
caller remains, `backtest_runner.py:680`, and it writes `combinations_project`
as metadata, not a gate input."

### 4.3 Unaffected by both bugs -- the large majority

Everything else went through `statistics/dsr.py`, which has neither defect:
all `docs/reports/futures/*_READINESS.md`, the ~40 `docs/reports/fx/**` gate
reports, `docs/reports/ramp/*` Phase-4 readiness, all
`output/backtests/**/gate.json`, `output/backtests/verdicts/verdicts.json`,
`output/deconcentration/*.json`, `output/ohlc/*.json`, `output/tierb/*`, and the
`docs/progress/` and `docs/strategies/research/` narratives of those runs.
**Confidence: high** (they record `dsr` as a probability in [0,1] alongside
`psr`/`pbo`/`oos_sharpe`, and their generators import `statistics.dsr`).

### 4.4 The defect that DID corrupt the futures/FX corpus -- and it is neither of these

In **every** `output/backtests/futures/**/gate.json`, `verdicts.json`,
`output/deconcentration/*.json` and `output/fx_trend_gate.json`, the recorded
`dsr` is **byte-identical to `psr`**. That means `expected_max_sharpe()` returned
0.0 -- a degenerate single-element trial-Sharpe list or `n_trials < 2` -- so
those runs were **undeflated**. Same direction as Bug 1 (too lenient), different
root cause, and **already documented** at
`docs/progress/SESSION-HANDOFF-futures-retest-2026-07-11.md:20` as "THE BIG
FINDING".

**Any recorded `DSR 1.0` in the futures/deconcentration corpus should be read as
an undeflated PSR, not a DSR.** That is a larger population than both bugs in
this brief combined, and it is out of scope here -- flagged, not fixed.

### 4.5 Unreliable for a third reason (carried forward, not re-graded)

- **RAMP Wave-3 family DSR** -- the trial chain was reset from 36 to 1, and
  `docs/strategies/RAMP_VARIANTS.md` records "Family DSR passes at n_trials <= 12
  (Wave-3 reset), fails at >= 36." A verdict flipped on the reset. Direction:
  **too lenient**. Confidence: high (the doc states it).
- **OMR/MP `DSR 0.9293`** -- the doc-only third implementation (Section 3).
  Direction: **unknown**, formula unsourced. Confidence: high that it is
  unreproducible.

---

## 5. Regression tests

`tests/backtesting/validation/test_dsr_consolidation.py` (14 tests) and the
rewritten `tests/experiments/test_registry.py` trial-count block (7 tests).

**Negative-control proof of power.** I restored the superseded
`deflated_sharpe.py` from git and re-ran the new suite against it:

```
11 failed, 3 passed
```

The suite demonstrably detects the bug it was written for. (The 3 that pass are
the pure-`statistics/dsr.py` analytic fixtures, which the old module never
touched.)

For Bug 1 the equivalent proof is direct: the old SQL returns **0** against the
live registry where the new function returns **367**. Every assertion in the
rewritten registry test returns 0 under the old implementation.

Coverage required by the brief:

| Case | Test |
|---|---|
| **Known-good analytic** | `test_expected_max_sharpe_reproduces_the_options_prereg_table` -- N=9/19/31/38 -> 0.4109/0.5074/0.5638/0.5860 to 5e-4 |
| **N = 0** | `test_n_trials_below_two_yields_no_deflation[0]`; registry: `test_n_trials_empty_registry_is_loud` (raises) |
| **N < 2** | `test_n_trials_below_two_yields_no_deflation[1]` |
| **Empty ledger** | `test_n_trials_empty_registry_is_loud` / `..._non_strict_returns_zero` |
| **Negative control** | `test_benchmark_is_a_sharpe_not_a_standard_normal_quantile` (asserts 1.9733 is wrong and ~0.75 is right); `test_the_old_formula_would_have_killed_a_viable_candidate`; `test_a_strong_honest_sharpe_passes_the_gate` |
| **Anti-allowlist** | `test_n_trials_counts_an_agent_name_nobody_has_invented_yet` |
| **Not-too-lenient** | `test_deflation_actually_bites_a_marginal_candidate` (Sharpe 0.55 at N=349 must FAIL) |
| **One formula only** | `test_validation_module_contains_no_second_dsr_formula` (source guard) |

Full run: **144 passed** across `tests/experiments`, `tests/backtesting/validation`,
`tests/backtesting/statistics`. Wider `tests/backtesting` run: 863 passed,
7 failed -- all 7 pre-existing and environmental (`FileNotFoundError`, no market
data in this worktree; plus one `test_parallel_equals_serial` dsr mismatch
**verified to fail identically at HEAD with my changes reverted**). Three
collection errors are missing optional deps (`dukascopy_python`, `holidays`) and
a stale import name, none touching changed modules.

---

## 6. Where the brief was wrong

1. **"Both corrupt strategy verdicts across futures, FX and equities -- not just
   options."** **False for Bug 2**, and overstated for Bug 1. Bug 2's entire
   recorded footprint is 3 files from one options-overlay campaign;
   `combined_gate()` has no production callers at all. Bug 1's gate footprint is
   3 FX cost-sensitivity reports. The futures/FX/equity corpus was computed by
   `statistics/dsr.py` and is untouched by both.
2. **"Two DSR implementations must not both stand."** There were **three** -- the
   doc-only hand-rolled formula in
   `docs/reports/general/20251205_BACKTEST_VALIDATION_ANALYSIS.md` (Section 3).
3. **The agent-name filter was not simply a bug.** It encoded a deliberate
   policy, evidenced by the test that asserted it. Treating it as a typo would
   have silently overturned a prior methodology decision (Section 1.2).
4. **"Bug 1 => under-deflated ... Bug 2 => over-deflated"** is correct as
   *direction*, but in every recorded case the affected verdict was already a
   FAIL by a wide margin, so **no recorded verdict flips** from either fix.
5. **The brief's premise that these two bugs are the main threat to DSR
   integrity is wrong.** The `dsr == psr` degeneracy (Section 4.4) affects far
   more recorded results than both combined.
6. **Minor**: the brief cites "roughly 3x" disagreement. Verified: 2.6x at N=18,
   3.0x at N=349 -- it grows with N.

---

## 7. Remaining work (not done here)

1. **Decide the counting policy** (Section 1.2). The code now surfaces it;
   Shuyang should ratify `every_spec` or revert to `optimizer_only` knowing it
   currently means N=0.
2. **Re-grade nothing yet** -- but the futures/FX corpus carrying `dsr == psr`
   (Section 4.4) is the highest-value re-grade target, not the two bugs here.
3. **RAMP Wave-3 family DSR** on the un-reset count (Section 4.5).
4. **Migrate remaining N sources.** `get_campaign_trial_distribution()`
   (`walkforward_common.py`) still derives N independently by counting rows with
   a numeric `oos_sharpe` -- 101 of 500 rows, giving N=141 against the
   reconstruction's 349. Two live definitions of N remain.
5. **Purge or annotate** the doc-only DSR in
   `20251205_BACKTEST_VALIDATION_ANALYSIS.md`.

---

## 8. Commits

- `6a1e36b` fix(dsr): n_trials_project_wide() has always returned 0
- `1015e95` fix(dsr): consolidate the two DSR implementations onto statistics/dsr.py

Registry (`output/experiments.duckdb`) was **read-only throughout**; no rows were
added, altered, or fabricated.
