# Homeguard Lifetime DSR Trial Count (N) -- Reconstruction

**Date**: 2026-07-28
**Status**: BLOCKING input for options Wave-1 grading. Supersedes the `N = 9`
used in the options-slate pre-registration chain.
**Scope**: integrity/accounting only. No backtest, no P&L, no strategy was run.

---

## 0. Headline

| Quantity | Pre-registration | Reconstructed |
|---|---|---|
| Trial count N feeding the options DSR | 9 | **349** (range **179 - 621**) |
| Sharpe-estimate sigma (window) | 0.27 (13.7 y) | **0.347** (8.3 y) |
| **E[max Sharpe \| null] -- the Wave-1 hurdle** | **0.41** | **1.02** (range **0.95 - 1.08**) |

The registered hurdle is understated by a factor of **~2.5x**. Both registered
inputs were wrong in the same direction, and they compound.

**The reconstruction's uncertainty does not matter for the decision.** N is
genuinely uncertain across a 3.5x span (179 to 621), but because
E[max Sharpe | null] grows only as `~sqrt(2 ln N)`, that entire span moves the
hurdle by **14%** -- from 0.95 to 1.08. Every defensible accounting rule puts
the bar near 1.0 and none puts it anywhere near 0.41. **Grading Wave 1 against
0.41 would pass noise.** A Wave-1 candidate must show an in-sample Sharpe near
**1.0** before it is distinguishable from the best of a null search of this
size.

---

## 1. Method

### 1.1 The formula, and why it is not a third implementation

The repo has two DSR implementations. I read both and reused one; neither was
forked.

- **`src/backtesting/statistics/dsr.py`** -- `expected_max_sharpe()`. The
  Bailey-Lopez de Prado expected maximum under the null:

  ```
  SR_0 = sigma_SR * [ (1 - gamma) * Phi^-1(1 - 1/N)
                      + gamma * Phi^-1(1 - 1/(N*e)) ],   gamma = 0.5772156649
  ```

  It takes `sigma_SR` as `sqrt(Var(trial_sharpes, ddof=1))` -- the *empirical*
  dispersion of the Sharpes actually observed across trials.

- **`src/backtesting/validation/deflated_sharpe.py`** -- `compute_deflated_sharpe()`.
  A different, older approximation used only by the 2026-04 RAMP options
  campaign. **It omits `sigma_SR` entirely** (implicitly 1), so at N=18 it
  produced `expected_max_sharpe = 1.973` in annualized Sharpe units. That is
  not the same statistic and is enormously more conservative. It is not used
  here; flagged in Section 6.

The options slate has **no trial-Sharpe distribution yet** (nothing has been
run), so the empirical dispersion is unavailable and the pre-registration's
theoretical prior `sigma_SR = 1/sqrt(years)` -- the asymptotic standard error
of an annualized Sharpe estimate under the null -- is the correct substitute.

I injected that sigma into the repo's own function without modifying it: for a
two-point sample `{0, s*sqrt(2)}` the `ddof=1` variance is exactly `s^2`, so
`expected_max_sharpe([0, s*sqrt(2)], N)` returns the Bailey-LdP maximum at
`sigma_SR = s`.

**Sanity check against the pre-registration's own table** (13.7 y, sigma 0.2702):

| N | reproduced here | pre-reg doc |
|---|---|---|
| 9 | 0.4109 | 0.41 |
| 19 | 0.5074 | 0.51 |
| 31 | 0.5638 | 0.56 |
| 38 | 0.5860 | 0.59 |

Exact to the docs' stated precision. **The pre-registration's multiplier is
correct; only its two inputs (N and the window) were wrong.**

### 1.2 The dedup rule (the central judgment call)

The registry mixes two writer conventions and neither identity field works
alone -- this is why the count could not simply be read off:

- `backtest_runner` / `robustness-runner` rows (equity) carry a real
  `config_sha` but **`params` is NULL**.
- futures/fx harness rows carry **`config_sha = 'unknown'`** but a full
  `params` JSON.

So the spec key is convention-aware:

```
if config_sha not in (NULL, 'unknown'):  key = (strategy, config_sha, notes)
elif params:                             key = (strategy, params - {dates,
                                                 trial_count_project_wide},
                                                 phase, window)
else:                                    key = (strategy, phase, window)
```

Stripping `dates` and `trial_count_project_wide` prevents a rerun under a
different bookkeeping counter from masquerading as a new specification.

**Rule applied (`CENTRAL`), stated explicitly:**

1. **One trial per distinct evaluated specification.** Every config whose
   Sharpe was *looked at* counts, whether or not it was promoted.
2. **Parameter sweeps count at full cardinality.** A 78-point robustness
   sweep contributes 78, not 1. This follows the methodology rule that a sweep
   multiplies a candidate's contribution by its sweep cardinality, and it is
   the statistically correct treatment: DSR asks how many draws the maximum
   was taken over.
3. **Exact reruns of an identical spec do NOT add a trial.** 191 of the 496
   rows are byte-identical respecifications -- bug-fix reruns, apparatus
   corrections, cost-leg replays. Per the methodology's explicit carve-out,
   *correcting the apparatus and re-running the same pre-registered spec is
   not a search*. Counting them would inflate N for work that involved no
   researcher degrees of freedom.
4. **Harness fixtures are excluded.** `ZeroForecastStub` and
   `ParamForecastStub` (31 rows) are engine unit-test fixtures -- a
   zero-forecast and a constant-forecast stub on a 2-instrument, 6-month
   window. **Verified**: they are named stubs, produce no `oos_sharpe`/`psr`/
   `dsr`, and carry no economic hypothesis. They are not trials.

Rules 2 and 3 pull in opposite directions, which is why the bounds are wide.

### 1.3 The overlapping-window rule

DSR's N counts trials evaluated against **overlapping** data. Applied by
intersecting each run's `[window_start, window_end]` with the options window
**2017-01-01 -> 2025-12-31**.

**This provides almost no relief, and that is the honest finding.** Nearly
every Homeguard campaign window spans 2017-2025: futures 2010-2026, FX
2011-2026, RAMP equity 2017-2026. Only **6 distinct specs** (30 rows) are
calendar-disjoint -- three futures single-pass 4-year windows ending before
2017 (`2010-06-07..2014-06-07`, `2011..2015`, `2012..2016`) for
`CarverMomentum` and `FuturesCarry`.

```
N_total       = 367
N_overlapping = 349     (= 367 - 18 rows'-worth of pre-2017-only specs)
```

**N_overlapping = 349 is the number that should feed the options DSR.**

---

## 2. What was counted, and from where

### 2.1 On-ledger (`output/experiments.duckdb`, 496 rows)

| Block | Rows | Distinct specs | Overlapping specs |
|---|---:|---:|---:|
| `futures_singlepass` (engine/dev + WF component windows) | 210 | 101 | 83 |
| `futures_wf_graded` (walk-forward, gradeable) | 57 | 22 | 22 |
| `fx_wf_graded` | 44 | 31 | 31 |
| `fx_singlepass` | 6 | 6 | 6 |
| `equity_robustness_sensitivity` (RAMP V26/V28/V31 sweeps) | 78 | 78 | 78 |
| `equity_wave3_readiness` | 57 | 28 | 28 |
| `equity_wave3_walkforward` | 9 | 5 | 5 |
| `demo_ma_crossover` | 4 | 3 | 3 |
| `fixture` (ZeroForecastStub / ParamForecastStub) -- **excluded** | 31 | 4 | 0 |
| **On-ledger total (counted)** | **465** | **274** | **256** |

### 2.2 Off-ledger (absent from the registry entirely)

| Block | Trials | Source | Window |
|---|---:|---|---|
| Futures SP-A/B/C/E + pre-campaign carry/crypto sweep | 40 | `src/backtesting/walkforward_common.py:38` `CAMPAIGN_CUMULATIVE_TRIALS`, itemized 7+4+4+14+11 in the comment at L25-37 | 2010-06 -> 2026-02 |
| RAMP equity v0-detector trial chain | 36 | `docs/strategies/RAMP_VARIANTS.md` (canonical file only in git history) audited decomposition: V11+pre-V11 22 + V12+sens 5 + V12c 1 + V13 1 + V14a/b/c 3 + V14a tau sens 2 + V14c dampen sens 2 | 2017 -> 2026 |
| RAMP options overlay campaign | 16 | `git show 910c567:docs/progress/20260402_RAMP_OPTIONS_PIPELINE_ARCHIVE.md` -- "31 candidates, 16 tested across 3 batches" | IS 2018-07 -> 2024-12, OOS 2025 |
| OpEx pinning | 1 | `docs/strategies/20251230_OPEX_PINNING_STRATEGY_STATUS.md:92` | Nov-2024 -> Dec-2025 data |
| **Off-ledger total** | **93** | | |

### 2.3 Build-up

```
CENTRAL   off-ledger 93 + on-ledger distinct specs 274 = N_total       367
          off-ledger 93 + on-ledger overlapping   256 = N_overlapping  349
```

### 2.4 Cross-checks against the campaigns' own counters

- **FX chain**: the FX campaign maintains its own honest monotone counter,
  visible both in the gate reports and embedded in the registry's `params` as
  `trial_count_project_wide`. It runs 94 -> 141 across 2026-07-19 .. 07-25 and
  terminates at **141**, exactly matching `docs/strategies/FX_60_CATALOG_TRACKER.md`
  ("N went 137 -> 141") and the commit log. **Reproduced live**:
  `get_campaign_trial_distribution()` returns `N = 141` today. My reconstruction
  is a strict superset of it (see 3.1).
- **Futures chain**: `66 -> 81 -> 94` across the retest + variation campaigns,
  per `docs/progress/20260711_FUTURES_VARIATION_CAMPAIGN.md:8,30,62`. The
  brief's "66 -> 94" is **verified**, with one caveat in Section 5.
- **The 40-trial static baseline is not double-counted**: it predates the
  registry by construction (first registry row 2026-05-13), as the
  `get_campaign_trial_distribution()` docstring asserts and as the timestamps
  confirm.

---

## 3. Findings the reconstruction surfaced

### 3.1 `n_trials_project_wide()` has always returned 0

`src/experiments/registry.py:309` computes:

```sql
SELECT COALESCE(SUM(combinations_in_run), 0) FROM runs
WHERE agent_name = 'backtest-optimizer'
```

**No row in the registry has ever had `agent_name = 'backtest-optimizer'`** --
the actual writers are `futures-harness`, `fx-harness-walkforward`,
`backtest-driver`, `robustness-runner`. The function returned **0** on every
call ever made, and `expected_max_sharpe` returns `0.0` for `N < 2`, so any
DSR routed through this function was computed with **zero deflation** (DSR
degenerating to PSR(0)).

This is why `max(combinations_project) = 0`: `backtest_runner.py:680` faithfully
wrote the return value of a function that always returned 0.

The FX/futures gates escaped this only because `get_campaign_trial_distribution()`
(`src/backtesting/walkforward_common.py:83`) independently derives N by
counting registry rows with a numeric `oos_sharpe`, bypassing the broken
function. Two FX cost-sensitivity reports (`docs/reports/fx/costsens/`) did
**not** escape it and openly report "trial count 2 (project-wide
`n_trials_project_wide()` = 0, + 2 local configs)".

**This is a live code defect, not just a historical accounting gap.** It is
documented here but not fixed -- fixing it is a code change outside this task's
scope. **Recommended fix**: drop the `agent_name` filter, or point the function
at `MAX(combinations_project)`, which the backfill in Section 4 now populates.

### 3.2 The live N=141 is a 2.5x undercount

`get_campaign_trial_distribution()` counts only registry rows carrying a
numeric `oos_sharpe` -- **101 of 496**. It therefore silently omits:

| Omitted | Trials |
|---|---:|
| RAMP equity registry rows (all 141 -- readiness, walk-forward, robustness sweeps) | 111 distinct specs |
| futures/fx single-pass rows | 107 distinct specs |
| RAMP equity v0-detector off-ledger chain | 36 |
| RAMP options campaign | 16 |
| OpEx pinning | 1 |

The RAMP equity omission is the single largest, and it is not accidental --
see 3.3.

### 3.3 An equity trial chain was reset from 36 to 1

`docs/strategies/RAMP_VARIANTS.md` records explicitly:

> "V20+ family starts at `n_trials_project = 1` (fresh chain), separate from
> the v0-detector family's 36-trial chain ... the two families have separate
> n_trials counters."

and, decisively:

> "Family DSR passes at n_trials <= 12 (Wave-3 reset), fails at >= 36."

**A gate verdict flipped on the reset.** Splitting counters per strategy
family is exactly the move the methodology's "never shrink N to make a gate
easier" rule forbids: the two families share a researcher, a universe, and a
2017-2026 window, and the second was designed after seeing the first's results.
For a *lifetime* N these 36 must be re-added, and they are (Section 2.2).

I have not re-graded RAMP on the un-reset count -- out of scope -- but **the
RAMP Wave-3 family DSR verdict should be treated as unreliable** until it is.

### 3.4 The two DSR implementations disagree

`deflated_sharpe.py` omits `sigma_SR`. At N=18 it reports
`expected_max_sharpe = 1.973`; the Bailey-LdP value at `sigma_SR = 1/sqrt(6.5y)`
is 0.75. The RAMP options campaign's `p_value = 1.0` verdict is unaffected (its
observed Sharpe was -0.766, failing under either), but the discrepancy would
matter for a live candidate. **The options slate should use
`src/backtesting/statistics/dsr.py`, not `validation/deflated_sharpe.py`.**

---

## 4. Ledger populated

`scripts/maintenance/backfill_trial_counts.py` (new, idempotent, single
serialized write transaction; backs the DB up first to
`output/experiments.duckdb.pre-trialcount-backfill.bak`):

1. Appended **4 aggregate `verdict='tested'` rows** for the off-ledger blocks
   (93 trials), backdated to 2026-05-01 so they precede the first real
   registry row and the counter stays monotonic in timestamp order. They are
   **aggregate provenance rows, not fabricated per-spec reconstructions** --
   the per-spec `params` were never recorded and are not inventable. Each
   carries its block size in `combinations_in_run` and its full provenance in
   `notes`.
2. Set `combinations_in_run` on all 496 existing rows: 1 for a distinct spec
   (274), 0 for an exact rerun (191) or a harness fixture (31).
3. Set `combinations_project` on all 500 rows to the running cumulative count.

Verified post-state:

```
rows                      500
max(combinations_project) 367   <- was 0
sum(combinations_in_run)  367   <- was 4
NULL combinations_project   0
monotonicity violations     0
N_overlapping (2017-2025) 349
```

---

## 5. What is uncertain, and how much it moves N

Ordered by impact. **None of it changes the decision** -- see 5.6.

### 5.1 Sweep counting -- moves N by ~195 (the dominant driver)

Whether a parameter sweep contributes its full cardinality or 1. The 78 RAMP
robustness rows carry a prior, explicit annotation:
`trial_count_treatment=NOT_PROMOTED_no_selection_trial_added` -- arguing that
because the sweep centre was not moved, no selection occurred. **I did not
honor that annotation.** DSR's N is a count of draws the maximum could have
been taken over, and all 78 Sharpes were observed (they are recorded verbatim
in `notes`, ranging 0.47 to 0.72). The same question applies to the 101
futures single-pass specs.

- Excluding sweeps + single-pass diagnostics: **N_overlapping = 179** (LOW)
- Including them: **N_overlapping = 349** (CENTRAL, adopted)

### 5.2 Rerun counting -- moves N by ~191

I treat 191 identical-spec reruns as apparatus corrections (0 trials). If every
executed row is instead counted, and the options optimizer grids (18 long-call
+ 75 CSP combos) are counted at cardinality:

- **N_overlapping = 621** (HIGH)

### 5.3 The RAMP options campaign's true contribution -- moves N by ~93

Three defensible numbers: **16** (distinct tested specs -- adopted), **18**
(the long-calls optimizer grid, which is what the brief cited and what
`20260402_statistical_validation.json` reports), or **109** (16 specs + both
optimizer grids at cardinality). **Note the brief's `n_trials = 18` is the
optimizer-combination count for *one* of the 16 tested strategies, not the
campaign total** -- reading it as the campaign's contribution undercounts.
Separately, "18" also appears as the OpEx *trade* count; the two are unrelated
and must not be summed.

### 5.4 Double-count risk between the 36-chain and the registry -- ~2 to 6

The off-ledger v0-detector chain (36) includes V11 and pre-V11. The registry
also holds `RAMP-V11` and `RAMP-V02+V05` `wave3_readiness` rows. Those are
re-runs under new cost/lag legs (distinct `config_sha`), so I counted both --
but if any are true duplicates, N is over-counted by up to ~6. **Direction:
this is the only identified over-count; every other uncertainty under-counts.**

### 5.5 Smaller items

- **The futures 66 seed is internally inconsistent by 3.** It was seeded as
  40 + 26 retest combos, but the retest count was later corrected in place to
  "23 graded + 4 ungradeable". On the corrected figure the seed is 63 (or 67).
  Immaterial: **+/-3 on N=349 moves the hurdle by 0.001.**
- **Whether the futures variation campaign's 28 runs are in the main registry.**
  They ran on `feat/futures-variations`, never merged. The FX chain's jump to
  N=94 on 2026-07-19 implies the main registry did hold 54 graded rows at that
  moment, so they are present. Confident but not proven.
- **FX Wave-2 per-spec N**: the session handoff tabulates 105/106 where the
  gate artifacts say 108/107. The endpoint (141) is unaffected.
- **Window length**: the options window is 8.3 y per the V-battery's direct
  measurement (`20260727_options_vbattery_report.md:506`, "~2017-09 -> 2025-12,
  not the 13.7 years the chain assumes"). QQQ alone has a full 13.7 y series.
  A per-root window would raise the hurdle for every root except QQQ.

### 5.6 Sensitivity -- why none of this changes the decision

E[max Sharpe | null] at the honest 8.3-year window:

| Rule | N_overlapping | Hurdle | + Wave 1 (N+9) | + full slate (N+50) |
|---|---:|---:|---:|---:|
| LOW (no sweeps, no single-pass) | 179 | **0.947** | 0.953 | 0.975 |
| **CENTRAL (adopted)** | **349** | **1.021** | **1.024** | **1.036** |
| HIGH (every executed config) | 621 | **1.082** | 1.083 | 1.090 |

**A 3.5x span in N moves the hurdle 14%.** For reference:

| Scenario | Hurdle |
|---|---:|
| Registered pre-reg (N=9, 13.7 y) | 0.411 |
| N=9 on the honest 8.3 y window | 0.528 |
| Repo's live `get_campaign_trial_distribution()` (N=141, 8.3 y) | 0.920 |
| **Reconstructed (N=349, 8.3 y)** | **1.021** |

Sensitivity to the window, at N=349: 7.0 y -> 1.112 | **8.3 y -> 1.021** |
9.0 y -> 0.981 | 13.7 y -> 0.795. **Even on the disputed 13.7-year window, the
honest N alone doubles the hurdle from 0.41 to 0.80.**

---

## 6. Recommendations

1. **Grade options Wave 1 against `E[max Sharpe | null] = 1.02`**, not 0.41.
   Amend the pre-registration chain (spec v2 Sec 7.2, amendment A1 Sec 8,
   feasibility screen v2) before any Wave-1 result is interpreted. Use N+9 =
   358 for the post-Wave-1 bar, and pre-commit the full-slate bar of 1.036 at
   N+50 = 399.
2. **Use `src/backtesting/statistics/dsr.py`**, not
   `src/backtesting/validation/deflated_sharpe.py` (Section 3.4).
3. **Fix `n_trials_project_wide()`** (Section 3.1) -- it returns 0 today. With
   `combinations_project` now populated, `MAX(combinations_project)` is a
   correct one-line replacement.
4. **Re-grade the RAMP Wave-3 family on the un-reset count** (Section 3.3), or
   mark its DSR verdict unreliable.
5. **Make registry writes carry `combinations_project`** going forward so this
   reconstruction is never needed again.

---

## 7. Reproduction

```
C:\Users\qwqw1\anaconda3\envs\fintech\python.exe scripts/maintenance/backfill_trial_counts.py
```

Registry: `output/experiments.duckdb` (gitignored). Pre-backfill snapshot:
`output/experiments.duckdb.pre-trialcount-backfill.bak`.

Hurdles reproduce via `src.backtesting.statistics.dsr.expected_max_sharpe`
with the two-point sigma injection described in Section 1.1:

```python
expected_max_sharpe([0.0, (1/np.sqrt(years)) * np.sqrt(2.0)], n_trials)
```
