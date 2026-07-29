# Lifetime DSR Trial Count Reconstruction - 2026-07-28

## Summary

Reconstructed Homeguard's honest lifetime Deflated-Sharpe trial count N from
the experiment registry plus off-ledger campaign docs, to unblock grading of
the queued 50-candidate options slate. The slate's pre-registration registered
a Wave-1 hurdle of `E[max Sharpe | null] ~ 0.41` from `N = 9` on a 13.7-year
window; the honest figures are **N_overlapping = 349** on an **8.3-year**
window, giving a hurdle of **1.02**. The registered bar is understated ~2.5x
and would have passed noise. No backtest was run -- this was integrity
accounting only.

## Changes Made

- **`scripts/maintenance/backfill_trial_counts.py`** (new): populates
  `combinations_project` (was 0 on every row) and `combinations_in_run` (was
  set on 4 of 496) in `output/experiments.duckdb`. Appends 4 aggregate
  `verdict='tested'` rows for the off-ledger trial blocks. Idempotent, backs
  up first, single serialized write transaction (DuckDB is single-writer).
- **`docs/strategies/research/options-slate/20260728_lifetime_trial_count.md`**
  (new): the deliverable -- method, dedup rule, every judgment call, N
  derivation, hurdle table, and an explicit uncertainty section with the range.
- **`output/experiments.duckdb`** (gitignored): backfilled. N_total = 367,
  N_overlapping = 349. Pre-change snapshot kept at
  `output/experiments.duckdb.pre-trialcount-backfill.bak`.

## Decisions and judgment calls

- **Dedup rule adopted (CENTRAL)**: one trial per distinct evaluated
  specification; parameter sweeps count at full cardinality; exact reruns of
  an identical spec count 0 (apparatus correction, not search); harness
  fixtures excluded. Rules 2 and 3 pull opposite ways, hence the wide bounds.
- **Spec identity must be convention-aware**: equity rows carry a real
  `config_sha` with NULL `params`; futures/fx harness rows carry
  `config_sha='unknown'` with a full `params` JSON. Neither field alone
  identifies a spec -- this is why the count could not just be queried.
- **Overrode a prior annotation**: the 78 RAMP robustness rows are tagged
  `trial_count_treatment=NOT_PROMOTED_no_selection_trial_added`. Not honored --
  all 78 Sharpes were observed, and DSR counts draws, not promotions. This is
  the single largest N driver (~195 trials) and is called out as such.
- **Sigma injection instead of a third DSR implementation**: reused
  `src.backtesting.statistics.dsr.expected_max_sharpe` by passing the
  two-point sample `{0, s*sqrt(2)}`, whose ddof=1 variance is exactly `s^2`.
- **Off-ledger blocks recorded as aggregate rows, not fabricated per-spec
  rows** -- the per-spec params were never recorded and are not inventable.

## Findings (beyond the headline number)

1. **`n_trials_project_wide()` has always returned 0.** It filters on
   `agent_name = 'backtest-optimizer'`, which no registry row has ever had.
   Any DSR routed through it was computed with zero deflation. NOT FIXED --
   code change out of scope; `MAX(combinations_project)` is now a valid
   one-line replacement.
2. **An equity trial chain was reset 36 -> 1** for the RAMP V20+ family, and
   `RAMP_VARIANTS.md` states the family "passes at n_trials <= 12, fails at
   >= 36" -- a gate verdict flipped on the reset. Re-added to lifetime N; the
   RAMP Wave-3 DSR verdict should be treated as unreliable until re-graded.
3. **The two DSR implementations disagree.** `validation/deflated_sharpe.py`
   omits `sigma_SR` entirely (reports 1.973 at N=18 where Bailey-LdP gives
   0.75). The options slate must use `statistics/dsr.py`.
4. **The brief's "options n_trials = 18" is one strategy's optimizer grid**,
   not the campaign total (16 distinct specs tested of 31 catalogued). Also
   collides with OpEx's unrelated "18 trades".
5. **The overlapping-window rule gives almost no relief** -- only 6 distinct
   specs are calendar-disjoint from 2017-2025. Every campaign window spans it.

## Validation

- Formula reproduces the pre-registration's own hurdle table exactly at its
  stated precision (N=9 -> 0.4109 vs 0.41; N=19 -> 0.5074 vs 0.51; N=31 ->
  0.5638 vs 0.56; N=38 -> 0.5860 vs 0.59), confirming the multiplier was never
  the problem -- only the two inputs.
- Live `get_campaign_trial_distribution()` returns N=141, matching the FX
  campaign docs and the commit log ("N 137 -> 141") independently; the
  reconstruction is a strict superset of it.
- Futures "66 -> 94" verified from `docs/progress/20260711_FUTURES_VARIATION_CAMPAIGN.md`
  (with a noted internal 3-trial inconsistency, immaterial at N=349).
- Post-backfill registry: 500 rows, `max(combinations_project) = 367 =
  sum(combinations_in_run)`, 0 NULLs, 0 monotonicity violations.
- Backfill re-run confirms idempotency (detects sentinel rows, no-ops).
- ASCII-only verified byte-wise on both new files.

## Commits

- `95b5518` feat(experiments): backfill project-wide DSR trial counter
- `d426996` docs(options-slate): reconstruct lifetime DSR trial count -- N=349, hurdle 1.02 not 0.41

Branch `chore/lifetime-trial-count`, NOT pushed.

## Known Issues / Remaining Work

- **Amend the options pre-registration chain** (spec v2 Sec 7.2, amendment A1
  Sec 8, feasibility screen v2) to the reconstructed hurdle before any Wave-1
  result is interpreted. Pre-commit N+9 = 358 (bar 1.024) and N+50 = 399
  (bar 1.036).
- **Fix `n_trials_project_wide()`** (Finding 1) -- still broken.
- **Re-grade RAMP Wave-3 on the un-reset count** (Finding 2), or mark its DSR
  verdict unreliable.
- **Make registry writers persist `combinations_project`** so this
  reconstruction is never needed again.
- Unresolved minor: whether the 28 futures variation runs (branch
  `feat/futures-variations`, never merged) are in the main registry -- the FX
  chain's jump to N=94 implies yes, but it is not proven.
- Per-root options windows differ (only QQQ has the full 13.7 y); a per-root
  hurdle would be higher than 1.02 for every root except QQQ.
