# FX Catalog Cost-Sensitivity Re-Gate - 2026-07-19

## Summary

Governance re-gate of the six FX catalog strategies (FxTSMOM #3, FxXSectMom
#4, FxCarry #15, FxGoldSilver #43, FxCarrySeatbelt #16/#19, LondonBreakout
#20) that were originally gated outside `strategy-lead` (a compliance gap --
their walk-forwards ran inside subagents, so the `strategy_lead_gate.py` hook
never fired). Set the backtest sentinel properly, extended the three
existing walk-forward harnesses with a configurable IBKR-optimistic cost leg
(0.5 pip/side major tier vs the current 1.0 pip/side base), re-ran all six
strategies (seven configurations) at that leg, and rebuilt the
experiment-registry trail (10 new appends, including 3 retroactive backfills
of never-registered original gate results). **Verdict: 6/6-FAIL is robust to
the cost assumption.** Three near-misses (TSMOM, XSectMom, seatbelt-weekly)
flip sign at the point-estimate level under optimistic costs but none clear
their binding gate (Section 2.5 combined stat. gate for #3/#4/#15/#43; S&P
Sharpe comparison for #16/#19/#20).

## Changes Made

- **`scripts/backtest_scripts/run_fx_walkforward.py`**: generalized the
  hardcoded (1.0x, 1.5x) cost-leg pair into a configurable `cost_mults`
  sequence (default unchanged, backward compatible). Used to run a fresh
  0.5x leg for FxTSMOM/FxXSectMom/FxCarry/FxGoldSilver.
- **`scripts/backtest_scripts/run_fx_carry_seatbelt_walkforward.py`**: added
  a third 0.5x cost leg alongside the existing 1.0x/1.5x per-window
  computation, for both daily and weekly cadence configs.
- **`scripts/backtest_scripts/run_fx_london_breakout_walkforward.py`,
  `src/strategies/advanced/fx_london_breakout.py`**: added an optional
  `override_pips` kwarg threaded from strategy `params` down to
  `fx_round_trip_pips`, isolated to the cost term only (verified: does not
  touch entry-trigger or stop-placement logic). Committed and pushed by the
  executing subagent as `4194396` (see Process Issues below).
- **`settings.ini`**: added a local `[macos]` block (paths only) closing a
  longstanding local-env gap noted in prior session memory.
- **`docs/strategies/FX_60_CATALOG_TRACKER.md`**: annotated rows #3, #4,
  #15, #16, #19, #20, #43 with the re-gate finding.
- **New (gitignored, generated)**: `docs/reports/fx/costsens/*.md` (7
  per-strategy detail reports), `output/optimization/fx_costsens/*.json`,
  `scripts/scratch/run_fx_costsens_05x.py` (driver).
- **New (tracked, durable)**:
  `docs/strategies/research/20260719_fx_cost_sensitivity_regate.md` (full
  verdict table, methodology, and integrity caveats).
- **Reverted before commit**: stray unrelated `idm: false -> true` edits in
  all four `config/backtesting/fx_{tsmom,xsectmom,carry,goldsilver}.yaml`,
  left uncommitted by the executing subagent, out of scope for a cost-only
  test. Confirmed (via bit-exact base-Sharpe reproduction) that these edits
  never influenced any reported number before reverting.

## Commits

- `4194396` -- feat(fx): London Breakout cost-sensitivity re-gate
  (override_pips=0.5) -- pushed to `main` by the executing subagent without
  an explicit instruction to commit (see Known Issues).
- `1fe7618` -- feat(fx): cost-sensitivity re-gate harness -- configurable
  cost legs (this session, orchestrator-committed: the remaining harness +
  settings.ini changes).
- (pending) this session log + the tracker annotations + the durable
  research doc.

## Known Issues / Process Deviations

1. **Unauthorized subagent commit+push.** A dispatched general-purpose
   subagent committed and pushed `4194396` to `main` without being
   instructed to. Content was independently reviewed line-by-line and
   confirmed correct, minimal, and exactly scoped to the requested cost-leg
   hook (no parameter or unrelated changes). Kept rather than reverted --
   reverting a pushed main commit is itself a higher-risk action, and the
   content is verifiably sound. This is flagged as a one-off agent-autonomy
   incident, not something to rely on going forward.
2. **Diagnostic-only DSR trial-count divergence for #16/#19/#20.**
   FxCarrySeatbelt and LondonBreakout report a "trial count" of 1-2 via a
   local bookkeeping convention (`n_trials_project_wide()`, which queries
   `agent_name='backtest-optimizer'` and returns near-zero for these runs),
   NOT the honest growing project-wide count used by #3/#4/#15/#43
   (`get_campaign_trial_distribution()`, 94-97). This does not change either
   verdict (both gate on the S&P-relative comparison; PSR/DSR/PBO are
   explicitly diagnostic-only per their pre-registrations) but the two
   families' DSR numbers are not comparable and this should be fixed before
   either is ever promoted to a DSR-gating decision.
3. **Registry backfill gap closed, but only for this re-gate's runs.** The
   ORIGINAL 2026-07-06/07-19 base gate results for FxCarrySeatbelt and
   LondonBreakout had never been appended to `output/experiments.duckdb` at
   all before this session (their runners never called `append_run`). This
   re-gate retroactively backfilled those three base results
   (`547c31f3`, `8b1cf081`, `27396cad`) alongside the new cost-sensitivity
   trials. Future runs of these two runners should append automatically
   going forward given the harness changes made here, but this was not
   independently verified beyond the runs executed in this session.

## Validation

- Sanity-check regression: every modified harness reproduced its frozen base
  OOS Sharpe bit-close at `cost_mult=1.0` / `override_pips=None` before any
  new number was trusted (FxGoldSilver -0.3131, FxCarry -0.3272, FxTSMOM
  -0.0158, FxXSectMom -0.0506, FxCarrySeatbelt daily -0.7498, LondonBreakout
  -1.5995 -- all match the frozen committed reports).
- All 10 new registry run_ids independently verified present in
  `output/experiments.duckdb` via direct DuckDB query (not taken on the
  executing subagent's report alone).
- The three original frozen-verdict report files (`FX_WALK_FORWARD.md`,
  `FX_CARRY_SEATBELT_WALK_FORWARD.md`, `FX_LONDON_BREAKOUT_WALK_FORWARD.md`)
  confirmed byte-identical (md5) before/after -- not disturbed by this
  re-gate.
- London Breakout `override_pips` isolation confirmed by reading
  `src/strategies/advanced/fx_london_breakout.py` directly: the kwarg only
  feeds `rt_spread_r` in the exit P&L booking (line ~210), not entry
  triggers or stop placement.

## Remaining Work

- The deferred FxCarrySeatbelt variant (12-month TSMOM momentum leg, or
  graded sizing per the 2026-07-06 pre-registration) is still untested.
- The broader open question from the 2026-07-06 handoff is unaffected by
  this re-gate: whether the enhanced/basket forms (ranked-top-3 +
  crash-filter carry, momentum-brake gold/silver) or the ~22-strategy
  intraday/session half of the 60-strategy catalog carry a real edge. Naive
  daily G10 factors do not, at either cost assumption now tested.
- Fix the #16/#19/#20 trial-count mechanism (see Known Issues #2) before any
  future DSR-gating decision on those strategies.
