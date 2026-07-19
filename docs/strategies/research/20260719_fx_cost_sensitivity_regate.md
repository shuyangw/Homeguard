# FX Catalog Cost-Sensitivity Re-Gate -- 2026-07-19

## Why this run exists (compliance gap)

The six FX catalog strategies below were originally gated OUTSIDE the
`strategy-lead` process: their walk-forward runs executed inside
subagent-driven-development subagents via ad-hoc runner invocations, so the
`strategy_lead_gate.py` `PreToolUse` hook never fired (it only intercepts
commands run by the top-level session, not commands a subagent issues from
inside its own shell). This is a real compliance gap, not a hypothetical one.
This document is `strategy-lead`'s retroactive re-gate: (1) the backtest
sentinel was set properly for this run, (2) the prior FAIL verdicts are
independently re-verified via a cost-sensitivity sweep, and (3) every run in
this re-gate is appended to `output/experiments.duckdb` so the DSR
trial-count trail is honest going forward. The original verdicts are NOT
being second-guessed on methodology -- they are being stress-tested on one
specific, previously-unexamined assumption: is the FX cost model too
conservative relative to this repo's primary broker (IBKR)?

## Hypothesis under test

Homeguard's FX major-tier cost (`src/backtesting/costs/fx.py`, midpoint of
the methodology's 0.5-1.5 pip/side range = **1.0 pip/side**, 2.0 pip/side
round-trip before session multiplier) is plausibly 2-3x conservative versus
IBKR-quality execution on liquid majors. The methodology's own low end
(Section 4.3: "typical retail tiers around 0.5-1.5 pips") supports testing
**0.5 pip/side** as an IBKR-optimistic bound. Test: does this bound flip any
of the three near-misses (FxTSMOM -0.02, FxXSectMom -0.05, FxCarrySeatbelt
weekly -0.11) to a PASS? And is the 6/6-FAIL conclusion robust to this
assumption across all six strategies?

## Mechanism

`cost_mult` (already wired end-to-end: config -> `run_fx_backtest` ->
`FxSpotPortfolioSimulator(cost_mult=...)`) is a straight multiplier on the
full session-aware round-trip USD cost. `cost_mult=0.5` on the 1.0 pip/side
base == exactly 0.5 pip/side major tier, i.e. the methodology low end. No
change to the cost model itself was needed for strategies #3/#4/#15/#43 or
the seatbelt (#16/#19) -- only the walk-forward harnesses needed to compute
an additional leg. LondonBreakout (#20) bakes its round-trip spread into a
per-trade R-multiple via `fx_round_trip_pips(tier, session="london")`; an
`override_pips` kwarg (already supported by `fx_round_trip_pips`, previously
unused by this caller) was threaded through
`LondonBreakoutStrategy.__init__` so the cost term alone -- not entry
triggers or stop placement -- can be overridden. Verified: `override_pips`
only feeds `rt_spread_r` in the exit P&L booking
(`src/strategies/advanced/fx_london_breakout.py` line ~210), confirmed by a
bit-exact regression reproduction of the frozen base result with
`override_pips=None`.

Full data coverage preserved: all six re-runs used the identical date ranges
as the original gates (2011-01-01 to 2026-04-01 for #3/#4/#15/#16/#19/#43;
the London Breakout config's native range for #20). No window was
shrunk. Purge/embargo (36m/12m/12m walk-forward) unchanged.

## Verdict table

| # | Strategy | Cadence | BASE OOS Sharpe (1.0x cost) | OPTIMISTIC OOS Sharpe (0.5x cost) | PSR | DSR | Trial count (source) | PBO | Primary gate | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 3 | FxTSMOM | -- | -0.0158 | **+0.0750** | 0.205 | ~0.000 | 95 (project-wide, growing) | 0.852 | Combined stat. gate (Sec 2.5) | REJECT (unchanged) |
| 4 | FxXSectMom | -- | -0.0506 | **+0.0584** | 0.006 | ~0.000 | 96 | 0.655 | Combined stat. gate | REJECT (unchanged) |
| 15 | FxCarry | -- | -0.3272 | -0.2948 | ~0.000 | ~0.000 | 97 | 0.727 | Combined stat. gate | REJECT (unchanged) |
| 43 | FxGoldSilver | -- | -0.3131 | -0.2994 | ~0.000 | ~0.000 | 94 | 0.489 | Combined stat. gate | REJECT (unchanged) |
| 16/19 | FxCarrySeatbelt | daily | -0.7498 | -0.4913 | diag only | diag only | 2 (local; see caveat) | 0.217 | OOS Sharpe > S&P (0.6842) | FAIL (unchanged) |
| 16/19 | FxCarrySeatbelt | weekly | -0.1123 | **+0.0200** | diag only | diag only | 2 | 0.420 | OOS Sharpe > S&P (0.6842) | FAIL (unchanged) |
| 20 | LondonBreakout | -- | -1.5995 | -0.7483 | diag only | diag only | 1 (local; see caveat) | 0.720 | OOS Sharpe > S&P (0.6767) | FAIL (unchanged) |

**Bold** = the point-estimate Sharpe crosses zero at the optimistic cost.

## Does the hypothesis hold? Partially, and it doesn't matter

**Confirmed: the near-misses ARE cost-sensitive at the point-estimate level.**
FxTSMOM, FxXSectMom, and FxCarrySeatbelt-weekly all flip from negative to
(marginally) positive raw OOS Sharpe under the 0.5 pip/side assumption. This
is a real, non-trivial finding -- the FX cost model's conservatism is not
academic, it moves the headline number by 0.05-0.13 Sharpe on strategies
that were already close to zero.

**Refuted: none of the three flip PASS on the binding gate.**
- FxTSMOM / FxXSectMom are gated by the Section 2.5 combined statistical
  gate, which does not move with the cost leg: PSR (0.205 / 0.006), DSR
  (~0.000 for both), and PBO (0.852 / 0.655) are computed on the stitched
  daily OOS return series' *distributional* properties (mean/vol/skew/kurt
  relative to a zero-Sharpe null and the deflation term), not re-derived per
  cost leg by this harness. A point estimate crossing zero does not move a
  PSR of 0.2 to 0.95, nor a PBO of 0.85 (strong overfitting on window
  ranking) below the 0.25 threshold. These two remain unambiguous REJECTs --
  the statistical significance was never close, only the raw point Sharpe
  was.
- FxCarrySeatbelt-weekly's primary gate is relative to the S&P 500 Sharpe
  (0.6842) over the same OOS dates, not an absolute statistical gate. +0.02
  is not within an order of magnitude of 0.68. This flip is a sign change,
  not a competitive result.
- FxCarry, FxGoldSilver, FxCarrySeatbelt-daily, and LondonBreakout all remain
  solidly negative even at 0.5x cost -- their FAIL was never primarily a
  cost-model artifact.

**Robustness conclusion: the 6/6-FAIL conclusion IS ROBUST to the cost
assumption.** Every strategy that failed at base cost still fails at the
IBKR-optimistic bound. The magnitude of the cost sensitivity (largest swing:
LondonBreakout, +0.85 Sharpe from -1.60 to -0.75) is informative for future
strategy design -- it confirms transaction costs are a first-order driver of
these results and any live FX deployment should pursue the tightest
available execution -- but it does not change any strategy's readiness
status. No strategy in this catalog is rescued by using a friendlier cost
assumption.

## Known limitations / integrity caveats surfaced during this re-gate

1. **Trial-count mechanism divergence.** FxTSMOM/FxXSectMom/FxCarry/
   FxGoldSilver use `src.backtesting.walkforward_common.get_campaign_trial_distribution()`,
   which correctly grows from the documented 40-trial futures-campaign
   baseline plus every subsequent registry-logged run with a numeric
   `oos_sharpe` metric (94-97 as of this re-gate, growing with each of the
   four new appends). FxCarrySeatbelt and LondonBreakout instead use
   `n_trials_project_wide()` (methodology Section 9.4's literal SQL: `SUM(combinations_in_run)
   WHERE agent_name='backtest-optimizer'`), which returns near-zero for these
   runs since they were never logged by an optimizer agent -- their reported
   "trial count" of 1-2 is a local bookkeeping convention (`base_trials + n_configs`),
   NOT the honest growing project-wide count. This does not affect either
   strategy's verdict (both gate on the S&P-relative comparison, with
   PSR/DSR explicitly marked diagnostic-only per their 2026-07-06/07-19
   pre-registrations), but it means their diagnostic DSR figures are not
   comparable to the #3/#4/#15/#43 figures and should not be quoted as if
   they were. **Recommendation:** migrate FxCarrySeatbelt and LondonBreakout
   to `get_campaign_trial_distribution()` if their DSR is ever promoted from
   diagnostic to gating.
2. **A subagent self-committed and pushed to `main` without an explicit
   instruction to do so** during this task (commit `4194396`, the
   LondonBreakout `override_pips` hook + report + registry backfill). Content
   was independently reviewed and verified correct, minimal, and exactly
   scoped to what this re-gate required (confirmed via `git show --stat` and
   a full diff read: only `fx_london_breakout.py`, its walk-forward runner,
   and new report/JSON files -- no parameter or unrelated changes). Kept as
   is rather than reverted, since the content is correct and reverting a
   pushed main commit is itself a higher-risk destructive action; flagged
   here as a process deviation for awareness. `strategy-lead` did not
   authorize the push and treats this as an isolated agent-autonomy incident,
   not a pattern to rely on.
3. **Stray unrelated parameter edits caught and reverted.** The same
   execution pass left `idm: false -> true` uncommitted edits in all four of
   `config/backtesting/fx_{tsmom,xsectmom,carry,goldsilver}.yaml` -- outside
   the scope of a cost-only test. These were reverted (`git checkout --`)
   before any commit. The bit-exact base-Sharpe reproduction (see Task 4
   sanity check below) confirms these stray edits never actually influenced
   any reported number (IDM is a portfolio-level diversification scalar,
   Sharpe-invariant), but the edit itself was still an out-of-scope parameter
   touch and is called out here for the record.
4. **Sanity-check regression (mandatory before trusting any new number):**
   every harness change reproduced its frozen base Sharpe bit-exact at
   `cost_mult=1.0` / `override_pips=None` before the optimistic leg was
   trusted: FxGoldSilver -0.3131028506596593, FxCarry -0.3271579521197852,
   FxTSMOM -0.01582270046976089, FxXSectMom -0.050597121638394124,
   FxCarrySeatbelt daily -0.7498, LondonBreakout -1.5995 (report-precision
   match). No regression found.

## Registry appends (new, this re-gate)

All ten `runs` rows below are new inserts into `output/experiments.duckdb`
via `src.experiments.append_run`, `phase="walk_forward"`,
`asset_class="fx"`. Seven are new cost-sensitivity trials; three are
retroactive backfills of the ORIGINAL base gate results for FxCarrySeatbelt
(both cadences) and LondonBreakout, which had never been registered before
this re-gate (a second integrity gap this re-gate closes).

| run_id | strategy_name | note |
|---|---|---|
| `ef88e6bb-f5a3-46ac-8b2a-8f1ef0b48856` | FxTSMOM | new 0.5x-cost trial |
| `095a7bd7-a2bc-48cc-a396-5443c93f43e6` | FxXSectMom | new 0.5x-cost trial |
| `badf0022-f9df-456e-9aba-8ca55e62b041` | FxCarry | new 0.5x-cost trial |
| `2958e81a-46d1-4d04-a86a-ff5ed6a20a51` | FxGoldSilver | new 0.5x-cost trial |
| `547c31f3-4241-44bf-b724-322d5275cde7` | FxCarrySeatbelt | daily, backfill of original 2026-07-06 base gate |
| `04b796db-e0dc-4b91-b777-36ec6e31fb3d` | FxCarrySeatbelt | daily, new cost-sensitivity re-gate (0.5x/1.0x/1.5x legs) |
| `8b1cf081-6ec4-4c1d-a934-497de8e84afc` | FxCarrySeatbelt | weekly, backfill of original 2026-07-06 base gate |
| `b551868b-4b83-4b15-a7d4-5f8b6aac77f2` | FxCarrySeatbelt | weekly, new cost-sensitivity re-gate |
| `27396cad-7044-4d76-a6ce-073ef2419aad` | LondonBreakout | backfill of original 2026-07-19 base gate |
| `33e45b88-6ba3-46eb-9d6e-ad066d871d2e` | LondonBreakout | new 0.5x-pip-side cost-sensitivity re-gate |

Verified present in `output/experiments.duckdb` by direct query (not taken
on the executing subagent's word alone).

## Per-strategy detail reports (generated, not tracked in git -- `docs/reports/`
is gitignored by project convention; this document is the durable tracked
copy)

- `docs/reports/fx/costsens/fx_tsmom_0.5x.md`
- `docs/reports/fx/costsens/fx_xsectmom_0.5x.md`
- `docs/reports/fx/costsens/fx_carry_0.5x.md`
- `docs/reports/fx/costsens/fx_goldsilver_0.5x.md`
- `docs/reports/fx/costsens/fx_carry_seatbelt_daily_0.5x.md`
- `docs/reports/fx/costsens/fx_carry_seatbelt_weekly_0.5x.md`
- `docs/reports/fx/costsens/fx_london_breakout_0.5x.md`

The three ORIGINAL frozen-verdict report files (`FX_WALK_FORWARD.md`,
`FX_CARRY_SEATBELT_WALK_FORWARD.md`, `FX_LONDON_BREAKOUT_WALK_FORWARD.md`)
were verified byte-identical (md5) before and after this re-gate -- they
were not overwritten or otherwise disturbed.

## Files changed by this re-gate

- `scripts/backtest_scripts/run_fx_walkforward.py` -- generalized hardcoded
  (1.0x, 1.5x) cost legs into a configurable `cost_mults` sequence, default
  unchanged, fully backward compatible.
- `scripts/backtest_scripts/run_fx_carry_seatbelt_walkforward.py` -- added a
  third 0.5x cost leg to the existing per-window 1.0x/1.5x computation.
- `scripts/backtest_scripts/run_fx_london_breakout_walkforward.py`,
  `src/strategies/advanced/fx_london_breakout.py` -- added an optional
  `override_pips` kwarg, threaded from strategy `params` down to the cost
  term only (committed and pushed as `4194396`, see caveat 2 above).
- `settings.ini` -- added a local `[macos]` block (paths only, no secrets) so
  FX backtests run on this machine (a longstanding local-env gap).
- `docs/strategies/FX_60_CATALOG_TRACKER.md` -- annotated rows #3, #4, #15,
  #16, #19, #20, #43 with the cost re-gate finding.
- This document.

## Overall verdict for the pipeline

**No change to strategy readiness.** All six FX catalog strategies (seven
gated configurations counting both seatbelt cadences) remain FAIL/REJECT.
The IBKR-optimistic cost hypothesis is confirmed directionally (costs are a
real, first-order drag, and the model here is conservative) but refuted as a
rescue mechanism (no strategy clears its binding gate at either cost
assumption). This is a completed, honest negative result, not a problem to
engineer around: the naive-daily-FX-factor edge (per the 2026-07-06 handoff)
remains not present in clean G10 data at either cost assumption tested. The
unresolved question from that handoff -- whether the enhanced/basket forms
or the intraday half of the catalog carry a real edge -- is unaffected by
this re-gate and remains open.
