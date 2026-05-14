# Open Decisions for the Methodology Rollout

**Status**: Awaiting decisions from Shuyang
**Created**: 2026-05-13
**Context**: Continuation of the v3 methodology rollout from `docs/planning/20260512_methodology_rollout_v3_plan.md`. Eight of the v3 PRs landed on 2026-05-12 plus a batch of five immediate follow-ups on 2026-05-13. This doc lists the remaining decisions before further work can proceed without guessing.

Each section has the same shape: **Question**, **Options**, **Default recommendation**, **Blocks / unblocks**.

---

## Decision 1 -- cost-tier assignment for the other 11 backtest YAMLs

**Question**: Which `costs.tier` (per methodology Section 4) does each of these YAMLs get?

| YAML | Likely tier | Notes / uncertainty |
|---|---|---|
| `config/backtesting/bmsb_crypto.yaml` | `crypto_major` or `crypto_alt` | depends on universe inside the YAML -- BTC/ETH only or altcoins included |
| `config/backtesting/dsts_btc.yaml` | `crypto_major` | BTC-only, clear |
| `config/backtesting/evr_crypto.yaml` | `crypto_alt` (likely) | "EVR" looks like an altcoin set |
| `config/backtesting/frs_crypto.yaml` | `crypto_alt` (likely) | same |
| `config/backtesting/hurst_mr_baseline.yaml` | depends on asset class | crypto, equity, or futures? |
| `config/backtesting/hv_orb_baseline.yaml` | `large_cap_liquid` | ORB usually on liquid stocks/ETFs |
| `config/backtesting/ict_production.yaml` | depends on universe | need to look inside |
| `config/backtesting/ma_single.yaml` | `large_cap_liquid` | generic MA test |
| `config/backtesting/ml_crypto_mr_baseline.yaml` | `crypto_alt` (likely) | crypto MR universe |
| `config/backtesting/opex_pinning.yaml` | options model (Section 4.5) | OPEX = options expiry pinning; not bps-tiered |
| `config/backtesting/orb_baseline.yaml` | `large_cap_liquid` | ORB on equities |

**Options**:
- **A**: Confirm each tier explicitly, file-by-file
- **B**: Authorize me to assign reasonable defaults based on each YAML's universe (I'll read every YAML, propose a tier, commit; you can override in a follow-up commit)
- **C**: Skip migration until the strategies are actually being backtested again -- the current deprecation warning is non-blocking

**Default recommendation**: **B**. The defaults are recoverable, the migration unlocks the 1.5x cost-sensitivity gate for these strategies, and the cost of getting one wrong is one extra commit. `opex_pinning.yaml` is the only ambiguous one -- options-specific cost models aren't in the per-side-fees translation layer yet, so it stays on raw fees/slippage with the deprecation warning until the options cost path is wired.

**Blocks**: Decision 5 (making `costs.tier` required) cannot happen until this is done.

---

## Decision 2 -- engine vs cost-tier sanity check

**Question**: Which strategy to use to confirm the cost-tier wiring (`costs.tier` -> per-side fees) doesn't perturb backtest metrics in surprising ways?

**Options**:
- **OMR** -- has a backtest config (`config/backtesting/omr_backtest.yaml`) wired with `costs.tier: leveraged_etf`. Pre-tier baseline metrics are recoverable from `docs/reports/omr/` if any historical report exists. Quick run, ~5-10 minutes.
- **CSCM** -- has `costs.tier: crypto_major`; tier changes are larger (4x previous cost assumption), so the impact will be more visible. Multi-symbol crypto run takes longer.
- **RAMP** -- no equity backtest config exists, so we'd have to write one first. Out of scope for a sanity check.

**Default recommendation**: **OMR** for the sanity check (small, fast, deterministic), then **CSCM** as a stress check (validates that the 4x cost jump is real and not an engine artifact).

**What to verify**:
1. Old metrics vs new metrics: only the Sharpe / expectancy differ by an amount explainable by the cost delta. (For OMR: ~22.5 bps vs prior ~30 bps RT -- expect slight improvement. For CSCM: ~126 bps vs prior ~30 bps RT -- expect a substantial Sharpe haircut.)
2. The 1.5x cost-sensitivity rerun (per Section 4.6) produces sensible numbers without manual intervention.
3. Stop-specific slippage (Section 11.5) is or is not applied -- our `_resolve_costs` currently puts everything into per-side `fees` with `slippage=0`, so stop multipliers from Section 11.5 might not fire correctly. **Sanity check should specifically verify this.**

**Blocks**: Decision 3 (tightening `_append_to_registry`) wants real-run validation; this sanity check provides it.

---

## Decision 3 -- tighten `_append_to_registry` to fatal

**Question**: When should we flip the registry-append from log-and-continue to hard-fail (per methodology Section 9.3 "no silent success")?

**Options**:
- **A**: Right after the OMR sanity check (Decision 2) succeeds end-to-end. Quick win, single-line change.
- **B**: After Decision 1 lands too (so we have multiple real runs in the registry before tightening).
- **C**: After Decision 6 (GridSearch fix) lands (so optimizer runs also append).

**Default recommendation**: **A**. The single-mode runner is the most common entry point; if its append works once cleanly, tightening is safe. Sweep / optimize / walk-forward append rows but `portfolio` is `None` for those modes -- they should be checked too. Spot-check, then tighten.

**Risk if tightened too early**: a transient DuckDB lock or disk issue fails the entire backtest run.
**Risk if not tightened**: the registry silently drops runs and Section 9 governance erodes.

---

## Decision 4 -- live-ops smoke test

**Question**: Which `live-ops` recipes do we exercise to validate the agent before relying on it for routine work?

**Recipes**:
- `status` (read-only)
- `metrics` (read-only)
- `journal` (read-only)
- `start-instance` (state-changing, confirms)
- `stop-instance` (state-changing, confirms)
- `restart` (state-changing, confirms)
- `sync-dashboards` (state-changing on the push step, confirms)

**Options**:
- **A**: Read-only recipes only (`status`, `metrics`, `journal`). Lowest-friction validation.
- **B**: Read-only PLUS one state-changing recipe with confirmation. Validates the full confirmation flow.
- **C**: All seven, with you standing by to approve each yes/no prompt.

**Default recommendation**: **B**, with `restart cscm` as the state-changing recipe (CSCM is the most expendable -- it runs weekly; a restart mid-week has zero market-hours impact). Validates the canonical pattern end-to-end without putting RAMP at risk during US market hours.

---

## Decision 5 -- make `costs.tier` required (`_resolve_costs` raises on missing)

**Question**: When do we switch `_resolve_costs` from warning-on-missing to raising?

**Options**:
- **A**: Immediately after Decision 1 is complete (all 13 YAMLs migrated). One-line flip in `src/backtest_runner.py`.
- **B**: After both Decision 1 AND Decision 3 (registry tightened first, then costs tightened).
- **C**: Defer indefinitely -- soft-warning is enough.

**Default recommendation**: **A**. Once all configs are migrated, the warning is dead code; turning it into a hard-fail enforces the "every strategy declares its tier" rule per the methodology.

**Risk**: any forgotten config (one slipped through the migration) will break next time it's run. Mitigation: grep `config/backtesting/*.yaml` for missing `costs:` block as part of Decision 1.

---

## Decision 6 -- GridSearch signature drift fix

**Question**: How to fix `GridSearchOptimizer` calling the engine's old `_run_single_symbol(strategy, data, symbol, price_type)` signature at six call sites? This breaks `test_validate_with_mock_data` and any in-flight optimizer run.

**Options**:
- **A**: Add `engine._run_single_symbol_with_data(strategy, data, symbol, price_type)` variant that accepts pre-loaded data. Restore the optimizer's per-config fast path (one data load, N config evaluations). ~30 min engine work + 6 call-site updates.
- **B**: Update the optimizer to call the engine's new `_run_single_symbol(strategy, symbol, start, end, price_type)` per config. Engine re-loads data each call -- slow on big sweeps (N x data-load time instead of 1x).
- **C**: Defer until a real optimization run is needed. The crash-loop affects only the optimizer test path, not anything currently in production.

**Default recommendation**: **A**. The optimizer is supposed to be fast; option B silently regresses sweep performance. The new method is internal (`_` prefix), additive (doesn't change existing API), and a faithful restoration of the pre-drift contract. The 6 call-site updates are trivial.

**Effort**: ~1 hour total including test re-run.

---

## Decision 7 -- pre-existing uncommitted WIP in the working tree

**Question**: What to do with the changes that have been sitting in the working tree from before this session?

```
M scripts/data/migrate_to_time_partitioned.py
M src/data/acquisition/base.py
M tests/data/test_acquisition/test_base.py
?? docs/progress/20260509_VALIDATION_FRAMEWORK_DEFERMENTS.md
?? docs/strategies/research/20260509_FUTURES_STRATEGY_TESTING_PLAN.md
?? scripts/data/normalize_equities_partitions.py
?? tests/data/test_acquisition/test_normalize_partitions.py
```

These appear to be a partly-finished data-acquisition migration (validation framework deferments) that someone (you, prior session) started before today.

**Options**:
- **A**: You finish it -- commit when ready, this doc just notes it's in flight.
- **B**: Stash for now and resume later. `git stash push -m "data-acq-WIP-2026-05-09"` and clean working tree.
- **C**: Revert if the partial state is wrong / abandoned. (Destructive; only if you're sure.)

**Default recommendation**: **A**. The naming (`20260509_VALIDATION_FRAMEWORK_DEFERMENTS.md` looks like a deliberate session-log start) suggests intentional in-flight work. Don't touch unless you've decided to abandon.

---

## Decision 8 -- new agents (`portfolio-integrator`, `strategy-architect`, `strategy-implementer`)

**Question**: Promote any of these from "future" to "now"?

**Status**: All three are deferred per decision B from the prior session. Methodology Section 6 (the rules) is in effect; the orchestrator handles their concerns inline. Promotion triggers documented in `docs/methodology/backtesting.md` Appendix.

**Options**:
- **A**: Stay deferred. (Current state.)
- **B**: Promote `portfolio-integrator` now (its rules already exist; the agent would just operationalize Section 6 / 11.11 / 12 reads against `output/experiments.duckdb`).
- **C**: Promote all three (full v2-plan scope).

**Default recommendation**: **A** until the explicit trigger fires. The triggers are:
- `portfolio-integrator` -- the first portfolio-integration question that requires multi-file return-stream analysis (loading >= 2 incumbent strategies' OOS returns from the registry, computing correlations, etc.). With one strategy live (RAMP) and CSCM in paper, the orchestrator can handle this inline.
- `strategy-architect` + `strategy-implementer` -- the first strategy where blueprint + implementation can't fit in a single session's context. RAMP, OMR, CSCM all fit today.

Re-evaluate when 5+ live strategies exist or correlation analysis becomes a recurring task.

---

## Decision 9 -- ZC futures roll-splicing helper

**Question**: Build the roll-discontinuity handling now, or defer until a futures backtest is on the agenda?

**Context**: ZC continuous (`.c.0`) data has 4 bars with >10% moves and 1 bar with a 22.68% move -- all at known July / September corn-contract roll boundaries. Computing `pct_change()` across roll dates produces fake P&L for any futures strategy.

**Options**:
- **A**: Build `src/backtesting/utils/futures_roll.py` now: roll-date detection + return-series splicing helpers. ~2-3 hours including tests.
- **B**: Defer until the first futures backtest is queued.
- **C**: Build a minimal warning (detect >5% close-to-close jumps and log them) and defer the splicing.

**Default recommendation**: **B**. No futures backtest is queued today. Building the helper without a concrete consumer means we'd guess at the API and might miss requirements. The 22.68% jump is documented in the data-validation session log so future work won't be surprised by it.

---

## Recommended next-session order

If you approve all "default recommendation" answers, execution order would be:

1. **Decision 1 (B)** -- assign tiers to the 11 remaining YAMLs (~30 min reading universes + commit)
2. **Decision 2 (OMR then CSCM)** -- sanity-check the cost-tier wiring (~30 min)
3. **Decision 3 (A)** -- tighten registry append to fatal (5 min)
4. **Decision 5 (A)** -- make `costs.tier` required (5 min)
5. **Decision 6 (A)** -- fix `GridSearchOptimizer` via `_run_single_symbol_with_data` (~1 hr)
6. **Decision 4 (B)** -- live-ops smoke test with `restart cscm` (~15 min, requires you on standby)
7. **Decisions 7, 8, 9** -- no immediate action

Total: ~2.5 hours focused work.
