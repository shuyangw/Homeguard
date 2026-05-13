# Methodology Rollout v3 -- 2026-05-12

## Summary

Continuation of the methodology rollout from earlier today. Executed all six v3 PRs (`0a`, `0b`, `0c`, `1a`, `1c`, `2`, `3`, `4`) per `docs/planning/20260512_methodology_rollout_v3_plan.md`. Methodology is now the single source of truth for backtest rules (cleanup landed), the experiment registry receives entries from single-mode backtests, OMR and CSCM use cost-tier-derived costs, methodology Sections 11 and 12 are written, all relevant agent prompts reference them, and a new `live-ops` agent provides paved-path access to routine EC2 operations.

## Changes Made

### PR 0a -- strip DATA LAYER duplicates (commit `e86d73f`)

- **`.claude/agents/backtest-driver.md`** lines 65-99 collapsed to a 5-line pointer to methodology Section 10.5 (-33 lines).
- **`.claude/agents/backtest-optimizer.md`** lines 57-96 collapsed similarly (-37 lines); kept one optimizer-specific paragraph on cross-asset universe choices.
- Canary commit for the methodology-as-single-source-of-truth pattern. Agents now resolve storage paths via Section 10.5 rather than carrying inline tables.

### PR 0b -- cleanup + rename (commit `59d239d`)

- **`.claude/agents/trade-log-analyzer.md`**: hardcoded EC2 identifiers (instance ID, EIP, SSH user, key path, region) replaced with `.env`-loaded values. Establishes the pattern PR 4's `live-ops` reuses.
- **`.claude/skills/implement-strategy/SKILL.md`**: added methodology pointer (Sections 1 / 7 / 11). The skill that authors strategy code now references the rules that govern it.
- **`trading-lead.md` -> `strategy-lead.md`**: `git mv` plus frontmatter update, and forward references updated in `.claude/hooks/save_state.py`, `.claude/rules/strategy-pipeline.md`, and `CLAUDE.md`. Methodology changelog and historical progress logs retain `trading-lead` as accurate historical record.

### PR 0c -- planning docs allow-listed (commit `c097fec`)

- **`.gitignore`**: `!docs/planning/` and `!docs/planning/20260512*.md` allow-list rule added. Other prior planning docs (Nov-Dec 2025) deliberately NOT brought in -- two of them contain hardcoded EC2 IPs, instance IDs, security group IDs, and home CIDR ranges that need scrubbing first.
- **`docs/planning/20260512_methodology_rollout_v3_plan.md`** committed as the authoritative v3 plan currently being executed.

### PR 1a -- wire `append_run` into single-mode backtests (commit `67bf703`)

- **`src/backtest_runner.py`**: new `_append_to_registry` helper plus call at the end of `run_single_from_config`. Captures wall-clock start, extracts metrics from `portfolio.stats()`, populates strategy / params / dates / metrics / config-sha / git-sha / env-hash / host / wall-clock window. `agent_name='backtest-runner'`, `phase='initial'`.
- Best-effort during initial rollout: log-and-continue on append failure so existing backtests aren't broken by a registry outage. Tighten to fatal once stable.
- Sweep / optimize / walk-forward modes NOT yet wired (they don't expose a single `portfolio` object; needs per-config row design); flagged as follow-up.

### PR 1c -- cost-tier wiring (commit `1b72551`)

- **`src/settings/schema.py`**: new `CostsSettings` block on `BacktestConfig` with `tier`, `bps_override`, `stop_slippage_multiplier`. Field is optional during transition.
- **`src/backtest_runner.py`**: new `_resolve_costs` helper translates `costs.tier` -> round-trip bps -> per-side fees via the `src/backtesting/costs/` modules shipped last session. When `costs.tier` is unset, falls back to raw `backtest.fees` / `backtest.slippage` with a deprecation warning.
- **`config/backtesting/omr_backtest.yaml`**: `costs.tier: leveraged_etf` (22.5 bps RT).
- **`config/backtesting/cscm_baseline.yaml`**: `costs.tier: crypto_major` (126 bps RT). The previous fees+slippage of ~30 bps RT was optimistic by ~4x relative to Coinbase retail taker fees.
- 11 other backtest YAMLs still use raw fees/slippage and emit the deprecation warning. Migration follow-up.

### PR 2 -- methodology Sections 11 and 12 (commit `4740aef`)

- **`docs/methodology/backtesting.md`** appended with:
  - **Section 11** (Exit Logic and Profit-Taking Methodology, 11 subsections): exit taxonomy, bar-resolution requirements, same-bar fill order (stops fill first), gap modeling, stop-specific slippage (1.5x-3.0x multipliers), MAE/MFE methodology with required trade-log fields and stop-sizing procedure, profit-taking by asset class (incl. options 50%-of-max / 21-DTE rules), parameter budget for exits, code-reviewer responsibilities, tightened sensitivity for exit-level parameters, registry schema extension.
  - **Section 12** (Required Diagnostic Outputs, 6 subsections): trade-level metrics with consistency gate, capacity curve at $50K / $250K / $1M / $5M / $25M, regime transition analysis, hyperparameter temporal stability via K-window CV, benchmark/IR per asset class, consolidated diagnostic checklist with severity gates.
  - Appendix table rewritten: actual on-disk agents in the main table; `portfolio-integrator`, `strategy-architect`, `strategy-implementer` moved to a "Future agents" block with explicit promotion triggers (decision B).
- **`docs/methodology/CHANGELOG.md`** created tracking v1 (initial) and v2 (this PR).
- **`.claude/rules/strategy-pipeline.md`** pointer table gains Section 11 and 12 rows.

### PR 3 -- agent prompts updated for Sections 11/12 (commit `65fbbb5`)

- **`.claude/agents/strategy-lead.md`**: methodology pointer extended (1, 5, 6, 10, 11, 12). New "Section 12 gates" subsection lists the seven operational gates the lead applies in Phase 6 / 9.
- **`.claude/agents/backtest-driver.md`**: methodology pointer extended (1, 2, 3, 4, 8, 9, 10, 11, 12). New "Required Diagnostic Outputs" subsection lists the Tier-1 + exit-logic diagnostics to produce. Hardcoded `C:/Users/qwqw1/anaconda3/...` path replaced with portable `conda run` / EC2-venv split. Options-slippage rule rewritten from "50-75% of bid-ask" to the alpha-fraction-of-half-spread model from methodology 4.5 with the by-liquidity table.
- **`.claude/agents/backtest-optimizer.md`**: methodology pointer extended. New "Tightened sensitivity" subsection (Section 11.10: BRITTLE drops from 0.5x to 0.7x best Sharpe for exit-logic params; optimizer-discovered stops without MAE/MFE rejected at Phase 9). New "Parameter temporal stability" subsection (Section 12.4 K-window procedure with research-phase waiver). Same conda-path correction.
- **`.claude/agents/code-reviewer.md`**: methodology pointer extended to include Section 11 for strategies with non-time-based exits. New "Exit Logic Reviews (Section 11.9)" subsection with the six review focus items.
- `trade-log-analyzer.md` already pointed at Section 10 from PR 0b -- no change needed.

### PR 4 -- `live-ops` agent (commit `018c0dc`)

- **`.claude/agents/live-ops.md`** created with six canned recipes (`status`, `metrics`, `journal`, `start-instance`, `stop-instance`, `restart`, `sync-dashboards`). All state-changing recipes require explicit user yes/no confirmation before execution. Loads EC2 identifiers from `.env`. Distinct from `trade-log-analyzer` which is diagnostics-only.
- **`CLAUDE.md`**: agent table gains the `live-ops` row.
- **`docs/architecture/infra_patterns.md`**: new "Operations agents" section distinguishing the two ops agents and noting that strategy-pipeline agents are separate.

## Commits

- `e86d73f` docs(agents): strip DATA LAYER duplicates per v3 plan PR 0a
- `59d239d` docs(agents): cleanup + rename trading-lead to strategy-lead (PR 0b)
- `c097fec` docs(planning): allow-list docs/planning/ and add v3 methodology rollout plan
- `67bf703` feat(experiments): wire append_run into run_single_from_config (PR 1a)
- `1b72551` feat(costs): wire cost-tier from methodology Section 4 (PR 1c)
- `4740aef` docs(methodology): append Sections 11 and 12 (PR 2)
- `65fbbb5` docs(agents): update prompts for methodology Sections 11 and 12 (PR 3)
- `018c0dc` feat(agents): add live-ops agent for routine EC2 operations (PR 4)

## Known Issues / Remaining Work

- **PR 1b not yet executed**: wire `append_run` into `run_sweep_from_config`, `run_optimize_from_config`, `run_walk_forward_from_config` and the five optimizer modules under `src/backtesting/optimization/`. These modes return aggregate results (not a single `portfolio`) so the wiring needs per-config row semantics with shared `parent_run_id`. Roughly 1 hr.
- **Cost-tier migration**: 11 of 13 backtest YAMLs in `config/backtesting/` still use raw `fees`/`slippage` and emit the deprecation warning. Each one needs a tier assignment (e.g., `bmsb_crypto.yaml` -> `crypto_major` / `crypto_alt` decision; `hv_orb_baseline.yaml` -> tier depends on universe; etc.). Once all migrated, `_resolve_costs` can be tightened to raise on missing tier instead of warning.
- **Engine cost-tier integration is via fee fraction, not slippage**: `_resolve_costs` puts entire round-trip into per-side `fees` and leaves `slippage=0`. If the engine treats these differently (e.g., slippage applied on every fill including stops), the 1.5x cost-sensitivity test and stop-specific multipliers from Section 11.5 may not interact correctly. Verify with a known-good backtest before relying on the 1.5x gate.
- **`append_run` is best-effort**: registry append failures log and continue rather than fail the run. Tighten to fatal per Section 9.3 once stability is confirmed.
- **Smoke tests for live-ops not yet run**: the agent prompt is shipped but the recipes have not been exercised against a live EC2 session. First user-driven invocation will validate.
- **Pre-existing test failures** unchanged from the earlier session log -- IBKR config defaults expect `client_id=1`, BacktestEngine signature mismatch in `test_validate_with_mock_data`.

## Validation

- After PR 0a, agent files dropped ~64 lines of inline data-layer content; smoke test deferred until next backtest dispatch.
- After PR 0b, `git grep "trading-lead"` outside `docs/progress/` returns only the CLAUDE.md backwards-compat hint line. `.env.example` already contains the EC2 keys.
- After PR 1a + 1c, schema parses cleanly; OMR resolves to per-side fees=0.001125 (22.5 bps RT) and CSCM resolves to per-side fees=0.006300 (126 bps RT). 48 tests pass across registry / statistics / costs / walk-forward.
- After PR 2, methodology grew from 42,993 to 73,631 chars; section ordering is clean (`## Section 11` immediately follows Section 10; Section 12 follows; Changelog and Appendix after).
- After PR 3, all five agent files have updated pointers; no inline magic-number rules remain in `.claude/`.
- After PR 4, `live-ops` agent file is in place. CLAUDE.md and infra_patterns.md updated.
- Full `pytest` for new modules: 48 passed; 1 pre-existing failure unrelated to this work.

## Decisions Made (recorded for future sessions)

- **Decision B confirmed**: `portfolio-integrator`, `strategy-architect`, `strategy-implementer` agents NOT created. Methodology Section 6 (the rules) stays in effect; the orchestrator handles those concerns inline until a concrete trigger surfaces.
- **Cost-tier defaults**: OMR -> `leveraged_etf`, CSCM -> `crypto_major`. RAMP has no equity backtest config yet so no tier assigned.
- **Hard-fail on missing cost tier**: deferred. Soft-rollout with deprecation warning while remaining 11 YAMLs are migrated.
- **`live-ops` confirmation discipline**: yes/no prompts, not stricter (no service-name retyping). Can be tightened later if needed.
- **Methodology section numbering**: 1-12 frozen. Future sections (Tier 2 / Tier 3 checks) must append, never insert.
