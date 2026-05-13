# Methodology Rollout v3: Cleanup, Registry Wiring, Methodology Expansion, Live Ops

**Status**: Draft -- awaiting review
**Owner**: Shuyang
**Author**: Claude
**Created**: 2026-05-12
**Location**: `docs/planning/20260512_methodology_rollout_v3_plan.md`
**Supersedes**: `docs/planning/20260512_methodology_rollout_v2_plan.md`
**Depends on**: Phases 1-3 of methodology rollout (landed 2026-05-12 -- see `docs/progress/20260512_RAMP_METRICS_FIX_AND_METHODOLOGY_PHASES_1_3.md`)

---

## Executive summary

The previous v2 plan focused on extending the methodology with two new sections (exit logic, required diagnostics) and creating a `portfolio-integrator` agent. After reviewing the actual on-disk agent files alongside operational feedback from a recent CC session, v3 reorders and rebalances the work.

**Three changes from v2:**

1. **Cleanup goes first.** The phase 1-3 rollout added methodology pointers to existing agents but left ~70 lines of duplicated content (DATA LAYER section in two backtest agents, hardcoded EC2 identifiers in trade-log-analyzer). Until these are removed, the methodology-as-single-source-of-truth pattern isn't actually in effect. PR 0 fixes this.
2. **`live-ops` agent is added.** A real gap surfaced from operational use: routine ops tasks (instance start/stop, dashboard sync, journalctl tails, metrics queries, service restarts) currently happen manually through one-off bash invocations. A read-mostly ops agent with canned recipes addresses this. Distinct from `trade-log-analyzer` (diagnostics-only, never modifies state).
3. **`portfolio-integrator` is deferred.** v2 created the agent file; v3 keeps methodology Section 6 (the *rules*) but marks the agent as "future, lead handles inline until triggered." Trigger: first portfolio-integration question that doesn't fit in the orchestrator's head. This aligns with CC's decision-B framing and avoids creating an agent that has nothing to do yet.

**Five things remain from v2:**

- Methodology Sections 11 (exit logic) and 12 (required diagnostics) -- still needed, still high-leverage
- Registry wiring (`append_run`) -- still the foundational gap; DSR uses N=1 until this lands
- Cost-model wiring -- strategy configs still don't pull tier from methodology Section 4
- Agent prompt updates for Sections 11-12 -- still required after methodology lands
- Naming-consistency rename of `trading-lead` -> `strategy-lead` -- added per CC suggestion

**Sequence (six PRs, ordered by leverage and risk):**

| PR | Content | Risk | Effort | Blocks |
|---|---|---|---|---|
| **PR 0a** | Strip DATA LAYER duplicates from two backtest agents | Very low | 30 min | Validates methodology-as-source-of-truth pattern |
| **PR 0b** | De-hardcode EC2 identifiers; add methodology pointer to implement-strategy skill; rename `trading-lead` -> `strategy-lead` | Low | 2-3 hr | None |
| **PR 1** | Wire `append_run` + cost models from methodology Section 4 | Medium | 1-2 days | All downstream methodology work |
| **PR 2** | Methodology Sections 11 (exit logic) and 12 (required diagnostics) | Low (doc-only) | 1 day | PR 3 |
| **PR 3** | Update existing agents to read Sections 11 and 12 | Low-medium | 1 day | None |
| **PR 4** | Create `live-ops` agent | Medium (new agent shape) | 1 day | None |

**Total estimated effort**: 5 working days. PR 0a should land first as its own commit to validate the architecture before further work depends on it.

---

## Goals and non-goals

### Goals

- Eliminate methodology-content duplication in agent files so methodology really is the single source of truth
- Remove hardcoded operational identifiers (EC2 instance, IP, SSH key, username) from `trade-log-analyzer` in favor of `.env`-loaded values
- Add methodology pointer to `.claude/skills/implement-strategy/SKILL.md` so the skill that authors strategy code references the rules that govern it
- Rename `trading-lead` -> `strategy-lead` to match methodology vocabulary
- Wire `append_run` into the backtest runner and every optimization entry point so the experiment registry actually fills with data
- Wire cost models (methodology Section 4) into strategy configs and the backtest engine
- Append methodology Sections 11 (exit logic, profit-taking, MAE/MFE) and 12 (required diagnostic outputs)
- Update existing agents to read Sections 11 and 12 and produce required outputs
- Create `live-ops` agent for routine ops tasks distinct from trade-log diagnostics

### Non-goals

- **`portfolio-integrator`, `strategy-architect`, `strategy-implementer` agents** -- deferred per decision B. Methodology Section 6 stays; the agent doesn't exist until the lead can't fit the work in its head. Trigger: first portfolio-integration question requiring multi-file return-stream analysis OR first strategy where the lead's blueprint design phase needs its own context budget. Methodology Appendix will be updated to mark these as "future" rather than referencing them as if they exist.
- **JSON-schema-based handoff contracts** for inter-agent communication. Registry covers 80% of the same need. Schema-based handoffs deferred until they're load-bearing.
- **Tier 2 and Tier 3 diagnostic checks** (outlier-trade contribution, ulcer index, parameter cross-correlation, bootstrap CIs, synthetic-data robustness, multi-objective optimization). Defer until pipeline runs end-to-end and reveals which are most needed.
- **Model-tier changes** for existing agents. Independent of this plan; lands via the uncommitted working-tree changes for `code-architect` / `code-explorer` / `codebase-analyzer`.

---

## PR 0a: Strip DATA LAYER duplicates

**Goal**: Remove inline DATA LAYER sections from `backtest-driver.md` and `backtest-optimizer.md`. Both reproduce methodology Section 10.5 with slight formatting variations and a stale "as of 2026-04" date stamp. Until removed, the methodology can't be the single source of truth.

**Why this PR is separate**: it's the smallest, lowest-risk change in the entire plan, and its success validates the methodology-as-single-source-of-truth pattern. If agents continue working correctly without the inline DATA LAYER context, the architectural choice from phase 1-3 is confirmed and the rest of the rollout proceeds with confidence. If something breaks (e.g., agents start writing scripts with wrong paths because they didn't actually read the methodology), it surfaces immediately rather than buried inside a larger PR.

**Estimated effort**: 30 minutes.

### Files touched

```
.claude/agents/backtest-driver.md
.claude/agents/backtest-optimizer.md
```

### Specific changes

#### Change 0a.1: backtest-driver.md DATA LAYER replacement

Current state: lines 63-99 contain a "DATA LAYER (read before any backtest)" section with the storage root, an "Available data on disk" table, pre-flight check code, and a loading-tools table. All of this duplicates methodology Section 10.5.

Replace with this 5-line block:

```
## Data layer

Storage paths, available data on disk, and unified `DataAcquisitionManager` usage are specified in `docs/methodology/backtesting.md` Section **10.5**. Always resolve storage root via `from src.settings import get_local_storage_dir`. Use `DataAcquisitionManager` from `src/data/acquisition.py` for downloads; never write ad-hoc downloaders.

For loading data in scripts you write, the available tools (Polars cache loader, streaming loader, direct partition reads, CompositeDataProvider fallback chain) are listed in methodology Section 10.5.
```

Net change: -33 lines from backtest-driver.md (582 -> 549).

#### Change 0a.2: backtest-optimizer.md DATA LAYER replacement

Current state: lines 55-96 contain the same DATA LAYER section with an additional "Unified data acquisition" code block. Even more duplication than backtest-driver.

Replace with:

```
## Data layer

Storage paths, available data on disk, and unified `DataAcquisitionManager` usage are specified in `docs/methodology/backtesting.md` Section **10.5**. Always resolve storage root via `from src.settings import get_local_storage_dir`. Use `DataAcquisitionManager` from `src/data/acquisition.py` for downloads; never write ad-hoc downloaders.

**Cross-asset implication for optimization**: when proposing parameter spaces or universes, do not default to equities-only. Futures (ES/NQ for index, CL/GC for commodities), FX, and crypto are immediately available -- propose at least one cross-asset variant for index, commodity, or pairs work. (Asset class list and coverage in methodology Section 10.5.)

For data loading in scripts (Polars cache, streaming loader, direct partition reads, CompositeDataProvider fallback), see methodology Section 10.5.
```

Net change: -37 lines from backtest-optimizer.md (983 -> 946).

Note: the "Cross-asset implication" sentence is the only non-duplicated content worth keeping in the agent prompt. The rest is pure methodology content.

### Acceptance criteria

- [ ] `backtest-driver.md` is ~30 lines shorter; no more "Available data on disk" table inline
- [ ] `backtest-optimizer.md` is ~35 lines shorter; same
- [ ] Both agents still reference methodology Section 10.5 in the data-layer pointer
- [ ] Smoke test: dispatch backtest-driver to run a simple backtest (e.g., 1-symbol daily-bar OMR backtest). Verify it correctly resolves the storage root, finds the data, and produces a report. Confirm it reads the methodology section by checking its tool-call log for a Read on `docs/methodology/backtesting.md`.
- [ ] Smoke test: dispatch backtest-optimizer with a small parameter grid (3-5 configs). Same verification.

### Rollback

Trivial. If agents start writing scripts with wrong paths or hallucinated data tables, revert this PR and restore the inline DATA LAYER sections. The agents continue functioning with the duplication. Risk surface is bounded to two files.

### Risks

- **Agents may not actually read the methodology section pointer.** This is the failure mode being tested. Mitigation: smoke tests above; if observed, the methodology pointer pattern itself needs reconsidering.
- **Drift between the two agents' data-layer pointer phrasing.** The replacement text above is intentionally identical for both files so future updates change one place. Reviewers should verify this on merge.

---

## PR 0b: Cleanup and rename

**Goal**: De-hardcode EC2 identifiers in `trade-log-analyzer.md`, add methodology pointer to the `implement-strategy` skill, and rename `trading-lead` to `strategy-lead` for consistency with methodology vocabulary.

**Why bundled**: all three are low-risk cleanup with no interdependencies. Bundling reduces PR overhead. Rename specifically benefits from being separate from the methodology-content changes in PR 0a so a rollback affects only the rename.

**Estimated effort**: 2-3 hours total (1 hour for EC2 de-hardcoding, 5 minutes for skill update, 1-2 hours for rename grep pass).

### Files touched

```
.claude/agents/trade-log-analyzer.md
.claude/skills/implement-strategy/SKILL.md
.claude/agents/trading-lead.md -> .claude/agents/strategy-lead.md   # git mv
.claude/rules/strategy-pipeline.md                                  # if it references trading-lead
docs/methodology/backtesting.md                                     # Appendix update
CLAUDE.md                                                           # if it references trading-lead
.env.example                                                        # new file documenting required keys
```

### Specific changes

#### Change 0b.1: De-hardcode EC2 identifiers in trade-log-analyzer

Current state: lines 25-28 in `trade-log-analyzer.md` hardcode Instance ID, Elastic IP, Username, SSH Key path. Plus line 41 hardcodes the instance ID in a `describe-instances` invocation.

Replace the connection table and Phase 1 startup block with:

```
## EC2 connection

Load these from `.env` at the start of every session. If any are missing, ask the user to populate them -- do not hardcode fallbacks.

INSTANCE_ID=$(grep '^EC2_INSTANCE_ID=' .env | cut -d= -f2)
ELASTIC_IP=$(grep '^EC2_ELASTIC_IP=' .env | cut -d= -f2)
SSH_USER=$(grep '^EC2_SSH_USER=' .env | cut -d= -f2)
SSH_KEY=$(grep '^EC2_SSH_KEY_PATH=' .env | cut -d= -f2)
AWS_REGION=$(grep '^AWS_REGION=' .env | cut -d= -f2)

Hardware: Amazon Linux 2023 (ARM64), t4g.medium. Memory threshold and other environment specifics per methodology Section **10.6**.

## Phase 1: EC2 instance startup

Instance may be stopped (scheduled 4:30 PM - 9:00 AM ET weekdays, all weekend). Check state first:

aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --region $AWS_REGION \
    --query "Reservations[0].Instances[0].State.Name" \
    --output text

If stopped, start it via the helper script (handles the AWS CLI call and waits for SSH readiness).

State handling:
- `running` -> proceed to Phase 2
- `stopped` -> start, wait ~60 seconds for SSH
- `pending` -> wait for running state
- `stopping` -> wait for stopped, then start
```

Also create `.env.example` at the repo root to document the required keys (if it doesn't already exist):

```
# .env.example
# Copy to .env and populate. .env is in .gitignore.

# AWS / EC2
EC2_INSTANCE_ID=i-XXXXXXXXXXXXX
EC2_ELASTIC_IP=X.X.X.X
EC2_SSH_USER=ec2-user
EC2_SSH_KEY_PATH=~/.ssh/your-key.pem
AWS_REGION=us-east-1

# (other existing keys preserved)
```

#### Change 0b.2: Add methodology pointer to implement-strategy skill

Current state: `.claude/skills/implement-strategy/SKILL.md` doesn't reference the methodology. Add this as the second line after the frontmatter:

```
**Methodology**: Consult `docs/methodology/backtesting.md` Sections **1** (bias prevention -- every signal must use `.shift(1)` or equivalent; no full-sample statistics in features; no `bfill` on price data) and **7** (point-in-time data conventions -- fundamentals lag, news timestamps, index membership) before writing strategy code. Section **11** also applies if the strategy has any non-time-based exit (stops, targets, trailing rules) -- the trade log must include `mae_pct`, `mfe_pct`, `hit_stop`, `hit_target`, `exit_reason`, `bars_held` fields per 11.6.
```

If `.claude/skills/implement-strategy/SKILL.md` doesn't exist yet (the phase 1-3 progress doc references it as a future deliverable), create the file with the pointer plus a brief skill scaffold.

#### Change 0b.3: Rename `trading-lead` -> `strategy-lead`

Sequence:

1. **Grep for all references first:**

```
# Code, config, docs
git grep -n "trading-lead" -- '.claude/' 'docs/' 'CLAUDE.md' 'TODO.md' '*.md' '*.yaml'

# Excluding historical progress logs (those reference the old name accurately)
git grep -n "trading-lead" -- '.claude/' 'docs/' 'CLAUDE.md' 'TODO.md' '*.md' '*.yaml' \
    | grep -v '^docs/progress/'
```

Expected hits per grep of the current files:
- `.claude/agents/trading-lead.md:2` (frontmatter name)
- `.claude/agents/trading-lead.md:3` (description)
- `.claude/rules/strategy-pipeline.md` (any references to "the orchestrator" by name)
- `docs/methodology/backtesting.md` (Appendix table -- currently lists `strategy-lead` as the expected name; this row stays the same after rename)

2. **Rename the file:**

`git mv .claude/agents/trading-lead.md .claude/agents/strategy-lead.md`

3. **Update the file's frontmatter:**

```
name: strategy-lead
description: Orchestrator for the algorithmic trading strategy pipeline. Reads TODO.md, dispatches to specialist agents, enforces backtest integrity at every phase, and manages session recovery across rate limit interruptions.
tools: Read, Write, Edit, MultiEdit, Bash, Glob, Grep, Agent
model: sonnet
```

4. **Update forward references:**

For each non-historical file that grep found, replace `trading-lead` with `strategy-lead` carefully (context-by-context, not blind sed -- some references may be inside quoted prose discussing the rename).

5. **Verify methodology Appendix:**

The methodology Appendix table already lists `strategy-lead` (it was written in advance of the rename). Confirm this row exists and the rename brings agent name into agreement.

6. **Leave historical progress logs alone.** `docs/progress/20260512_RAMP_METRICS_FIX_AND_METHODOLOGY_PHASES_1_3.md` and similar files describe past state when the agent was named `trading-lead`. Their references stay accurate as historical record.

### Acceptance criteria

- [ ] `trade-log-analyzer.md` contains no hardcoded instance ID, EIP, SSH key path, or username
- [ ] `.env.example` exists at repo root with EC2 keys documented
- [ ] `.claude/skills/implement-strategy/SKILL.md` exists (creating if needed) and references methodology Sections 1, 7, and 11
- [ ] `.claude/agents/strategy-lead.md` exists; `.claude/agents/trading-lead.md` no longer exists
- [ ] `git grep "trading-lead"` outside of `docs/progress/` returns no hits
- [ ] Methodology Appendix `strategy-lead` row matches the actual agent name
- [ ] Smoke test: open a Claude Code session and invoke `strategy-lead` by name; verify it loads correctly
- [ ] Smoke test: dispatch `trade-log-analyzer` with `.env` populated; verify it loads identifiers correctly and connects

### Rollback

Rename is reversible: `git mv strategy-lead.md trading-lead.md` plus revert the frontmatter and forward references. EC2 de-hardcoding is reversible by restoring the table. Skill update is one line.

### Risks

- **Stale references in user shell aliases / scripts** that reference `trading-lead` by name. Mitigation: grep wider (`~/.claude/` user config, any user-side hooks) before merging.
- **`.env` doesn't exist on the dev machine** when trade-log-analyzer first tries to load. Mitigation: the prompt instructs the agent to ask the user to populate `.env` rather than fail silently; verify the prompt change actually has this fallback.
- **Methodology Appendix and other docs may have lingering "lead" terminology** that conflates `strategy-lead` with `trading-lead`. Grep pass should catch this.

---

## PR 1: Wire `append_run` and cost models

**Goal**: Make methodology Sections 4 (cost models), 8 (reproducibility), and 9 (registry) operationally real. Until this lands, `append_run` is never called, DSR uses N=1 (effectively no multiple-testing correction), and strategy configs use ad-hoc costs that may not match the methodology.

**Why this PR matters most**: it's the foundational gap noted in the phase 1-3 progress doc's "Known Issues" section. Every downstream methodology gate (PSR, DSR, PBO, capacity, cost sensitivity) depends on the registry being populated.

**Estimated effort**: 1-2 days. Mostly mechanical wiring, but touches many files.

### Files touched

```
src/experiments/registry.py                              # extend with query helpers
src/backtest_runner.py                                   # wire append_run
src/backtesting/optimization/grid_search.py              # wire append_run per config
src/backtesting/optimization/sweep_runner.py             # wire append_run per config
src/backtesting/optimization/random_search.py            # wire append_run per config
src/backtesting/optimization/bayesian_optimizer.py       # wire append_run per config
src/backtesting/optimization/genetic_optimizer.py        # wire append_run per config
src/backtesting/engine/backtest_engine.py                # read cost tier from config
src/utils/git.py                                         # new -- git SHA helper
src/utils/config_hash.py                                 # new -- config hash helper
src/utils/env_hash.py                                    # new -- pip freeze hash helper
src/settings.py                                          # add get_experiment_registry_path()
config/strategies/omr_strategy.yaml                      # add costs.tier
config/strategies/ramp_strategy.yaml                     # add costs.tier
config/strategies/cscm_strategy.yaml                     # add costs.tier
config/strategies/<each existing strategy>.yaml          # add costs.tier
tests/experiments/test_registry_wiring.py                # new
tests/backtesting/test_cost_model_wiring.py              # new
```

### Specific changes

#### Change 1.1: Registry query helpers

Add to `src/experiments/registry.py`:

```python
import duckdb
import logging
from src.settings import get_experiment_registry_path

logger = logging.getLogger(__name__)


def query_project_trial_count() -> int:
    """Query cumulative project-wide trial count from the registry.

    Used by DSR computation per methodology Section 2.3 -- the N in
    expected_max_sharpe() is project-wide, not per-run.

    Returns 1 if the registry is empty (defensive: a brand-new project
    with no prior optimization runs has effectively N=1, and DSR ~ PSR).
    """
    try:
        path = get_experiment_registry_path()
        con = duckdb.connect(str(path), read_only=True)
        result = con.execute("""
            SELECT COALESCE(SUM(combinations_in_run), 1)
            FROM runs
            WHERE agent_name = 'backtest-optimizer'
        """).fetchone()
        con.close()
        return max(1, int(result[0]))
    except Exception as e:
        logger.warning(f"Registry query failed, defaulting to N=1: {e}")
        return 1


def query_incumbent_return_streams(incumbent_names: list[str], min_history_days: int = 504):
    """Pull daily OOS return streams for incumbent strategies.

    Used by orchestrator (until portfolio-integrator agent exists) for
    correlation sanity checks. Returns a dict {strategy_name: pd.DataFrame}.
    """
    # Implementation per methodology Section 9
    ...


def append_run(**kwargs):
    """Append a single run row to experiments.duckdb.

    Schema per methodology Section 9.1. All identity fields (git_sha,
    config_sha, data_snapshot_date, python_env_hash, random_seeds) are
    required per Section 8.1. The function does NOT silently drop a row
    if required fields are missing -- it raises ValueError.

    On any other failure (DB locked, disk full, etc.), logs the error
    and re-raises. Callers wrap in try/except if they want non-fatal
    behavior; default is fatal.
    """
    # Existing implementation extended with PR 2's new columns:
    #   exit_logic_summary, mae_mfe_validated (default None / False)
    ...
```

#### Change 1.2: Wire `append_run` in `src/backtest_runner.py`

At the end of the main `run()` function, before returning, append a row:

```python
from src.experiments.registry import append_run, query_project_trial_count
from src.utils.git import get_current_sha
from src.utils.config_hash import compute_config_sha
from src.utils.env_hash import get_python_env_hash
import socket, uuid
from datetime import datetime

wall_clock_end = datetime.utcnow()
run_id = str(uuid.uuid4())

append_run(
    run_id=run_id,
    timestamp_utc=wall_clock_end,
    strategy_name=strategy_name,
    agent_name='backtest-runner',  # 'backtest-driver' if invoked through that agent
    phase='initial',
    parent_run_id=None,
    params=strategy_params,
    universe_name=universe,
    asset_class=asset_class,
    data_frequency=data_frequency,
    window_start=window_start,
    window_end=window_end,
    is_start=is_start,
    is_end=is_end,
    oos_start=oos_start,
    oos_end=oos_end,
    n_folds=n_folds,
    metrics=metrics_dict,
    regime_breakdown=regime_dict,
    fold_metrics=fold_metrics_dict,
    cost_tier_used=engine.cost_tier_used,
    cost_bps=engine.entry_bps,
    cost_sensitivity=cost_sensitivity_dict,
    combinations_in_run=1,
    combinations_project=query_project_trial_count(),
    git_sha=get_current_sha(),
    config_sha=compute_config_sha(config_path),
    data_snapshot_date=data_snapshot_date,
    python_env_hash=get_python_env_hash(),
    random_seeds=seeds_dict,
    wall_clock_start=wall_clock_start,
    wall_clock_end=wall_clock_end,
    host=socket.gethostname(),
    verdict='PENDING',
    verdict_reasons=None,
    notes=run_notes,
)
```

#### Change 1.3: Wire `append_run` in each optimizer

Each of the five optimizer modules gets the same pattern inside its per-configuration loop:

```python
optimization_run_id = str(uuid.uuid4())  # shared parent for the whole sweep

for config_idx, params in enumerate(param_space):
    config_run_id = str(uuid.uuid4())
    metrics = run_single_backtest(strategy_class, params, ...)

    append_run(
        run_id=config_run_id,
        agent_name='backtest-optimizer',
        phase=f'optimization_round_{round_number}',
        parent_run_id=optimization_run_id,
        params=params,
        metrics=metrics,
        combinations_in_run=len(param_space),
        combinations_project=query_project_trial_count(),
        # ... rest of identity fields
    )
```

Performance: DuckDB appends are ~ms. The backtest itself is seconds-to-minutes per config. No batching needed.

Concurrency: DuckDB serializes writes via file-level locking. For multi-worker optimizers, each worker opens its own connection; contention is real but bounded.

#### Change 1.4: Cost-tier defaults per asset class

Methodology Section 4 defines tier bps. Engine reads from strategy config:

```python
# src/backtesting/engine/backtest_engine.py

TIER_BPS_DEFAULTS = {
    # Equities
    'large_cap_liquid': 12,
    'mid_cap': 22,
    'leveraged_etf': 22,
    'small_cap_illiquid': 100,
    # FX
    'fx_major': 5,
    'fx_minor': 15,
    # Crypto
    'crypto_major': 60,
    'crypto_alt': 100,
    # Futures and options handled separately
    'futures': None,
    'options_liquid': None,
    'options_illiquid': None,
}


def _initialize_costs(self, strategy_config):
    cost_config = strategy_config.get('costs', {})
    tier = cost_config.get('tier')

    if tier is None:
        raise ValueError(
            f"Strategy config missing required 'costs.tier' field. "
            f"See methodology Section 4 for tier values."
        )

    if tier not in TIER_BPS_DEFAULTS:
        raise ValueError(f"Unknown cost tier '{tier}'. See methodology Section 4.")

    bps_override = cost_config.get('bps_override')
    self.entry_bps = bps_override if bps_override is not None else TIER_BPS_DEFAULTS[tier]
    self.stop_slippage_multiplier = cost_config.get('stop_slippage_multiplier', 1.5)
    self.cost_tier_used = tier
```

#### Change 1.5: Strategy configs

Each existing strategy gains `costs.tier`:

```yaml
# config/strategies/omr_strategy.yaml
strategy:
  name: omr
  # ... existing fields preserved ...
  costs:
    tier: leveraged_etf
    bps_override: null
    stop_slippage_multiplier: 1.5

# config/strategies/ramp_strategy.yaml
strategy:
  name: ramp
  costs:
    tier: mid_cap
    bps_override: null
    stop_slippage_multiplier: 1.5

# config/strategies/cscm_strategy.yaml
strategy:
  name: cscm
  costs:
    tier: crypto_major
    bps_override: null
    stop_slippage_multiplier: 1.5
```

Confirm tier choices with Shuyang before merging.

#### Change 1.6: Setting helper

Add to `src/settings.py`:

```python
def get_experiment_registry_path() -> Path:
    """Path to the project experiment registry DuckDB file.

    Defaults to <repo>/output/experiments.duckdb. Override via
    HOMEGUARD_EXPERIMENT_REGISTRY environment variable for tests
    or per-environment isolation.
    """
    override = os.environ.get('HOMEGUARD_EXPERIMENT_REGISTRY')
    if override:
        return Path(override)
    return get_repo_root() / 'output' / 'experiments.duckdb'
```

### Acceptance criteria

- [ ] Running `python -m src.backtest_runner --config config/backtesting/omr_backtest.yaml` results in exactly one new row in `output/experiments.duckdb`
- [ ] Running `python -m src.backtesting.optimization.grid_search ...` with N configs results in N new rows, all sharing the same `parent_run_id`
- [ ] `query_project_trial_count()` returns correct cumulative count
- [ ] Every existing strategy config has a populated `costs.tier`
- [ ] Engine raises a clear error if `costs.tier` is missing from config
- [ ] DSR computation in optimizer uses registry-queried N (verified by reading optimizer log: should not say "N=1 fallback")
- [ ] Existing test suite passes; no regressions
- [ ] New tests `test_registry_wiring.py` pass: verify append happens on backtest run, on each optimizer config, with all required identity fields
- [ ] New tests `test_cost_model_wiring.py` pass: verify cost is read from config; verify error on missing tier; verify override works

### Rollback

If `append_run` causes issues mid-run, the call sites have try/except that re-raise (default fatal). To make non-fatal during initial rollout, wrap in try/except that logs and continues. Switch back to fatal once stability is confirmed.

For cost-tier wiring: if a strategy config fails to load due to missing `costs.tier`, the strategy is broken -- the engine refuses to run. This is intentional but means PR 1 must update all existing configs before merging.

### Risks

- **Schema drift**: methodology Section 9 may grow new columns. Mitigation: `schema_version` column with default 1; bump on schema changes.
- **DuckDB concurrent writes**: multi-worker optimizers may see contention. Mitigation: each process opens its own connection; monitor write latency on the first big optimization sweep.
- **Cost-tier choices wrong**: defaults assigned to OMR (`leveraged_etf`), RAMP (`mid_cap`), CSCM (`crypto_major`) are reasonable but should be confirmed with Shuyang before merge.
- **`config_hash` is non-deterministic if YAML round-trips reorder keys**. Mitigation: use a deterministic serializer (`yaml.safe_dump` with `sort_keys=True`) before hashing.

---

## PR 2: Methodology Sections 11-12

**Goal**: Append exit logic + required diagnostic outputs sections to `docs/methodology/backtesting.md`. Doc-only change.

**Estimated effort**: 1 day.

**Depends on**: PR 1 (Section 12.4 parameter stability requires registry to be live).

**Blocks**: PR 3 (agents reference new sections).

### Files touched

```
docs/methodology/backtesting.md          # append Sections 11 and 12, update Appendix
docs/methodology/CHANGELOG.md            # new -- track methodology version
.claude/rules/strategy-pipeline.md       # add 11 and 12 to pointer table
```

### Specific changes

#### Change 2.1: Append Section 11 (Exit Logic and Profit-Taking)

Full text from v2 plan. Subsections:
- 11.1 Exit-logic taxonomy (12 named exit types in a reference table)
- 11.2 Bar-resolution requirements
- 11.3 Same-bar fill-order convention (stops fill first)
- 11.4 Gap modeling
- 11.5 Stop-specific slippage (1.5x-3.0x multipliers by condition)
- 11.6 MAE/MFE methodology (required trade log fields; stop-sizing procedure)
- 11.7 Profit-taking by asset class (equities, futures, FX, crypto, options)
- 11.8 Stops and the parameter budget
- 11.9 Code-reviewer responsibilities for exit logic
- 11.10 Optimizer behavior with exit-level parameters (tightened sensitivity)
- 11.11 Exit logic and the experiment registry

**Adjustment from v2 plan**: Section 11.11 references to `portfolio-integrator` become "the orchestrator (until a portfolio-integrator agent is created) reads `exit_logic_summary`..."

#### Change 2.2: Append Section 12 (Required Diagnostic Outputs)

Full text from v2 plan. Subsections:
- 12.1 Trade-level metrics alongside portfolio metrics
- 12.2 Capacity curve
- 12.3 Regime transition analysis
- 12.4 Hyperparameter temporal stability
- 12.5 Benchmark comparison and information ratio
- 12.6 Consolidated diagnostic checklist table

#### Change 2.3: Update methodology Appendix

Stop pretending future agents exist. The single biggest correction from v2 plan:

```
## Appendix: Reading priority for agents

Agents currently in `.claude/agents/`:

| Agent | Must read | Should read |
|---|---|---|
| strategy-lead | 1, 5, 6, 10, 11, 12 | 2 (for verdicts), 9 |
| code-architect (when used for strategy work) | 1, 10, 11 | 4 (cost-aware design) |
| code-explorer | 10 | -- |
| code-reviewer | 1, 7, 11 (for strategies with exits) | 10 (paths) |
| backtest-driver | 1, 2, 3, 4, 8, 9, 10, 11, 12 | 5 (sanity check) |
| backtest-optimizer | 1, 2, 3, 5, 8, 9, 11, 12 | 4, 10 |
| trade-log-analyzer | 10 (services, brokers, env) | -- |
| live-ops (PR 4) | 10 | -- |
| codebase-analyzer | -- | -- |

**Future agents** (decision B -- defer until trigger):
- `portfolio-integrator`: trigger = first portfolio-integration question requiring multi-file return-stream analysis the orchestrator can't fit in its head. Methodology Section 6 (the rules) is in effect; lead handles inline until then.
- `strategy-architect`, `strategy-implementer`: trigger = first strategy where the blueprint phase needs its own context budget. Currently `code-architect` and the general-purpose agent handle these.

When a future agent is created, update this table.
```

#### Change 2.4: New methodology changelog

```
# docs/methodology/CHANGELOG.md

## v2 (2026-XX-XX)
- Section 11 added: Exit Logic and Profit-Taking Methodology (11 subsections)
- Section 12 added: Required Diagnostic Outputs (6 subsections)
- Registry schema extended: `exit_logic_summary`, `mae_mfe_validated` columns
- Appendix reading-priority table updated to reflect actual on-disk agents;
  marks `portfolio-integrator`, `strategy-architect`, `strategy-implementer`
  as future per decision B
- Gates added: trade-expectancy consistency, capacity, regime transitions,
  parameter temporal stability, information ratio
- Stop-loss governance: MAE/MFE-derived stops required for live deployment;
  optimizer-discovered stops without MAE/MFE backing are rejected

## v1 (2026-05-12)
- Initial consolidated methodology (Sections 1-10)
- Bias prevention, statistical framework (PSR/DSR/PBO), walk-forward (purge+embargo),
  cost models per asset class, stopping conditions, portfolio integration rules,
  point-in-time data, reproducibility, registry schema, Homeguard reference
- Replaces inline rules previously scattered across agent prompts
```

#### Change 2.5: Update strategy-pipeline.md pointer table

Add to the existing topic-to-section pointer table:

```
| Exit logic, stops, profit-taking, MAE/MFE, asset-class profit-taking rules | Section 11 |
| Required diagnostic outputs (trade-level metrics, capacity, regime transitions, parameter stability, benchmark/IR) | Section 12 |
```

Plus the trade-log schema requirement note from v2 plan.

### Acceptance criteria

- [ ] Methodology renders correctly (markdown lint passes)
- [ ] All Section 1-10 references in existing agent prompts still resolve
- [ ] `.claude/rules/strategy-pipeline.md` references Sections 11 and 12
- [ ] Methodology changelog reflects v2
- [ ] Methodology Appendix lists only actually-existing agents in main table; future agents in separate block
- [ ] No agent file changes in this PR

### Rollback

Doc-only PR. Trivial git revert.

### Risks

- **Section numbers stay stable.** Sections 1-10 must not renumber. New sections only append.
- **Section 6 references to `portfolio-integrator`** stay (the rules are valid). Section 11.11 must say "orchestrator (until portfolio-integrator exists) reads..." rather than "portfolio-integrator reads..."

---

## PR 3: Update agents to read Sections 11 and 12

**Goal**: Make Sections 11 and 12 operational by updating each agent's prompt to read the relevant sections and produce required outputs.

**Estimated effort**: 1 day.

**Depends on**: PR 2 (sections must exist before agents reference them).

**Blocks**: None -- final methodology-related PR.

### Files touched

```
.claude/agents/strategy-lead.md          # after PR 0b rename
.claude/agents/backtest-driver.md
.claude/agents/backtest-optimizer.md
.claude/agents/code-reviewer.md
```

(`trade-log-analyzer.md` and the read-only `code-explorer` / `code-architect` / `codebase-analyzer` agents are unchanged in this PR.)

### Specific changes

Carried forward from v2 plan PR 3. Summary of changes per agent:

#### Change 3.1: `strategy-lead.md`

Update methodology pointer to include Sections 11, 12. Add a new gating subsection covering all Section 12 gates (trade-expectancy consistency, capacity, regime transitions, parameter stability, information ratio, exit logic summary, MAE/MFE validation). Update agent roster table to reflect reality: strip references to non-existent strategy-architect / strategy-implementer; mark portfolio-integrator as future.

#### Change 3.2: `backtest-driver.md`

Update methodology pointer to include Sections 11, 12. Add "Required Diagnostic Outputs" section listing what to produce: trade-level metrics, capacity curve (~5-10 min extra compute per backtest), regime transition analysis, benchmark/IR, exit logic diagnostics if applicable.

Two existing-state corrections still pending from v2 plan:
- Replace hardcoded `C:/Users/qwqw1/anaconda3/...` path with `conda run -n fintech python` (Windows) or `~/Homeguard/venv/bin/python` (EC2)
- Replace options-slippage rule "50-75% of bid-ask" with the alpha-fraction-of-half-spread model from methodology Section 4.5

#### Change 3.3: `backtest-optimizer.md`

Update methodology pointer to include Sections 11, 12. Add Section 11.10 tightened-sensitivity behavior for exit-logic parameters (BRITTLE threshold drops from 0.5x to 0.7x best Sharpe). Add Section 12.4 parameter temporal stability requirement for live-bound strategies with >= 2 parameters.

Same conda path correction.

#### Change 3.4: `code-reviewer.md`

Update methodology pointer to include Section 11 for any strategy code under `src/strategies/` with non-time-based exits. Add Section 11.9 review checklist: bar-resolution match, same-bar fill-order documented, gap modeling present, trade log fields complete, stop slippage multiplier applied, parameter budget compliance.

### Acceptance criteria

- [ ] All four agent files reference the new methodology sections
- [ ] backtest-driver prompt includes Section 12 diagnostic spec
- [ ] backtest-optimizer prompt includes Section 11.10 and Section 12.4
- [ ] code-reviewer prompt includes Section 11.9 exit-logic checks
- [ ] strategy-lead's gating table includes the new Section 12 gates
- [ ] No agent prompt exceeds 50% of context budget when loaded fresh
- [ ] Smoke test: run a backtest end-to-end; verify all Section 12 diagnostics appear in the report; verify strategy-lead correctly gates based on them

### Rollback

PR 3 is mergeable in independent commits per agent. Revert one file at a time if needed.

### Risks

- **Agent prompts continue growing.** Methodology pointers prevent rule drift; operational gate specifications prevent gate drift. Both needed. Monitor total prompt size; backtest-driver and strategy-lead are the largest.
- **In-flight optimizer runs** before this PR don't produce parameter stability data. Mark as `parameter_stability: "not_assessed_pre_pr3"` and the lead handles gracefully.

---

## PR 4: Create `live-ops` agent

**Goal**: Add a read-mostly operations agent with canned recipes for routine tasks: status checks, metrics queries, journal tails, EC2 instance start/stop, Grafana dashboard sync, systemd service restarts (with user confirmation).

**Why this agent**: a recent CC session involved ~12 manual SSH + scp + journalctl + curl invocations to handle routine ops. The friction is real and the work is patterned -- a paved-path agent solves it. Distinct from `trade-log-analyzer`, which is diagnostics-only and explicitly forbidden from modifying state.

**Estimated effort**: 1 day.

**Depends on**: PR 0b (`.env` loading pattern established for EC2 identifiers).

**Blocks**: None.

### Files touched

```
.claude/agents/live-ops.md                                # new
docs/architecture/infra_patterns.md                       # mention live-ops in ops section
docs/methodology/backtesting.md                           # Appendix update (live-ops row)
tests/agents/test_live_ops_smoke.py                       # smoke test
```

Note: `strategy-lead.md` is NOT updated -- `live-ops` is not part of the strategy pipeline. It's a parallel ops surface invoked directly by the user.

### Specific changes

#### Change 4.1: New agent file `live-ops.md`

```
---
name: live-ops
description: |
  Read-mostly operations agent for routine Homeguard live-system tasks on EC2.
  Has canned recipes for: status checks, metrics queries, journal tails,
  EC2 instance start/stop, Grafana dashboard sync, systemd service restarts.

  Distinct from trade-log-analyzer (diagnostics-only, never modifies state).
  This agent CAN modify state for declared operations, but only with explicit
  user confirmation. Never modifies code, strategy configs, or trading state.

  ## When to use
  - Routine ops: instance starting/stopping, dashboard syncing, services restarting
  - Pulling metrics from Prometheus
  - Tailing journalctl for a specific service
  - Quick status checks

  ## When NOT to use
  - Diagnosing trade errors -> use trade-log-analyzer
  - Modifying strategy code or configs -> use general-purpose agent
  - Trade decisions -> manual
  - Methodology / backtest questions -> strategy-lead and its specialists

  ## Trigger phrases
  - "start the EC2 instance"
  - "what's the bot status"
  - "tail the OMR journal"
  - "get the latest metrics"
  - "sync the Grafana dashboards"
  - "restart the RAMP service" (requires confirmation)

tools: Read, Bash, Write
model: sonnet
color: orange
---

You are the Homeguard live-ops agent. Your job is to run routine operational tasks on the EC2-deployed trading system. You are read-mostly: any state-changing action requires explicit user confirmation before execution.

**Methodology**: Consult `docs/methodology/backtesting.md` Section **10** for service names, brokers, paths, and environment specifics. No backtest methodology is in scope.

## Core constraints

1. **NEVER modify code, configs, or trading state.** Code changes go through general-purpose agent. Strategy config changes are explicit human decisions.
2. **Confirm state changes.** Any action that mutates state (start instance, restart service, modify .env, push dashboard) MUST be explicitly confirmed by the user with a yes/no prompt before execution.
3. **Read-mostly default.** Status, metrics, journal queries do not require confirmation.
4. **Load identifiers from .env at session start.** Never hardcode instance ID, EIP, SSH key path, etc.

## .env loading

At session start, load:

INSTANCE_ID=$(grep '^EC2_INSTANCE_ID=' .env | cut -d= -f2)
ELASTIC_IP=$(grep '^EC2_ELASTIC_IP=' .env | cut -d= -f2)
SSH_USER=$(grep '^EC2_SSH_USER=' .env | cut -d= -f2)
SSH_KEY=$(grep '^EC2_SSH_KEY_PATH=' .env | cut -d= -f2)
AWS_REGION=$(grep '^AWS_REGION=' .env | cut -d= -f2)

If any are missing from .env, ask the user to populate them. Do not hardcode fallbacks.

## Canned recipes

### `status`
Check overall system health. Read-only.

ssh -i $SSH_KEY $SSH_USER@$ELASTIC_IP 'systemctl status homeguard-*.service --no-pager' || echo "instance may be stopped"

Report: which services are running, which are failed, instance uptime.

### `metrics [strategy] [metric]`
Query Prometheus on the EC2 instance. Read-only.

ssh -i $SSH_KEY $SSH_USER@$ELASTIC_IP \
  'curl -sG http://localhost:9090/api/v1/query --data-urlencode "query=$METRIC"'

If no metric specified, default to `hg_portfolio_equity_usd` and `hg_strategy_equity_usd` for all strategies.

### `journal <service> [--since=N] [--grep=PATTERN]`
Tail journalctl for a specific service. Read-only.

ssh -i $SSH_KEY $SSH_USER@$ELASTIC_IP \
  "TZ=America/New_York sudo journalctl -u homeguard-$SERVICE.service \
   --since '$SINCE' --no-pager $GREP_ARG"

### `start-instance`
Start the EC2 instance if stopped. REQUIRES CONFIRMATION.

Step 1: Check current state. Step 2: Print proposed action ("Will start instance $INSTANCE_ID in $AWS_REGION"). Step 3: Wait for user yes/no. Step 4: Execute.

aws ec2 start-instances --instance-ids $INSTANCE_ID --region $AWS_REGION

Then wait for SSH readiness (~60 seconds), confirm reachability.

### `stop-instance`
Stop the EC2 instance. REQUIRES CONFIRMATION.

Step 1: Check current state. Step 2: Print proposed action. Step 3: Confirm there are no open positions or active trading windows (consult schedule per methodology Section 10.4). Step 4: Wait for user yes/no. Step 5: Execute.

### `restart <service>`
Restart a specific systemd service. REQUIRES CONFIRMATION.

Step 1: Print proposed action ("Will restart homeguard-$SERVICE.service"). Step 2: Confirm with user. Step 3: Execute via SSH:

ssh -i $SSH_KEY $SSH_USER@$ELASTIC_IP \
  "sudo systemctl restart homeguard-$SERVICE.service"

Step 4: Verify the service came back up cleanly (read systemctl status).

### `sync-dashboards`
Pull Grafana dashboard JSON from EC2 to local. Read on remote, write locally.

scp -i $SSH_KEY $SSH_USER@$ELASTIC_IP:~/grafana_dashboards/*.json \
    ./dashboards/

Report which dashboards were updated.

## Escalation triggers

Report to user immediately, do not proceed:
- Any service is failed during market hours
- Instance is stopped during scheduled trading hours
- Memory usage > 3GB (per methodology Section 10.6 threshold)
- SSH connection fails on multiple retries
- AWS API returns auth errors

## Output format

Always end your turn with one of:

- "OPERATION COMPLETE: <what was done>" -- for executed state changes
- "STATUS: <summary>" -- for read-only queries
- "AWAITING CONFIRMATION: <proposed action>" -- for state changes that need user approval
- "ESCALATION: <issue>" -- for trigger conditions
```

#### Change 4.2: Mention `live-ops` in `infra_patterns.md`

Add to the ops section of `docs/architecture/infra_patterns.md`:

```
## Operations agents

Two agents handle EC2/live-system interaction:

- **trade-log-analyzer**: diagnostics-only, read-only. Analyzes today's logs in ET, identifies errors, proposes (does not implement) fixes. Best for "what went wrong today" questions.
- **live-ops** (PR 4): routine ops with state-changing capability. Canned recipes for status, metrics, journal tails, instance start/stop, dashboard sync, service restarts. State changes require explicit user confirmation. Best for "do X to the system" tasks.

Both load EC2 identifiers from `.env`. Neither modifies code, configs, or trading state. Strategy-pipeline agents (strategy-lead and its specialists) are separate from these ops agents.
```

#### Change 4.3: Update methodology Appendix

Already covered in PR 2 Change 2.3 -- `live-ops` row added with "must read: 10".

### Acceptance criteria

- [ ] `.claude/agents/live-ops.md` exists with full recipe set
- [ ] `infra_patterns.md` distinguishes live-ops from trade-log-analyzer
- [ ] Smoke test: invoke `live-ops` with `status` recipe; verify it reads `.env`, connects to EC2, returns service status
- [ ] Smoke test: invoke `live-ops` with `start-instance` recipe; verify it asks for confirmation before executing
- [ ] Smoke test: invoke `live-ops` with `restart <service>` against a paper service; verify confirmation flow and post-restart status check
- [ ] No state-changing recipe executes without user confirmation in the smoke tests

### Rollback

New agent file; trivial to delete. No other agent depends on `live-ops`.

### Risks

- **Confirmation bypass**: if the user habitually approves all confirmations without reading, the agent's safety guarantee is undermined. Mitigation: confirmation messages are explicit and include the AWS region / instance ID / service name being affected. Cannot prevent rubber-stamp approvals; document the risk and rely on user discipline.
- **Recipe coverage gaps**: 6 recipes covers ~80% of routine ops. Other patterns will emerge. Add new recipes incrementally based on usage.
- **Write authority**: `live-ops` has Write tool access for `.env` modifications and dashboard sync. This is the broadest write authority on the team after general-purpose. Audit periodically; consider scope-restricting Write to specific paths via a hook if abuse emerges.
- **SSH keys in `.env`**: assumes the user has correct SSH key paths in `.env`. If wrong, ops fail loudly (good). Document the required keys in `.env.example` (already in PR 0b).

---

## Validation plan

After all PRs land, the end-to-end validation procedure:

1. **PR 0a smoke**: dispatch backtest-driver and backtest-optimizer; verify they read methodology Section 10.5 and find data correctly. Validates methodology-as-source-of-truth.
2. **PR 0b smoke**: dispatch strategy-lead and trade-log-analyzer; verify rename took effect and `.env` loading works.
3. **PR 1 verification**: run a real backtest and verify a row appears in `experiments.duckdb` with all identity fields populated. Run a 10-config optimization sweep; verify 10 child rows linked to one parent. Query project trial count; should return >= 10.
4. **PR 2 doc lint**: methodology renders, all references resolve.
5. **PR 3 end-to-end**: run a complete backtest of a strategy with stops; verify Section 12 diagnostics appear in report (capacity curve, regime transitions, trade-level metrics, IR if applicable, MAE/MFE if applicable). Verify strategy-lead's gates fire correctly with intentional failures injected.
6. **PR 4 ops drill**: walk through all 6 canned recipes; verify confirmation flow works on state changes.
7. **Existing test suite**: full `pytest`; no regressions across PRs.

---

## Risks and known limitations

### Risks

- **Methodology pointer pattern fails in practice.** PR 0a is the canary; if agents stop reading the methodology after duplication is removed, the entire architectural choice from phase 1-3 needs reconsidering. Mitigation: monitor PR 0a smoke tests carefully; if failure observed, consider keeping a one-table excerpt inline as a "fast-path" while still pointing at methodology for full details.
- **Cost-tier defaults wrong**: confirm with Shuyang. Mitigation: PR 1 acceptance criteria includes explicit confirmation step.
- **Section number stability**: future methodology sections (Section 13+ for Tier 2 checks) must append, not insert. Documented in methodology Section 0 (purpose and usage).
- **Live-ops confirmation discipline**: confirmation prompts are necessary but not sufficient; depends on user reading them. No automation can fix this.
- **Methodology version drift across rollout**: methodology Changelog must be updated on each PR that touches it. PRs 0b, 1, 2, 4 all touch methodology; ensure changelog entry per PR.

### Known limitations

- **No JSON-schema-based handoff contracts**. Registry covers most need.
- **No strategy-architect / strategy-implementer agents**. Decision B defers; trigger documented.
- **No `portfolio-integrator` agent**. Same as above. Methodology Section 6 (rules) still in effect; lead handles inline.
- **No synthetic-data robustness check**. Tier 3 deferred.
- **No bootstrap CIs on metrics**. Could promote from Tier 3 to Tier 2 if reviews keep asking "is this difference meaningful?"
- **No Tier 2 diagnostic checks** (outlier contribution, ulcer index, parameter cross-correlation, equity-curve smoothness, trade frequency vs capacity). Add by accretion based on pipeline experience.

---

## Decision log

Decisions made in drafting this v3 plan:

- **PR 0a stands alone, not bundled.** Smallest, lowest-risk change; success validates the entire methodology-as-source-of-truth pattern. Worth a separate commit for clean signal.
- **`portfolio-integrator` is removed from PR scope.** v2 created the agent; v3 defers per decision B. Methodology Section 6 stays as rules; agent file not created until trigger. This is the biggest substantive change from v2.
- **Methodology Appendix gets a "future agents" block.** Stops pretending agents exist that don't. When a future agent is created, it moves into the main table.
- **`live-ops` is added.** Real gap from operational experience; agent shape is read-mostly with explicit confirmation for state changes. Distinct from `trade-log-analyzer`.
- **Naming rename (`trading-lead` -> `strategy-lead`) is in scope.** v2 had decided against; v3 reverses based on CC's argument that methodology vocabulary is the canonical reference. Risk is bounded with a grep-and-update pass.
- **Cleanup before methodology expansion.** Phase 1-3 left content duplication that defeats the architectural goal. Strip first, then add.
- **Registry wiring is foundational.** Every methodology gate depends on it. PR 1 has to land before PRs 2-4 are meaningful even though it's mechanical-feeling work.
- **No new sub-agents for exit logic or diagnostic outputs.** Exit logic is methodology, not a new responsibility. Driver computes, reviewer verifies, lead gates. No proliferation.
- **Stop-loss governance is firmest in MAE/MFE requirement.** Optimizer-discovered stops without MAE/MFE backing are rejected. Negotiable but strong default.
- **K-window parameter stability cost.** 5x compute for live-bound strategies. Scoped exemption for research strategies.
- **Conda path correction kept in PR 3.** Listed in v2 plan; carries forward. Hardcoded Windows path in backtest-driver still needs fixing.

---

## Open questions for Shuyang

1. **Confirm cost-tier defaults** for OMR (`leveraged_etf`), RAMP (`mid_cap`), CSCM (`crypto_major`) before PR 1.
2. **Capacity scale points** ($50K, $250K, $1M, $5M, $25M) -- do these match realistic deployment range?
3. **K for parameter stability** -- 5 walk-forward windows default; do shorter-history strategies (e.g., crypto pairs) need K=3 override?
4. **Information ratio gates** in Section 12.5 table -- starting points; recalibrate after 1-2 strategies.
5. **PR 0a smoke test thresholds** -- what counts as "agents still working correctly"? Suggested: backtest produces a non-empty report with at least Sharpe, max DD, and trade count fields, and the report path is correct.
6. **Confirmation discipline for live-ops** -- does the user want stricter than yes/no (e.g., require typing the service name)? Default is yes/no.
7. **Trade-log schema migration** -- PR 2 adds `mae_pct`, `mfe_pct`, etc. to required trade log fields. Existing backtest engine may not produce these. Should PR 2 also include a `src/backtesting/engine/` update to start emitting them, or is that a follow-up PR?

---

## Appendix: Effort estimates and sequencing

| PR | Effort | Cumulative | Can ship without |
|---|---|---|---|
| PR 0a: Strip DATA LAYER duplicates | 0.5 hr | 0.5 hr | Anything |
| PR 0b: Cleanup and rename | 2-3 hr | ~3.5 hr | Anything |
| PR 1: Registry + cost-model wiring | 1.5 days | ~2 days | Anything (foundational) |
| PR 2: Methodology Sections 11-12 | 1 day | ~3 days | Without registry, Section 12.4 is aspirational |
| PR 3: Agent prompt updates | 1 day | ~4 days | Without Section 11/12, has nothing to reference |
| PR 4: live-ops agent | 1 day | ~5 days | Without `.env` pattern (PR 0b), can't load identifiers |

**Total**: 5 working days. Each PR independently reviewable and rollback-safe.

**Critical path**: PR 0a -> PR 0b -> PR 1 -> PR 2 -> PR 3. PR 4 can be done in parallel with PR 2 or PR 3 since it doesn't depend on methodology expansion (just on PR 0b's `.env` pattern).

---

## Sign-off

- [ ] v3 plan reviewed by Shuyang
- [ ] Open questions resolved
- [ ] PR 0a ready to execute (canary)
- [ ] PR 0b ready to execute (after PR 0a green)
- [ ] PR 1 ready to execute (after PR 0b)
- [ ] PR 2 ready to execute (after PR 1)
- [ ] PR 3 ready to execute (after PR 2)
- [ ] PR 4 ready to execute (parallel with PR 2/3)
- [ ] End-to-end validation completed
- [ ] v2 plan archived to `docs/planning/archive/`
- [ ] v3 plan archived to `docs/planning/archive/` after completion
