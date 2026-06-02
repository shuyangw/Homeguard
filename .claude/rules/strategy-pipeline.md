# Strategy Pipeline Rules

These rules load alongside CLAUDE.md at session start. They apply to the orchestrator. Subagents do NOT read this file — the orchestrator must include relevant rules in every dispatch prompt.

## Session recovery protocol

1. On EVERY session start (fresh or resumed), read TODO.md FIRST
2. If `.claude-recovery.log` exists, read it for interrupt context
3. Any phase marked `[~]` was interrupted — restart it from the beginning
4. Read output files from completed phases to rebuild context — don't re-run them
5. For interrupted optimizer runs: check `output/optimization/<strategy>/` for the progress chronicle
6. For interrupted backtests: check `docs/reports/<strategy>/` for partial reports

## Agent dispatch rules

### Read-only agents (code-explorer, code-architect, code-reviewer)
- Return text only — orchestrator MUST write their output to `docs/agent-learnings/<strategy>/`
- Their analysis is gone if you don't save it

### Self-writing agents (backtest-driver, backtest-optimizer)
- backtest-driver writes to `docs/reports/<strategy>/`
- backtest-optimizer writes to `output/optimization/<strategy>/`
- Orchestrator reads and extracts key metrics into TODO.md
- Verify output files exist before marking phase complete

### All agents
- Subagents have NO context from your session — prompts must be self-contained
- Include ALL relevant Homeguard file paths in every dispatch prompt
- Include the applicable integrity rules (from strategy-lead.md Section 2) in every dispatch
- Include data frequency from the strategy spec in every backtest/optimizer dispatch
- Include data coverage rule: use FULL available data range, never a subset window
- After return: read the output files, don't trust the summary alone

## TODO.md discipline

- Mark `[~]` BEFORE starting a phase (so recovery knows what's in progress)
- Mark `[x]` AFTER verifying output files exist
- Never skip filling in the backtest iterations table — record ALL metrics
- Never overwrite previous results — increment run numbers

## Documentation gates (maximal-documentation discipline)

Every phase must leave a durable, MAXIMAL record across FOUR dimensions. "Maximal"
means: if the chat were deleted right now, the next session could fully reconstruct
what was done, why, and what it found, from committed files alone. Nothing that
matters may live only in an agent's chat summary.

**The four dimensions (document ALL four, every phase):**
1. **METHODOLOGY** — the design/approach, WHY it was chosen, the integrity rules +
   `docs/methodology/backtesting.md` sections applied, data frequency, full-data-coverage
   confirmation, the acceptance bar, and the DSR / project-wide trial-count justification.
2. **TESTS** — the TDD tests written, what each asserts, and the pass/fail count.
3. **MODIFICATIONS** — every file created/modified (FULL paths) + the commit hashes.
4. **RESULTS** — the full metrics row, the verdict, artifact paths, and the
   experiment-registry `run_id`s.

**Three gates. Do NOT mark a phase `[x]` until its gate passes. Verify artifacts by
reading the files / querying the registry — NEVER on a subagent's summary alone.**

- **GATE D0 — METHODOLOGY (before running anything):** the phase methodology is written
  down (in the dispatch prompt AND echoed into the report/spec header) before the first
  run. Mark `[~]`. A run whose design / acceptance-bar / trial-justification is not
  documented does not start.
- **GATE D1 — IMPLEMENTATION (after code, before backtest):** MODIFICATIONS documented
  (file paths) AND TESTS written and GREEN. Never run a backtest on red/absent tests.
- **GATE D2 — RESULTS (after the run, before `[x]`):** RESULTS recorded in ALL of:
  - (a) the readiness report `.md` + machine-readable `.json`;
  - (b) the experiment registry via `append_run` (methodology Section 9.3 — a failed
    append FAILS the run; no silent success);
  - (c) the **canonical variant/strategy glossary** (e.g. `docs/strategies/<STRAT>_VARIANTS.md`)
    — **UPDATE IT PER VERDICT.** A result not in the glossary is not recorded;
  - (d) `TODO.md` AND its tracked twin in `docs/progress/` (root `TODO.md` is gitignored —
    the tracked twin is the durable record; keep them in sync);
  - (e) the session log `docs/progress/YYYYMMDD_<TOPIC>.md`.

  The orchestrator VERIFIES each artifact exists before `[x]`.
- **GATE D3 — DEFINITION OF DONE (`[x]` gate):** all four dimensions have durable
  artifacts; the canonical docs (glossary + tracked TODO + session log + registry) are
  updated; nothing material is chat-only. Only then mark `[x]`.

### Documentation contract — paste into EVERY self-writing dispatch

Subagents do NOT read this file. Paste this block verbatim into every backtest-driver /
backtest-optimizer / implementation dispatch, and verify the return against it:

> **DOCUMENTATION CONTRACT (report back ALL of these AND write the durable artifacts):**
> 1. **METHODOLOGY:** restate the design, the `docs/methodology/backtesting.md` sections
>    applied, data frequency, full-window confirmation, the acceptance bar, and the
>    DSR / trial-count justification — in the report header.
> 2. **TESTS:** list the tests you wrote, what each asserts, and the pass/fail count.
> 3. **MODIFICATIONS:** list every file created/modified (full paths) + commit hashes.
> 4. **RESULTS:** the full metrics row (every metric in the Metrics section), the verdict,
>    the report `.md` + `.json` paths, and the experiment-registry `run_id`s
>    (`append_run` per Section 9.3 — raise on failure, no silent success).
> Do NOT report the phase done until all four artifacts exist on disk.

After return, the orchestrator: reads the report, queries the registry for the
`run_id`s, **updates the canonical glossary + TODO + tracked twin + session log**, and
only then marks `[x]`. The most common failure is documenting RESULTS in the report but
NOT in the glossary / tracked TODO — Gate D2 (c)+(d) exist specifically to catch that.

## Backtest integrity (authoritative)

**`docs/methodology/backtesting.md` is the single source of truth.** When this file and the methodology disagree, the methodology wins. Read the relevant section directly; do not paraphrase from memory.

| Topic | Methodology section |
|---|---|
| Bias prevention (lookahead, survivorship, selection, normalization, vol-target, full-data coverage) | Section 1 |
| Statistical gate (Sharpe, PSR, DSR with correct formula, PBO, combined gate) | Section 2 |
| Walk-forward (purging, embargo -- embargo is NOT the feature lookback) | Section 3 |
| Cost models by asset class + 1.5x cost-sensitivity gate | Section 4 |
| Stopping conditions (statistical floor, diminishing returns, overfitting trip, parameter sensitivity) | Section 5 |
| Data quality, point-in-time, news timestamps, data-snapshot reproducibility | Section 7 |
| Reproducibility identity fields (git SHA, config SHA, snapshot date, env hash, seeds) | Section 8 |
| Experiment registry schema (`output/experiments.duckdb`) | Section 9 |
| Homeguard paths, regime detectors, brokers, services, EC2 env | Section 10 |
| Exit logic, stops, profit-taking, MAE/MFE, asset-class profit-taking rules | Section 11 |
| Required diagnostic outputs (trade-level metrics, capacity, regime transitions, parameter stability, benchmark/IR) | Section 12 |

Inline magic-number thresholds ("Sharpe > 3.0 REJECT", "CAGR > 20% INVESTIGATE") are removed -- they were replaced by the combined statistical gate in methodology Section 2.5.

## Metrics -- record all of these for every backtest run

Sharpe, PSR, DSR (using project-wide cumulative trial count), PBO, CAGR, Max DD, Max DD duration, Calmar, win rate (monthly), profit factor, trade count, avg hold time, regime robustness, cost sensitivity (1.5x), IS/OOS Sharpe ratio, backtest window, data frequency. Append to the experiment registry per methodology Section 9.3.

## Homeguard file structure

- Strategy specs: `docs/strategies/production/<n>.md`
- Strategy code: `src/strategies/advanced/<n>.py`
- Strategy base classes: `src/backtesting/base/`
- Strategy configs: `config/strategies/<n>.yaml`
- Backtest configs: `config/backtesting/`
- Backtest engine: `src/backtesting/engine/`
- Walk-forward chunking: `src/backtesting/chunking/`
- Optimization framework: `src/backtesting/optimization/`
- Regime detection: `src/backtesting/regimes/detector.py`
- Reporting: `src/backtesting/reporting/`
- Position sizing/risk: `src/backtesting/utils/`
- Symbol universes: `config/universes/`
- Backtest scripts: `scripts/backtest_scripts/`
- Tests: `tests/strategies/test_<n>.py`
- Backtesting guidelines: `docs/guidelines/`
- Phase analysis output: `docs/agent-learnings/<strategy>/`
- Backtest reports: `docs/reports/<strategy>/`
- Optimization output: `output/optimization/<strategy>/`
- Backtest run data: `output/backtests/<strategy>/`
- Infra patterns: `docs/architecture/infra_patterns.md`
- Implementation skill: `.claude/skills/implement-strategy/`
- Recovery log: `.claude-recovery.log`
