# Session Handoff: Futures Campaign SP-C + SP-D + Governance + Retest TODO

**Date:** 2026-07-11 | **Working dir:** C:\Users\qwqw1\Dropbox\cs\github\Homeguard | **Branch:** main @ bfaecfc (clean, pushed)

## Resume Here (read this first)

- **Goal:** Finish the Futures Strategy Testability Campaign and correct a governance gap. This session shipped SP-C (spread engine) and SP-D (VRP + the honest-deflation fix), added a hard-enforcement hook so strategy testing routes through `strategy-lead`, and produced an APPROVED plan + a comprehensive TODO doc to RE-TEST all built SP-A..E strategies through `strategy-lead`.
- **Status:** SP-C and SP-D DONE + merged + pushed. Governance hook DONE + committed. The retest TODO doc is WRITTEN + committed. The retest itself has NOT been run yet.
- **Next step (the one open action):** execute the retest by running a DEDICATED `strategy-lead` session pointed at `docs/strategies/research/20260711_FUTURES_RETEST_TODO.md`. It does Gate 0 (a repo-wide deflation fix + committed sleeve drivers) then re-validates 19 strategies through Phases 5->8 + optimization. Awaiting the user's choice: they run `claude --agent strategy-lead`, OR I dispatch strategy-lead as a background agent. I did NOT start it (it is a large multi-hour run).
- **Blockers / open questions:** none blocking. One decision pending: who kicks off the strategy-lead retest.
- **To resume, you need:** the `fintech` conda env; run pytest via the direct interpreter `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest ...` (conda is NOT on the Bash PATH); the Dropbox worktree-gitlink git hazard rules (targeted git only, never `git add -A`); awareness of the new backtest hook (below).

## Original Task (this session's arc)

Continuation of the campaign. In order: (1) execute SP-C multi-leg spread engine (brainstormed/designed/planned in the prior session); (2) "continue with both" -> execute SP-D options-IV #28 AND the VIX #26 deflation as one VRP-finalization sub-project; (3) user asked how many of the ~60 catalog strats we tested and whether any went through `strategy-lead`; (4) "make it so strategy-lead is invoked even with superpowers but only if testing strategies" -> build the hook; (5) "write a comprehensive todo doc to invoke strategy-lead on ALL SP-A..E strategies, retesting them comprehensively" -> the retest TODO (plan-mode, approved).

## Subtasks & Progress

- [x] **SP-C multi-leg spread engine** -- DONE, merged ff to main (a07336e), 15 commits. Shared spread-construction layer + continuous engine (#35/#36) + convergence state machine (#31-#34), gated as return streams. HEADLINE FINDING: #31 calendar "edge" (Sharpe 1.0-1.18, NG nominally > carry) was ROLL-JUMP CONTAMINATION; masking front/second symbol-change days collapsed it (CL 1.183->0.394, NG 1.017->-0.150). Nothing beats carry. Ledger `docs/strategies/research/20260710_FUTURES_SP_C_TRIALS.md`; log `docs/progress/20260710_FUTURES_SP_C.md`.
- [x] **SP-D + VRP finalization** -- DONE, merged ff (52ce408), 12 commits (d53c4f7..52ce408). Built `src/backtesting/vol/` (option_symbol parser, Black-76 atm_iv + VIX validation, Corsi HAR har_rv, vrp_strategy). THE BIG FINDING: the gate's DSR had NEVER deflated (`TRIAL_COUNT_PARAMETER_FREE = 1` AND `dsr(sharpe, [sharpe], ...)` single-element list -> `expected_max_sharpe` returns 0). Fixed with `CAMPAIGN_CUMULATIVE_TRIALS = 40` + `CAMPAIGN_TRIAL_SHARPES` (29 real OOS Sharpes, var 0.112) -> SR_zero = 0.733; also PBO 2*s drop-threshold fix. Under honest deflation NOTHING clears DSR >= 0.95 INCLUDING carry (0.588/0.765 at/below 0.733; PBO 0.093 not overfit). #26 FAIL (DSR 8.9e-06, max DD -81.1% tail); #28 FAIL + re-expression of #26 (marginal Sharpe 0.015). Ledger `20260711_FUTURES_SP_D_TRIALS.md`; log `20260711_FUTURES_SP_D.md`.
- [x] **CLAUDE.md worktree rule** -- superpowers implementation MUST run in a separate worktree (committed with the North Star block, main).
- [x] **Governance hook (strategy-lead gate)** -- DONE, committed+pushed (c35cf42). `PreToolUse` Bash hook hard-blocks strategy backtest/gate/verdict commands unless the `.claude/.strategy-lead-active` sentinel exists. strategy-lead creates it first / removes it last. Prose in CLAUDE.md + strategy-pipeline.md (build-vs-verdict boundary). Verified LIVE this session (it blocked a test command).
- [x] **Coverage analysis** -- of ~55 numbered catalog strats, ~19 TOUCHED (16 with real gradeable verdicts, 2 ungradeable #35/#39, 1 no-data #49); ~36 NEVER tested. NONE of the tested ones went through `strategy-lead` (all via superpowers SDD + general-purpose + controller-run gates).
- [x] **Retest TODO doc + plan** -- plan APPROVED (`.claude/plans/ok-lets-discard-that-validated-blum.md`); doc WRITTEN + committed (c009b0a, ASCII fix bfaecfc): `docs/strategies/research/20260711_FUTURES_RETEST_TODO.md`.
- [ ] **Run the strategy-lead retest** -- NOT started. The one remaining action.

## Key Decisions & Tradeoffs

- **Governance = hard-block hook (option B), not warn.** Instructions alone had already failed (the rule was in CLAUDE.md and still skipped). A `PreToolUse` deny with a sentinel gate is deterministic. Tradeoff: strategy-lead must set the sentinel or its own backtest-driver is blocked; a stale sentinel fails open. Escape hatch: `touch .claude/.strategy-lead-active`.
- **Retest depth: FULL** (Phases 5->8 + improvement 6.5 + optimization 7), with strategy-lead's guardrails (max 2 improve rounds/strategy, each a pre-committed hypothesis, each raising the DSR trial count).
- **Deflation fix: repo-wide** (Gate 0.1) -- thread `CAMPAIGN_TRIAL_SHARPES` into the carver/session/fx/satellite gates so ALL deflate against SR_zero 0.733 (currently only `gate_return_stream` does). Only lowers DSR -> no prior PASS flips.
- **Retest TODO lives in a dedicated file** (not root TODO.md, which is the concluded RAMP Wave-3 batch); strategy-lead is pointed at it explicitly.
- **SP-C/SP-D roll-jump + underlying-source + overnight-gap fixes** were all review-caught during SDD (see Gotchas).

## Discussion Summary

The campaign's honest verdict crystallized this session: the futures testability apparatus works and surfaced real mechanisms (carry, VRP, calendar spreads), but under contamination-free, deflation-correct evaluation NOTHING clears the statistical gate, including the carry incumbent. Two contamination classes were caught by the SDD reviews (roll-jump in #31 calendar; ratio-adjusted-vs-raw underlying in the #28 IV extractor; overnight-gap in HAR RV) and one methodology bug (DSR never deflating). The governance question ("did we use strategy-lead?" -> no) led to the hook and then to the retest plan, whose real point is the FIRST honest, uniform evaluation: SP-A (carver gate) and SP-B (session gate) were never deflated, several verdicts are broken/stale (#16 mis-sampled, PBO-NaN predating the fix, #36 book-corr never run), and the Path-2 sleeves have no committed drivers.

## Commands & Outputs (load-bearing)

```
# Governance hook proven live (blocked a matching command):
$ echo "harmless test string gate_return_stream"
BLOCKED: strategy backtest/gate/verdict command detected outside strategy-lead ...

# The hook is BLUNT: it blocks any Bash command whose text CONTAINS a pattern token,
# even a read-only grep FOR those tokens. Use the Grep/Read TOOLS (not Bash) to inspect
# gate code, or set the sentinel. This is why Gate 0 must run under strategy-lead.

# Carry incumbent, gated through the honest deflated gate (the finding):
oos_sharpe 0.588 | psr 1.0 | dsr 5.4e-14 (FAIL) | pbo 0.093 (PASS) | SR_zero 0.733
```

## Git / Push state

- main @ bfaecfc, pushed to origin/shuyangw/Homeguard. This session's line: SP-C (thru a07336e), SP-D (d53c4f7..52ce408), SP-D log (42d9967), governance (c35cf42), retest TODO (c009b0a, bfaecfc).
- Both SP-C and SP-D were built in isolated worktrees (`.worktrees/futures-sp-c`, `.worktrees/futures-sp-d`), merged ff via `git merge --ff-only` from the main dir, worktrees removed. No open worktrees now.

## Files Touched (this session, key)

- `docs/strategies/research/20260711_FUTURES_RETEST_TODO.md` -- THE active artifact (the retest work-list). Gate 0 + tiered strategies + acceptance bar + iterations table + EXCLUDED (#49/#9).
- `.claude/plans/ok-lets-discard-that-validated-blum.md` -- the approved retest plan of record.
- `.claude/hooks/strategy_lead_gate.py` + `.claude/settings.json` -- the hook (NOTE: settings.json is GITIGNORED -> the hook REGISTRATION is local-only; the mechanism/script IS committed).
- `.claude/agents/strategy-lead.md` -- sentinel lifecycle added (create first / remove last).
- `CLAUDE.md`, `.claude/rules/strategy-pipeline.md` -- build-vs-verdict boundary + worktree rule.
- SP-D code: `src/backtesting/walkforward_common.py` (deflation + PBO fix, `CAMPAIGN_TRIAL_SHARPES`), `src/backtesting/vol/*`, `src/backtesting/vix/vix_rolldown_eval.py` (subperiod_audit).
- SP-C code: `src/backtesting/spreads/*`, `src/strategies/advanced/spread_*.py`, `src/data/futures/front_next.py`.

## Key Takeaways & Gotchas

- **The backtest hook is intentionally blunt.** It blocks any Bash command CONTAINING `backtest_runner|run_futures_backtest|walk_forward|run_carver_walkforward|gate_return_stream|gate_convergence|gate_session_stream|run_vix_rolldown|run_vrp|run_standard_report|scripts/backtest_scripts/sp_` unless the sentinel exists -- including a grep FOR those tokens. To inspect gate code, use the Grep/Read tools (not shell grep) or set `.claude/.strategy-lead-active`. This is BY DESIGN: it routes gate work through strategy-lead.
- **`.claude/settings.json` is gitignored** -> the hook registration is LOCAL to this machine (persists as a file, not committed). A fresh clone/teammate must re-add the 8-line PreToolUse block. The hook SCRIPT + docs ARE committed.
- **Deflation is NOT yet uniform.** SP-D's fix only reached `gate_return_stream` (VIX/VRP/SP-C spreads). The carver gate (`run_carver_walkforward.py`), `session_walkforward.gate_session_stream`, `run_fx_walkforward`, `satellite_blend` still pass single-element trial lists -> DSR == PSR, un-deflated. Gate 0.1 in the retest TODO fixes this.
- **Path-2 sleeves have no committed drivers** (`scripts/backtest_scripts/sp_*` never existed in-tree). Gate 0.3 creates them (RunStatus-wrapped, writing returns.csv + gate.json + `_verdict`).
- **Dropbox worktree-gitlink hazard:** bare `git status`/`git diff`/`git checkout`/`reset --hard`/`git add -A` can be FATAL or partially clobber. Use targeted git (`git add <paths>`, `git commit`, `git log`, `git merge --ff-only`, `git push`).
- **Two data exclusions (cannot retest):** #49 funding (Binance geo-blocked HTTP 451), #9 multi-horizon carry (never implemented).
- **Retest honest expectation:** confirm the negative (nothing clears DSR >= 0.95 including carry; SR_zero grows as each run appends a trial). The value is rigor/governance/uniform-deflation/fixing-broken-verdicts, not finding a winner. Surfacing that IS the completed objective (North Star).

## References

- Approved plan: `.claude/plans/ok-lets-discard-that-validated-blum.md`
- Retest TODO (execute this): `docs/strategies/research/20260711_FUTURES_RETEST_TODO.md`
- Prior handoff (this session continued it): `docs/progress/SESSION-HANDOFF-futures-campaign-2026-07-10.md`
- Trial ledgers: `docs/strategies/research/20260710_FUTURES_SP_C_TRIALS.md`, `20260711_FUTURES_SP_D_TRIALS.md`
- Session logs: `docs/progress/20260710_FUTURES_SP_C.md`, `20260711_FUTURES_SP_D.md`
- Campaign memory: `C:\Users\qwqw1\.claude\projects\C--Users-qwqw1-Dropbox-cs-github-Homeguard\memory\project_futures_testability_campaign.md`
