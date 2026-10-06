# Homeguard Console Phase 0 + Phase 1 (build) - 2026-10-05

## Summary
Took the 2026-10-05 console design doc and prototype through brainstorming, a re-verified spec, a plan, and inline execution. Built Phase 0 (fail-closed toggle defaults, untracked toggle, fail-closed template) and Phase 1 (the read-only on-instance agent `src/console_agent/` with `GET /status` and `GET /decisions`, its systemd unit and installer). Merged to local main. Nothing is deployed yet: the instance rollout (R1-R5 in the plan) waits for operator go-ahead.

## Changes Made
- **Spec** `docs/superpowers/specs/2026-10-05-homeguard-console-design.md`: the design doc adopted as the umbrella spec, re-verified against main c63e5dd and deploy branch f009df5. Revision notes: EC2 runs `ramp-phase4-turnover-regime-research` (574 commits off main), so instance code ships by cherry-pick; IAM/S3 and `/series`,`/logs` moved to Phase 2; RAMP's 15:55 is hardcoded, not in config; the agent cannot import `src.trading` (75MB via BrokerFactory, over MemoryMax=96M), so it reads decision JSON directly.
- **Plan** `docs/superpowers/plans/2026-10-05-homeguard-console-p0-p1.md`: 7 build tasks plus operator-gated rollout R1-R6. The plan's code was extracted and run in a scratch tree before execution (60 passed).
- **`src/trading/state/strategy_state_manager.py`**: a missing toggle file regenerates every strategy disabled (was OMR/RAMP/CSCM enabled) and the error names a safe restore and the variant check.
- **`config/trading/strategy_toggle.yaml`** untracked (`.gitignore` comment updated); **`strategy_toggle.example.yaml`** now fails closed with an explicit variant per strategy (RAMP v11). The template fix came from the final review: with the toggle untracked, copying the old template would have turned OMR on and run RAMP V01.
- **`src/console_agent/`** (units, status, server, `__main__`, README): stdlib ThreadingHTTPServer on 127.0.0.1:8090; `Tailscale-User-Login` must equal `CONSOLE_OPERATOR_LOGIN`; non-GET 405; per-source `errors`; never writes, never constructs `StrategyStateManager`.
- **`infra/ec2/services/homeguard-console-agent.service`** (MemoryMax=96M, OOMScoreAdjust=500, not PartOf the trading target) and **`infra/ec2/setup/install_console_agent.sh`** (idempotent, adds the `tailscale serve` 8443 mapping); `.env.example` gains `CONSOLE_OPERATOR_LOGIN`.
- **Docs**: `CLAUDE.md` package list and `docs/architecture/ARCHITECTURE_OVERVIEW.md` register the agent.

## Commits
Main (feature branch `feat/console-p0-p1`, fast-forwarded):
- `11295b6` docs(console): Homeguard Console design spec, re-verified for Phase 0+1
- `cda30cc` docs(console): Phase 0+1 implementation plan; spec notes agent cannot import src.trading
- `dc3cfb1` / `496bd43` test + fix(state): fail closed when strategy_toggle.yaml is missing
- `10083ba` fix(config): untrack strategy_toggle.yaml so instance toggles survive deploys
- `b62ce31` / `cd662d8` test + feat(console-agent): systemd queries and unit-to-strategy mapping
- `d0e7478` / `7913d5f` test + feat(console-agent): read toggle, lock, snapshots and decisions into /status
- `52f1105` / `dc0b57c` test + feat(console-agent): read-only HTTP server with /status and /decisions
- `199eb12` feat(infra): console agent systemd unit and idempotent installer
- `ebdb6ab` docs(console-agent): package README
- `e4ab1e8` docs(architecture): register the console agent package
- `c7361d8` / `882d07d` test + fix(config): fail-closed toggle template with explicit variants
- `35705b9` docs(console): rollout uses a prepared deploy series and a direct pull

Deploy series, prepared on local branch `deploy/console-p0-p1` (worktree `.worktrees/deploy-console`), cut from origin/ramp-phase4-turnover-regime-research f009df5, NOT pushed: the 13 code commits above (d67e85c..230cc77; template conflict resolved to main's version) plus `6223f56` test(ramp): skip the toggle variant check when the untracked toggle is absent (deploy-only test).

## Known Issues / Remaining Work
- **Rollout R1-R5 not started** (each needs operator go-ahead): push `deploy/console-p0-p1` to the deploy branch (re-check fast-forward first), check instance Python and `tailscale serve --https`, the one-time toggle copy / direct `git pull` / restore, add `CONSOLE_OPERATOR_LOGIN` and run the installer, exit-gate curls (including the forged-header check that proves `tailscale serve` overwrites the header).
- **origin/main has diverged from local main** (pre-existing, not from this work): 56 local-only commits, 17 remote-only (2026-07-27/28 monitoring, alerting and RAMP logging fixes). A trial merge with this branch is clean and none of the 17 touch this branch's files. Main is not pushed until that is reconciled.
- **Deferred review minors**: non-JSON YAML values (dates) 500 `/status`; HEAD/OPTIONS return 501; no handler timeout; `/decisions` opaque 500 on bad toggle; installer can report success for a crash-looping agent and accepts placeholder logins; no `LoadState`; `errors` never empty in steady state (`snapshot:mp`, `snapshot:omr`); optional systemd sandboxing.
- **Phase 3 design input**: agent POST routes need their own Origin/token check (serve-injected login is ambient authority in the operator's browser). Recorded in the spec's open questions.
- **Environment flake**: `tests/trading/test_adapters.py::TestMomentumLiveAdapter::test_run_once_calls_fetch_todays_closes` intermittently fails with WinError 5 on `os.replace` of the real `data/trading/strategy_positions.json` inside Dropbox; it writes the repo's real state path instead of a temp dir.
- **Tooling gotcha**: the Bash PreToolUse hook resolves `.claude/hooks/strategy_lead_gate.py` relative to the shell's working directory, so any Bash call whose cwd is a deploy-branch worktree (no such file there) is blocked; drive that worktree with `git -C` from another directory.
- Phases 2 and 3 (local FastAPI/htmx app, S3 snapshots, IAM, controls) are approved in outline only and need their own plan.

## Validation
- Feature tests: `tests/console_agent` + `tests/trading/test_state_manager_toggle_defaults.py` 66 passed at head.
- Wider suite on the feature branch (`tests/trading tests/monitoring tests/utils tests/settings tests/console_agent`): 1222 passed / 5 skipped before the review fixes; 1224 passed + the one Dropbox flake after them.
- Wider suite on the prepared deploy series: 1325 passed, 13 skipped.
- Baseline state-manager suites: 167 passed before and after every Phase 0 change.
- Whole-branch review by a fresh Opus reviewer: ready with fixes, 0 critical; both Important findings fixed with RED->GREEN tests (I1 on the deploy series, I2 on main); M7 re-graded and fixed in the rollout procedure.
- Not yet validated on the instance: memory under the 96M cap in practice, the tailscale serve header behavior, and the toggle untrack sequence (all are R5 exit gates).
