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

## Rollout (2026-10-06, 00:00-00:50 ET, operator-approved at each step)
- **R1**: pushed `deploy/console-p0-p1` to `ramp-phase4-turnover-regime-research` (`f009df5..6223f56`, fast-forward).
- **Instance start**: started manually at 04:10 UTC (outside the power lock). The OS came up hung: EC2 instance status `impaired` (reachability failed from 04:14), no Tailscale, port 22 closed to the public IP, SSM not registered, serial console silent after 3 s of kernel boot. A plain `reboot-instances` at 04:32 fixed it; back on the tailnet at 04:37 with every unit active. Cause unknown; nothing from this work had touched the instance yet. If it recurs at the 08:00 scheduled start, the start is a no-op on a running-but-hung instance and RAMP would miss its 15:55 rebalance.
- **R2**: instance venv Python 3.11.14, Tailscale 1.102.2 with `serve --https`.
- **R3**: toggle saved to `~/strategy_toggle.yaml.pre-untrack`, direct `git pull --ff-only`, copied back; sha256 `925beee6...` identical before and after, file untracked, RAMP still enabled on v11.
- **R4**: `CONSOLE_OPERATOR_LOGIN="shuyangw@github"` appended to the instance `.env`; installer enabled `homeguard-console-agent` and published `https://homeguard-ec2.tail3e202b.ts.net:8443/` next to the Grafana mapping.
- **R5**: all exit gates passed. `/status` over the tailnet lists cscm/mp/omr/ramp, MP enabled with no unit, RAMP's 2026-10-05 15:55 decision all-passed, lock free, only error `snapshot:mp`; forged header from the client returned 200 (serve replaced it); loopback without header 403; `/decisions?strategy=../x` 400; agent MemoryCurrent 28MB of 96MB, OOMScoreAdjust 500, 0 restarts; toggle untracked and unchanged across a later pull.
- The instance keeps running until the Tuesday 20:00 ET scheduled stop (the 08:00 start is a no-op).

## Follow-up cleanup (2026-10-06, operator-approved)
- **Agent review minors fixed and deployed** (main `2a8ce12`, `40f2d7e`, `9f58c75`, `dcf1caf`; deploy branch `7df428d..c0e18f2`): YAML dates no longer break `/status`; HEAD/OPTIONS return 405; `/decisions` returns 503 naming the unreadable source; 15 s idle-connection timeout; units report `load_state`; a missing snapshot is an error only when a unit runs the strategy (steady-state `errors` is now `[]`); the installer rejects empty/placeholder logins and only reports success after the live agent answers 403. Verified over the tailnet after a restart of `homeguard-console-agent` only.
- **Auto-reboot alarm** (`30a05e0`, docs `665ad20`): `homeguard-trading-bot-instance-check-reboot` on `StatusCheckFailed_Instance` > 0 for 3 x 60 s, actions EC2 reboot + SNS email. Applied with `terraform apply -target` (1 add, 0 change, 0 destroy).
- **Terraform drift found, not changed**: the live security group allows SSH from a different /32 than this machine's `terraform.tfvars`. A plain targeted plan would have rewritten the SSH rule, so the apply pinned `ssh_allowed_cidrs` to the live value. Update `terraform.tfvars` (or the security group) before the next untargeted apply.
- Not deployed: M9 systemd sandboxing (needs on-instance testing of the logger's file writes).

## Known Issues / Remaining Work
- **Hung boot on manual start** (see Rollout); the new reboot alarm should now recover a repeat automatically: watch the 2026-10-06 20:00 stop and the 2026-10-07 08:00 start; the instance has no out-of-band access (no SSM, public SSH closed), so a hang can only be fixed by reboot or stop/start from the AWS API. Consider attaching an SSM-capable role (ties into the open question of which role is attached).
- Tailscale SSH to the instance is in "check" mode and needs a browser approval per check period.
- **origin/main divergence resolved 2026-10-06**: merged origin/main (17 commits from 2026-07-27/28) into main as `f96145e` with no conflicts and pushed. An older untracked draft of docs/progress/20260727_GRAFANA_ALERTING_FIX.md blocked the fast-forward; it was kept as `.local-draft.md` next to the merged version.
- **Network-dependent tests**: 6 tests (`tests/trading/test_adapters.py` MP preload/fetch, `test_streaming_integration.py::...same_schema`) download real data from Alpaca and failed identically on both merge parents and the merge after midnight ET on 2026-10-06 ("Failed to download historical price data from Alpaca"); they passed the evening before. Not a code regression; the tests should mock the data fetch.
- **Deferred review minors**: all fixed on 2026-10-06 (see Follow-up cleanup) except M9, optional systemd sandboxing.
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
