# Homeguard Console Phase 2a (build) - 2026-10-08

## Summary
Built Phase 2a of the Homeguard Console: a read-only local app (FastAPI + htmx on 127.0.0.1:8765) that shows whether Homeguard is up and trading, and keeps showing the last known state from an S3 snapshot after the instance stops. Spec, plan, nine build tasks and a final fix wave are merged to main and pushed (`09568e3..65056c5`). Nothing is deployed yet: the Terraform apply, the access keys, the instance install and the 20:00 exit gate are operator steps still to run.

## Changes Made
- **Instance uploader** (`src/console_agent/upload.py`, three units in `infra/ec2/services/`, `infra/ec2/setup/install_console_upload.sh`): builds the agent's `/status` document plus the latest decision per strategy and copies it to `s3://<bucket>/console/latest/status.json` with the AWS CLI every 5 minutes and at shutdown (shutdown unit ordered after the trading units so its `ExecStop` runs while they are up).
- **Terraform** (`infra/terraform/console.tf`): private bucket (SSE-S3, public access blocked), `s3:PutObject` on the prefix for `homeguard-ec2-cloudwatch`, read-only IAM user `homeguard-console` (DescribeInstances, GetSchedule, GetObject on the prefix, unconditioned ListBucket on the bucket so a missing snapshot reads as NoSuchKey, FilterLogEvents on the two scheduler Lambda log groups). No Start/Stop until Phase 3.
- **Local app** (`tools/console/`): `poller.py` (agent every 10 s, S3 only while the agent is down, EC2 every 30 s, schedules every 10 min; state published copy-on-write), `schedule.py` (four cron schedules, expected state, NYSE session and power lock, power tile), `checks.py` (eight exception checks), `freshness.py`, `views.py`, `app.py` (TrustedHost check), templates, vendored htmx 2.0.4, Windows logon task and macOS launchd installers.
- **Decisions in the design**: one status document shape from either source (no second parser); no SQLite until Phase 3; decision times in a local table pinned to the adapters by a drift test; Phase 2 split into 2a (this) and 2b (`/series`, `/logs`, charts, metrics-scrape check).
- **Docs**: spec `docs/superpowers/specs/2026-10-08-homeguard-console-phase2a-design.md`, plan `docs/superpowers/plans/2026-10-08-homeguard-console-phase2a.md`, `docs/INFRASTRUCTURE_OVERVIEW.md`, `infra/terraform/README.md`, `tools/console/README.md`, `CLAUDE.md`, `docs/architecture/ARCHITECTURE_OVERVIEW.md`, `.env.example`.

## Commits
- `9b60fc1` docs(spec): Homeguard Console Phase 2a design; `09568e3` docs(plan)
- Build: `a5359b5` uploader, `0a43af0` units, `e5caa2f` + `c5df725` terraform, `1533201` settings, `24ffccb` schedule, `37e2e0d` checks, `1c31f29` + `b3371c5` poller, `c53ef2a` + `e92530c` app and templates, `7637d26` + `d76bbf7` installers
- Final-review fix wave: `fa4b036` live age limit, `70d5b65` per-strategy heartbeat, `d37b0d0` stale and unit-less snapshots, `026b40b` unit state in the process column, `3e0b00a` source stamps on panels, `b41bf67` unreachability timed from launch plus start grace, `1e6634f` S3 snapshot chosen on the instance clock, `65056c5` clear the S3 error on recovery

## Known Issues / Remaining Work
- **Operator rollout (not started)**: targeted `terraform apply` of console.tf; create `homeguard-console` access keys and the profile on the PC and the Mac; cherry-pick `a5359b5` and `0a43af0` onto `ramp-phase4-turnover-regime-research`, add `CONSOLE_SNAPSHOT_BUCKET` to the instance .env, check `aws --version`, run `install_console_upload.sh` (outside market hours); add `CONSOLE_AGENT_URL` and `CONSOLE_SNAPSHOT_BUCKET` to the local .env and install the logon task / launchd agent.
- **Exit gate**: after a 20:00 ET stop the console must show "Snapshot from S3 ... (shutdown)" stamped within a minute of 20:00 with unknown tiles for the live-only checks.
- **Verify live**: a GetObject on an absent key with the real profile returns NoSuchKey; TLS to the agent over the tailnet works from httpx.
- **Deferred minors** (the final review triaged them as fine to defer): Day P&L format "$-500", gates panel not auto-refreshed, accessibility polish, launchd respawn loop on a misconfigured app, ANSI codes in the Windows console log, installer parsing edge cases, a malformed snapshot timestamp would 500 every panel.
- **Cleanup**: `.worktrees/console-p2a` and branch `feat/console-p2a` could not be removed (Dropbox lock on its `.superpowers` folder); remove later with `git worktree remove --force .worktrees/console-p2a && git branch -d feat/console-p2a`.
- **Phase 2b next**: agent `/series` and `/logs`, chart and log panels, the metrics-scrape check.

## Validation
- Every task test-first with a per-task spec and quality review; fix rounds on Tasks 3, 7, 8 and 9; a final whole-branch review (Opus) found 1 Critical and 5 Important issues, all fixed in one wave and confirmed by a scoped re-review.
- `tests/trading tests/monitoring tests/console_agent tests/console_app`: 1278 passed, 12 skipped, 1 failed (`test_adapters.py::TestMomentumLiveAdapter::test_run_once_calls_fetch_todays_closes`, the known Dropbox WinError 5 lock flake; passes alone; the branch changes nothing under `src/trading` or `tests/trading`).
- `terraform validate` and `fmt -check` pass; no plan or apply run.
- Entry point smoke: a missing AWS profile refuses to start with exit 1 under both python.exe and pythonw.exe (Windows test writes to `%LOCALAPPDATA%\Homeguard\console.log`).
