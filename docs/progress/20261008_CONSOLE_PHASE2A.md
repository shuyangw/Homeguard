# Homeguard Console Phase 2a (build) - 2026-10-08

## Summary
Built Phase 2a of the Homeguard Console: a read-only local app (FastAPI + htmx on 127.0.0.1:8765) that shows whether Homeguard is up and trading, and keeps showing the last known state from an S3 snapshot after the instance stops. Spec, plan, nine build tasks and a final fix wave are merged to main and pushed (`09568e3..65056c5`). Rolled out the same afternoon (operator-approved): Terraform applied, keys and profile on the Windows PC, uploader live on the instance, console running at logon on the PC. The Mac install and the 20:00 exit gate remain.

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

## Rollout (2026-10-08, about 15:15-15:40 ET)
- Terraform: targeted plan 6 to add, 0 change, 0 destroy; applied (bucket `homeguard-console-<account id>`, set in the local untracked terraform.tfvars).
- `homeguard-console` access key created; profile written on the Windows PC. Profile verified: DescribeInstances and all four GetSchedule calls succeed, GetObject on the absent key returns NoSuchKey (the unconditioned ListBucket works), StopInstances is denied.
- Deploy branch: `7dafc86` + `73c94d3` (uploader cherry-picks) and `22b848d` (installer fix); instance pulled to `73c94d3`, `CONSOLE_SNAPSHOT_BUCKET` added to its .env, aws-cli 2.30.4 present, `install_console_upload.sh` installed and enabled the timer and the shutdown unit; first upload landed (15 KB at 19:23 UTC). The installer's final `aws s3 ls` failed because the instance role has PutObject only; fixed in `d69a180` (main) / `22b848d` (deploy branch).
- Windows: local .env has the console values; `install_windows_task.ps1` registered `HomeguardConsole` (pythonw), serving "Live from agent" and logging to %LOCALAPPDATA%\Homeguard\console.log.
- Process note: the instance step ran at about 15:22 ET, inside market hours, because Git Bash ignored TZ and the clock check printed UTC as ET. It restarted no trading unit and RAMP is paused; check the clock with the instance's `date -u` next time.

## Incident found by the console: homeguard-multi down since 08:03 ET
- The first live view showed RAMP's snapshot frozen at 08:03 ET (heartbeat and stream tiles unknown, not green). `homeguard-multi` was `failed` with NRestarts=5: at boot its startup reconciliation saw "state says N on ibkr, broker reports 0" for every RAMP holding and exited 1; five fast restarts hit the start limit by 08:03 ET. IBC finished logging in at 08:01 ET, so the runner most likely read positions before IBKR delivered them.
- A read-only check at about 15:30 ET (clientId 98) shows the account intact: 26 stock positions (18 long, the same 8 shorts), 0 open orders.
- Impact: none on trading (RAMP is paused), but the process that would trade RAMP is down until restarted, and the same race can recur on any boot.
- Root cause: the preflight retry covered only an empty book; a freshly logged-in gateway served a partial one (no "retrying" lines in the boot journal), so every start failed at once.
- Fix (operator-approved): `277ab18` retries while any this-broker holding is missing (12 attempts x 10 s); `f313dc6` also checks short positions, which the guard had never checked (independent review finding). Deploy branch `d8892c0`, `9573c3f`. Trading suite: main 1068 passed, deploy branch 1102 passed.
- Restarted 16:17 ET (20:17 UTC, after the 16:15 lock), RAMP still disabled: active, NRestarts=0, "[Reconcile] Pre-flight check passed for ramp" over 26 positions, RAMP v11 loaded; IBKR paper smoke test PASSED (2 successful, 0 failed, 0 retries, no lingering orders); read-only check 26 stock positions (18 long, 8 short), 0 open orders; console heartbeat tile live again (ramp 53 s, cscm 35 s).

## Known Issues / Remaining Work
- **Boot race fix**: proven only on a warm gateway so far; confirm at the next 08:00 boot that the preflight passes (look for "retrying" lines and a single start).
- **Mac**: copy the `homeguard-console` keys to the Mac (`aws configure --profile homeguard-console`), add the three CONSOLE_ values to its repo .env, run `bash tools/console/install_macos_launchd.sh <env>/bin/python`.
- **Exit gate FAILED 10-08**: the shutdown snapshot never reached S3 (see Validation). Diagnose from the previous boot's journal on 10-09, fix, and re-run the gate at the 10-09 20:00 ET stop.
- **Verify live**: a GetObject on an absent key with the real profile returns NoSuchKey; TLS to the agent over the tailnet works from httpx.
- **Deferred minors** (the final review triaged them as fine to defer): Day P&L format "$-500", gates panel not auto-refreshed, accessibility polish, launchd respawn loop on a misconfigured app, ANSI codes in the Windows console log, installer parsing edge cases, a malformed snapshot timestamp would 500 every panel.
- **Cleanup**: `.worktrees/console-p2a` and branch `feat/console-p2a` could not be removed (Dropbox lock on its `.superpowers` folder); remove later with `git worktree remove --force .worktrees/console-p2a && git branch -d feat/console-p2a`.
- **Phase 2b next**: agent `/series` and `/logs`, chart and log panels, the metrics-scrape check.

## Validation
- Every task test-first with a per-task spec and quality review; fix rounds on Tasks 3, 7, 8 and 9; a final whole-branch review (Opus) found 1 Critical and 5 Important issues, all fixed in one wave and confirmed by a scoped re-review.
- `tests/trading tests/monitoring tests/console_agent tests/console_app`: 1278 passed, 12 skipped, 1 failed (`test_adapters.py::TestMomentumLiveAdapter::test_run_once_calls_fetch_todays_closes`, the known Dropbox WinError 5 lock flake; passes alone; the branch changes nothing under `src/trading` or `tests/trading`).
- `terraform validate` and `fmt -check` pass; no plan or apply run.
- Entry point smoke: a missing AWS profile refuses to start with exit 1 under both python.exe and pythonw.exe (Windows test writes to `%LOCALAPPDATA%\Homeguard\console.log`).
- **Exit gate, 20:07 ET: FAIL (2 of 4 pass).** The instance stopped at 00:00:18 UTC (StopInstances, user initiated).
  - [-] (1) S3 `console/latest/status.json` LastModified 23:58:09 UTC, `reason: periodic`. That is the last timer upload; no shutdown snapshot landed.
  - [-] (2) The header reads "Agent unreachable; last agent reading 6 min ago", not "Snapshot from S3 ... (shutdown)". This follows from (1): the console keeps its 00:00:09 UTC live reading over an older S3 snapshot.
  - [+] (3) Exceptions: IB Gateway, Broker heartbeat, Market data stream and Host memory all UNKNOWN "since 20:00".
  - [+] (4) Power tile: "Instance stopped, as scheduled", next start Fri 08:00 ET.
  - 10-09 follow-up (from Loki, no SSH needed): the shutdown unit DID run at 00:00:19 UTC; `aws s3 cp` hung until the 30 s timeout killed it at 00:00:49. The network and DNS were up throughout (systemd-networkd and systemd-resolved stopped at 00:01:49), and periodic uploads take about 1.2 s, so this hang happens only at shutdown. Fix `bea369e` (deploy `57c1dfe`): `--cli-connect-timeout 5 --cli-read-timeout 10`, so the CLI retries and names the unreachable endpoint, plus a 45 s process timeout inside TimeoutStopSec=60. A manual periodic upload with the new flags succeeded on the instance at 22:02:48 UTC. Gate re-run at the 10-09 20:00 ET stop.
  - (superseded) Cause not yet known. The shutdown unit was active (since 19:22:58 UTC) and loads the bucket the same way as the periodic unit, so a missing env var is ruled out. At the next boot read `journalctl -b -1 -u homeguard-console-upload-shutdown.service` to see whether ExecStop ran and what failed.

## Alert noise fix (19:25 ET)

- **Symptom:** `RampDecisionMissedToday` (critical) re-posted to Discord every hour from 18:15 ET. The alert was correct: homeguard-multi was down 08:03-16:17 ET, so the 15:55 run wrote no decision record (paused RAMP still writes a placeholder record daily).
- **Cause:** all critical alerts share one route with `repeat_interval: 1h`. This alert cannot clear before the next day's 15:55 run, so the repeats carried no new information.
- **Fix:** `config/monitoring/grafana/alerting/homeguard_notifications.yaml` gets a route for this alert alone, `repeat_interval: 24h`, ordered ahead of the critical route. Test `test_ramp_decision_missed_notifies_once_a_day` (RED then GREEN); `tests/monitoring` 90 passed, 7 skipped.
- **Commits:** main `16b879f`, deploy branch `f36cbf6`. Deployed via `infra/ec2/sync_grafana_alerts.sh` (Grafana restart only, no trading restart). The sync also installed the pending comment-only rules change `1a3ce85`. Live policy verified: the new route shows `repeat_interval 1d`, and all 7 rules are healthy.
- **Silence:** `f205059e` on `RampDecisionMissedToday` until 2026-10-09 16:00 ET, for tonight's known miss; it survived the Grafana restart.
