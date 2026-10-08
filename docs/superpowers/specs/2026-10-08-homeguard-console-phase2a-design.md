# Homeguard Console Phase 2a - Design

Date: 2026-10-08
Status: Draft for review
Author: Shuyang
Parent spec: docs/superpowers/specs/2026-10-05-homeguard-console-design.md (Phases 0 and 1 shipped and deployed 2026-10-06)

## Summary

Phase 2a builds the first usable console: a read-only local app on the operator's machines that shows whether Homeguard is up and trading as intended, and keeps showing the last known state after the instance stops. It adds a snapshot uploader on the instance, a private S3 bucket, a read-only IAM user for the local app, and the local FastAPI and htmx app with its overview panels and the expected versus actual instance state. Charts and logs (the agent's `GET /series` and `GET /logs` and their panels) are Phase 2b. Controls stay in Phase 3.

| Decision | Choice |
| --- | --- |
| Machines | This Windows PC and the Mac, each with its own start-at-login entry |
| State transport | One status document: the uploader puts the agent's `/status` shape plus the latest decision records into a single S3 object |
| Local cache | None in 2a; SQLite arrives in Phase 3 with the action log |
| Decision times | A table in the local app, pinned to the adapters' hardcoded times by a drift test |
| Instance role | `homeguard-ec2-cloudwatch`, managed by Terraform as `aws_iam_role.ec2_cloudwatch` |
| Uploader transport | The AWS CLI on the instance (`aws s3 cp`), so no boto3 in the instance venv |
| Local AWS access | A read-only IAM user `homeguard-console`, keys created by the operator outside Terraform |

## Changes from the parent spec

- **One document shape instead of raw files.** The parent spec has the uploader copy the snapshot JSONs, the toggle file, the state file and the latest decision records to S3, which would give the local app two parsers (agent JSON and raw files) that must agree. Instead the uploader calls the agent's own `build_status()` and decision reader and uploads one document, so the local app renders a single shape whichever source it came from.
- **No SQLite in 2a.** With the S3 snapshot covering an app restart while the instance is off, the cache's first real need is the Phase 3 action log, so it moves there.
- **Open questions answered.** The local app runs on the Windows PC and the Mac. The instance role is `homeguard-ec2-cloudwatch` (verified with `aws ec2 describe-instances` on 2026-10-07; Terraform's `ignore_changes` note is about the attachment, the role itself is in monitoring.tf). Decision times come from a local table.
- **Phase 2 is split.** 2a meets the Phase 2 exit gate on its own; 2b adds `/series`, `/logs` and their panels.

## Instance side

### Uploader

`src/console_agent/upload.py`, run as `python -m src.console_agent.upload --reason periodic|shutdown`.

- Builds `{"uploaded_at": <ISO UTC>, "reason": <reason>, "status": build_status(config, now), "decisions": {<strategy>: <full latest record or null>}}`, reusing `status.build_status` and `status.read_latest_decision`, so it imports nothing beyond what the agent imports.
- Writes the document to a temporary file and runs `aws s3 cp <tmp> s3://<bucket>/console/latest/status.json` as an argument list with a 30 second timeout. The bucket comes from `CONSOLE_SNAPSHOT_BUCKET` in the instance .env, and a missing value exits non-zero with an error naming the variable.
- A failed upload logs an error and exits non-zero; it never retries in a loop, so it never holds up shutdown beyond its timeout.
- Before the plan relies on the CLI, the rollout checks `aws --version` on the instance.

### Units

All in `infra/ec2/services/`.

| Unit | Shape |
| --- | --- |
| `homeguard-console-upload.service` | `Type=oneshot`, runs the uploader with `--reason periodic`, `User=ec2-user`, `MemoryMax=256M`, `OOMScoreAdjust=500` |
| `homeguard-console-upload.timer` | `OnBootSec=2min`, `OnUnitActiveSec=5min` |
| `homeguard-console-upload-shutdown.service` | `Type=oneshot`, `RemainAfterExit=yes`, `ExecStart=/bin/true`, the uploader with `--reason shutdown` in `ExecStop=`, `TimeoutStopSec=60`, `After=` and `Wants=` network-online.target, and `After=` homeguard-multi, homeguard-cscm, homeguard-gateway and homeguard-console-agent |

systemd stops units in the reverse of their start order, so the shutdown upload runs while the trading units, the gateway, the agent and the network are still up, and the snapshot shows the last live reading. The 5 minute timer covers a crash or a forced stop, neither of which runs `ExecStop=`.

`infra/ec2/setup/install_console_upload.sh` installs and enables the three units, idempotently, in the style of install_console_agent.sh.

### AWS

New file `infra/terraform/console.tf`.

- **Bucket:** private, all public access blocked, SSE-S3, no versioning, and it only ever holds `console/latest/status.json`. The name comes from a variable with no default, set in terraform.tfvars.
- **Instance role:** an inline policy on `aws_iam_role.ec2_cloudwatch` granting `s3:PutObject` on `arn:aws:s3:::<bucket>/console/latest/*` only.
- **IAM user `homeguard-console`:** read-only in 2a, with `ec2:DescribeInstances` and `scheduler:GetSchedule` on all resources, `s3:GetObject` on the prefix, and `logs:FilterLogEvents` on the two scheduler Lambda log groups. `ec2:StartInstances` and `ec2:StopInstances` are added in Phase 3.
- **Access keys** are created by the operator with `aws iam create-access-key`, so they never enter Terraform state, and stored as the `homeguard-console` profile on both machines.
- docs/INFRASTRUCTURE_OVERVIEW.md and infra/terraform/README.md are updated.

## Local app

### Structure

`tools/console/`, run as `python -m tools.console` under uvicorn on 127.0.0.1:8765.

| Module | Responsibility |
| --- | --- |
| `app.py` | FastAPI app; its lifespan starts the poller. `GET /` renders the page and `GET /panels/<name>` the htmx fragments. Middleware rejects any request whose Host is not `127.0.0.1:<port>` or `localhost:<port>` with 400. |
| `poller.py` | One async task that keeps a `ConsoleState` current |
| `schedule.py` | Parses the four EventBridge cron expressions and computes the expected instance state at a given time |
| `checks.py` | The eight exception checks and their levels |
| `freshness.py` | Classifies each value as frozen, drifting, unknown or live from its source and age |
| `decision_times.py` | `DECISION_TIMES = {"ramp": ["15:55"], "omr": ["09:31", "15:50"]}` |
| `templates/`, `static/` | Jinja templates, server-rendered SVG for the schedule rail, CSS tokens, and htmx vendored as a local file |
| `requirements.txt` | fastapi, uvicorn, jinja2, httpx, boto3, so the Mac can install them |
| `install_windows_task.ps1`, `install_macos_launchd.sh`, `com.homeguard.console.plist` | Start-at-login entries |

`schedule.py`, `checks.py` and `freshness.py` are pure functions over plain data and the current time, which is where the tests concentrate.

### Data flow

| Source | Cadence | Notes |
| --- | --- | --- |
| Agent `GET /status` at `CONSOLE_AGENT_URL` | Every 10 s | 5 s timeout. The full `GET /decisions` record for a strategy is fetched only when its latest decision time changes. |
| S3 `console/latest/status.json` | Every 60 s, only while the agent is unreachable | The same document shape as the agent, plus `uploaded_at` and `reason` |
| EC2 `DescribeInstances` | Every 30 s | Instance state and its launch time |
| Scheduler `GetSchedule` for the four schedules | At startup and every 10 minutes | homeguard-start-instance `cron(0 8 ? * MON-FRI *)` and homeguard-stop-instance `cron(0 20 ? * MON-FRI *)` in America/New_York, homeguard-start-instance-sunday `cron(0 23 ? * SAT *)` and homeguard-stop-instance-sunday `cron(10 0 ? * SUN *)` in UTC |

`ConsoleState` holds the newest document with its source (`agent` or `s3`) and the time it was read or uploaded, the instance state, the parsed schedules and the poll errors per source. Panels render only from it, so several open tabs never multiply the calls. Configuration comes from the repo .env (`EC2_INSTANCE_ID`, `EC2_REGION`, `CONSOLE_AGENT_URL`, `CONSOLE_SNAPSHOT_BUCKET`) and the `homeguard-console` AWS profile, with `<YOUR_VALUE>` placeholders added to .env.example.

### Panels

- **Header:** the source badge, for example "Live from agent, 4 s ago" or "Snapshot from S3, taken 20:00:12 (shutdown)", and a thin strip listing the agent's `errors` and the poller's errors.
- **Today's schedule:** an SVG rail with the instance window from the schedules, the NYSE session from pandas_market_calendars (holidays and early closes included), the 09:15 to 16:15 power lock band, decision ticks from `DECISION_TIMES` for strategies with a unit running, and a now line. Beside it, the power tile shows actual against expected state using the parent spec's table, with no buttons in 2a.
- **Exceptions:** eight fixed tiles, below.
- **Strategies:** name, switch, process (unit state or "no unit"), variant, and the latest decision's time and pass or fail. A switch that is on with no unit running it shows as a caution. Expanding a row fetches that strategy's gate results.
- **Account:** equity, day P&L, drawdown and position count from the snapshot gauges, greyed out with their age when the source is S3.
- **Host:** each unit's active and sub state, restart count, memory and active-since time.

The layout and dark theme follow the prototype, and every state carries a text label as well as a color.

### Exception checks

| Check | Source | Rule |
| --- | --- | --- |
| IB Gateway | `homeguard-gateway` unit | Not active: warning |
| Broker heartbeat | `hg_broker_last_heartbeat_timestamp` | Older than 120 s while the instance is up: warning |
| Market data stream | `hg_websocket_connected` | 0 during the session: caution; gauge absent: "not reported" in grey |
| Decisions on schedule | Latest decision time and `DECISION_TIMES` | On a trading day, a scheduled time has passed and the latest decision is older than it plus 5 minutes: warning |
| Order rejects | `hg_orders_rejected_total` | Above 0 since the process started: caution, with the count |
| Drawdown | `hg_portfolio_drawdown_pct` | Negative by construction (METRIC_SPEC.md): caution at or below -10%, warning at or below -20%; the plan confirms whether the gauge is a fraction or a percent before fixing the bounds |
| Host memory | homeguard-multi `MemoryCurrent` against its 1G cap | Above 80%: caution; above 95%: warning |
| Metrics scrape | VictoriaMetrics | Unknown, labelled "measured from 2b" |

When the source is S3, the live-only checks (gateway, heartbeat, stream, memory) render as hatched tiles reading "Unknown since <upload time>" with the last reading underneath, while decisions and rejects render normally, stamped with the upload time.

### Expected versus actual state

| Actual | Expected by schedule | Console shows |
| --- | --- | --- |
| Stopped | Stopped | Neutral, with the next scheduled start |
| Stopped | Running | Warning that the instance should be running, with a link to the start Lambda's log |
| Running | Stopped | Caution that the instance is running outside its schedule |
| Running, agent unreachable for over 2 minutes | Running or stopped | Warning that the instance is up but the console cannot reach it, never shown as stopped |

The expected state follows the schedules, not the exchange calendar, since the schedule also starts the instance on market holidays.

## Error handling and security

- Each poll source fails on its own; a failure is logged, recorded in `ConsoleState.errors` and shown in the header strip, and the other sources keep updating. The page never returns 500 because a source is down.
- With no S3 object at all, the panels show "No snapshot yet" instead of empty grey tiles.
- Every external call has a timeout: 5 s for the agent, and botocore's standard retry mode with connect and read timeouts for AWS.
- The app binds to 127.0.0.1 and checks the Host header. 2a has no POST routes, so the HX-Request, Origin and confirm token checks arrive with the Phase 3 actions. AWS credentials stay in the named profile and are never rendered.

## Test plan

| Level | Case | Expected |
| --- | --- | --- |
| Unit | Expected state from the four schedules | Weekday 07:59 ET stopped, 08:00 running, 20:00 stopped; Saturday 23:30 UTC running; Sunday 00:10 UTC stopped; 2026-11-26 (holiday) still running 08:00 to 20:00 |
| Unit | Cron parsing | The four real expressions parse; an unsupported field raises instead of guessing |
| Unit | Each exception check at its boundaries | Heartbeat 119 s and 121 s, memory 80% and 95%, drawdown at -10% and -20% with the negative bound, absent gauges, decisions on a holiday and on an early-close day |
| Unit | Freshness | Live-only checks become unknown on the S3 source, drifting values are greyed with their age, and unknown never uses the normal style |
| Unit | Decision-times drift | The table matches the times hardcoded in ramp_live_adapter.py and the OMR adapter |
| Unit | Uploader | The document has the agreed keys and shape; a missing bucket variable and a failed `aws s3 cp` each log an error and exit non-zero |
| Integration | App against a fake agent (a local `ThreadingHTTPServer`) with botocore Stubber for EC2, the Scheduler and S3 | Falls back to S3 when the agent times out, labels every value with its source, and returns to live when the agent recovers |
| Integration | Host check | A foreign Host returns 400 |
| Integration | Uploader units on the instance | `systemd-analyze verify` passes for all three units |
| End to end | Exit gate | See below |

## Rollout

Steps marked [operator] wait for the operator's explicit go-ahead.

1. Build on a worktree branch `feat/console-p2a`, then merge to main under the repo's standing permission.
2. [operator] Targeted `terraform apply` of console.tf. The operator creates the access keys and the `homeguard-console` profile on both machines.
3. [operator] Cherry-pick the uploader and its units onto `ramp-phase4-turnover-regime-research`, push, add `CONSOLE_SNAPSHOT_BUCKET` to the instance .env, check `aws --version`, and run install_console_upload.sh.
4. Install and run the console on the Windows PC, then on the Mac.
5. Write the session log in docs/progress/.

## Exit gate

- The `homeguard-console` profile can describe the instance and read all four schedules.
- After a 20:00 ET scheduled stop, the console shows the shutdown snapshot with an `uploaded_at` within a minute of the stop, source labels on every panel, and unknown tiles for the live-only checks.
- A stopped instance inside the schedule window raises the missed start warning, verified with a stubbed instance state rather than by breaking a real start.

## Rollback

Stop the local app and remove its start-at-login entries, disable the three uploader units, and optionally delete the bucket and the IAM user with a targeted `terraform destroy`. Nothing in 2a writes to trading state, so the trading units are never touched.
