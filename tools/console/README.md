# Homeguard Console (local app)

Read-only operations console for the EC2 trading instance. It polls the console agent over the
tailnet every 10 s, falls back to the S3 snapshot (`console/latest/status.json`) when the agent is
unreachable, and reads EC2 and the four EventBridge schedules with the read-only `homeguard-console`
AWS profile. Design: `docs/superpowers/specs/2026-10-08-homeguard-console-phase2a-design.md`.

## Setup

1. Add `CONSOLE_AGENT_URL`, `CONSOLE_SNAPSHOT_BUCKET` (and optionally `CONSOLE_AWS_PROFILE`) to the repo
   `.env`; `EC2_INSTANCE_ID` and `EC2_REGION` are already there.
2. `aws configure --profile homeguard-console` with the keys from `aws iam create-access-key --user-name homeguard-console`.
3. Run once by hand: `python -m tools.console`, then open http://127.0.0.1:8765.
4. Start at login:
   - Windows: `powershell -File tools\console\install_windows_task.ps1 -Python <env>\pythonw.exe`. The task runs without a console, so the app logs to `%LOCALAPPDATA%\Homeguard\console.log`.
   - macOS: `bash tools/console/install_macos_launchd.sh <env>/bin/python`

## Modules

| Module | Role |
| --- | --- |
| `poller.py` | ConsoleState from the agent, S3, EC2 and the Scheduler |
| `schedule.py` | Cron parsing, expected instance state, NYSE session and power lock |
| `checks.py` | The eight exception checks |
| `freshness.py` | Live, frozen, drifting and unknown classes |
| `views.py` | Template contexts |
| `app.py` | FastAPI app and Host check |
| `decision_times.py` | Decision times, pinned to the adapters by a test |
