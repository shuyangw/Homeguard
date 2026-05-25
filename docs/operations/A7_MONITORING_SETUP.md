# A7 Paper-Validation Passive Monitoring -- Setup

## Purpose

The V11 RAMP paper-validation pipeline ("A7") on EC2 emits a counter at `/var/lib/homeguard/a7_clean_sessions` and increments it after each weekday clean rebalance. Target: 5 clean sessions before V11 graduates from paper to live.

The post-campaign next-steps doc (`docs/progress/20260525_RAMP_POST_CAMPAIGN_NEXT_STEPS.md`) chose **Option D: cron + log file** as the most minimal passive monitor: a daily check appends counter state to a local log; the analyst reads the log on their own cadence.

## What this sets up

- `scripts/ops/check_a7_counter.sh` -- bash check that SSHes to EC2 and appends to `$HOME/.homeguard/a7_log/a7_check.log`.
- `infra/ec2/check_a7.bat` -- Windows Task Scheduler wrapper that invokes the bash script via Git Bash.
- Daily Mon-Fri at 16:10 ET (5 minutes after the 16:05 ET A7 helper); the helper writes the counter, the check reads it.

## One-time setup

Open PowerShell or `cmd.exe` as a regular user (does not need admin). From the repo root:

```
schtasks /create /tn HomeguardA7Check /tr "%CD%\infra\ec2\check_a7.bat" /sc weekly /d MON,TUE,WED,THU,FRI /st 16:10 /f
```

Verify:

```
schtasks /query /tn HomeguardA7Check
```

The task will run weekdays at 16:10 in your local-machine timezone. If your machine's clock is ET, you're done. If your machine is in another tz, adjust the `/st` time accordingly.

### Manual check (ad-hoc)

To run the check immediately and verify it works:

```
bash scripts/ops/check_a7_counter.sh
```

Or from Git Bash:

```
./scripts/ops/check_a7_counter.sh
```

Expected output: a single line like `[2026-05-25T16:10:00-04:00] A7 counter: 0`.
Check the log: `cat ~/.homeguard/a7_log/a7_check.log`.

## Reviewing the log

```
tail -n 50 ~/.homeguard/a7_log/a7_check.log
```

Look for:
- **Counter incrementing**: 0 -> 1 -> 2 -> ... -> 5 means V11 is making progress toward paper graduation.
- **Counter resetting**: any drop in the counter value means a clean-session check failed; investigate the comparator output captured in the same log entry.
- **SSH_FAIL**: EC2 is unreachable. If sustained, the EC2 instance may be stopped (start it via `infra/ec2/local_start_instance.bat`) or the SSH config has drifted.

## Failure path

Per the A2 runbook at `docs/operations/V11_PAPER_VALIDATION.md`:
- 1 reset: continue monitoring; no action.
- 2 consecutive resets, same root-cause class: investigate-and-redeploy.
- 3+ resets, mixed root causes: roll back to V01 baseline.
- Structural RAMP issues V11 can't address: halt production paper.

## Removal

```
schtasks /delete /tn HomeguardA7Check /f
```

## Why this design

The next-steps doc evaluated four monitoring options. Option D was chosen for "minimum touch / minimum infrastructure" reasons:

- No EC2 code changes (B / C options require modifying the A7 helper on EC2).
- No long-running Claude Code routine (A option requires a daily CC agent).
- The log is sufficient: V11's A7 outcome is reviewed on the analyst's own cadence, not in real-time.

If the analyst later wants real-time push notifications, the Option B (EC2 Discord webhook) design is documented in the same next-steps doc.
