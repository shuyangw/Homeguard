# V11 Paper-Validation Monitoring -- Context Handoff

**Purpose**: read this on session start (or after context clear) to recover state on V11 paper-validation monitoring without re-reading the full 2026-05-24 campaign history.

**Last updated**: 2026-05-25 (immediately after RAMP regime-detector campaign closure + WS-3d halt)

---

## Where we are

**V11** (RAMP turnover-lite variant: V01 + rank_buffer + min_hold + delta_threshold) is the only RAMP variant currently in production paper. Deployed 2026-05-23 04:30 UTC via `homeguard-multi` service on EC2. The 5-experiment WS-3 / WS-3d detector-replacement campaign closed TIER 4 (see `docs/progress/20260524_RAMP_REGIME_DETECTOR_CAMPAIGN_CLOSURE.md`); V11 is now in its A7 5-session paper validation.

**Target**: A7 counter `/var/lib/homeguard/a7_clean_sessions` reaches **5 clean sessions** before V11 graduates to live consideration.

**Earliest pass date**: 2026-05-30 (Fri, if every Mon-Fri session is clean from 2026-05-26 onward).
**Realistic pass date**: mid-June (allowing for occasional reset).

**Current counter value**: 0 (as of 2026-05-25 02:03 ET, after EC2 cold-restart). First increment expected after tonight's 15:55 ET rebalance.

---

## Daily check (one command)

```bash
tail -n 20 ~/.homeguard/a7_log/a7_check.log
```

The log auto-populates via Windows Task Scheduler (cron equivalent) Mon-Fri at 16:10 ET, 5 minutes after the 16:05 ET A7 helper writes the counter. The cron entry runs `scripts/ops/check_a7_counter.sh`, which SSHes to EC2, reads the counter, captures recent journalctl lines, and appends to the log.

**To register the cron** (one-time, ~5 sec):

```powershell
schtasks /create /tn HomeguardA7Check /tr "%CD%\infra\ec2\check_a7.bat" /sc weekly /d MON,TUE,WED,THU,FRI /st 16:10 /f
```

(Run from repo root in PowerShell or cmd. Not yet registered as of 2026-05-25 -- pending user action.)

**To verify**: `bash scripts/ops/check_a7_counter.sh` -- should print one line with the current counter value.

---

## Reading the log

Look for:

- **Counter incrementing** (`0 -> 1 -> 2 -> ... -> 5`): V11 progressing toward graduation. No action.
- **Counter resetting** (drop in value): clean-session check failed. See decision rule below.
- **`SSH_FAIL`**: EC2 is unreachable. Either stopped (start via `infra/ec2/local_start_instance.bat`) or SSH config drifted.

---

## Decision rule (pre-registered, do NOT modify after A7 outcome known)

Full taxonomy at `docs/operations/V11_PAPER_VALIDATION.md`. Reset event classes: `C-DATA` (data outage), `C-FILTER` (V11 filter edge case), `C-EXEC` (broker drift), `C-INFRA` (EC2/scheduler), `C-OTHER` (anything else).

| Pattern | Action |
|---|---|
| 0-1 resets, any class | Continue monitoring; no action |
| 2 consecutive resets, **same class** | **Investigate-and-redeploy** -- patch the variant or comparator, restart A7 timer |
| 2 consecutive resets, different classes | Continue monitoring (noise, not structural) |
| 3+ resets, **same class** | Investigate-and-redeploy with elevated severity |
| 3+ resets, **mixed classes** | **Roll back to V01 baseline** as production paper |
| Repeated `C-OTHER` | **Halt production paper** -- shut RAMP's slot in `strategy_toggle.yaml`; redirect to non-RAMP work |

**A7 cleared = counter reaches 5 with no resets in between.** Live promotion is a separate analyst decision; not automatic.

---

## What NOT to do during the A7 wait

- **No new V11-family variants on the v0 detector.** Trial chain is at 36; further variants inherit a DSR threshold that cannot be cleared without forward OOS evidence. See campaign closure for detail.
- **No retrying WS-3d.** Three Gate 1 rounds failed; the leading-indicator hypothesis is falsified at consumer-relevant threshold. Closure is decisive.
- **No deleting the local `archive/regime-detector-campaign-2026-05` branch.** It's the authoritative reference for the campaign; ~28 commits of variant code, specs, reports.
- **No starting C1/C2/C3 spec work** until A7 outcome is known. Pre-registered sequencing in `docs/progress/20260525_RAMP_POST_CAMPAIGN_NEXT_STEPS.md`.

---

## Open operational items (todo)

1. **Register the schtasks cron** (PowerShell one-liner above). User action; ~5 sec.
2. **Review and merge PR #6** (https://github.com/shuyangw/Homeguard/pull/6 -- cherry-pick of leading-indicators package + detector freshness timestamp to main).
3. **(Optional) clean up vestigial `mp.enabled: true`** in `strategy_toggle.yaml` -- the homeguard-mp systemd unit is disabled and inactive; the YAML flag is noise.

---

## Conditional next steps (post-A7)

Once A7 outcome is known, sequence per the post-campaign next-steps doc:

| A7 outcome | Primary action | Secondary |
|---|---|---|
| **Cleared (5 sessions)** | C1: Universe expansion spec (S&P 500 + NDX-100 Tier 1) | C2: RAMP-OMR portfolio (only if OMR healthy -- currently OMR is INACTIVE per A3 audit, so C2 needs OMR revival first) |
| **2+ resets same class** | Investigate-and-redeploy per failure-path rule | Continue A7 timer |
| **3+ resets mixed** | Roll back to V01 baseline | Then C3 Darwinex FX (independent of RAMP) |
| **Repeated C-OTHER** | Halt production paper | Redirect to non-RAMP (FX, monitoring, etc.) |

---

## Key file pointers

| Topic | File |
|---|---|
| Cron setup details | `docs/operations/A7_MONITORING_SETUP.md` |
| V11 failure-path runbook | `docs/operations/V11_PAPER_VALIDATION.md` |
| Campaign closure (full history) | `docs/progress/20260524_RAMP_REGIME_DETECTOR_CAMPAIGN_CLOSURE.md` (on archive branch) |
| Post-campaign next-steps plan | `docs/progress/20260525_RAMP_POST_CAMPAIGN_NEXT_STEPS.md` |
| Variant glossary | `docs/strategies/RAMP_VARIANTS.md` (V01-V14 + V20+ archive entries) |
| Cron script | `scripts/ops/check_a7_counter.sh` |
| Windows wrapper | `infra/ec2/check_a7.bat` |
| Branch with full campaign | `archive/regime-detector-campaign-2026-05` (origin, 30+ commits) |
| Branch with cherry-picked subset | `feat/leading-indicators-and-detector-timestamp` -> PR #6 to main |

---

## EC2 state notes (as of 2026-05-25 02:03 ET)

- Instance: started, IP `100.30.95.146`. (Public IP changes on stop/start; check `infra/ec2/.env` or the start-script output.)
- `homeguard-multi`: active (running), PID 2091, running `homeguard-ramp` process. Confirms `--strategy ramp` invocation per `strategy_toggle.yaml`.
- `homeguard-omr`: **inactive (disabled)**. Memory incorrectly flagged OMR as active; A3 audit corrected this.
- `homeguard-mp`: **inactive (disabled)**. YAML flag is vestigial.
- `homeguard-cscm`: active (running), PID 1453. Runs independently of toggle YAML.
- A7 counter file exists, value = 0.
- Today's 15:55 ET rebalance window: NOT missed (homeguard-multi is up well before).

---

## When in doubt

- **What's the next user action?** Register schtasks cron + merge PR #6 + wait. Do not start new RAMP work.
- **What if A7 resets?** Look at `~/.homeguard/a7_log/a7_check.log`, classify the reset per `V11_PAPER_VALIDATION.md` taxonomy, apply the decision rule.
- **What if EC2 stops again?** Restart via `infra/ec2/local_start_instance.bat`. The IP will change.
- **What if I want to start C1/C2/C3 anyway?** Don't -- the pre-registered sequencing exists to prevent wasted effort on a path A7's outcome may invalidate. If A7 has cleared (counter = 5), then proceed per the C-conditional table above.
