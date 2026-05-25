# V11 Paper-Validation Runbook (A7)

## Status

V11 deployed to production paper 2026-05-23 04:30 UTC on EC2 via `homeguard-multi`. A7 paper-validation counter at `/var/lib/homeguard/a7_clean_sessions` increments on each weekday clean rebalance. Target: 5 clean sessions before graduation from paper to live.

## Daily monitor

- Cron: weekday at 16:10 ET via Windows Task Scheduler (see `docs/operations/A7_MONITORING_SETUP.md`).
- Log: `~/.homeguard/a7_log/a7_check.log`.
- Manual ad-hoc: `bash scripts/ops/check_a7_counter.sh`.

## Pre-registered failure-path decision rule

This rule is committed BEFORE A7 outcome is known to prevent hindsight bias.

### Reset event taxonomy

A "reset" is any decrement (or no-increment when expected) of the A7 counter. The same-root-cause-class classification:

- **Class C-DATA**: comparator failed due to data unavailability (Alpaca / yfinance / FRED outage).
- **Class C-FILTER**: V11 filter stack (rank_buffer / min_hold / delta_threshold) hit an edge case the comparator didn't expect.
- **Class C-EXEC**: live execution drifted from plan (broker order rejection, partial fill, rounding).
- **Class C-INFRA**: EC2 / systemctl / scheduler issue unrelated to RAMP.
- **Class C-OTHER**: anything not in the above 4 classes.

### Decision rule

| Reset count + class profile | Action |
|---|---|
| 0 resets | Continue monitoring; no action. |
| 1 reset (any class) | Continue monitoring; no action. Log the class for future correlation. |
| 2 consecutive resets, **same class** | **Option 1: Investigate-and-redeploy.** Diagnose the root cause in the recurring class. Patch the variant or comparator. Restart the A7 timer. |
| 2 consecutive resets, **different classes** | Continue monitoring; no action. Different classes suggest noise, not a structural issue. |
| 3+ resets, **mixed classes** (no 2 in same class) | **Option 2: Roll back to V01 baseline.** Redeploy the pre-V11 baseline as production paper while RAMP regime work is paused. V11 is not stable enough to ship. |
| 3+ resets, **same class** | Option 1 with elevated severity. The recurring class is the binding constraint; fix or fall back to V01. |
| Any reset in **Class C-OTHER**, repeated | **Option 3: Halt production paper.** If "other" issues are repeated, RAMP's slot in `strategy_toggle.yaml` is shut down and effort redirects to non-RAMP work (FX, RAMP-OMR portfolio, etc.). |

### Pass condition

A7 cleared = counter reaches 5 with no resets in between. On pass:
- V11 is eligible for live promotion.
- Live promotion is a separate decision (analyst's judgment based on paper-Sharpe + risk tolerance), not automatic.
- The campaign-closure doc and RAMP_VARIANTS.md are updated to reflect V11 as ready-for-live.

### Decision is locked

This rule is committed in advance. Modifying it after A7 outcome is known constitutes hindsight bias and requires a separate spec amendment.

## Operational details

### Where the counter lives

- File: `/var/lib/homeguard/a7_clean_sessions` on EC2 (read by `cat`).
- Writer: the A7 helper script (runs at 16:05 ET Mon-Fri after the rebalance).
- Reset trigger: comparator finds a discrepancy between live state and the V11 plan output.

### Inspection commands

```
# Counter value
ssh ec2 "cat /var/lib/homeguard/a7_clean_sessions"

# Recent comparator activity (last 2 hours)
ssh ec2 "sudo journalctl -u homeguard-multi --since '2 hours ago' --no-pager | grep -iE 'a7|clean_session|variant.*v11|reset'"

# Position ledger state
ssh ec2 "cat /home/ubuntu/Homeguard/output/state/ramp_position_state.json"
```

(The `ssh ec2` alias is defined in EC2 setup; see `infra/ec2/`.)

### Live-promotion decision (after A7 cleared)

Out of scope for this runbook. Tracked separately. The runbook covers A7 only.
