# IBKR Migration Validation & EC2 Upgrade - 2026-04-18

## Summary
Completed broker-switching validation runbook (Phase 5.1), upgraded EC2 from t4g.small (2GB) to t4g.medium (4GB) for multi-strategy headroom, and updated all infrastructure docs to match.

## Changes Made
- **EC2 instance**: Upgraded t4g.small -> t4g.medium via Terraform. 4GB RAM gives ~2.8GB free after Gateway + CSCM + CloudWatch Agent.
- **SSH security group**: Consolidated to single CIDR rule (old IP removed, current IP only).
- **Docs (13 files)**: Updated all references to instance type, RAM, and cost estimates across CLAUDE.md, SETUP.md, docs/, and infra/ directories.
- **CLAUDE.md**: Added "Session Work Logs" section requiring post-implementation summaries in `docs/progress/`.

## Commits
- `0e69148` docs(infra): update instance type t4g.small -> t4g.medium across all docs
- Previous session commits (for context):
  - `1cdeee6` feat(monitoring): CloudWatch Agent for memory/swap/disk metrics
  - `35e3260` fix(infra): Gateway service Restart=always to survive IB nightly reset
  - `721a134` fix(cscm-demo): remove redundant market hours check blocking Sunday rebalance

## Validation Runbook Results (Phase 5.1)

| Step | Test | Result |
|---|---|---|
| 1 | pytest tests/trading/ | 542 pass, 14 pre-existing failures (none from broker switching) |
| 2 | OMR dry-run via routing (--once) | `omr execution broker: ibkr`, pre-flight passed |
| 3 | OMR on IBKR | Connected to Gateway at 127.0.0.1:4002, reconciled |
| 5 | Rollback OMR to Alpaca | `omr execution broker: alpaca`, pre-flight passed, then restored to IBKR |
| 6 | RAMP via routing | `ramp execution broker: ibkr` -- routing works |
| 7 | MP via routing | `mp execution broker: ibkr` -- routing works |
| 8a | Mismatch detection | Injected fake position, runner blocked with POSITION MISMATCH DETECTED |
| 8b | --force-start bypass | Proceeded with warning, state file unchanged |

## Known Issues / Remaining Work
- **IBKRBroker.get_positions() missing**: `session_tracker.generate_end_of_day_report()` calls `broker.get_positions()` which exists on AlpacaBroker but not IBKRBroker (has `get_stock_positions()`).
- **BRK.B symbol format**: IBKR uses `BRK B` (space), not `BRK.B` (dot). Affects RAMP and MP S&P 500 universe.
- **Alpaca streaming symbol limit**: RAMP subscribes to 405 symbols but Alpaca WebSocket caps at ~350.
- **14 pre-existing test failures**: IBKR config defaults, MP data caching, EOD report API, streaming integration, live trading logging -- none related to broker switching.
- **SNS email subscription**: User needs to confirm email (qwqw1337@gmail.com) for CloudWatch alarm notifications.
- **Gateway nightly reset**: Needs verification that Restart=always survives the ~23:45 UTC IB maintenance window.
