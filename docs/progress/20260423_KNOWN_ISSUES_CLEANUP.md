# Known-Issues Cleanup From 2026-04-23 Grafana/IBKR Session - 2026-04-23

## Summary
Followed up on the "Known Issues / Remaining Work" list left by the earlier
2026-04-23 Grafana/IBKR session (see `20260423_GRAFANA_AND_IBKR_FIXES.md`).
Addressed four actionable items: misleading Alpaca log lines when running on
IBKR, missing orphan-kill protection on `homeguard-multi`, IBKR config tests
failing on env-var leakage, and five IBKR integration tests erroring because a
shared fixture didn't exist.

## Changes Made

### Broker-agnostic log lines
- **`scripts/trading/run_live_paper_trading.py:1297, 1300`**: replaced
  hardcoded "Alpaca" with the `broker_name` already resolved at line 1287 from
  `routing.get_broker_name(args.strategy)`. Previously, RAMP running on IBKR
  logged "Connected to Alpaca Paper Trading" — factually wrong and misleading
  during post-mortem of the 2026-04-23 routing-fallback incident. Now logs
  "Connected to ibkr" / "Connected to alpaca" etc.

### homeguard-multi orphan protection
- **`infra/ec2/homeguard-multi.service`**: added three directives to survive
  the failure mode from the 2026-04-23 incident, where an orphaned
  `run_live_paper_trading.py` PID 5554 from the disabled `homeguard-ramp` unit
  was holding the Alpaca WS slot (Alpaca free tier allows 1 concurrent WS per
  key) and blocking the new `homeguard-multi` process with HTTP 429.
  - `ExecStartPre=-/bin/bash -c '/usr/bin/pkill -9 -f "run_live_paper_trading.py" || true'`
    placed BEFORE the gateway-port wait, so orphans die before we block on
    gateway readiness. Leading `-` makes it non-fatal if nothing matches. Safe
    because `homeguard-{omr,mp,ramp}.service` units are all disabled and
    superseded — this unit is the only authorized runner.
  - `KillMode=mixed` — SIGTERM to the main PID on stop (lets it close the WS
    cleanly), then SIGKILL to any remaining children after the timeout.
  - `TimeoutStopSec=30` — bounds the kill window.

### IBKR config test env isolation
- **`tests/trading/brokers/ibkr/test_config_and_errors.py`**: `IBKRConfig`'s
  env-override logic (src/trading/brokers/ibkr/config.py:91-94) applies
  `IBKR_*` environment variables AFTER kwargs merge, so a developer with
  `IBKR_CLIENT_ID=10` / `IBKR_PORT=4002` sourced from `.env` would silently
  override both the class default and explicit `IBKRConfig(port=...)` kwargs
  in tests. The 2026-04-23 log flagged three tests failing for this reason
  (`test_defaults`, `test_paper_detection`, `test_gateway_type_label`).
  Extracted a `_clean_ibkr_env(monkeypatch)` module helper and called it from
  each of those tests. No production code change — the
  env-overrides-kwargs behavior is documented and intentional for operators.

### IBKR integration-test fixture
- **`tests/trading/brokers/ibkr/conftest.py`**: added an `ibkr_connection`
  fixture. Five tests in `test_contracts.py::TestContractResolution` request
  this fixture but it was never defined, so pytest errored with "fixture not
  found" (five collection errors in the 2026-04-23 log). Fixture now auto-
  skips unless `HOMEGUARD_RUN_IBKR_TESTS=1` is set and the gateway starts
  successfully. Uses `IBKRConnectionManager.start()/stop()` per the API at
  `src/trading/brokers/ibkr/connection.py:33-149`.

## Commits
- (to be filled in after commit)

## Known Issues / Remaining Work

- **`strategy_toggle.yaml` drift** — NOT fixed in this pass; still "user's
  call" per the original 2026-04-23 log. Repo has `mp.enabled: false`, EC2 was
  manually toggled to `true` on 2026-04-20. Since `homeguard-multi` is pinned
  to `--strategy ramp`, the flag is dead config either way. Recommendation:
  after this change deploys, `scp config/trading/strategy_toggle.yaml` from
  the repo to EC2 to reset the drift.
- **MP/OMR scrape targets remain `down`** — expected (units are
  disabled-and-superseded by `homeguard-multi --strategy ramp`). Reviving
  either strategy would require separate unit files with `ENABLE_METRICS=true`
  and distinct `METRICS_PORT` values, following commit `3f27f8a`.
- **`--strategy multi` mode in `run_live_paper_trading.py:1447-1464`** still
  only launches a single strategy by priority order. True multi-strategy
  concurrency is a separate piece of work, not covered here.

## Validation

- **Syntax check**: `python -m ast` parses all three modified Python files
  cleanly.
- **Systemd unit parse**: `systemd-analyze verify infra/ec2/homeguard-multi.service`
  returns only the expected EC2-only error about the venv python path not
  existing on the local dev host; no syntax errors from the new directives.
- **Pytest** — could not run locally (no pandas in this dev env). User should
  verify with:
  ```
  IBKR_CLIENT_ID=10 IBKR_PORT=4002 pytest tests/trading/brokers/ibkr/ -v
  ```
  Expected: three previously-failing tests in `test_config_and_errors.py`
  pass; five tests in `test_contracts.py::TestContractResolution` report as
  SKIPPED (not error) unless `HOMEGUARD_RUN_IBKR_TESTS=1` is set.

- **EC2 deploy steps** (post-merge, by user):
  ```
  sudo cp /home/ec2-user/Homeguard/infra/ec2/homeguard-multi.service /etc/systemd/system/
  sudo systemctl daemon-reload
  sudo systemctl restart homeguard-multi
  journalctl -u homeguard-multi --since "5 min ago" | grep -E "(Connected to|Failed to connect)"
  ```
  Expected: log line reads "Connected to ibkr" (not "Connected to Alpaca
  Paper Trading").

- **Orphan-kill smoke test** (optional, on EC2):
  ```
  sudo systemctl stop homeguard-multi
  nohup python scripts/trading/run_live_paper_trading.py --strategy ramp &
  disown
  sudo systemctl start homeguard-multi
  pgrep -af run_live_paper_trading.py
  ```
  Expected: exactly one matching PID (systemd-supervised, PPID=1).
