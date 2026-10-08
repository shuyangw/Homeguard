# Console agent

Read-only HTTP agent on the EC2 instance that serves Homeguard's live state to
the local Homeguard Console. Design: `docs/superpowers/specs/2026-10-05-homeguard-console-design.md`.

## Routes (Phase 1)

| Route | Returns |
| --- | --- |
| `GET /status` | Units (systemd), per-strategy toggle state, units running each strategy, metric snapshot, latest decision gates, execution lock, and an `errors` list for any source that failed to read |
| `GET /decisions?strategy=<name>` | The full latest decision record from `data/trading/decisions/_latest/<name>.json` |

Every request must carry `Tailscale-User-Login` equal to `CONSOLE_OPERATOR_LOGIN`
(403 otherwise). Any write method returns 405.

## Rules

- Never import `src.trading`: its package `__init__` loads about 75MB of broker code,
  over the unit's `MemoryMax=96M`. `tests/console_agent/test_server.py` enforces this.
- Never write files and never construct `StrategyStateManager` (its `__init__` writes).
- Every subprocess goes through `units.run_command`.

## Run

```bash
# On the instance (via the unit):
bash infra/ec2/setup/install_console_agent.sh
curl -s -H "Tailscale-User-Login: $CONSOLE_OPERATOR_LOGIN" http://127.0.0.1:8090/status

# From the operator's machine over the tailnet:
curl -s https://<tailnet-host>:8443/status
```

## Tests

```bash
python -m pytest tests/console_agent -q
```

## Uploader (Phase 2a)

`python -m src.console_agent.upload --reason periodic|shutdown` builds the same document as `GET /status`
plus the full latest decision per strategy and copies it to `s3://$CONSOLE_SNAPSHOT_BUCKET/console/latest/status.json`
with the AWS CLI (the instance role grants `s3:PutObject` on that prefix only). It exits non-zero on any failure.
The units `homeguard-console-upload.timer` (every 5 minutes) and `homeguard-console-upload-shutdown.service`
(at shutdown) run it; install them with `infra/ec2/setup/install_console_upload.sh`.
