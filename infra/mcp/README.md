# Grafana MCP wrappers

Stdio wrappers that let Claude Code and Claude Desktop query the tailnet-scoped
Grafana on `homeguard-ec2` read-only.

Full setup, verification, and rollback: [`docs/monitoring/20260727_GRAFANA_MCP_SETUP.md`](../../docs/monitoring/20260727_GRAFANA_MCP_SETUP.md)

| File | Platform |
|---|---|
| `homeguard-grafana-mcp.cmd` | Windows (`swangpc2`) |
| `homeguard-grafana-mcp.sh` | macOS (`sws-macbook-pro`), Linux |

## Why a wrapper at all

Neither `~/.claude.json` nor `claude_desktop_config.json` encrypts embedded
environment variables, and both default to world-readable. The wrapper reads the
service-account token from `~/.config/homeguard/grafana-mcp.env` instead, so the
client configs contain only a path. The env file is deliberately outside the repo
and outside Dropbox.

Keeping both wrappers in git, rather than at `/usr/local/bin`, means the flag set
is version-controlled and identical on both machines. Change flags here and both
clients pick them up on next start. **If you edit one wrapper, edit the other.**

## Flags

Resolved against `mcp-grafana v0.17.2`; re-check `mcp-grafana --help` after any
upgrade, because a new default-enabled category would otherwise appear silently.

| Flag | Reason |
|---|---|
| `--enabled-tools "search,datasource,dashboard,prometheus,loki,alerting,navigation,annotations"` | Allowlist, not a denylist. New upstream tool categories cannot silently attach themselves to a production trading integration. |
| `--disable-write` | Blocks create/update tools. Second layer over the Viewer role. |
| `--disable-rendering` | Needs the Grafana Image Renderer plugin, which `install_grafana.sh` does not deploy. The tool would fail at call time. |
| `--max-loki-log-limit 200` | Default is 100. Loki's `max_entries_limit_per_query` is unset in `config/monitoring/loki/config.yaml`, so the server-side ceiling is the Loki default of 5000; 200 sits well under it. |
| `--slow-request-threshold 5s` | Distinguishes "EC2 is stopped" from "query is slow" in stderr without full `--debug`. |

### Why an allowlist

`v0.17.2` enables these by default:

```
search datasource incident prometheus loki alerting dashboard folder oncall
asserts sift pyroscope navigation proxied annotations rendering snapshot
plugin api config provisioning
```

A denylist covering only the Grafana Cloud features (`incident`, `oncall`,
`asserts`, `sift`, `pyroscope`) leaves `folder`, `proxied`, `snapshot`, `plugin`,
`api`, `config`, and `provisioning` enabled. `proxied` in particular exposes
tools from external MCP servers registered as Grafana datasources. The allowlist
is a smaller, stated surface.

Yields 26 tools. Verify after changes:

```bash
printf '%s\n%s\n%s\n' \
 '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"probe","version":"0"}}}' \
 '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
 '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
 | ./homeguard-grafana-mcp.sh 2>/dev/null | tail -1 | python3 -m json.tool | grep '"name"'
```

Both wrappers write diagnostics to stderr only. stdout is the MCP JSON-RPC
channel, so anything printed there corrupts the protocol.

## Datasource UIDs

| UID | Type | Backend | Retention |
|---|---|---|---|
| `victoriametrics` | `prometheus` | `http://127.0.0.1:8428` | 90 days |
| `loki` | `loki` | `http://127.0.0.1:3100` | 30 days |

Provisioned `editable: false`, so these are stable. Name the UID and bound the
time window in every prompt: both stores return an empty result set past their
retention horizon rather than an error, so "no data" is ambiguous by default.
