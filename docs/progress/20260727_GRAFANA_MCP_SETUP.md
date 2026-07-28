# Grafana MCP Setup - 2026-07-27

## Summary

Ingested an external Grafana MCP setup draft, verified every claim against the repo
and the live host, and installed the integration on Windows. The draft was largely
sound but contained two factual errors, one dangerous shell bug, and an unstated
tool-exposure gap. Read-only Grafana MCP is now connected in Claude Code and
verified end-to-end against live production metrics.

## Changes Made

- **`infra/ec2/setup/install_grafana.sh`**: replaced three non-idempotent `sed`
  calls with a `set_ini_key` helper. The old `sed 's/^;key =.*/'` form matched only
  the commented default, so it silently no-opped on every re-run after the first.
  Added `root_url` (never set at all). Routed `admin_password` through the same
  helper, which the draft missed.
- **`infra/ec2/setup/install_tailscale.sh`**: added guarded `tailscale serve --bg
  3000` to codify a config that had been applied by hand and never captured.
  Corrected the closing echo block, which advertised two unreachable URLs.
- **`docs/INFRASTRUCTURE_OVERVIEW.md`**: Loki retention 14 -> 30 days; rewrote the
  remote-access section, which claimed Grafana/VM/Loki bind to the tailnet when all
  three bind loopback.
- **`infra/mcp/`** (new): `homeguard-grafana-mcp.cmd` (Windows),
  `homeguard-grafana-mcp.sh` (macOS/Linux), `README.md`. Wrappers live in git
  rather than `/usr/local/bin` so the flag set is version-controlled and cannot
  drift between the two operator machines.
- **`docs/monitoring/20260727_GRAFANA_MCP_SETUP.md`** (new): corrected setup doc.
  Appendix D tabulates every correction to the draft.

## Commits

- `a7923b1` fix(infra): make grafana.ini edits idempotent and codify tailscale serve
- `663d004` docs(infra): correct Loki retention and the monitoring bind/access model
- `11b8a7c` feat(mcp): read-only Grafana MCP wrappers for Windows and macOS
- `e84d50c` docs(monitoring): Grafana MCP setup, verified against repo and live host

Merged to `main` by fast-forward (`44220a5..e84d50c`) and pushed.

## Key Decisions

- **Allowlist over denylist.** The draft used `--disable-*` flags and deferred the
  safer `--enabled-tools` allowlist because the category strings were "not fully
  enumerated upstream". They are enumerated in `mcp-grafana --help` as of v0.17.2,
  so the allowlist was adopted. The draft's denylist would have left `folder`,
  `proxied`, `snapshot`, `plugin`, `api`, `config`, and `provisioning` enabled
  without saying so, `proxied` being notable since it exposes tools from external
  MCP servers.
- **Wrappers in git, not `/usr/local/bin`.** Makes the flag list reviewable and
  keeps both machines identical. Only the credential file stays outside the repo.
- **Service account created via HTTP API, not the UI.** Reproducible and
  scriptable; the UI path is not.
- **Go toolchain for the binary.** Sidesteps per-OS release-asset discovery on both
  platforms. Both machines already have Go.

## Errors Found in the Draft

| Draft claim | Actual |
|---|---|
| Loki retention 14 days | `720h` = 30 days. `INFRASTRUCTURE_OVERVIEW.md` was also wrong |
| `--max-loki-log-limit` sits below a configured `max_entries_limit_per_query` | That key is unset; ceiling is the Loki default 5000 |
| mcp-grafana v0.14.0 | v0.17.2 |
| Escape with `sed 's/[&|\\]/\\&/g'` | Wrong under this sed. `&` expands to the whole matched line, so a password containing `&` would corrupt `admin_password`. Needs `\\\\&` |
| Only `http_addr`/`http_port` need idempotency | `admin_password` had the same bug |
| One unreachable URL in `install_tailscale.sh` | Two, including `http://homeguard-ec2:8428/vmui` |
| Phase 0 unresolved | `tailscale serve` already configured and HTTPS certs already enabled, just uncaptured. Phase 1 is codification, not a behavior change |
| macOS only | Primary machine is Windows |

## Validation

Phase 0, resolved without SSH:
- `https://homeguard-ec2.tail3e202b.ts.net/api/health` -> `{"database":"ok","version":"12.4.3"}`
- `http://<fqdn>:3000/api/health` -> connection refused, confirming loopback bind

`set_ini_key` proven against a fixture with password `p@ss|w&rd\x` over three
idempotent passes: exactly one line per key, password round-tripped byte-exact,
and the already-uncommented drift case corrected. The first version of the helper
failed this test, which is how the escaping bug was caught.

Phase 2, service account id 2, role Viewer:
- `GET /api/datasources` -> `loki`, `victoriametrics`
- `GET /api/v1/provisioning/alert-rules` -> 200
- `GET /api/search?type=dash-db` -> 5 dashboards in folder Homeguard
- `POST /api/dashboards/db` -> **403**

Phase 3/4, `mcp-grafana v0.17.2` built to `C:\Users\qwqw1\bin\mcp-grafana.exe`:
- stdio handshake through the wrapper returns 26 tools, no write tools
- `claude mcp get grafana` -> Connected
- `grep -c glsa_ ~/.claude.json` -> 0, config holds only a path
- ACL on `~/.config/homeguard` restricted to `SWANGPC2\qwqw1`

Phase 6, live tool calls against production:
- `list_datasources` -> both UIDs
- `list_prometheus_metric_names` regex `hg_.*` -> 15 metrics
- `search_dashboards("Portfolio")` -> 2 hits with real UIDs
- `query_prometheus` instant `hg_portfolio_equity_usd` -> live equity for `homeguard-ramp` and `homeguard-cscm`
- `query_prometheus` range `max(hg_portfolio_drawdown_pct) by (job)` 24h -> real series
- `update_dashboard` and `create_annotation` -> `tool not found`, blocked at the exposure layer

Confirmed no impact on the concurrent session: md5 of its three modified files
identical before and after the fast-forward, and all still listed as modified.

## Phase 1 Applied (later the same session)

The user completed the Tailscale SSH browser check, which unblocked shell access.

On-host diagnostics confirmed every remote inference: all three services bound to
`127.0.0.1` (`3000`, `8428`, `3100`), `tailscale serve` already proxying
`https://homeguard-ec2.tail3e202b.ts.net -> http://localhost:3000`, and
`root_url` still commented at `grafana.ini:59`.

**Applied `root_url` only, not the documented full-script run.** Two reasons found
on the host:

- **EC2 tracks `ramp-phase4-turnover-regime-research`, not `main`.** `git pull`
  there brought in an unrelated alerts commit and none of this work; the host's
  `install_grafana.sh` has zero `set_ini_key` occurrences. The full apply would
  have needed a cherry-pick onto the production branch first.
- The full installer also runs `dnf install`, re-syncs datasources and dashboards
  from the deploy branch's config, overwrites the systemd unit, and restarts
  Grafana. More blast radius than Phase 1 requires on a live host.

Backed up the config, applied the change with the identical verified `set_ini_key`
logic inline, restarted Grafana. Diff against the backup was exactly one line (59).

Verified: `/api/health` ok; `/api/org` with admin auth 200, so the password
survived; `appUrl` now `https://homeguard-ec2.tail3e202b.ts.net/`; a fresh wrapper
logs `public_url=https://homeguard-ec2.tail3e202b.ts.net` and `generate_deeplink`
returns `https://homeguard-ec2.tail3e202b.ts.net/d/homeguard-portfolio`.

### R11: a bug my own fix created

Making `set_ini_key` work turned a latent no-op into a live hazard.
`install_grafana.sh` had `GRAFANA_PASS="${GRAFANA_ADMIN_PASSWORD:-admin}"`. Under
the old broken `sed`, `admin_password` was already uncommented so `^;admin_password`
never matched and the line did nothing. With the working helper, re-running the
installer without the env var exported would have **overwritten the Grafana admin
password with `admin`** on a host serving live trading dashboards. Caught before
running anything on the host. Fixed in `27ee4ff`: skip the key when the env var is
unset rather than defaulting.

### Gotcha: mcp-grafana caches public_url

It fetches `public_url` once at startup. A server process predating the `root_url`
change keeps returning `http://localhost:3000` from `generate_deeplink`, which the
in-session MCP server did. Restart the client session, not just Grafana.

## Known Issues / Remaining Work

- **Deploy branch not reconciled.** The installer fix is on `main` only, while EC2
  tracks `ramp-phase4-turnover-regime-research`. A rebuild from the deploy branch
  reintroduces both the non-idempotent `sed` and the unset `root_url`. Cherry-pick
  `a7923b1` and `27ee4ff` onto it when next touched (R12).
- **`install_tailscale.sh` serve block only fires on rebuild.** Its
  `TAILSCALE_AUTH_KEY` guard `exit 1`s before reaching the new code. Harmless,
  since a rebuild is where the auth key is supplied, and serve is already active.
- **Config backup left on host** at `/etc/grafana/grafana.ini.bak.20260727`.
- **Phase 5, Claude Desktop, not started** on either machine.
- **macOS machine not set up.** Wrapper is committed and ready; needs the binary,
  the env file, and `claude mcp add`.
- **MCP tools not live in the current session.** Servers are discovered at session
  start, so a new session is needed to use them conversationally. All verification
  above went through the raw protocol instead.
- **EventBridge stop time unverified.** Docs and draft both say 4:30 PM ET, but the
  tailnet reported the peer alive until roughly 8:00 PM ET. Read the rule before
  relying on the window.
- **Instance left running.** It was started during this session and the overnight
  stop already passed. Stop it manually if the schedule does not catch it.
