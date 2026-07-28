# Grafana MCP Server -- Claude Code + Claude Desktop Setup

**Date**: 2026-07-27
**Status**: Phases 0-4 APPLIED and verified on Windows (`swangpc2`), including `root_url` on the host. Phase 5 (Claude Desktop) not started; macOS machine not set up.
**Scope**: Local stdio Model Context Protocol (MCP) integration between operator machines and the tailnet-scoped Grafana on `homeguard-ec2`
**Machines**: `swangpc2` (Windows 11, primary) and `sws-macbook-pro` (macOS)
**Related**: `docs/INFRASTRUCTURE_OVERVIEW.md`, `docs/monitoring/METRIC_SPEC.md`, `infra/mcp/README.md`, `infra/ec2/setup/install_grafana.sh`, `infra/ec2/setup/install_tailscale.sh`

This revises an external draft (`~/Downloads/20260727_GRAFANA_MCP_SETUP.md`) after
verifying every claim against the repo and the live host. Corrections to that
draft are collected in Appendix D.

---

## TL;DR

1. Claude.ai custom connectors are dialed from Anthropic's cloud and **cannot** reach a tailnet-only host. Local stdio is the only viable transport. Works in Claude Code and Claude Desktop, not claude.ai web or mobile.
2. **Phase 0 is resolved.** `tailscale serve` is already running on the host and tailnet HTTPS certificates are already enabled, but neither is captured in the repo. Grafana is loopback-bound; `https://homeguard-ec2.tail3e202b.ts.net/api/health` returns `{"database":"ok","version":"12.4.3"}` while `http://...:3000` is refused. Phase 1 therefore **codifies existing behavior** rather than changing it.
3. `root_url` was unset on the host, confirmed on-host at `grafana.ini:59`. Now **set and verified**: `appUrl` and `generate_deeplink` both return `https://homeguard-ec2.tail3e202b.ts.net/`.
4. Grafana **Viewer** service account plus an `--enabled-tools` allowlist and `--disable-write`. Verified: `POST /api/dashboards/db` with the token returns **403**.
5. Secrets live in `~/.config/homeguard/grafana-mcp.env`, never in `~/.claude.json` or `claude_desktop_config.json`, both of which store env vars in world-readable plaintext.
6. Datasource UIDs are `victoriametrics` (type `prometheus`) and `loki`, both `editable: false`.

## What this document does NOT do

- No remote MCP / custom connector. Requires public exposure via Tailscale Funnel plus a hand-rolled inbound auth layer. Out of scope, recommended against (Appendix C).
- No change to `homeguard-trading-bot-sg`. No inbound port opened.
- No write access to Grafana. No dashboard, alert-rule, or annotation mutation.
- Does not bring the monitoring stack under Terraform. OS-level daemons stay in `infra/ec2/setup/`.
- Does not resolve overlap with `homeguard-discord.service`, which covers similar ground on-host. Deferred.
- Does not change retention, scrape cadence, or the metric naming contract.

---

## Prerequisites

| Requirement | Check | Status |
|---|---|---|
| Operator machine on tailnet | `tailscale status \| grep homeguard-ec2` | Verified on `swangpc2` |
| Tailnet HTTPS certificates | Implied by a working `https://` fetch | Verified (TLS handshake succeeds) |
| Grafana >= 9.0 | Pinned 12.4.3 in `install_grafana.sh` | Verified 12.4.3 live |
| EC2 running | See Section 7 | Instance is stopped overnight |
| Go toolchain | `go version` | 1.25.4 on `swangpc2` |

---

## Phase 0 -- Bind-address divergence (RESOLVED)

The draft flagged a contradiction, and it is real. `install_grafana.sh` bound
Grafana to `127.0.0.1`, while `install_tailscale.sh` and
`INFRASTRUCTURE_OVERVIEW.md` both advertised `http://homeguard-ec2:3000`.

Resolved without SSH, by probing the endpoint from the operator machine:

```bash
FQDN=homeguard-ec2.tail3e202b.ts.net
curl -sS --max-time 15 "https://${FQDN}/api/health"
# {"database":"ok","version":"12.4.3","commit":"86c83248"}

curl -sS --max-time 10 "http://${FQDN}:3000/api/health"
# curl: (7) Failed to connect ... port 3000: Could not connect to server
```

**Verdict**: loopback bind plus a `tailscale serve` config that was applied by
hand and never captured in the repo. The repo was right about the bind; the docs
were right that the tailnet URL works, but wrong about *how*. Phase 1 codifies
the serve config so a rebuild reproduces it.

Note `root_url` was never set at all, which the draft did not catch. Confirmed
live from the MCP server's own startup log:

```
level=INFO msg="Fetched public URL from Grafana frontend settings" public_url=http://localhost:3000
```

### If you need shell access

Direct SSH over the Elastic IP currently fails: the security group allows
`73.68.21.247/32` and the operator IP has drifted to `73.218.180.119`. Tailscale
SSH intercepts port 22 and requires interactive browser auth
(`tailscale ssh ec2-user@homeguard-ec2` prints a `login.tailscale.com` URL).
Neither is a blocker for MCP, which only needs HTTPS. Options if shell is needed:
complete the Tailscale check-mode auth in a browser, or update the SG CIDR.

---

## Phase 1 -- EC2-side changes (COMMITTED, NOT YET RUN)

Both changes are in idempotent setup scripts rather than manual host edits.

### 1.1 `install_grafana.sh`

The original `sed 's/^;http_addr =.*/.../'` form matched **only the commented
default**, so it silently no-opped on every re-run after the first. A value that
had drifted or been hand-edited was never corrected. Replaced with an idempotent
`set_ini_key` helper that handles commented, uncommented, and absent keys, and
adds `root_url`.

The same latent bug applied to `admin_password`, which the draft did not mention.
It is now routed through the same helper.

`set_ini_key` escapes `&`, `|`, and `\` before substitution. This matters: the
naive escape `sed 's/[&|\\]/\\&/g'` is wrong under this sed, and a password
containing `&` would be replaced by the entire matched line, silently corrupting
`admin_password`. The correct form is `sed 's/[&|\\]/\\\\&/g'`. Verified against
a fixture with the password `p@ss|w&rd\x` over three idempotent passes.

`root_url` is derived from `tailscale status --json` via `python3` rather than
`jq`, because `python3` ships with Amazon Linux 2023 and `jq` may not. If the
FQDN cannot be resolved the script warns and leaves `root_url` untouched instead
of aborting under `set -euo pipefail`.

### 1.2 `install_tailscale.sh`

Adds a guarded `tailscale serve --bg 3000` (skipped if already published,
non-fatal on failure since tailscaled is already up and SSH still works), and
corrects the closing echo block. The old block advertised **two** unreachable
URLs, not one: `http://homeguard-ec2:3000` and `http://homeguard-ec2:8428/vmui`.
VictoriaMetrics and Loki are loopback-only and reached through Grafana's
datasource proxy (`access: proxy`), so both now print SSH-tunnel instructions.

### 1.3 Apply (APPLIED 2026-07-27, `root_url` only)

Shell access is via Tailscale SSH, which requires a one-time interactive browser
check (`tailscale ssh ec2-user@homeguard-ec2` prints a `login.tailscale.com` URL).
Direct SSH over the Elastic IP does not work: the SG allows `73.68.21.247/32`
while the operator IP has drifted to `73.218.180.119`.

**The documented full-script apply was deliberately NOT used.** Two reasons found
on the host:

1. **EC2 tracks a deploy branch, not `main`.** `~/Homeguard` is on
   `ramp-phase4-turnover-regime-research` (per `docs/progress` history, this is
   the stable V11 production branch). `git pull` there does not bring in commits
   merged to `main`; the host's `install_grafana.sh` has no `set_ini_key`. Running
   the full installer would have required cherry-picking onto the production
   branch first.
2. **The full installer does more than Phase 1 needs** on a live host: `dnf
   install`, a datasource and dashboard re-sync from the *deploy branch's*
   `config/monitoring/`, a systemd unit overwrite, and a Grafana restart.

So only the `root_url` change was applied, using the identical verified
`set_ini_key` logic inline over SSH, after backing up the config:

```bash
sudo cp -n /etc/grafana/grafana.ini /etc/grafana/grafana.ini.bak.20260727
# set_ini_key root_url "https://${TS_FQDN}/" /etc/grafana/grafana.ini
sudo systemctl restart grafana-server
```

Resulting diff against the backup was exactly one line:

```
59c59
< ;root_url = %(protocol)s://%(domain)s:%(http_port)s/
---
> root_url = https://homeguard-ec2.tail3e202b.ts.net/
```

`install_tailscale.sh` was not run and does not need to be: `tailscale serve` is
already active on the host (`https://homeguard-ec2.tail3e202b.ts.net (tailnet
only) |-- / proxy http://localhost:3000`). Note that its `TAILSCALE_AUTH_KEY`
guard `exit 1`s before reaching the new serve block, so the patch only takes
effect on a rebuild, which is where the auth key is supplied anyway.

### 1.4 Verify (DONE)

```
GET /api/health            -> {"database":"ok","version":"12.4.3"}
GET /api/org (admin auth)  -> 200      # password survived the edit
GET /api/frontend/settings -> appUrl: https://homeguard-ec2.tail3e202b.ts.net/
grep -c '^admin_password = '           -> 1, unchanged
sudo ss -lntp                          -> 127.0.0.1:{3000,8428,3100}, all loopback
```

Fresh wrapper process:

```
startup log       -> public_url=https://homeguard-ec2.tail3e202b.ts.net
generate_deeplink -> https://homeguard-ec2.tail3e202b.ts.net/d/homeguard-portfolio
```

**Gotcha:** `mcp-grafana` fetches `public_url` **once at startup** and caches it.
An MCP server process that was running before the `root_url` change keeps
returning `http://localhost:3000` from `generate_deeplink`. Restart the client
session, not just Grafana, after changing `root_url`.

### 1.5 Remaining: reconcile the deploy branch

`main` has the idempotent `set_ini_key` installer;
`ramp-phase4-turnover-regime-research` does not. Until those converge, a rebuild
from the deploy branch reintroduces the non-idempotent `sed` and the unset
`root_url`. Cherry-pick `a7923b1` and `27ee4ff` onto the deploy branch when it is
next touched, or fold it into the eventual branch reconciliation.

---

## Phase 2 -- Grafana service account (APPLIED)

Created via the HTTP API with admin basic auth rather than the UI, so it is
reproducible:

```bash
set -a; . ./.env; set +a
FQDN=homeguard-ec2.tail3e202b.ts.net

curl -sS -u "admin:${GRAFANA_ADMIN_PASSWORD}" -H 'Content-Type: application/json' \
  -d '{"name":"mcp-grafana","role":"Viewer","isDisabled":false}' \
  "https://${FQDN}/api/serviceaccounts"
# -> id 2

curl -sS -u "admin:${GRAFANA_ADMIN_PASSWORD}" -H 'Content-Type: application/json' \
  -d '{"name":"mcp-grafana-claude"}' \
  "https://${FQDN}/api/serviceaccounts/2/tokens"
# -> {"key":"glsa_..."}  (shown once; write it straight to the env file)
```

### Why Viewer and not Editor

The upstream README suggests Editor for convenience and is explicit that it is
the less restrictive option. Viewer grants exactly what the enabled tool set
needs: `dashboards:read`, `datasources:read`, `datasources:query`,
`annotations:read`, `alert.rules:read`. It does not grant any `:write`.

### Verified

```
GET  /api/datasources            -> uid=loki (loki), uid=victoriametrics (prometheus)
GET  /api/v1/provisioning/alert-rules -> HTTP 200
GET  /api/search?type=dash-db    -> 5 dashboards in folder "Homeguard":
       homeguard-incident, homeguard-infra, homeguard-portfolio,
       homeguard-strategies, homeguard-system
POST /api/dashboards/db          -> HTTP 403   <-- the write barrier
```

The 403 is the control that proves the Viewer role. Re-run it after any token or
role change.

---

## Phase 3 -- Binary and wrapper

### 3.1 Binary (both platforms)

Use the Go toolchain. It sidesteps per-OS release-asset discovery entirely, and
both machines already have Go.

```bash
# Windows (swangpc2)
GOBIN="$HOME/bin" go install github.com/grafana/mcp-grafana/cmd/mcp-grafana@latest
#   -> C:\Users\qwqw1\bin\mcp-grafana.exe

# macOS (sws-macbook-pro)
GOBIN=/usr/local/bin go install github.com/grafana/mcp-grafana/cmd/mcp-grafana@latest
```

Installed and verified on Windows: **v0.17.2**.

Do **not** use `uvx mcp-grafana`. Claude Desktop launches from Finder and does
not inherit the shell `PATH`, which is the documented cause of
`Error: spawn mcp-grafana ENOENT`. `uvx` also adds package-resolution latency to
every cold start, which can trip Claude Code's 30-second stdio startup timeout.
Always reference an absolute path.

### 3.2 Secret file (per machine)

```bash
mkdir -p ~/.config/homeguard
cat > ~/.config/homeguard/grafana-mcp.env <<'EOF'
GRAFANA_URL=https://homeguard-ec2.tail3e202b.ts.net
GRAFANA_SERVICE_ACCOUNT_TOKEN=glsa_xxxxxxxxxxxxxxxxxxxx
EOF
chmod 600 ~/.config/homeguard/grafana-mcp.env   # macOS/Linux
```

`~/.config` is outside Dropbox and outside any git tree on both machines. On
Windows there is no `chmod`; `C:\Users\qwqw1\.config` inherits the user-profile
ACL, which already excludes other non-admin users. To tighten it explicitly:

```powershell
icacls "$env:USERPROFILE\.config\homeguard" /inheritance:r /grant:r "$env:USERNAME:(OI)(CI)F"
```

The same token works from both machines. Copy it rather than minting a second
one, so revocation stays a single action.

### 3.3 Wrapper

Both wrappers live in the repo at `infra/mcp/`, not `/usr/local/bin`, so the flag
set is version-controlled and cannot drift between machines:

- `infra/mcp/homeguard-grafana-mcp.cmd` (Windows)
- `infra/mcp/homeguard-grafana-mcp.sh` (macOS/Linux, needs `chmod +x`)

Flag rationale and the allowlist argument are in `infra/mcp/README.md`. The short
version: this uses `--enabled-tools` as an allowlist rather than the draft's
`--disable-*` denylist, because `mcp-grafana --help` now enumerates the category
strings. The denylist left `folder`, `proxied`, `snapshot`, `plugin`, `api`,
`config`, and `provisioning` enabled without saying so.

### 3.4 Smoke test the wrapper standalone

Do this before wiring any client. If it fails here, the client is not the problem.

```bash
printf '%s\n%s\n%s\n' \
 '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"probe","version":"0"}}}' \
 '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
 '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
 | <wrapper> 2>/dev/null | tail -1 | python3 -m json.tool | grep '"name"'
```

Verified on Windows: 26 tools, `serverInfo` `mcp-grafana v0.17.2`, and no write
tools present (`update_dashboard`, `create_folder`, `create_annotation` all
absent). `npx @modelcontextprotocol/inspector <wrapper>` is the interactive
equivalent.

---

## Phase 4 -- Claude Code (APPLIED on Windows)

Options precede the server name; `--` separates the name from the command.

```bash
# Windows: invoke the .cmd through cmd /c
claude mcp add --scope user --transport stdio grafana -- \
  cmd /c "C:\Users\qwqw1\Dropbox\cs\github\Homeguard\infra\mcp\homeguard-grafana-mcp.cmd"

# macOS
claude mcp add --scope user --transport stdio grafana -- \
  "$HOME/Dropbox/cs/github/Homeguard/infra/mcp/homeguard-grafana-mcp.sh"
```

**Scope**: `user`. `local` is private to one project directory; `project` writes a
committable `.mcp.json`, which must not point at production monitoring. Because
the wrapper reads credentials from the env file, `~/.claude.json` holds only a
path.

```bash
claude mcp list && claude mcp get grafana
```

Tools are discovered at session start, so a server added mid-session needs a new
session. If startup times out: `MCP_TIMEOUT=60000 claude`.

---

## Phase 5 -- Claude Desktop (NOT STARTED)

| Platform | Config path |
|---|---|
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |

```json
{
  "mcpServers": {
    "grafana": {
      "command": "cmd",
      "args": ["/c", "C:\\Users\\qwqw1\\Dropbox\\cs\\github\\Homeguard\\infra\\mcp\\homeguard-grafana-mcp.cmd"]
    }
  }
}
```

On macOS use `"command": "/Users/<you>/Dropbox/cs/github/Homeguard/infra/mcp/homeguard-grafana-mcp.sh"` with `"args": []`.

Merge into any existing `mcpServers` object rather than replacing it. Quit fully
(Cmd-Q on macOS, tray exit on Windows) and relaunch; closing the window is not
enough.

**Logs**, the first place to look on failure:

```
macOS:   ~/Library/Logs/Claude/mcp-server-grafana.log
Windows: %APPDATA%\Claude\logs\mcp-server-grafana.log
```

`claude mcp add-from-claude-desktop` imports Desktop definitions into Claude Code.
Either direction works since both point at the same wrapper.

---

## Phase 6 -- End-to-end verification

Run as prompts in a fresh session, per client. Requires the instance up.

1. `List the Grafana datasources.` -> `victoriametrics` (prometheus), `loki`.
2. `Summarize the homeguard-portfolio dashboard.` -> exercises `get_dashboard_summary`.
3. `Query datasource victoriametrics for hg_portfolio_drawdown_pct over the last 24 hours, broken out by job.` -> the Prometheus range path through the datasource proxy.
4. `Query Loki for the last 50 lines from the homeguard-ramp unit.` -> `query_loki_logs`.
5. `Try to update the portfolio overview dashboard title.` -> **must fail.**

Test 5 is not optional; it is the control proving the write barrier. Note that
with the allowlist, no write tool is even exposed, so the model should report the
capability as absent rather than getting a 403. Both outcomes pass; a success
does not.

---

## Risks

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | LLM mutates a Grafana alert rule on a live trading host | High | Viewer role (403 verified) + allowlist excludes write categories + `--disable-write`. Phase 6 test 5 |
| R2 | Live P&L, position, drawdown series egress to Anthropic as model context | Medium | Inherent to the integration and accepted by choosing it. Decide explicitly. Prefer aggregate queries over trade-log dumps |
| R3 | Token in plaintext in a client config | Medium | Wrapper + env file outside repo and Dropbox. `~/.claude.json` holds only a path |
| R4 | `tailscale serve` lost on rebuild | Low | Now codified in `install_tailscale.sh`. `--bg` persists in tailscaled state across stop/start |
| R5 | Full dashboard JSON blows the context window | Low | Prefer `get_dashboard_summary` / `get_dashboard_property` over `get_dashboard_by_uid` |
| R6 | "No data" reported for queries beyond VictoriaMetrics 90-day retention | Medium | Boundary is silent: VM returns empty, not an error. State the window explicitly |
| R7 | Same silent-empty past Loki's **30-day** retention | Medium | Same handling as R6. The draft said 14 days; actual is `720h` |
| R8 | Tool-set drift on upgrade adds unreviewed categories | Low | Allowlist, not denylist. Re-read `--help` and re-run Phase 6 test 5 after any upgrade |
| R9 | Config drift between the two clients or two machines | Low | One wrapper per platform, both in git, byte-identical flag lists |
| R10 | `root_url` loopback, so `generate_deeplink` returns unusable links | Low | **RESOLVED 2026-07-27.** `appUrl` is the tailnet FQDN and deeplinks verified. Note the startup-cache gotcha in Phase 1.4 |
| R11 | Installer resets `admin_password` to `admin` | High | **Fixed in `27ee4ff`.** Making `set_ini_key` work turned a latent no-op into a live overwrite: `GRAFANA_ADMIN_PASSWORD:-admin` would have replaced a good password with a known one on re-run. The key is now skipped when the env var is unset |
| R12 | Rebuild from the deploy branch reintroduces both defects | Medium | The fix is on `main` only; EC2 tracks `ramp-phase4-turnover-regime-research`. See Phase 1.5 |

---

## Section 7 -- Operational notes

### Instance schedule

Verified against EventBridge **Scheduler** (not Rules; `aws events list-rules`
returns nothing):

| Schedule | Expression | TZ |
|---|---|---|
| `homeguard-start-instance` | `cron(0 8 ? * MON-FRI *)` | America/New_York |
| `homeguard-stop-instance` | `cron(0 20 ? * MON-FRI *)` | America/New_York |
| `homeguard-start-instance-sunday` | `cron(0 23 ? * SAT *)` | UTC |
| `homeguard-stop-instance-sunday` | `cron(10 0 ? * SUN *)` | UTC |

So the window is 8:00 AM to 8:00 PM ET Mon-Fri, plus a ~70 minute Sat/Sun slot for
the CSCM tick. Both the draft and `INFRASTRUCTURE_OVERVIEW.md` said 9:00 AM to
4:30 PM ET with the instance idling all weekend; both were wrong and are corrected.

Practical consequence: an instance started manually **after** the 8:00 PM stop has
no schedule to stop it until the following 8:00 PM, so it runs overnight. Stop it
by hand.

The stdio server still launches when the host is down; it is a local process and
does not connect at startup. Every *tool call* in that window returns a
connection error. Expected, not a misconfiguration.

### VictoriaMetrics is Prometheus-compatible, not Prometheus

- `list_prometheus_metric_metadata` depends on `/api/v1/metadata`, which VM handles differently and may return sparse or empty results. Prefer `list_prometheus_metric_names` (backed by `/api/v1/label/__name__/values`) for `hg_*` discovery.
- Retention boundaries return empty result sets, not errors (R6).

### Upgrade discipline

Pin `mcp-grafana` alongside the Grafana pin. `install_grafana.sh` pins 12.4.3
with a documented rationale; treat the MCP binary the same way. `@latest` was
used for the initial install and resolved to v0.17.2. Re-read `--help` after any
bump, since a new default-enabled category would otherwise land silently.

---

## Rollback

```bash
claude mcp remove grafana                       # Claude Code
# Claude Desktop: remove the "grafana" key from mcpServers, quit fully, relaunch

rm -f ~/bin/mcp-grafana.exe                     # Windows binary
sudo rm -f /usr/local/bin/mcp-grafana           # macOS binary
rm -f ~/.config/homeguard/grafana-mcp.env       # credentials

# Grafana: revoke the token
curl -sS -u "admin:${GRAFANA_ADMIN_PASSWORD}" -X DELETE \
  "https://homeguard-ec2.tail3e202b.ts.net/api/serviceaccounts/2"

# EC2, only if reverting the serve change
ssh ec2-user@homeguard-ec2 'sudo tailscale serve --https=443 off'
```

Reverting `tailscale serve` restores the state in which the documented Grafana URL
does not resolve. **Keep the `root_url` and echo-block corrections regardless** of
whether the MCP integration survives; they fix a real defect independent of it.

---

## Appendix A -- Enabled tool inventory (26, verified live)

**Dashboards**: `search_dashboards`, `search_folders`, `get_dashboard_by_uid`, `get_dashboard_summary`, `get_dashboard_property`, `get_dashboard_panel_queries`
**Datasources**: `list_datasources`, `get_datasource`, `check_datasources_health`
**Prometheus**: `query_prometheus`, `query_prometheus_histogram`, `list_prometheus_metric_names`, `list_prometheus_metric_metadata`, `list_prometheus_label_names`, `list_prometheus_label_values`
**Loki**: `query_loki_logs`, `query_loki_stats`, `query_loki_patterns`, `list_loki_label_names`, `list_loki_label_values`, `analyze_loki_labels`
**Alerting**: `alerting_manage_rules`, `alerting_manage_routing` (read paths only under Viewer)
**Annotations**: `get_annotations`, `get_annotation_tags`
**Navigation**: `generate_deeplink`

## Appendix B -- Prompt patterns

- *"Using datasource `victoriametrics`, plot `hg_portfolio_drawdown_pct` by `job` over the last 7 days and tell me which strategy contributed most of the max drawdown."*
- *"Pull `homeguard-ramp` logs from Loki for the 3:55 PM ET rebalance window yesterday and identify any IBKR reconnection events."*
- *"List all Grafana-managed alert rules and their current state. Do not modify anything."*
- *"Generate a deeplink to `homeguard-incident` scoped to 2026-07-24 15:00-16:30 ET."*

Always name the datasource UID and always bound the time window. Unbounded
queries against a 90-day store consume context without adding information.

## Appendix C -- Why not a remote custom connector

Custom connectors are reached from Anthropic's cloud, not the local machine, even
in Claude Desktop. A tailnet-scoped service is unreachable by design.

Making it reachable requires `tailscale funnel`, which publishes to the open
internet. `mcp-grafana` implements no inbound authentication: the service-account
token authenticates the *server to Grafana* and does nothing to authenticate
*clients to the server*. Anyone with the `.ts.net` hostname would get
unauthenticated read access to live position, P&L, and log data.

Closing that gap needs either Claude's request-header authentication (beta,
access by request) plus a reverse proxy to validate the header, or a self-built
OAuth layer. Anthropic's published inbound IP ranges narrow the source to shared
Anthropic infrastructure, which is a layer, not a control.

Combined with the overnight shutdown, the cost/benefit does not clear. Revisit
only if mobile incident response becomes a real requirement, and only with header
auth in place.

## Appendix D -- Corrections to the external draft

| Draft claim | Actual | Impact |
|---|---|---|
| Loki retention 14 days | `720h` = **30 days** (`config/monitoring/loki/config.yaml:33`). `INFRASTRUCTURE_OVERVIEW.md` also said 14; both corrected | R7 boundary was wrong by 16 days |
| `--max-loki-log-limit` should sit "1 below Loki's `max_entries_limit_per_query`" | That key is **unset**, so the ceiling is the Loki default 5000 | Value 200 still fine; the stated reason did not hold |
| `mcp-grafana` current release v0.14.0 | **v0.17.2** | Flag surface differs |
| `--enabled-tools` category strings "not fully enumerated upstream", so use the denylist | `--help` enumerates all 21 | Adopted the allowlist the draft itself preferred |
| Denylist of 6 `--disable-*` flags is sufficient | Leaves `folder`, `proxied`, `snapshot`, `plugin`, `api`, `config`, `provisioning` enabled | Unstated surface, incl. `proxied` (external MCP servers) |
| Phase 0 unresolved, three possible outcomes | `tailscale serve` already configured, HTTPS certs already enabled | Phase 1 is codification, not a behavior change |
| `install_grafana.sh` needs `http_addr`/`http_port`/`root_url` made idempotent | `admin_password` has the **same** latent bug | Would have silently ignored a rotated password |
| Proposed escape `sed 's/[&|\\]/\\&/g'` | Wrong under this sed; needs `\\\\&` | Would have **corrupted `admin_password`** for any password containing `&` |
| `install_tailscale.sh` advertises one bad URL | **Two**: also `http://homeguard-ec2:8428/vmui` | Both corrected |
| macOS-only, `/usr/local/bin` wrapper | Primary machine is Windows; wrappers now in `infra/mcp/` under git | Cross-platform, no drift |
| EventBridge stop at 4:30 PM ET, up all weekend | Actually 8:00 PM ET Mon-Fri via EventBridge Scheduler, plus a ~70 min Sat/Sun CSCM slot | Both times wrong in the draft and in INFRASTRUCTURE_OVERVIEW.md |
