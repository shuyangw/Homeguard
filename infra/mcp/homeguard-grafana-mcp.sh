#!/bin/bash
# macOS / Linux stdio wrapper for the Grafana MCP server.
#
# Loads Grafana credentials from a 0600 file OUTSIDE the repo and outside any
# synced directory, then execs mcp-grafana in stdio mode. Claude Code and Claude
# Desktop both invoke this same wrapper so the two clients cannot drift in tool
# exposure or credentials.
#
# MUST NOT write anything to stdout: stdout is the MCP JSON-RPC channel.
# All diagnostics go to stderr.
#
# Docs: docs/monitoring/20260727_GRAFANA_MCP_SETUP.md
set -euo pipefail

ENV_FILE="${HOME}/.config/homeguard/grafana-mcp.env"
if [[ ! -r "${ENV_FILE}" ]]; then
    echo "homeguard-grafana-mcp: cannot read ${ENV_FILE}" >&2
    echo "  Create it with GRAFANA_URL and GRAFANA_SERVICE_ACCOUNT_TOKEN." >&2
    echo "  See docs/monitoring/20260727_GRAFANA_MCP_SETUP.md Phase 3.2." >&2
    exit 1
fi

set -a
# shellcheck disable=SC1090
. "${ENV_FILE}"
set +a

: "${GRAFANA_URL:?homeguard-grafana-mcp: GRAFANA_URL not set in ${ENV_FILE}}"
: "${GRAFANA_SERVICE_ACCOUNT_TOKEN:?homeguard-grafana-mcp: GRAFANA_SERVICE_ACCOUNT_TOKEN not set in ${ENV_FILE}}"

# Absolute path, never a bare name: Claude Desktop is launched from Finder and
# does not inherit the shell PATH, which is the documented cause of
# "Error: spawn mcp-grafana ENOENT".
MCP_GRAFANA_BIN="${MCP_GRAFANA_BIN:-/usr/local/bin/mcp-grafana}"
if [[ ! -x "${MCP_GRAFANA_BIN}" ]]; then
    echo "homeguard-grafana-mcp: binary not found or not executable at ${MCP_GRAFANA_BIN}" >&2
    echo "  Build it: GOBIN=/usr/local/bin go install github.com/grafana/mcp-grafana/cmd/mcp-grafana@latest" >&2
    exit 1
fi

# Flag rationale lives in infra/mcp/README.md. Keep this list byte-identical to
# the .cmd wrapper so the two machines expose the same tools.
exec "${MCP_GRAFANA_BIN}" \
    --enabled-tools "search,datasource,dashboard,prometheus,loki,alerting,navigation,annotations" \
    --disable-write \
    --disable-rendering \
    --max-loki-log-limit 200 \
    --slow-request-threshold 5s \
    --log-level info \
    "$@"
