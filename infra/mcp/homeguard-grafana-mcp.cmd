@echo off
REM Windows stdio wrapper for the Grafana MCP server.
REM
REM Loads Grafana credentials from a file OUTSIDE the repo and outside any synced
REM directory, then execs mcp-grafana in stdio mode. Claude Code and Claude
REM Desktop both invoke this same wrapper so the two clients cannot drift in tool
REM exposure or credentials.
REM
REM MUST NOT write anything to stdout: stdout is the MCP JSON-RPC channel.
REM All diagnostics go to stderr.
REM
REM Docs: docs/monitoring/20260727_GRAFANA_MCP_SETUP.md

setlocal EnableExtensions

set "ENV_FILE=%USERPROFILE%\.config\homeguard\grafana-mcp.env"
if not exist "%ENV_FILE%" (
    echo homeguard-grafana-mcp: cannot read "%ENV_FILE%" 1>&2
    echo   Create it with GRAFANA_URL and GRAFANA_SERVICE_ACCOUNT_TOKEN. 1>&2
    echo   See docs/monitoring/20260727_GRAFANA_MCP_SETUP.md Phase 3.2. 1>&2
    exit /b 1
)

REM Parse KEY=VALUE lines, skipping blanks and # comments.
for /f "usebackq eol=# tokens=1,* delims==" %%A in ("%ENV_FILE%") do (
    if not "%%~A"=="" set "%%A=%%B"
)

if not defined GRAFANA_URL (
    echo homeguard-grafana-mcp: GRAFANA_URL not set in "%ENV_FILE%" 1>&2
    exit /b 1
)
if not defined GRAFANA_SERVICE_ACCOUNT_TOKEN (
    echo homeguard-grafana-mcp: GRAFANA_SERVICE_ACCOUNT_TOKEN not set in "%ENV_FILE%" 1>&2
    exit /b 1
)

if not defined MCP_GRAFANA_BIN set "MCP_GRAFANA_BIN=%USERPROFILE%\bin\mcp-grafana.exe"
if not exist "%MCP_GRAFANA_BIN%" (
    echo homeguard-grafana-mcp: binary not found at "%MCP_GRAFANA_BIN%" 1>&2
    echo   Build it: GOBIN=%%USERPROFILE%%\bin go install github.com/grafana/mcp-grafana/cmd/mcp-grafana@latest 1>&2
    exit /b 1
)

REM Flag rationale lives in infra/mcp/README.md. Keep this list byte-identical
REM to the .sh wrapper so the two machines expose the same tools.
"%MCP_GRAFANA_BIN%" ^
    --enabled-tools "search,datasource,dashboard,prometheus,loki,alerting,navigation,annotations" ^
    --disable-write ^
    --disable-rendering ^
    --max-loki-log-limit 200 ^
    --slow-request-threshold 5s ^
    --log-level info ^
    %*
exit /b %ERRORLEVEL%
