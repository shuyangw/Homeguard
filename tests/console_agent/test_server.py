"""Integration tests: a real ThreadingHTTPServer on an ephemeral port."""

import http.client
import json
import socket
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from src.console_agent import server as server_module
from src.console_agent.server import LOGIN_HEADER, make_server
from tests.console_agent.conftest import OPERATOR_LOGIN

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def agent_url(agent_config, fake_systemd):
    server = make_server(agent_config, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def call(url, path, login=OPERATOR_LOGIN, method="GET"):
    headers = {LOGIN_HEADER: login} if login is not None else {}
    data = b"{}" if method != "GET" else None
    request = urllib.request.Request(url + path, headers=headers, method=method, data=data)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_status_with_the_operator_login_returns_every_strategy(agent_url):
    code, body = call(agent_url, "/status")

    assert code == 200
    assert set(body["strategies"]) == {"cscm", "mp", "omr", "ramp"}
    assert body["strategies"]["mp"]["units"] == []


@pytest.mark.parametrize("login", [None, "", "intruder@example.com", OPERATOR_LOGIN.upper()])
def test_requests_without_the_operator_login_are_forbidden(agent_url, login):
    code, body = call(agent_url, "/status", login=login)

    assert code == 403
    assert "strategies" not in body


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_write_methods_are_not_allowed(agent_url, method):
    code, _ = call(agent_url, "/strategies/ramp/enabled", method=method)
    assert code == 405


def test_unknown_path_is_not_found(agent_url):
    assert call(agent_url, "/metrics")[0] == 404


def test_decisions_returns_the_full_latest_record(agent_url):
    code, body = call(agent_url, "/decisions?strategy=ramp")

    assert code == 200
    assert body["decision_id"] == "ramp-20261005-1555"
    assert body["preconditions"]["health_check"]["error"] == "Insufficient buying power"


@pytest.mark.parametrize(
    "query",
    ["", "?strategy=", "?strategy=../x", "?strategy=RAMP", "?strategy=ramp%2F..", "?strategy=%C3%A9", "?strategy=" + "x" * 17],
    ids=["no-param", "empty", "path", "upper", "encoded-slash", "non-ascii", "too-long"],
)
def test_decisions_rejects_bad_strategy_names(agent_url, query):
    assert call(agent_url, "/decisions" + query)[0] == 400


def test_decisions_for_a_name_not_in_the_toggle_is_not_found(agent_url):
    assert call(agent_url, "/decisions?strategy=zzz")[0] == 404


def test_decisions_for_a_strategy_with_no_record_is_not_found(agent_url):
    code, body = call(agent_url, "/decisions?strategy=mp")
    assert code == 404
    assert "no decision" in body["error"]


def test_agent_writes_nothing_to_the_tree(agent_config, agent_url):
    def tree_contents():
        return {
            path: path.read_bytes() for path in sorted(agent_config.repo_root.rglob("*")) if path.is_file()
        }

    before = tree_contents()
    call(agent_url, "/status")
    call(agent_url, "/decisions?strategy=ramp")
    call(agent_url, "/decisions?strategy=cscm")

    assert tree_contents() == before


def test_agent_does_not_import_the_trading_stack():
    probe = (
        "import sys\n"
        "import src.console_agent.server, src.console_agent.status, src.console_agent.units\n"
        "import src.console_agent.__main__\n"
        "heavy = sorted(m for m in sys.modules if m.startswith('src.trading')"
        " or m.split('.')[0] in ('pandas', 'numpy', 'alpaca', 'ib_async'))\n"
        "print(heavy)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=REPO_ROOT, capture_output=True, text=True, timeout=60, check=True
    )
    assert result.stdout.strip().splitlines()[-1] == "[]"


@pytest.mark.parametrize("method", ["HEAD", "OPTIONS"])
def test_head_and_options_are_not_allowed(agent_url, method):
    host, port = agent_url.removeprefix("http://").split(":")
    connection = http.client.HTTPConnection(host, int(port), timeout=5)
    connection.request(method, "/status", headers={LOGIN_HEADER: OPERATOR_LOGIN})
    response = connection.getresponse()
    connection.close()

    assert response.status == 405


def test_status_survives_non_json_yaml_values(agent_config, agent_url):
    agent_config.toggle_path.write_text("strategies:\n  ramp:\n    enabled: true\n    variant: 2026-05-23\n")

    code, body = call(agent_url, "/status")

    assert code == 200
    assert body["strategies"]["ramp"]["variant"] == "2026-05-23"


def test_decisions_with_an_unreadable_toggle_is_503_naming_the_source(agent_config, agent_url):
    agent_config.toggle_path.write_text("strategies: [unclosed\n")

    code, body = call(agent_url, "/decisions?strategy=ramp")

    assert code == 503
    assert body["source"] == "toggle"


def test_decisions_with_a_corrupt_record_is_503_naming_the_source(agent_config, agent_url):
    (agent_config.latest_dir / "ramp.json").write_text('{"schema_version": 2, "decision_id": "ramp-tru')

    code, body = call(agent_url, "/decisions?strategy=ramp")

    assert code == 503
    assert body["source"] == "decision:ramp"


def test_handler_timeout_defaults_to_fifteen_seconds():
    assert server_module.AgentHandler.timeout == 15


def test_idle_connection_is_closed_by_the_server(agent_config, fake_systemd, monkeypatch):
    monkeypatch.setattr(server_module.AgentHandler, "timeout", 0.5)
    server = make_server(agent_config, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        idle = socket.create_connection(("127.0.0.1", server.server_address[1]), timeout=5)
        assert idle.recv(1) == b""
        idle.close()
    finally:
        server.shutdown()
        server.server_close()
