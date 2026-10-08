from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from tests.console_app.test_views import DOCUMENT, RECORD, SCHEDULES
from tools.console.app import create_app
from tools.console.poller import ConsoleState

NOW = datetime(2026, 10, 8, 16, 30, tzinfo=timezone.utc)


class StubPoller:
    def __init__(self, state):
        self.state = state


def make_client(state, host="127.0.0.1:8765"):
    app = create_app(StubPoller(state), "us-east-1", clock=lambda: NOW, start_polling=False)
    return TestClient(app, base_url=f"http://{host}")


def live_state():
    return ConsoleState(document=DOCUMENT, source="agent", as_of=NOW - timedelta(seconds=4),
                        instance_state="running", schedules=SCHEDULES, decisions={"ramp": RECORD})


def test_page_renders_every_panel():
    response = make_client(live_state()).get("/")

    assert response.status_code == 200
    for heading in ("Today's schedule", "Exceptions", "Strategies", "Account", "Host"):
        assert heading in response.text
    assert "Live from agent, 4 s ago" in response.text


def test_each_panel_fragment_renders():
    client = make_client(live_state())
    for panel in ("header", "schedule", "exceptions", "strategies", "account", "host"):
        assert client.get(f"/panels/{panel}").status_code == 200


def test_unknown_panel_is_404():
    assert make_client(live_state()).get("/panels/nope").status_code == 404


def test_gates_fragment():
    client = make_client(live_state())

    assert "strategy_enabled" in client.get("/panels/gates/ramp").text
    assert client.get("/panels/gates/mp").status_code == 404


def test_empty_state_renders_no_snapshot_yet():
    response = make_client(ConsoleState()).get("/")

    assert response.status_code == 200
    assert "No snapshot yet" in response.text
    assert "No status yet" in response.text


def test_snapshot_state_renders_unknown_tiles():
    state = live_state()
    state.source, state.agent_down_since, state.reason = "s3", NOW, "shutdown"

    response = make_client(state).get("/panels/exceptions")

    assert "tile unknown" in response.text
    assert "Unknown since" in response.text


def test_foreign_host_is_rejected():
    assert make_client(live_state(), host="evil.example:8765").get("/").status_code == 400


def test_localhost_is_accepted():
    assert make_client(live_state(), host="localhost:8765").get("/").status_code == 200


def test_post_is_not_allowed():
    assert make_client(live_state()).post("/").status_code == 405
