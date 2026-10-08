import io
import json
from datetime import datetime, timedelta, timezone

import boto3
import httpx
import pytest
from botocore.response import StreamingBody
from botocore.stub import Stubber

from tools.console.config import Settings
from tools.console.poller import SNAPSHOT_KEY, AgentClient, AwsClients, Poller

SETTINGS = Settings("i-0123456789abcdef0", "us-east-1", "https://agent.test:8443", "bucket")
NOW = datetime(2026, 10, 8, 16, 0, tzinfo=timezone.utc)
LAUNCHED = datetime(2026, 10, 8, 12, 0, 30, tzinfo=timezone.utc)
STATUS = {
    "generated_at": NOW.isoformat(),
    "units": [],
    "strategies": {"ramp": {"units": ["homeguard-multi.service"], "snapshot": None,
                            "last_decision": {"decision_id": "ramp-1", "timestamp": NOW.isoformat()}}},
    "execution_lock": {"state": "free"},
    "errors": [],
}
DECISION = {"decision_id": "ramp-1", "preconditions": {"all_passed": True}}


class FakeAgent:
    def __init__(self):
        self.mode = "ok"
        self.decision_calls = 0
        self.decision_fails = False
        self.status_body = STATUS
        self.decision_body = DECISION

    def handler(self, request):
        if self.mode == "timeout":
            raise httpx.ConnectTimeout("timed out", request=request)
        if self.mode == "403":
            return httpx.Response(403, json={"error": "forbidden"})
        if request.url.path == "/status":
            return httpx.Response(200, json=self.status_body)
        self.decision_calls += 1
        if self.decision_fails:
            return httpx.Response(500, json={"error": "boom"})
        return httpx.Response(200, json=self.decision_body)


def client(name):
    return boto3.client(name, region_name="us-east-1", aws_access_key_id="x", aws_secret_access_key="x")


@pytest.fixture
def aws():
    clients = AwsClients(client("ec2"), client("scheduler"), client("s3"))
    stubs = {name: Stubber(getattr(clients, name)) for name in ("ec2", "scheduler", "s3")}
    for stub in stubs.values():
        stub.activate()
    yield clients, stubs
    for stub in stubs.values():
        stub.deactivate()


def make_poller(aws_clients, agent=None):
    agent = agent or FakeAgent()
    agent_client = AgentClient(SETTINGS.agent_url, transport=httpx.MockTransport(agent.handler))
    return Poller(SETTINGS, agent_client, aws_clients, clock=lambda: NOW), agent


def s3_body(uploaded_at, reason="shutdown", generated_at=None):
    status = {**STATUS, "generated_at": (generated_at or uploaded_at).isoformat()}
    data = json.dumps({"uploaded_at": uploaded_at.isoformat(), "reason": reason,
                       "status": status, "decisions": {"ramp": DECISION}}).encode()
    return {"Body": StreamingBody(io.BytesIO(data), len(data))}


def test_live_agent_reading_fetches_the_decision_once(aws):
    poller, agent = make_poller(aws[0])

    poller.poll_agent(NOW)
    poller.poll_agent(NOW + timedelta(seconds=10))

    assert poller.state.source == "agent"
    assert poller.state.is_live(NOW + timedelta(seconds=10)) is True
    assert poller.state.decisions["ramp"] == DECISION
    assert agent.decision_calls == 1


def test_agent_timeout_falls_back_to_s3_and_labels_the_source(aws):
    clients, stubs = aws
    poller, agent = make_poller(clients)
    agent.mode = "timeout"
    uploaded = NOW - timedelta(minutes=3)
    stubs["s3"].add_response("get_object", s3_body(uploaded), {"Bucket": "bucket", "Key": SNAPSHOT_KEY})

    poller.poll_agent(NOW)
    poller.poll_s3(NOW)

    assert poller.state.source == "s3"
    assert poller.state.is_live(NOW + timedelta(seconds=10)) is False
    assert poller.state.as_of == uploaded
    assert poller.state.reason == "shutdown"
    assert poller.state.agent_down_since == NOW
    assert "agent" in poller.state.errors


def test_agent_403_falls_back_to_s3_and_reports_the_error(aws):
    clients, stubs = aws
    poller, agent = make_poller(clients)
    agent.mode = "403"
    stubs["s3"].add_response("get_object", s3_body(NOW - timedelta(minutes=1)), {"Bucket": "bucket", "Key": SNAPSHOT_KEY})

    poller.poll_agent(NOW)
    poller.poll_s3(NOW)

    assert "403" in poller.state.errors["agent"]
    assert poller.state.source == "s3"


def test_agent_recovery_returns_to_live(aws):
    clients, stubs = aws
    poller, agent = make_poller(clients)
    agent.mode = "timeout"
    stubs["s3"].add_response("get_object", s3_body(NOW - timedelta(minutes=3)), {"Bucket": "bucket", "Key": SNAPSHOT_KEY})
    poller.poll_agent(NOW)
    poller.poll_s3(NOW)

    agent.mode = "ok"
    poller.poll_agent(NOW + timedelta(seconds=10))

    assert poller.state.source == "agent"
    assert poller.state.is_live(NOW + timedelta(seconds=10)) is True
    assert poller.state.agent_down_since is None
    assert "agent" not in poller.state.errors


def test_older_s3_snapshot_does_not_replace_a_newer_agent_reading(aws):
    clients, stubs = aws
    poller, agent = make_poller(clients)
    poller.poll_agent(NOW)
    agent.mode = "timeout"
    poller.poll_agent(NOW + timedelta(seconds=10))
    stubs["s3"].add_response("get_object", s3_body(NOW - timedelta(minutes=5)), {"Bucket": "bucket", "Key": SNAPSHOT_KEY})

    poller.poll_s3(NOW + timedelta(seconds=10))

    assert poller.state.source == "agent"
    assert poller.state.is_live(NOW + timedelta(seconds=10)) is False


def test_missing_snapshot_reads_as_no_snapshot_yet(aws):
    clients, stubs = aws
    poller, _ = make_poller(clients)
    stubs["s3"].add_client_error("get_object", service_error_code="NoSuchKey", http_status_code=404)

    poller.poll_s3(NOW)

    assert poller.state.errors["s3"] == "No snapshot yet"
    assert poller.state.document is None


def test_s3_access_denied_is_an_error_not_no_snapshot(aws):
    clients, stubs = aws
    poller, _ = make_poller(clients)
    stubs["s3"].add_client_error("get_object", service_error_code="AccessDenied", http_status_code=403)

    poller.poll_s3(NOW)

    assert "AccessDenied" in poller.state.errors["s3"]


def test_ec2_state_and_failure(aws):
    clients, stubs = aws
    poller, _ = make_poller(clients)
    stubs["ec2"].add_response(
        "describe_instances",
        {"Reservations": [{"Instances": [{"InstanceId": SETTINGS.instance_id, "LaunchTime": LAUNCHED,
                                          "State": {"Code": 80, "Name": "stopped"}}]}]},
        {"InstanceIds": [SETTINGS.instance_id]},
    )
    stubs["ec2"].add_client_error("describe_instances", service_error_code="UnauthorizedOperation")

    poller.poll_ec2(NOW)
    assert poller.state.instance_state == "stopped"
    assert poller.state.launched_at == LAUNCHED

    poller.poll_ec2(NOW)
    assert poller.state.instance_state is None
    assert poller.state.launched_at is None
    assert "UnauthorizedOperation" in poller.state.errors["ec2"]


def add_schedules(stub, disabled=()):
    real = {
        "homeguard-start-instance": ("cron(0 8 ? * MON-FRI *)", "America/New_York"),
        "homeguard-stop-instance": ("cron(0 20 ? * MON-FRI *)", "America/New_York"),
        "homeguard-start-instance-sunday": ("cron(0 23 ? * SAT *)", "UTC"),
        "homeguard-stop-instance-sunday": ("cron(10 0 ? * SUN *)", "UTC"),
    }
    for name, (expression, zone) in real.items():
        stub.add_response(
            "get_schedule",
            {"Name": name, "ScheduleExpression": expression, "ScheduleExpressionTimezone": zone,
             "State": "DISABLED" if name in disabled else "ENABLED"},
            {"Name": name},
        )


def test_schedules_are_read_and_disabled_ones_skipped(aws):
    clients, stubs = aws
    poller, _ = make_poller(clients)
    add_schedules(stubs["scheduler"], disabled={"homeguard-start-instance-sunday"})

    poller.poll_schedules(NOW)

    assert [s.name for s in poller.state.schedules] == [
        "homeguard-start-instance", "homeguard-stop-instance", "homeguard-stop-instance-sunday"]


def test_a_failed_schedule_read_keeps_the_previous_schedules(aws):
    clients, stubs = aws
    poller, _ = make_poller(clients)
    add_schedules(stubs["scheduler"])
    poller.poll_schedules(NOW)
    stubs["scheduler"].add_client_error("get_schedule", service_error_code="AccessDeniedException")

    poller.poll_schedules(NOW + timedelta(minutes=10))

    assert len(poller.state.schedules) == 4
    assert "schedule:homeguard-start-instance" in poller.state.errors


def test_tick_polls_s3_only_while_the_agent_is_down(aws):
    clients, stubs = aws
    poller, agent = make_poller(clients)
    stubs["ec2"].add_response(
        "describe_instances",
        {"Reservations": [{"Instances": [{"InstanceId": SETTINGS.instance_id, "LaunchTime": LAUNCHED,
                                          "State": {"Code": 16, "Name": "running"}}]}]},
        {"InstanceIds": [SETTINGS.instance_id]},
    )
    add_schedules(stubs["scheduler"])

    poller.tick()

    # A poll_s3 call would hit the unstubbed S3 client and record an "s3" error.
    assert "s3" not in poller.state.errors
    stubs["ec2"].assert_no_pending_responses()
    stubs["scheduler"].assert_no_pending_responses()
    assert poller.state.instance_state == "running"
    assert len(poller.state.schedules) == 4


def status_with_decision_id(decision_id):
    strategies = {"ramp": {"units": [], "snapshot": None, "last_decision": {"decision_id": decision_id}}}
    return {**STATUS, "strategies": strategies}


def test_failed_decision_fetch_is_retried_on_the_next_poll(aws):
    poller, agent = make_poller(aws[0])
    poller.poll_agent(NOW)
    newer = {"decision_id": "ramp-2"}
    agent.status_body = status_with_decision_id("ramp-2")
    agent.decision_body = newer
    agent.decision_fails = True
    poller.poll_agent(NOW + timedelta(seconds=10))
    agent.decision_fails = False

    poller.poll_agent(NOW + timedelta(seconds=20))

    assert poller.state.decisions["ramp"] == newer
    assert agent.decision_calls == 3


def test_changed_decision_id_triggers_a_refetch(aws):
    poller, agent = make_poller(aws[0])
    poller.poll_agent(NOW)
    agent.status_body = status_with_decision_id("ramp-2")

    poller.poll_agent(NOW + timedelta(seconds=10))

    assert agent.decision_calls == 2


def test_published_state_is_not_mutated_by_later_polls(aws):
    poller, agent = make_poller(aws[0])
    poller.poll_agent(NOW)
    before = poller.state
    agent.mode = "timeout"

    poller.poll_agent(NOW + timedelta(seconds=10))

    assert "agent" not in before.errors
    assert before is not poller.state


@pytest.mark.parametrize("body", [[], {"strategies": []}])
def test_unexpected_status_shape_marks_the_agent_down(aws, body):
    poller, agent = make_poller(aws[0])
    agent.status_body = body

    poller.poll_agent(NOW)

    assert poller.state.is_live(NOW + timedelta(seconds=10)) is False
    assert poller.state.agent_down_since == NOW
    assert "agent" in poller.state.errors


def agent_then_down(clients, generated_at):
    poller, agent = make_poller(clients)
    agent.status_body = {**STATUS, "generated_at": generated_at.isoformat()}
    poller.poll_agent(NOW + timedelta(seconds=30))
    agent.mode = "timeout"
    poller.poll_agent(NOW + timedelta(seconds=40))
    return poller


def test_the_shutdown_snapshot_replaces_the_last_agent_reading(aws):
    clients, stubs = aws
    poller = agent_then_down(clients, NOW + timedelta(seconds=6))
    stubs["s3"].add_response("get_object", s3_body(NOW, generated_at=NOW + timedelta(seconds=2)),
                             {"Bucket": "bucket", "Key": SNAPSHOT_KEY})

    poller.poll_s3(NOW + timedelta(seconds=40))

    assert (poller.state.source, poller.state.reason, poller.state.as_of) == ("s3", "shutdown", NOW)


def test_an_older_periodic_snapshot_keeps_the_agent_document(aws):
    clients, stubs = aws
    poller = agent_then_down(clients, NOW + timedelta(seconds=6))
    stubs["s3"].add_response("get_object", s3_body(NOW, reason="periodic", generated_at=NOW + timedelta(seconds=2)),
                             {"Bucket": "bucket", "Key": SNAPSHOT_KEY})

    poller.poll_s3(NOW + timedelta(seconds=40))

    assert poller.state.source == "agent"


def test_naive_snapshot_timestamps_are_read_as_utc(aws):
    clients, stubs = aws
    poller = agent_then_down(clients, NOW)
    naive = (NOW + timedelta(minutes=1)).replace(tzinfo=None)
    stubs["s3"].add_response("get_object", s3_body(naive, reason="periodic"), {"Bucket": "bucket", "Key": SNAPSHOT_KEY})

    poller.poll_s3(NOW + timedelta(minutes=2))

    assert poller.state.source == "s3"
    assert poller.state.as_of == NOW + timedelta(minutes=1)
