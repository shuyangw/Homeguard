"""Keeps one ConsoleState current from the agent, S3, EC2 and the Scheduler.

Each source fails on its own: a failure is logged once, recorded in
state.errors, and the other sources keep updating.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

import boto3
import httpx
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError

from src.utils.logger import logger
from tools.console.config import Settings
from tools.console.schedule import SCHEDULE_ACTIONS, CronSchedule, parse_cron

SNAPSHOT_KEY = "console/latest/status.json"
AGENT_TIMEOUT_SECONDS = 5.0
AGENT_INTERVAL = timedelta(seconds=10)
S3_INTERVAL = timedelta(seconds=60)
EC2_INTERVAL = timedelta(seconds=30)
SCHEDULE_INTERVAL = timedelta(minutes=10)
AWS_ERRORS = (BotoCoreError, ClientError)
AGENT_ERRORS = (httpx.HTTPError, ValueError)


@dataclass
class ConsoleState:
    document: dict | None = None
    source: str | None = None
    as_of: datetime | None = None
    reason: str | None = None
    decisions: dict = field(default_factory=dict)
    instance_state: str | None = None
    schedules: list = field(default_factory=list)
    errors: dict = field(default_factory=dict)
    agent_down_since: datetime | None = None

    @property
    def live(self) -> bool:
        return self.source == "agent" and self.agent_down_since is None


class AgentClient:
    def __init__(self, base_url: str, transport: httpx.BaseTransport | None = None):
        self._http = httpx.Client(base_url=base_url, timeout=AGENT_TIMEOUT_SECONDS, transport=transport)

    def status(self) -> dict:
        response = self._http.get("/status")
        response.raise_for_status()
        return response.json()

    def decision(self, strategy: str) -> dict:
        response = self._http.get("/decisions", params={"strategy": strategy})
        response.raise_for_status()
        return response.json()


@dataclass(frozen=True)
class AwsClients:
    ec2: object
    scheduler: object
    s3: object


def make_aws_clients(settings: Settings) -> AwsClients:
    session = boto3.Session(profile_name=settings.aws_profile, region_name=settings.region)
    config = BotoConfig(retries={"mode": "standard", "max_attempts": 3}, connect_timeout=5, read_timeout=10)
    return AwsClients(*(session.client(name, config=config) for name in ("ec2", "scheduler", "s3")))


def _decision_id(document: dict | None, strategy: str) -> str | None:
    entry = ((document or {}).get("strategies") or {}).get(strategy) or {}
    return (entry.get("last_decision") or {}).get("decision_id")


class Poller:
    def __init__(self, settings: Settings, agent: AgentClient, aws: AwsClients, clock: Callable[[], datetime]):
        self.settings = settings
        self.agent = agent
        self.aws = aws
        self.clock = clock
        self.state = ConsoleState()
        self._last_run: dict[str, datetime] = {}

    def _fail(self, source: str, error: object) -> None:
        if source not in self.state.errors:
            logger.warning(f"[console] {source} poll failed: {error!r}")
        self.state.errors[source] = repr(error)

    def poll_agent(self, now: datetime) -> None:
        try:
            document = self.agent.status()
        except AGENT_ERRORS as e:
            self._fail("agent", e)
            self.state.agent_down_since = self.state.agent_down_since or now
            return
        previous = self.state.document if self.state.source == "agent" else None
        self._refresh_decisions(document, previous)
        self.state.document, self.state.source, self.state.as_of, self.state.reason = document, "agent", now, None
        self.state.agent_down_since = None
        self.state.errors.pop("agent", None)

    def _refresh_decisions(self, document: dict, previous: dict | None) -> None:
        for name in (document.get("strategies") or {}):
            latest = _decision_id(document, name)
            if latest is None:
                self.state.decisions[name] = None
                continue
            if latest == _decision_id(previous, name) and self.state.decisions.get(name) is not None:
                continue
            try:
                self.state.decisions[name] = self.agent.decision(name)
                self.state.errors.pop(f"decision:{name}", None)
            except AGENT_ERRORS as e:
                self._fail(f"decision:{name}", e)

    def poll_s3(self, now: datetime) -> None:
        try:
            body = self.aws.s3.get_object(Bucket=self.settings.snapshot_bucket, Key=SNAPSHOT_KEY)["Body"].read()
            snapshot = json.loads(body)
            uploaded_at = datetime.fromisoformat(snapshot["uploaded_at"])
            document = snapshot["status"]
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "NoSuchKey":
                self.state.errors["s3"] = "No snapshot yet"
            else:
                self._fail("s3", e)
            return
        except (BotoCoreError, ValueError, KeyError, TypeError) as e:
            self._fail("s3", e)
            return
        self.state.errors.pop("s3", None)
        if self.state.as_of is not None and self.state.as_of >= uploaded_at:
            return
        self.state.document, self.state.source, self.state.as_of = document, "s3", uploaded_at
        self.state.reason = snapshot.get("reason")
        self.state.decisions = snapshot.get("decisions") or {}

    def poll_ec2(self, now: datetime) -> None:
        try:
            reservations = self.aws.ec2.describe_instances(InstanceIds=[self.settings.instance_id])["Reservations"]
            self.state.instance_state = reservations[0]["Instances"][0]["State"]["Name"]
        except AWS_ERRORS + (IndexError, KeyError) as e:
            self._fail("ec2", e)
            self.state.instance_state = None
            return
        self.state.errors.pop("ec2", None)

    def poll_schedules(self, now: datetime) -> None:
        schedules: list[CronSchedule] = []
        failed = False
        for name, action in SCHEDULE_ACTIONS.items():
            try:
                entry = self.aws.scheduler.get_schedule(Name=name)
                if entry["State"] == "ENABLED":
                    schedules.append(parse_cron(name, action, entry["ScheduleExpression"],
                                                entry.get("ScheduleExpressionTimezone", "UTC")))
                self.state.errors.pop(f"schedule:{name}", None)
            except AWS_ERRORS + (KeyError, ValueError) as e:
                self._fail(f"schedule:{name}", e)
                failed = True
        # A partial list would compute the wrong expected state, so keep the last full one.
        if not failed:
            self.state.schedules = schedules

    def _due(self, name: str, interval: timedelta, now: datetime) -> bool:
        last = self._last_run.get(name)
        if last is not None and now - last < interval:
            return False
        self._last_run[name] = now
        return True

    def tick(self) -> None:
        now = self.clock()
        self.poll_agent(now)
        if self.state.agent_down_since is not None and self._due("s3", S3_INTERVAL, now):
            self.poll_s3(now)
        if self._due("ec2", EC2_INTERVAL, now):
            self.poll_ec2(now)
        if self._due("schedules", SCHEDULE_INTERVAL, now):
            self.poll_schedules(now)

    async def run(self) -> None:
        while True:
            try:
                await asyncio.to_thread(self.tick)
                self.state.errors.pop("poller", None)
            except Exception as e:
                # Keep the console up and say so; one bad tick must not stop every panel.
                logger.error(f"[console] poller tick failed: {e!r}")
                self.state.errors["poller"] = repr(e)
            await asyncio.sleep(AGENT_INTERVAL.total_seconds())
