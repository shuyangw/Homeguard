"""Keeps one ConsoleState current from the agent, S3, EC2 and the Scheduler.

Each source fails on its own: a failure is logged once, recorded in
state.errors, and the other sources keep updating.
"""
from __future__ import annotations

import asyncio
import json
import functools
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
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
LIVE_MAX_AGE = 3 * AGENT_INTERVAL
S3_INTERVAL = timedelta(seconds=60)
EC2_INTERVAL = timedelta(seconds=30)
SCHEDULE_INTERVAL = timedelta(minutes=10)
SHUTDOWN_SNAPSHOT_SLACK = timedelta(seconds=60)
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
    launched_at: datetime | None = None
    schedules: list = field(default_factory=list)
    errors: dict = field(default_factory=dict)
    agent_down_since: datetime | None = None

    def is_live(self, now: datetime) -> bool:
        if self.source != "agent" or self.agent_down_since is not None or self.as_of is None:
            return False
        return now - self.as_of <= LIVE_MAX_AGE


class AgentClient:
    def __init__(self, base_url: str, transport: httpx.BaseTransport | None = None):
        self._http = httpx.Client(base_url=base_url, timeout=AGENT_TIMEOUT_SECONDS, transport=transport)

    def status(self) -> dict:
        response = self._http.get("/status")
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict) or not isinstance(body.get("strategies", {}), dict):
            raise ValueError("unexpected /status shape")
        return body

    def decision(self, strategy: str) -> dict:
        response = self._http.get("/decisions", params={"strategy": strategy})
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError("unexpected /decisions shape")
        return body


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


def _utc(text: str) -> datetime:
    moment = datetime.fromisoformat(text)
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)


def _publishes(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        finally:
            self._publish()
    return wrapper


class Poller:
    def __init__(self, settings: Settings, agent: AgentClient, aws: AwsClients, clock: Callable[[], datetime]):
        self.settings = settings
        self.agent = agent
        self.aws = aws
        self.clock = clock
        self._work = ConsoleState()
        self._publish()
        self._last_run: dict[str, datetime] = {}

    def _publish(self) -> None:
        work = self._work
        self.state = replace(work, errors=dict(work.errors), decisions=dict(work.decisions),
                             schedules=list(work.schedules))

    def _fail(self, source: str, error: object) -> None:
        if source not in self._work.errors:
            logger.warning(f"[console] {source} poll failed: {error!r}")
        self._work.errors[source] = repr(error)

    @_publishes
    def poll_agent(self, now: datetime) -> None:
        try:
            document = self.agent.status()
        except AGENT_ERRORS as e:
            self._fail("agent", e)
            self._work.agent_down_since = self._work.agent_down_since or now
            return
        self._refresh_decisions(document)
        self._work.document, self._work.source, self._work.as_of, self._work.reason = document, "agent", now, None
        self._work.agent_down_since = None
        self._work.errors.pop("agent", None)
        self._work.errors.pop("s3", None)

    def _refresh_decisions(self, document: dict) -> None:
        for name in (document.get("strategies") or {}):
            latest = _decision_id(document, name)
            if latest is None:
                self._work.decisions[name] = None
                continue
            if latest == (self._work.decisions.get(name) or {}).get("decision_id"):
                continue
            try:
                self._work.decisions[name] = self.agent.decision(name)
                self._work.errors.pop(f"decision:{name}", None)
            except AGENT_ERRORS as e:
                self._fail(f"decision:{name}", e)

    @_publishes
    def poll_s3(self, now: datetime) -> None:
        try:
            body = self.aws.s3.get_object(Bucket=self.settings.snapshot_bucket, Key=SNAPSHOT_KEY)["Body"].read()
            snapshot = json.loads(body)
            uploaded_at = _utc(snapshot["uploaded_at"])
            document = snapshot["status"]
            replaces = self._s3_replaces_current(document, snapshot.get("reason"))
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "NoSuchKey":
                self._work.errors["s3"] = "No snapshot yet"
            else:
                self._fail("s3", e)
            return
        except (BotoCoreError, ValueError, KeyError, TypeError) as e:
            self._fail("s3", e)
            return
        self._work.errors.pop("s3", None)
        if not replaces:
            return
        self._work.document, self._work.source, self._work.as_of = document, "s3", uploaded_at
        self._work.reason = snapshot.get("reason")
        self._work.decisions = snapshot.get("decisions") or {}

    def _s3_replaces_current(self, document: dict, reason: str | None) -> bool:
        # Both generated_at values come from the instance clock; the console's own clock is not comparable.
        current = self._work.document
        if current is None:
            return True
        if self._work.agent_down_since is None:
            return False
        ours, theirs = _utc(current["generated_at"]), _utc(document["generated_at"])
        return theirs > ours or (reason == "shutdown" and ours - theirs <= SHUTDOWN_SNAPSHOT_SLACK)

    @_publishes
    def poll_ec2(self, now: datetime) -> None:
        try:
            reservations = self.aws.ec2.describe_instances(InstanceIds=[self.settings.instance_id])["Reservations"]
            instance = reservations[0]["Instances"][0]
            self._work.instance_state = instance["State"]["Name"]
            self._work.launched_at = instance.get("LaunchTime")
        except AWS_ERRORS + (IndexError, KeyError) as e:
            self._fail("ec2", e)
            self._work.instance_state, self._work.launched_at = None, None
            return
        self._work.errors.pop("ec2", None)

    @_publishes
    def poll_schedules(self, now: datetime) -> None:
        schedules: list[CronSchedule] = []
        failed = False
        for name, action in SCHEDULE_ACTIONS.items():
            try:
                entry = self.aws.scheduler.get_schedule(Name=name)
                if entry["State"] == "ENABLED":
                    schedules.append(parse_cron(name, action, entry["ScheduleExpression"],
                                                entry.get("ScheduleExpressionTimezone", "UTC")))
                self._work.errors.pop(f"schedule:{name}", None)
            except AWS_ERRORS + (KeyError, ValueError) as e:
                self._fail(f"schedule:{name}", e)
                failed = True
        # A partial list would compute the wrong expected state, so keep the last full one.
        if not failed:
            self._work.schedules = schedules

    def _due(self, name: str, interval: timedelta, now: datetime) -> bool:
        last = self._last_run.get(name)
        if last is not None and now - last < interval:
            return False
        self._last_run[name] = now
        return True

    def tick(self) -> None:
        now = self.clock()
        self.poll_agent(now)
        if self._work.agent_down_since is not None and self._due("s3", S3_INTERVAL, now):
            self.poll_s3(now)
        if self._due("ec2", EC2_INTERVAL, now):
            self.poll_ec2(now)
        if self._due("schedules", SCHEDULE_INTERVAL, now):
            self.poll_schedules(now)

    async def run(self) -> None:
        while True:
            try:
                await asyncio.to_thread(self.tick)
                self._work.errors.pop("poller", None)
            except Exception as e:
                # Keep the console up and say so; one bad tick must not stop every panel.
                logger.error(f"[console] poller tick failed: {e!r}")
                self._work.errors["poller"] = repr(e)
            self._publish()
            await asyncio.sleep(AGENT_INTERVAL.total_seconds())
