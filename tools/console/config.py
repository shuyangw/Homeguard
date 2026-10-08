"""Console settings from the repo .env and the environment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

_REQUIRED = {
    "instance_id": "EC2_INSTANCE_ID",
    "region": "EC2_REGION",
    "agent_url": "CONSOLE_AGENT_URL",
    "snapshot_bucket": "CONSOLE_SNAPSHOT_BUCKET",
}


@dataclass(frozen=True)
class Settings:
    instance_id: str
    region: str
    agent_url: str
    snapshot_bucket: str
    aws_profile: str = "homeguard-console"
    port: int = 8765


def settings_from_env(environ: Mapping[str, str]) -> Settings:
    values = {name: environ.get(variable, "").strip() for name, variable in _REQUIRED.items()}
    missing = [_REQUIRED[name] for name, value in values.items() if not value or value.startswith("<")]
    if missing:
        raise ValueError(f"missing or placeholder values in .env: {', '.join(missing)}")
    return Settings(**values, aws_profile=environ.get("CONSOLE_AWS_PROFILE", "homeguard-console"))
