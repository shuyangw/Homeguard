"""Entry point: python -m tools.console (run from the repo root)."""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from botocore.exceptions import ProfileNotFound
from dotenv import load_dotenv

from src.utils.logger import logger
from tools.console.app import create_app
from tools.console.config import settings_from_env
from tools.console.poller import AgentClient, Poller, make_aws_clients

REPO_ROOT = Path(__file__).resolve().parents[2]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _redirect_missing_stdio() -> None:
    if sys.stdout is not None and sys.stderr is not None:
        return
    log_dir = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "Homeguard"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(log_dir / "console.log", "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = log_file


def main() -> None:
    _redirect_missing_stdio()
    load_dotenv(REPO_ROOT / ".env")
    try:
        settings = settings_from_env(os.environ)
        aws = make_aws_clients(settings)
    except (ValueError, ProfileNotFound) as e:
        logger.error(f"[console] refusing to start: {e}")
        raise SystemExit(1)
    poller = Poller(settings, AgentClient(settings.agent_url), aws, utc_now)
    logger.info(f"[console] serving on http://127.0.0.1:{settings.port}")
    uvicorn.run(create_app(poller, settings.region, utc_now), host="127.0.0.1", port=settings.port, log_level="warning")


if __name__ == "__main__":
    main()
