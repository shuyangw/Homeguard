"""Entry point: python -m src.console_agent (run from the repo root)."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from src.console_agent.server import make_server
from src.console_agent.status import config_from_env
from src.settings import get_local_storage_dir
from src.utils.logger import logger

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    snapshot_dir = Path(get_local_storage_dir()) / "metrics_snapshots"
    try:
        config = config_from_env(os.environ, REPO_ROOT, snapshot_dir)
    except ValueError as e:
        logger.error(f"[console-agent] refusing to start: {e}")
        raise SystemExit(1)
    server = make_server(config)
    host, port = server.server_address[:2]
    logger.info(f"[console-agent] serving {REPO_ROOT} on {host}:{port}, snapshots from {snapshot_dir}")
    server.serve_forever()


if __name__ == "__main__":
    main()
