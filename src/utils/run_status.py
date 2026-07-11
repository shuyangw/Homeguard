"""Run-status logging that survives a SIGKILL.

Long/background runs (walk-forwards, cache builds, big backtests) can be torn
down externally -- an OS kill, a harness reaping a background job, a working
tree yanked out from under a running process. A `SIGKILL`'d process cannot log
its own death, so the process's own stdout log just *stops*, leaving no reason.

`RunStatus` writes a small JSON status file per run and updates it from a
background heartbeat thread. On a clean finish it records `DONE` (with the exit
detail); on an exception it records `ERROR`. If the process is killed, the file
is frozen at the last heartbeat with `status == "RUNNING"` -- so a stale RUNNING
file plus its `heartbeat_at` timestamp tells you the run died and roughly when,
instead of leaving you to guess.

Usage:
    from src.utils.run_status import RunStatus
    with RunStatus("carver_walkforward", meta={"config": path, "jobs": jobs}) as st:
        result = do_long_work()
        st.heartbeat(note="gate computed")
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.timezone import tz
from src.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_HEARTBEAT_SECONDS = 15
_STATUS_DIR = Path("output") / "run_status"


class RunStatus:
    def __init__(self, name: str, meta: Optional[Dict[str, Any]] = None,
                 heartbeat_seconds: int = _DEFAULT_HEARTBEAT_SECONDS):
        self.name = name
        self.meta = dict(meta or {})
        self.heartbeat_seconds = heartbeat_seconds
        stamp = tz.now().strftime("%Y%m%d_%H%M%S")
        self.path = _STATUS_DIR / f"{name}_{stamp}.json"
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started_at = ""
        self._note = ""

    def _write(self, status: str, extra: Optional[Dict[str, Any]] = None) -> None:
        data: Dict[str, Any] = {
            "name": self.name,
            "pid": os.getpid(),
            "status": status,
            "started_at": self._started_at,
            "heartbeat_at": tz.now().isoformat(),
            "note": self._note,
            "meta": self.meta,
        }
        if extra:
            data.update(extra)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        # `output/` can live inside a Dropbox-synced folder; Dropbox's indexer
        # periodically opens a just-written file for read, which collides with
        # the atomic rename below (WinError 5 / PermissionError). Retry
        # briefly rather than letting a transient lock abort an otherwise
        # successful run or heartbeat -- mirrors the identical pattern already
        # used for the same failure mode in
        # src/experiments/registry.py::_connect_with_retry.
        last_exc: Optional[BaseException] = None
        for attempt in range(5):
            try:
                tmp.replace(self.path)  # atomic swap so a reader never sees a partial file
                return
            except (OSError, PermissionError) as exc:
                last_exc = exc
                if attempt == 4:
                    raise
                time.sleep(0.2 * (2 ** attempt))
        raise RuntimeError(f"unreachable: run_status write retries exhausted (last={last_exc})")

    def heartbeat(self, note: str = "") -> None:
        if note:
            self._note = note
        self._write("RUNNING")

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_seconds):
            try:
                self._write("RUNNING")
            except Exception:  # never let the heartbeat thread crash the run
                pass

    def __enter__(self) -> "RunStatus":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._started_at = tz.now().isoformat()
        self._write("RUNNING")
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()
        logger.info(f"[run_status] {self.name} STARTED (pid {os.getpid()}) -> {self.path}")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        ended = {"ended_at": tz.now().isoformat()}
        if exc_type is None:
            self._write("DONE", ended)
            logger.info(f"[run_status] {self.name} DONE -> {self.path}")
        else:
            self._write("ERROR", {**ended, "error": repr(exc)})
            logger.error(f"[run_status] {self.name} ERROR: {exc!r} -> {self.path}")
        return False  # never suppress the exception
