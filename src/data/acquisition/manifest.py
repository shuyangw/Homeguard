"""Unified JSON manifest tracker for download progress."""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DownloadManifest:
    """Thread-safe manifest for tracking download status per symbol-month.

    Stores state at {base_dir}/_manifests/{source}.json with structure:
    {
        "source": "futures",
        "entries": {
            "ES|2024-01": {"status": "complete", "rows": 28450, "updated_at": "..."}
        }
    }
    """

    def __init__(self, base_dir: Path, source: str):
        self.source = source
        self._dir = base_dir / "_manifests"
        self._path = self._dir / f"{source}.json"
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {"source": source, "entries": {}}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                self._data = raw
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Corrupt manifest {self._path}: {e}, starting fresh")
                self._data = {"source": self.source, "entries": {}}

    def save(self) -> None:
        with self._lock:
            self._dir.mkdir(parents=True, exist_ok=True)
            tmp_path = self._path.with_suffix(".tmp")
            tmp_path.write_text(
                json.dumps(self._data, indent=2, default=str),
                encoding="utf-8",
            )
            tmp_path.replace(self._path)

    def set_entry(self, key: str, **kwargs) -> None:
        with self._lock:
            if key not in self._data["entries"]:
                self._data["entries"][key] = {}
            self._data["entries"][key].update(kwargs)
            self._data["entries"][key]["updated_at"] = (
                datetime.now(timezone.utc).isoformat()
            )

    def get_entry(self, key: str) -> Optional[dict]:
        with self._lock:
            return self._data["entries"].get(key)

    def get_all_entries(self) -> dict:
        with self._lock:
            return dict(self._data["entries"])

    def get_entries_by_status(self, status: str) -> dict:
        with self._lock:
            return {
                k: v
                for k, v in self._data["entries"].items()
                if v.get("status") == status
            }

    def summary(self) -> dict[str, int]:
        with self._lock:
            entries = self._data["entries"]
            statuses = [e.get("status", "unknown") for e in entries.values()]
            return {
                "total": len(entries),
                "complete": statuses.count("complete"),
                "failed": statuses.count("failed"),
                "pending": statuses.count("pending"),
                "in_progress": statuses.count("in_progress"),
            }
