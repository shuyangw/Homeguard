"""Validation runner: orchestrates checks for a domain."""

from __future__ import annotations

import time
import traceback
from datetime import datetime, timezone
from typing import Literal

from src.data.validation.core.base import BaseCheck, _registry
from src.data.validation.core.result import RunReport, Severity, ValidationResult


class ValidationRunner:
    """Executes registered checks for a given domain.

    Continues past CRITICAL failures and aggregates everything into one report.
    Single-threaded; checks are I/O-bound but error-handling complexity isn't
    worth threading until a full run exceeds 5 minutes.
    """

    def __init__(
        self,
        domain: str,
        layer: int | list[int] | None = None,
        check_filter: str | None = None,
        mode: Literal["initial", "quarterly"] = "quarterly",
        flags: dict | None = None,
    ) -> None:
        self.domain = domain
        self.layer = layer
        self.check_filter = check_filter
        self.mode = mode
        self.flags = flags or {}

    def run(self) -> RunReport:
        started_at = datetime.now(timezone.utc)
        check_classes = _registry.get(
            domain=self.domain,
            layer=self.layer,
            name_match=self.check_filter,
        )
        results: list[ValidationResult] = []
        for cls in check_classes:
            results.append(self._run_one(cls))
        ended_at = datetime.now(timezone.utc)
        return RunReport(
            domain=self.domain,
            results=results,
            started_at=started_at,
            ended_at=ended_at,
        )

    def _run_one(self, cls: type[BaseCheck]) -> ValidationResult:
        t0 = time.time()
        try:
            instance = cls(mode=self.mode, flags=self.flags)
            return instance.run()
        except Exception as e:
            return ValidationResult(
                name=cls.name,
                domain=cls.domain,
                layer=cls.layer,
                severity=Severity.CRITICAL,
                passed=False,
                message=f"check raised {type(e).__name__}: {e}",
                details={"error": str(e), "traceback": traceback.format_exc()},
                duration_seconds=time.time() - t0,
                timestamp=datetime.now(timezone.utc),
            )
