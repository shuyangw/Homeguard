"""Result types for the validation framework."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any


class Severity(StrEnum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class ValidationResult:
    name: str
    domain: str
    layer: int
    severity: Severity
    passed: bool
    message: str
    details: dict[str, Any]
    duration_seconds: float
    timestamp: datetime


@dataclass(frozen=True)
class RunReport:
    domain: str
    results: list[ValidationResult]
    started_at: datetime
    ended_at: datetime

    @property
    def critical_failures(self) -> int:
        return sum(
            1 for r in self.results
            if not r.passed and r.severity == Severity.CRITICAL
        )

    @property
    def warnings(self) -> int:
        return sum(
            1 for r in self.results
            if not r.passed and r.severity == Severity.WARNING
        )

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def has_critical_failures(self) -> bool:
        return self.critical_failures > 0
