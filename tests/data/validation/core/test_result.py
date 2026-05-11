"""Tests for ValidationResult, Severity, and RunReport."""
from datetime import datetime, timezone
from src.data.validation.core.result import (
    Severity,
    ValidationResult,
    RunReport,
)


def test_severity_values():
    assert Severity.CRITICAL.value == "critical"
    assert Severity.WARNING.value == "warning"
    assert Severity.INFO.value == "info"


def test_validation_result_construction():
    r = ValidationResult(
        name="futures.statistical.density_GC",
        domain="futures",
        layer=2,
        severity=Severity.CRITICAL,
        passed=False,
        message="density 7 bars/day < expected 900",
        details={"rows_per_day": 7, "expected_min": 900},
        duration_seconds=0.42,
        timestamp=datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc),
    )
    assert r.passed is False
    assert r.severity == Severity.CRITICAL
    assert r.details["rows_per_day"] == 7


def test_validation_result_is_frozen():
    r = ValidationResult(
        name="x", domain="d", layer=1, severity=Severity.INFO,
        passed=True, message="", details={},
        duration_seconds=0.0,
        timestamp=datetime.now(timezone.utc),
    )
    import dataclasses
    with __import__("pytest").raises(dataclasses.FrozenInstanceError):
        r.passed = False  # type: ignore[misc]


def test_run_report_summary():
    now = datetime.now(timezone.utc)
    results = [
        ValidationResult(
            name="a", domain="futures", layer=1, severity=Severity.CRITICAL,
            passed=False, message="", details={}, duration_seconds=0.1, timestamp=now,
        ),
        ValidationResult(
            name="b", domain="futures", layer=1, severity=Severity.WARNING,
            passed=False, message="", details={}, duration_seconds=0.1, timestamp=now,
        ),
        ValidationResult(
            name="c", domain="futures", layer=1, severity=Severity.INFO,
            passed=True, message="", details={}, duration_seconds=0.1, timestamp=now,
        ),
    ]
    report = RunReport(domain="futures", results=results, started_at=now, ended_at=now)
    assert report.critical_failures == 1
    assert report.warnings == 1
    assert report.passed_count == 1
    assert report.has_critical_failures is True
