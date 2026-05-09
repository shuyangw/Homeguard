"""Tests for ValidationRunner."""
from datetime import datetime, timezone

from src.data.validation.core.base import BaseCheck, _registry
from src.data.validation.core.result import Severity, ValidationResult
from src.data.validation.core.runner import ValidationRunner


def _make_check(name: str, domain: str, layer: int, passed: bool):
    class _C(BaseCheck):
        def run(self):
            return ValidationResult(
                name=self.name, domain=self.domain, layer=self.layer,
                severity=Severity.WARNING, passed=passed,
                message="", details={}, duration_seconds=0.01,
                timestamp=datetime.now(timezone.utc),
            )
    _C.name = name
    _C.domain = domain
    _C.layer = layer
    _registry.add(_C)
    return _C


def test_runner_executes_registered_checks():
    _registry.clear()
    _make_check("test.l1.a", "test", 1, passed=True)
    _make_check("test.l1.b", "test", 1, passed=False)

    runner = ValidationRunner(domain="test")
    report = runner.run()
    assert len(report.results) == 2
    assert report.passed_count == 1


def test_runner_filters_by_layer():
    _registry.clear()
    _make_check("test.l1", "test", 1, passed=True)
    _make_check("test.l2", "test", 2, passed=True)

    runner = ValidationRunner(domain="test", layer=2)
    report = runner.run()
    assert len(report.results) == 1
    assert report.results[0].layer == 2


def test_runner_continues_past_critical():
    _registry.clear()
    _make_check("test.bad", "test", 1, passed=False)
    _make_check("test.good", "test", 1, passed=True)

    runner = ValidationRunner(domain="test")
    report = runner.run()
    assert len(report.results) == 2  # both ran despite failure


def test_runner_handles_check_exception():
    _registry.clear()

    class BrokenCheck(BaseCheck):
        name = "test.broken"; domain = "test"; layer = 1
        def run(self):
            raise RuntimeError("oops")

    runner = ValidationRunner(domain="test")
    report = runner.run()
    assert len(report.results) == 1
    assert report.results[0].passed is False
    assert report.results[0].severity == Severity.CRITICAL
    assert "oops" in report.results[0].details.get("error", "")
