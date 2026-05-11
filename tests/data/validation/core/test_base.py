"""Tests for BaseCheck ABC and the auto-registry."""
import pytest
from datetime import datetime, timezone

from src.data.validation.core.base import BaseCheck, _registry, register_check
from src.data.validation.core.result import Severity, ValidationResult


def test_subclass_auto_registers():
    _registry.clear()

    class MyCheck(BaseCheck):
        name = "test.layer1.dummy"
        domain = "test"
        layer = 1

        def run(self) -> ValidationResult:
            return ValidationResult(
                name=self.name, domain=self.domain, layer=self.layer,
                severity=Severity.INFO, passed=True, message="ok",
                details={}, duration_seconds=0.0,
                timestamp=datetime.now(timezone.utc),
            )

    found = _registry.get(domain="test")
    assert MyCheck in found


def test_registry_filter_by_layer():
    _registry.clear()

    class L1Check(BaseCheck):
        name = "x.l1"; domain = "x"; layer = 1
        def run(self): ...

    class L2Check(BaseCheck):
        name = "x.l2"; domain = "x"; layer = 2
        def run(self): ...

    l1 = _registry.get(domain="x", layer=1)
    assert L1Check in l1
    assert L2Check not in l1


def test_registry_filter_by_name():
    _registry.clear()

    class FooCheck(BaseCheck):
        name = "x.density_foo"; domain = "x"; layer = 2
        def run(self): ...

    class BarCheck(BaseCheck):
        name = "x.density_bar"; domain = "x"; layer = 2
        def run(self): ...

    matched = _registry.get(domain="x", name_match="foo")
    assert FooCheck in matched
    assert BarCheck not in matched


def test_register_check_decorator_explicit():
    """For checks that should not auto-register (e.g., adaptation_f)."""
    _registry.clear()

    class DeferredCheck(BaseCheck):
        name = "x.deferred"; domain = "x"; layer = 4
        _auto_register = False
        def run(self): ...

    assert DeferredCheck not in _registry.get(domain="x")
    register_check(DeferredCheck)
    assert DeferredCheck in _registry.get(domain="x")
