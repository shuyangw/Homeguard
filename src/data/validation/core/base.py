"""Base check ABC and the auto-registry."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.data.validation.core.result import Severity, ValidationResult


class _CheckRegistry:
    def __init__(self) -> None:
        self._checks: list[type[BaseCheck]] = []

    def add(self, cls: type[BaseCheck]) -> None:
        if cls not in self._checks:
            self._checks.append(cls)

    def get(
        self,
        domain: str | None = None,
        layer: int | list[int] | None = None,
        name_match: str | None = None,
    ) -> list[type[BaseCheck]]:
        results = list(self._checks)
        if domain:
            results = [c for c in results if c.domain == domain]
        if layer is not None:
            layers = layer if isinstance(layer, list) else [layer]
            results = [c for c in results if c.layer in layers]
        if name_match:
            results = [c for c in results if name_match in c.name]
        return results

    def clear(self) -> None:
        self._checks.clear()


_registry = _CheckRegistry()


class BaseCheck(ABC):
    """Abstract base for all validation checks.

    Subclasses define class-level `name`, `domain`, `layer` and implement
    `run()`. Subclasses auto-register on definition unless they set
    `_auto_register = False`.
    """

    name: str = ""
    domain: str = ""
    layer: int = 0
    severity_on_fail: Severity = Severity.WARNING
    _auto_register: bool = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if getattr(cls, "_auto_register", True):
            _registry.add(cls)

    def __init__(self, mode: str = "quarterly", flags: dict | None = None) -> None:
        self.mode = mode
        self.flags = flags or {}

    @abstractmethod
    def run(self) -> ValidationResult: ...


def register_check(cls: type[BaseCheck]) -> None:
    """Manually register a check that opted out of auto-registration."""
    _registry.add(cls)
