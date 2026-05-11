"""Adaptation F (SPX Iron Condor extension) gating checks.

This module is NOT auto-imported by `futures.checks.__init__`. Load explicitly
via `register_check` from the CLI when --adaptation-f flag is set.
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.data.validation.core.base import BaseCheck
from src.data.validation.core.result import Severity, ValidationResult


class _AdaptationFCheck(BaseCheck):
    domain = "futures"
    layer = 4
    _auto_register = False  # opt-in via --adaptation-f


class ChainDensityCheck(_AdaptationFCheck):
    name = "futures.l4.adaptation_f.chain_density"

    def run(self) -> ValidationResult:
        # Implementation deferred for the strategy build phase.
        # Stub returns INFO so it doesn't fail when flag is set without strategy ready.
        return ValidationResult(
            name=self.name, domain=self.domain, layer=self.layer,
            severity=Severity.INFO, passed=True,
            message="Adaptation F chain density check -- implementation pending strategy integration",
            details={"adaptation": "F", "stub": True},
            duration_seconds=0.0,
            timestamp=datetime.now(timezone.utc),
        )


class IvRankComputabilityCheck(_AdaptationFCheck):
    name = "futures.l4.adaptation_f.iv_rank_computability"

    def run(self) -> ValidationResult:
        return ValidationResult(
            name=self.name, domain=self.domain, layer=self.layer,
            severity=Severity.INFO, passed=True,
            message="Adaptation F IV rank computability check -- implementation pending",
            details={"adaptation": "F", "stub": True},
            duration_seconds=0.0,
            timestamp=datetime.now(timezone.utc),
        )


class IvSmileConsistencyCheck(_AdaptationFCheck):
    name = "futures.l4.adaptation_f.iv_smile_consistency"

    def run(self) -> ValidationResult:
        return ValidationResult(
            name=self.name, domain=self.domain, layer=self.layer,
            severity=Severity.INFO, passed=True,
            message="Adaptation F IV smile consistency check -- implementation pending",
            details={"adaptation": "F", "stub": True},
            duration_seconds=0.0,
            timestamp=datetime.now(timezone.utc),
        )
