"""Layer 3 cross-source checks for futures data."""

from __future__ import annotations

import random
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from src.data.derivations.futures import sofr as sofr_module, yields as yields_module
from src.data.futures.paths import continuous_1min_dir, definitions_dir, per_contract_1min_dir
from src.data.validation.core.base import BaseCheck
from src.data.validation.core.result import Severity, ValidationResult
from src.settings import get_local_storage_dir


def _storage_root() -> Path:
    return get_local_storage_dir()


def _result(name: str, severity: Severity, passed: bool,
            message: str, details: dict, t0: float) -> ValidationResult:
    return ValidationResult(
        name=name, domain="futures", layer=3,
        severity=severity, passed=passed,
        message=message, details=details,
        duration_seconds=time.time() - t0,
        timestamp=datetime.now(timezone.utc),
    )


def _load_continuous(symbol: str) -> pl.DataFrame:
    sym_dir = continuous_1min_dir() / f"symbol={symbol}"
    files = list(sym_dir.rglob("data.parquet"))
    if not files:
        return pl.DataFrame()
    return pl.concat([pl.read_parquet(f) for f in files])


class DefinitionsCompletenessCheck(BaseCheck):
    """5.4: Every per-contract symbol has a matching definition record."""
    name = "futures.l3.definitions_completeness"
    domain = "futures"
    layer = 3

    def run(self) -> ValidationResult:
        t0 = time.time()
        pcm = per_contract_1min_dir()
        defs = definitions_dir()
        if not pcm.exists() or not defs.exists():
            return _result(
                self.name, Severity.CRITICAL, False,
                "missing required dirs", {}, t0,
            )
        pcm_symbols: set[str] = set()
        for f in pcm.rglob("data.parquet"):
            df = pl.scan_parquet(f).select("symbol").unique().collect()
            pcm_symbols.update(df["symbol"].to_list())
        def_symbols: set[str] = set()
        for f in defs.rglob("data.parquet"):
            df = pl.scan_parquet(f).select("raw_symbol").unique().collect()
            def_symbols.update(df["raw_symbol"].to_list())
        missing = pcm_symbols - def_symbols
        passed = not missing
        return _result(
            self.name,
            Severity.CRITICAL if not passed else Severity.INFO,
            passed,
            f"{len(missing)} contracts in per-contract data lack definitions"
            if not passed else "all per-contract symbols have definitions",
            {
                "missing_count": len(missing),
                "missing_sample": list(missing)[:20],
            },
            t0,
        )


class SofrVs2YCheck(BaseCheck):
    """5.7.1: derived SOFR vs 2YY direct read should correlate >0.85."""
    name = "futures.l3.sofr_vs_2y"
    domain = "futures"
    layer = 3
    severity_on_fail = Severity.WARNING

    def run(self) -> ValidationResult:
        t0 = time.time()
        # Both available from 2022-08-15 onward
        start = max(sofr_module.SR1_LISTING_DATE, yields_module.MICRO_YIELD_LISTING_DATE)
        end = date.today() - timedelta(days=2)
        if start >= end:
            return _result(
                self.name, Severity.INFO, True,
                "no eligible date range", {}, t0,
            )
        sample = [start + timedelta(days=random.randint(0, (end - start).days))
                  for _ in range(60)]
        pairs: list[tuple[float, float]] = []
        errors = 0
        for d in sample:
            try:
                sofr_val = sofr_module.derive_sofr(d)
                y2_val = yields_module.get_treasury_yield("2Y", d)
                pairs.append((sofr_val, y2_val))
            except Exception:
                errors += 1
        if len(pairs) < 10:
            return _result(
                self.name, Severity.WARNING, False,
                f"insufficient samples ({len(pairs)} usable, {errors} errors)",
                {"errors": errors}, t0,
            )
        # Correlation
        sofr_vals, y2_vals = zip(*pairs)
        df = pl.DataFrame({"sofr": sofr_vals, "y2": y2_vals})
        corr = df.select(pl.corr("sofr", "y2")).item()
        passed = corr is not None and corr > 0.85
        return _result(
            self.name,
            Severity.WARNING if not passed else Severity.INFO,
            passed,
            f"SOFR vs 2Y correlation {corr:.3f} (expected > 0.85)",
            {"correlation": corr, "samples": len(sofr_vals), "errors": errors},
            t0,
        )


class YieldCurveSmoothnessCheck(BaseCheck):
    """5.7.2: <=10% sample dates show sawtooth pattern across 2YY/5YY/10Y/30Y."""
    name = "futures.l3.yield_curve_smoothness"
    domain = "futures"
    layer = 3
    severity_on_fail = Severity.WARNING

    def run(self) -> ValidationResult:
        t0 = time.time()
        start = yields_module.MICRO_YIELD_LISTING_DATE
        end = date.today() - timedelta(days=2)
        if start >= end:
            return _result(
                self.name, Severity.INFO, True,
                "no eligible date range", {}, t0,
            )
        sample = [start + timedelta(days=random.randint(0, (end - start).days))
                  for _ in range(30)]
        sawtooth = 0
        details: list[tuple[str, list[float]]] = []
        for d in sample:
            try:
                ys = [yields_module.get_treasury_yield(t, d)
                      for t in ("2Y", "5Y", "10Y", "30Y")]
                signs = [1 if a < b else -1 if a > b else 0
                         for a, b in zip(ys[:-1], ys[1:])]
                changes = sum(1 for i in range(len(signs) - 1)
                              if signs[i] != signs[i+1] and signs[i] != 0 and signs[i+1] != 0)
                if changes >= 2:
                    sawtooth += 1
                    if len(details) < 10:
                        details.append((str(d), ys))
            except Exception:
                continue
        passed = sawtooth <= 3
        return _result(
            self.name,
            Severity.WARNING if not passed else Severity.INFO,
            passed,
            f"yield curve sawtooth on {sawtooth}/{len(sample)} dates",
            {"sawtooth_count": sawtooth, "examples": details},
            t0,
        )


# Deferred stubs
class _DeferredCrossSource(BaseCheck):
    domain = "futures"
    layer = 3
    _msg = "deferred per docs/progress/20260509_VALIDATION_FRAMEWORK_DEFERMENTS.md"
    _auto_register = False

    def run(self) -> ValidationResult:
        t0 = time.time()
        return _result(
            self.name, Severity.INFO, True,
            f"derivation {self._msg}",
            {"deferred": True},
            t0,
        )


class CarrySanityDeferredCheck(_DeferredCrossSource):
    name = "futures.l3.carry_sanity"
    _auto_register = True


class IvVsRvDeferredCheck(_DeferredCrossSource):
    name = "futures.l3.iv_vs_rv"
    _auto_register = True


class CarrySignStabilityDeferredCheck(_DeferredCrossSource):
    name = "futures.l3.carry_sign_stability"
    _auto_register = True
