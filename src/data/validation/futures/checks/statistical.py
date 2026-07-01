"""Layer 2 statistical checks for futures data."""

from __future__ import annotations

import random
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from src.data.derivations.futures import sofr as sofr_module, yields as yields_module
from src.data.futures.paths import continuous_1min_dir
from src.data.validation.core.base import BaseCheck
from src.data.validation.core.result import Severity, ValidationResult
from src.data.validation.futures.expectations import (
    EXPECTED_DENSITY,
    EXPECTED_RANGES,
    KNOWN_GAPS,
)
from src.settings import get_local_storage_dir


def _storage_root() -> Path:
    return get_local_storage_dir()


def _now_result(name: str, severity: Severity, passed: bool,
                message: str, details: dict, t0: float) -> ValidationResult:
    return ValidationResult(
        name=name, domain="futures", layer=2,
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


def _make_density_check(symbol: str) -> type[BaseCheck]:
    """Factory: a per-symbol density check class."""

    class _DensityCheck(BaseCheck):
        name = f"futures.l2.density_{symbol}"
        domain = "futures"
        layer = 2
        _symbol = symbol

        def run(self) -> ValidationResult:
            t0 = time.time()
            df = _load_continuous(self._symbol)
            if df.is_empty():
                return _now_result(
                    self.name, Severity.CRITICAL, False,
                    f"{self._symbol}: no data", {}, t0,
                )
            n = len(df)
            n_days = df["timestamp"].dt.date().n_unique()
            rows_per_day = n / max(1, n_days)
            expected_min, expected_max = EXPECTED_DENSITY[self._symbol]
            if rows_per_day >= expected_min and rows_per_day <= expected_max:
                severity, passed = Severity.INFO, True
            elif rows_per_day < expected_min * 0.5:
                severity, passed = Severity.CRITICAL, False
            else:
                severity, passed = Severity.WARNING, False
            return _now_result(
                self.name, severity, passed,
                f"{self._symbol}: {rows_per_day:.0f} rows/day "
                f"(expected {expected_min}-{expected_max})",
                {
                    "rows_per_day": rows_per_day,
                    "rows_total": n,
                    "days": n_days,
                    "expected_min": expected_min,
                    "expected_max": expected_max,
                },
                t0,
            )

    return _DensityCheck


# Public alias used by tests: instantiate per-symbol with kwarg.
class DensityCheck(BaseCheck):
    """Direct test-friendly variant: pass symbol as constructor arg."""
    name = "futures.l2.density"
    domain = "futures"
    layer = 2
    _auto_register = False

    def __init__(self, symbol: str, mode: str = "quarterly", flags: dict | None = None):
        super().__init__(mode=mode, flags=flags)
        self.symbol = symbol

    def run(self) -> ValidationResult:
        cls = _make_density_check(self.symbol)
        # Avoid double-registration
        from src.data.validation.core.base import _registry
        if cls in _registry._checks:
            _registry._checks.remove(cls)
        return cls(mode=self.mode, flags=self.flags).run()


# Auto-register one density check per symbol in EXPECTED_DENSITY
for _sym in EXPECTED_DENSITY:
    _make_density_check(_sym)


class OhlcvInvariantsCheck(BaseCheck):
    name = "futures.l2.ohlcv_invariants"
    domain = "futures"
    layer = 2
    _auto_register = False

    def __init__(self, symbol: str = "ES", mode: str = "quarterly", flags: dict | None = None):
        super().__init__(mode=mode, flags=flags)
        self.symbol = symbol

    def run(self) -> ValidationResult:
        t0 = time.time()
        df = _load_continuous(self.symbol)
        if df.is_empty():
            return _now_result(
                self.name, Severity.WARNING, False,
                f"{self.symbol}: no data", {}, t0,
            )
        bad = df.filter(
            (pl.col("low") > pl.col("open"))
            | (pl.col("low") > pl.col("close"))
            | (pl.col("high") < pl.col("open"))
            | (pl.col("high") < pl.col("close"))
            | (pl.col("high") < pl.col("low"))
        )
        passed = bad.is_empty()
        return _now_result(
            self.name,
            Severity.CRITICAL if not passed else Severity.INFO,
            passed,
            f"{self.symbol}: {len(bad)} bars violate OHLCV invariants"
            if not passed else f"{self.symbol}: all {len(df)} bars OK",
            {"bad_count": len(bad), "total": len(df)},
            t0,
        )


# One auto-registered OHLCV invariants check per symbol
for _sym in EXPECTED_DENSITY:
    class _OhlcvInvariantsForSym(OhlcvInvariantsCheck):
        name = f"futures.l2.ohlcv_invariants_{_sym}"
        _auto_register = True
        def __init__(self, mode="quarterly", flags=None, _sym_local=_sym):
            super().__init__(symbol=_sym_local, mode=mode, flags=flags)


class DateFloorCheck(BaseCheck):
    name = "futures.l2.date_floor"
    domain = "futures"
    layer = 2

    def run(self) -> ValidationResult:
        t0 = time.time()
        # Pick a long-listed symbol (ES) to verify the dataset floor is at 2010-06-06
        df = _load_continuous("ES")
        if df.is_empty():
            return _now_result(
                self.name, Severity.WARNING, False,
                "no ES data found", {}, t0,
            )
        earliest = df["timestamp"].min()
        floor = datetime(2010, 6, 7, tzinfo=timezone.utc)  # 1-day grace
        passed = earliest <= floor
        return _now_result(
            self.name,
            Severity.WARNING if not passed else Severity.INFO,
            passed,
            f"earliest data {earliest} (expected <= 2010-06-07)",
            {"earliest": str(earliest), "expected_max_floor": str(floor)},
            t0,
        )


class LatestDataFreshnessCheck(BaseCheck):
    name = "futures.l2.latest_data_freshness"
    domain = "futures"
    layer = 2

    def run(self) -> ValidationResult:
        t0 = time.time()
        df = _load_continuous("ES")
        if df.is_empty():
            return _now_result(
                self.name, Severity.CRITICAL, False,
                "no ES data found", {}, t0,
            )
        latest = df["timestamp"].max()
        now = datetime.now(timezone.utc)
        # Stale if > 14 days
        staleness_days = (now - latest).days if latest else 999
        passed = staleness_days < 14
        return _now_result(
            self.name,
            Severity.CRITICAL if not passed else Severity.INFO,
            passed,
            f"latest ES bar {latest}, {staleness_days} days old",
            {"latest": str(latest), "staleness_days": staleness_days},
            t0,
        )


class GapDetectionCheck(BaseCheck):
    name = "futures.l2.gap_detection"
    domain = "futures"
    layer = 2
    severity_on_fail = Severity.WARNING
    _auto_register = False

    def __init__(self, symbol: str = "ES", mode: str = "quarterly", flags: dict | None = None):
        super().__init__(mode=mode, flags=flags)
        self.symbol = symbol

    def run(self) -> ValidationResult:
        t0 = time.time()
        df = _load_continuous(self.symbol)
        if df.is_empty():
            return _now_result(
                self.name, Severity.WARNING, False,
                f"{self.symbol}: no data", {}, t0,
            )
        df_daily = df.group_by(pl.col("timestamp").dt.date().alias("d")).len().sort("d")
        suspicious: list[tuple[str, int]] = []
        prev = None
        for row in df_daily.iter_rows(named=True):
            d = row["d"]
            if prev is not None:
                gap = (d - prev).days
                if gap > 5:
                    if not _is_known_gap(prev, d):
                        suspicious.append((str(prev), gap))
            prev = d
        passed = not suspicious
        return _now_result(
            self.name,
            Severity.WARNING if not passed else Severity.INFO,
            passed,
            f"{self.symbol}: {len(suspicious)} suspicious gaps"
            if suspicious else f"{self.symbol}: no suspicious gaps",
            {"suspicious": suspicious[:10]},
            t0,
        )


def _is_known_gap(start: date, end: date) -> bool:
    for s, e, _ in KNOWN_GAPS:
        if s <= start and end <= e + timedelta(days=2):
            return True
    return False


# Per-symbol gap checks
for _sym in EXPECTED_DENSITY:
    class _GapCheckForSym(GapDetectionCheck):
        name = f"futures.l2.gap_detection_{_sym}"
        _auto_register = True
        def __init__(self, mode="quarterly", flags=None, _sym_local=_sym):
            super().__init__(symbol=_sym_local, mode=mode, flags=flags)


class VolumeSanityCheck(BaseCheck):
    """At most 5% of bars should have zero volume."""
    name = "futures.l2.volume_sanity"
    domain = "futures"
    layer = 2
    severity_on_fail = Severity.WARNING
    _auto_register = False

    def __init__(self, symbol: str = "ES", mode: str = "quarterly", flags: dict | None = None):
        super().__init__(mode=mode, flags=flags)
        self.symbol = symbol

    def run(self) -> ValidationResult:
        t0 = time.time()
        df = _load_continuous(self.symbol)
        if df.is_empty():
            return _now_result(
                self.name, Severity.WARNING, False,
                f"{self.symbol}: no data", {}, t0,
            )
        zero_pct = (df["volume"] == 0).mean() or 0.0
        passed = zero_pct < 0.05
        return _now_result(
            self.name,
            Severity.WARNING if not passed else Severity.INFO,
            passed,
            f"{self.symbol}: {zero_pct:.1%} zero-volume bars",
            {"zero_pct": float(zero_pct)},
            t0,
        )


# Per-symbol volume sanity
for _sym in EXPECTED_DENSITY:
    class _VolumeSanityForSym(VolumeSanityCheck):
        name = f"futures.l2.volume_sanity_{_sym}"
        _auto_register = True
        def __init__(self, mode="quarterly", flags=None, _sym_local=_sym):
            super().__init__(symbol=_sym_local, mode=mode, flags=flags)


class SofrDerivationSanityCheck(BaseCheck):
    """4.7.1: SR1-derived SOFR within plausible 0-8% range."""
    name = "futures.l2.derived_sofr_sanity"
    domain = "futures"
    layer = 2

    def run(self) -> ValidationResult:
        t0 = time.time()
        # 30 random weekday dates from 2018-05-07 onward. Weekdays only;
        # market holidays still occur but at a low enough rate to stay
        # under the WARNING threshold (<=5 issues).
        start = sofr_module.SR1_LISTING_DATE
        end = date.today() - timedelta(days=2)
        if start >= end:
            return _now_result(
                self.name, Severity.INFO, True,
                "no eligible date range", {}, t0,
            )
        total_days = (end - start).days
        weekdays = [
            start + timedelta(days=i)
            for i in range(total_days + 1)
            if (start + timedelta(days=i)).weekday() < 5
        ]
        sample = random.sample(weekdays, min(30, len(weekdays)))
        issues: list[tuple[str, str]] = []
        for d in sample:
            try:
                rate = sofr_module.derive_sofr(d)
                if not (-0.5 <= rate <= 8.0):
                    issues.append((str(d), f"SOFR {rate:.2f}% out of plausible range"))
            except Exception as e:
                issues.append((str(d), f"derivation failed: {type(e).__name__}: {e}"))
        passed = len(issues) <= 2
        severity = Severity.CRITICAL if len(issues) > 5 else Severity.WARNING if not passed else Severity.INFO
        return _now_result(
            self.name, severity, passed,
            f"{len(issues)} issues across {len(sample)} sample dates",
            {"issues": issues[:10]},
            t0,
        )


class TreasuryYieldsSanityCheck(BaseCheck):
    """4.7.2: Micro Yield prices look like yields (0-8%), not bond prices."""
    name = "futures.l2.derived_treasury_yields_sanity"
    domain = "futures"
    layer = 2

    def run(self) -> ValidationResult:
        t0 = time.time()
        issues: list[str] = []
        for tenor in ("2Y", "5Y", "10Y", "30Y"):
            sym = yields_module.TENOR_TO_SYMBOL[tenor]
            df = _load_continuous(sym)
            if df.is_empty():
                continue
            mn, mx, avg = df["close"].min(), df["close"].max(), df["close"].mean()
            if mx > 50:
                issues.append(
                    f"{tenor} ({sym}): max close {mx:.2f} suggests bond price, not yield"
                )
            if not (0.0 <= avg <= 8.0):
                issues.append(
                    f"{tenor} ({sym}): mean close {avg:.2f} outside [0, 8] yield range"
                )
        passed = not issues
        return _now_result(
            self.name,
            Severity.CRITICAL if not passed else Severity.INFO,
            passed,
            "yields look like yields" if passed else f"{len(issues)} suspicious",
            {"issues": issues},
            t0,
        )


class _DeferredDerivationCheck(BaseCheck):
    """Stub for derivations deferred per the deferments doc."""
    _deferred_message = "deferred per docs/progress/20260509_VALIDATION_FRAMEWORK_DEFERMENTS.md"
    _auto_register = False

    def run(self) -> ValidationResult:
        t0 = time.time()
        return _now_result(
            self.name, Severity.INFO, True,
            f"derivation {self._deferred_message}",
            {"deferred": True},
            t0,
        )


class EsRealizedVolDeferredCheck(_DeferredDerivationCheck):
    name = "futures.l2.derived_es_rv_sanity"
    domain = "futures"
    layer = 2
    _auto_register = True


class CarryComputationDeferredCheck(_DeferredDerivationCheck):
    name = "futures.l2.derived_carry_sanity"
    domain = "futures"
    layer = 2
    _auto_register = True
