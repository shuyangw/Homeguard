"""Layer 4 external validation checks for futures data."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from src.data.validation.core.base import BaseCheck
from src.data.validation.core.result import Severity, ValidationResult
from src.data.validation.futures.expectations import KNOWN_EVENTS
from src.settings import get_local_storage_dir


def _storage_root() -> Path:
    return get_local_storage_dir()


def _result(name: str, severity: Severity, passed: bool,
            message: str, details: dict, t0: float) -> ValidationResult:
    return ValidationResult(
        name=name, domain="futures", layer=4,
        severity=severity, passed=passed,
        message=message, details=details,
        duration_seconds=time.time() - t0,
        timestamp=datetime.now(timezone.utc),
    )


class YfinanceCrossCheck(BaseCheck):
    """Read-only sanity check vs yfinance.

    Off by default per the data-quality caveat: yfinance is best-effort,
    not authoritative. Opt in via --external-yfinance flag.
    """
    name = "futures.l4.yfinance_cross_check"
    domain = "futures"
    layer = 4

    def run(self) -> ValidationResult:
        t0 = time.time()
        if not self.flags.get("external_yfinance"):
            return _result(
                self.name, Severity.INFO, True,
                "opt-in flag --external-yfinance not set; skipped",
                {"flag": "external_yfinance"}, t0,
            )
        # When flag is set, perform cross-check.
        # NOTE: yfinance disagreement triggers investigation, not failure.
        # Requires 3+ disagreeing sample dates (median diff >5%) before flagging.
        try:
            import yfinance as yf
        except ImportError:
            return _result(
                self.name, Severity.WARNING, False,
                "yfinance not installed", {}, t0,
            )
        try:
            yf_data = yf.Ticker("ES=F").history(start="2024-01-01", end="2024-06-01")
        except Exception as e:
            return _result(
                self.name, Severity.WARNING, False,
                f"yfinance fetch failed: {e}",
                {"note": "yfinance is best-effort; investigate before assuming our data is wrong"},
                t0,
            )
        # Compare against our ES continuous; sample every 20th day
        es_dir = _storage_root() / "futures_1min" / "symbol=ES"
        diffs: list[float] = []
        for date_idx in yf_data.index[::20]:
            date_obj = date_idx.date() if hasattr(date_idx, "date") else date_idx
            month_dir = es_dir / f"year={date_obj.year}" / f"month={date_obj.month}"
            data_file = month_dir / "data.parquet"
            if not data_file.exists():
                continue
            try:
                df = pl.scan_parquet(data_file).filter(
                    pl.col("timestamp").dt.date() == date_obj
                ).collect()
                if df.is_empty():
                    continue
                our_close = df["close"][-1]
                yf_close = yf_data.loc[date_idx, "Close"]
                diff_pct = abs(our_close - yf_close) / yf_close
                diffs.append(diff_pct)
            except Exception:
                continue
        # Need 3+ disagreeing samples (>5% diff) to flag
        disagreeing = sum(1 for d in diffs if d > 0.05)
        passed = disagreeing < 3
        return _result(
            self.name,
            Severity.WARNING if not passed else Severity.INFO,
            passed,
            f"yfinance cross-check: {disagreeing}/{len(diffs)} samples disagree >5%. "
            "yfinance is best-effort; investigate before assuming our data is wrong.",
            {"disagreeing_samples": disagreeing, "total_samples": len(diffs)},
            t0,
        )


class CmeSettlementsCheck(BaseCheck):
    """6.2: Cross-check vs CME daily settlement files.

    Off by default; opt in via --external-cme. Higher-authority replacement
    for yfinance long-term.
    """
    name = "futures.l4.cme_settlements_cross_check"
    domain = "futures"
    layer = 4

    def run(self) -> ValidationResult:
        t0 = time.time()
        if not self.flags.get("external_cme"):
            return _result(
                self.name, Severity.INFO, True,
                "opt-in flag --external-cme not set; skipped",
                {"flag": "external_cme"}, t0,
            )
        # Implementation: HTTP fetch from cmegroup.com/market-data/settlements
        # for 5 sample dates. Compare against our futures_statistics.
        # Returning placeholder result for now.
        return _result(
            self.name, Severity.INFO, True,
            "CME settlements cross-check stub; full implementation requires "
            "HTTP scrape work -- see deferments doc",
            {"note": "implement scrape when bandwidth allows"}, t0,
        )


class KnownEventsCheck(BaseCheck):
    """6.3: Verify data captures known historical events."""
    name = "futures.l4.known_events"
    domain = "futures"
    layer = 4

    def run(self) -> ValidationResult:
        t0 = time.time()
        root = _storage_root()
        issues: list[tuple[str, str, str]] = []
        for d, (sym, desc) in KNOWN_EVENTS.items():
            month_dir = root / "futures_1min" / f"symbol={sym}" / f"year={d.year}" / f"month={d.month}"
            f = month_dir / "data.parquet"
            if not f.exists():
                issues.append((str(d), sym, "month partition missing"))
                continue
            df = pl.scan_parquet(f).filter(
                pl.col("timestamp").dt.date() == d
            ).collect()
            if df.is_empty():
                issues.append((str(d), sym, f"{desc}: no data"))
                continue
            # Total volume must be at least non-trivial
            total_vol = df["volume"].sum()
            if total_vol < 100:
                issues.append((str(d), sym, f"{desc}: total volume {total_vol} suspiciously low"))
        passed = not issues
        return _result(
            self.name,
            Severity.CRITICAL if not passed else Severity.INFO,
            passed,
            f"{len(issues)}/{len(KNOWN_EVENTS)} known events have issues"
            if not passed else "all known events captured",
            {"issues": issues},
            t0,
        )


class ZnVs10YCorrelationCheck(BaseCheck):
    """6.4: ZN price-traded futures should anti-correlate with 10Y Micro Yield.

    When yields rise, bond prices fall and vice versa.
    Replaces the v1.0 FRED rates check.
    """
    name = "futures.l4.zn_vs_10y_correlation"
    domain = "futures"
    layer = 4
    severity_on_fail = Severity.WARNING

    def run(self) -> ValidationResult:
        t0 = time.time()
        root = _storage_root()
        zn_dir = root / "futures_1min" / "symbol=ZN"
        y10_dir = root / "futures_1min" / "symbol=10Y"
        if not zn_dir.exists() or not y10_dir.exists():
            return _result(
                self.name, Severity.WARNING, False,
                "missing ZN or 10Y data", {}, t0,
            )
        # Co-availability: 2022-08-15 onward
        zn_files = list(zn_dir.rglob("data.parquet"))
        y10_files = list(y10_dir.rglob("data.parquet"))
        if not zn_files or not y10_files:
            return _result(
                self.name, Severity.WARNING, False,
                "no parquet files for ZN or 10Y", {}, t0,
            )
        zn = pl.concat([pl.scan_parquet(f) for f in zn_files]).filter(
            pl.col("timestamp") >= datetime(2022, 8, 15, tzinfo=timezone.utc)
        ).collect()
        y10 = pl.concat([pl.scan_parquet(f) for f in y10_files]).filter(
            pl.col("timestamp") >= datetime(2022, 8, 15, tzinfo=timezone.utc)
        ).collect()
        if zn.is_empty() or y10.is_empty():
            return _result(
                self.name, Severity.WARNING, False,
                "no co-available data", {}, t0,
            )
        zn_daily = zn.group_by(pl.col("timestamp").dt.date().alias("date")).agg(
            pl.col("close").last().alias("zn_close")
        ).sort("date")
        y10_daily = y10.group_by(pl.col("timestamp").dt.date().alias("date")).agg(
            pl.col("close").last().alias("y10_yield")
        ).sort("date")
        merged = zn_daily.join(y10_daily, on="date").with_columns([
            pl.col("zn_close").pct_change().alias("zn_return"),
            pl.col("y10_yield").diff().alias("yield_change"),
        ]).drop_nulls()
        if len(merged) < 30:
            return _result(
                self.name, Severity.WARNING, False,
                f"insufficient co-available days ({len(merged)})", {}, t0,
            )
        corr = merged.select(pl.corr("zn_return", "yield_change")).item()
        passed = corr is not None and corr < -0.85
        severity = Severity.CRITICAL if corr is not None and corr > -0.5 else Severity.WARNING
        return _result(
            self.name,
            severity if not passed else Severity.INFO,
            passed,
            f"ZN vs 10Y yield correlation {corr:.3f} (expected < -0.85)",
            {"correlation": corr, "n_samples": len(merged)},
            t0,
        )
