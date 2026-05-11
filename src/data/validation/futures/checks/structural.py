"""Layer 1 structural checks for futures data."""

from __future__ import annotations

import random
import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from src.data.validation.core.base import BaseCheck
from src.data.validation.core.result import Severity, ValidationResult
from src.data.validation.futures.expectations import EXPECTED_SCHEMAS
from src.settings import get_local_storage_dir

EXPECTED_SECTIONS = list(EXPECTED_SCHEMAS.keys())


def _storage_root() -> Path:
    return get_local_storage_dir()


def _now_result(name: str, layer: int, severity: Severity, passed: bool,
                message: str, details: dict, t0: float) -> ValidationResult:
    return ValidationResult(
        name=name, domain="futures", layer=layer,
        severity=severity, passed=passed,
        message=message, details=details,
        duration_seconds=time.time() - t0,
        timestamp=datetime.now(timezone.utc),
    )


class SectionPresenceCheck(BaseCheck):
    name = "futures.l1.section_presence"
    domain = "futures"
    layer = 1

    def run(self) -> ValidationResult:
        t0 = time.time()
        root = _storage_root()
        missing: list[str] = []
        for section in EXPECTED_SECTIONS:
            section_dir = root / section
            if not section_dir.exists():
                missing.append(section)
                continue
            # also confirm it's non-empty
            if not any(section_dir.rglob("*.parquet")):
                missing.append(f"{section} (empty)")
        passed = not missing
        return _now_result(
            self.name, self.layer,
            Severity.CRITICAL if not passed else Severity.INFO,
            passed,
            "all sections present" if passed else f"missing/empty sections: {missing}",
            {"missing": missing, "expected": EXPECTED_SECTIONS},
            t0,
        )


class PartitionKeysValidCheck(BaseCheck):
    name = "futures.l1.partition_keys_valid"
    domain = "futures"
    layer = 1

    def run(self) -> ValidationResult:
        t0 = time.time()
        root = _storage_root()
        invalid: list[str] = []
        for section in EXPECTED_SECTIONS:
            section_dir = root / section
            if not section_dir.exists():
                continue
            for path in section_dir.rglob("*.parquet"):
                rel = path.relative_to(section_dir)
                for part in rel.parts:
                    if part.startswith("year="):
                        try:
                            year = int(part.split("=", 1)[1])
                            if year < 2010 or year > 2026:
                                invalid.append(f"{path}: year={year}")
                        except ValueError:
                            invalid.append(f"{path}: bad year part {part}")
                    if part.startswith("month="):
                        try:
                            month = int(part.split("=", 1)[1])
                            if month < 1 or month > 12:
                                invalid.append(f"{path}: month={month}")
                        except ValueError:
                            invalid.append(f"{path}: bad month part {part}")
        passed = not invalid
        return _now_result(
            self.name, self.layer,
            Severity.CRITICAL if not passed else Severity.INFO,
            passed,
            "all partition keys valid" if passed else f"{len(invalid)} invalid keys",
            {"invalid_partitions": invalid[:20]},  # cap for readability
            t0,
        )


class FileCountPerSymbolCheck(BaseCheck):
    name = "futures.l1.file_count_per_symbol"
    domain = "futures"
    layer = 1
    severity_on_fail = Severity.WARNING

    MIN_FILES_PER_SYMBOL = 100  # ~16 years x 12 months minus listing-date adjustments

    def run(self) -> ValidationResult:
        from src.data.validation.futures.expectations import LISTING_DATES
        t0 = time.time()
        root = _storage_root() / "futures_1min"
        if not root.exists():
            return _now_result(
                self.name, self.layer, Severity.CRITICAL, False,
                "futures_1min missing", {}, t0,
            )
        deficient: list[tuple[str, int]] = []
        for symbol_dir in root.iterdir():
            if not symbol_dir.name.startswith("symbol="):
                continue
            symbol = symbol_dir.name.split("=", 1)[1]
            n = sum(1 for _ in symbol_dir.rglob("data.parquet"))
            # Adjust expected min for late-listed symbols
            expected_min = self.MIN_FILES_PER_SYMBOL
            if symbol in LISTING_DATES:
                listed = LISTING_DATES[symbol]
                months_available = (2026 - listed.year) * 12 + (2 - listed.month)
                expected_min = max(50, int(months_available * 0.9))
            if n < expected_min:
                deficient.append((symbol, n))
        passed = not deficient
        return _now_result(
            self.name, self.layer,
            Severity.WARNING if not passed else Severity.INFO,
            passed,
            "file counts adequate" if passed else f"{len(deficient)} deficient symbols",
            {"deficient": deficient},
            t0,
        )


class SchemaMatchCheck(BaseCheck):
    name = "futures.l1.schema_match"
    domain = "futures"
    layer = 1

    def run(self) -> ValidationResult:
        t0 = time.time()
        root = _storage_root()
        mismatches: list[str] = []
        for section, expected_cols in EXPECTED_SCHEMAS.items():
            section_dir = root / section
            sample = next(section_dir.rglob("*.parquet"), None) if section_dir.exists() else None
            if sample is None:
                continue
            try:
                schema = pl.scan_parquet(sample).schema
            except Exception as e:
                mismatches.append(f"{section}: cannot read schema ({e})")
                continue
            for col_name, expected_dtype in expected_cols:
                if col_name not in schema:
                    mismatches.append(f"{section}: missing column {col_name}")
                else:
                    actual_dtype = str(schema[col_name])
                    if expected_dtype not in actual_dtype:
                        mismatches.append(
                            f"{section}.{col_name}: expected {expected_dtype}, got {actual_dtype}"
                        )
        passed = not mismatches
        return _now_result(
            self.name, self.layer,
            Severity.CRITICAL if not passed else Severity.INFO,
            passed,
            "all schemas match" if passed else f"{len(mismatches)} schema mismatches",
            {"mismatches": mismatches},
            t0,
        )


class FileIntegrityCheck(BaseCheck):
    name = "futures.l1.file_integrity"
    domain = "futures"
    layer = 1

    def run(self) -> ValidationResult:
        t0 = time.time()
        root = _storage_root()
        all_files: list[Path] = []
        for section in EXPECTED_SECTIONS:
            section_dir = root / section
            if section_dir.exists():
                all_files.extend(section_dir.rglob("*.parquet"))
        if not all_files:
            return _now_result(
                self.name, self.layer, Severity.CRITICAL, False,
                "no parquet files found in any section", {}, t0,
            )
        # Initial: 100% sweep. Quarterly: 5% sample.
        sample_pct = 1.0 if self.mode == "initial" else 0.05
        sample = all_files if sample_pct >= 1.0 else random.sample(
            all_files, max(1, int(len(all_files) * sample_pct)),
        )
        corrupt: list[str] = []
        for f in sample:
            try:
                pl.scan_parquet(f).select(pl.len()).collect()
            except Exception as e:
                corrupt.append(f"{f.relative_to(root)}: {type(e).__name__}: {e}")
        passed = not corrupt
        return _now_result(
            self.name, self.layer,
            Severity.CRITICAL if not passed else Severity.INFO,
            passed,
            f"all {len(sample)} sampled files OK"
            if passed else f"{len(corrupt)} corrupt files",
            {"corrupt": corrupt[:20], "sample_size": len(sample), "total": len(all_files)},
            t0,
        )
