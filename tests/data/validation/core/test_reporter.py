"""Tests for MarkdownReporter."""
from datetime import datetime, timezone
from pathlib import Path

from src.data.validation.core.reporter import MarkdownReporter
from src.data.validation.core.result import RunReport, Severity, ValidationResult


def _sample_report():
    now = datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc)
    return RunReport(
        domain="futures",
        results=[
            ValidationResult(
                name="futures.l1.section_presence", domain="futures", layer=1,
                severity=Severity.CRITICAL, passed=True,
                message="all sections present", details={}, duration_seconds=0.05,
                timestamp=now,
            ),
            ValidationResult(
                name="futures.l2.density_GC", domain="futures", layer=2,
                severity=Severity.CRITICAL, passed=False,
                message="GC density 7 bars/day < expected 900",
                details={"rows_per_day": 7, "expected_min": 900},
                duration_seconds=1.2, timestamp=now,
            ),
        ],
        started_at=now,
        ended_at=now,
    )


def test_writes_markdown_with_frontmatter(tmp_path: Path):
    report = _sample_report()
    out = tmp_path / "report.md"
    MarkdownReporter().write(report, out)
    text = out.read_text()
    assert text.startswith("---\n")
    assert "domain: futures" in text
    assert "critical_failures: 1" in text


def test_includes_summary_and_failed_checks(tmp_path: Path):
    report = _sample_report()
    out = tmp_path / "report.md"
    MarkdownReporter().write(report, out)
    text = out.read_text()
    assert "## Summary" in text
    assert "## Failed Checks" in text
    assert "futures.l2.density_GC" in text
    assert "rows_per_day" in text


def test_comparison_vs_previous(tmp_path: Path):
    # Previous run: density was passing. Current run: density fails (regression).
    prev_report = RunReport(
        domain="futures",
        results=[
            ValidationResult(
                name="futures.l2.density_GC", domain="futures", layer=2,
                severity=Severity.CRITICAL, passed=True,
                message="density 1100 bars/day", details={},
                duration_seconds=0.1,
                timestamp=datetime(2026, 2, 9, tzinfo=timezone.utc),
            ),
        ],
        started_at=datetime(2026, 2, 9, tzinfo=timezone.utc),
        ended_at=datetime(2026, 2, 9, tzinfo=timezone.utc),
    )
    prev_path = tmp_path / "prev.md"
    MarkdownReporter().write(prev_report, prev_path)

    cur = _sample_report()
    cur_path = tmp_path / "cur.md"
    MarkdownReporter().write(cur, cur_path, previous_path=prev_path)
    text = cur_path.read_text()
    assert "## Comparison vs Previous Run" in text
    assert "Regressions" in text
    assert "futures.l2.density_GC" in text
