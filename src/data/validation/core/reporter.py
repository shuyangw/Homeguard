"""Markdown reporter for validation runs."""

from __future__ import annotations

import re
from pathlib import Path

from src.data.validation.core.result import RunReport, Severity


class MarkdownReporter:
    """Writes a RunReport as a markdown file with YAML frontmatter.

    Frontmatter is machine-parseable so a future run can find and parse
    the previous report for comparison.
    """

    def write(
        self,
        report: RunReport,
        output_path: Path,
        previous_path: Path | None = None,
    ) -> None:
        previous = self._load_previous(previous_path) if previous_path else None
        text = self._render(report, previous)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")

    def _load_previous(self, path: Path) -> dict | None:
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8")
        # Parse names of failing/passing checks from frontmatter or content
        passing = set()
        failing = set()
        for m in re.finditer(r"^- \[(PASS|FAIL)\] `(.+?)`", text, re.MULTILINE):
            (passing if m.group(1) == "PASS" else failing).add(m.group(2))
        return {"passing": passing, "failing": failing}

    def _render(self, report: RunReport, previous: dict | None) -> str:
        lines: list[str] = []

        # Frontmatter
        lines.append("---")
        lines.append(f"domain: {report.domain}")
        lines.append(f"started_at: {report.started_at.isoformat()}")
        lines.append(f"ended_at: {report.ended_at.isoformat()}")
        lines.append(f"total_checks: {len(report.results)}")
        lines.append(f"passed: {report.passed_count}")
        lines.append(f"critical_failures: {report.critical_failures}")
        lines.append(f"warnings: {report.warnings}")
        lines.append("---")
        lines.append("")

        # Summary
        lines.append(f"# Validation Report -- {report.domain}")
        lines.append("")
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- Domain: `{report.domain}`")
        lines.append(f"- Total checks: {len(report.results)}")
        lines.append(f"- Passed: {report.passed_count}")
        lines.append(f"- CRITICAL failures: {report.critical_failures}")
        lines.append(f"- Warnings: {report.warnings}")
        duration = (report.ended_at - report.started_at).total_seconds()
        lines.append(f"- Duration: {duration:.1f}s")
        lines.append("")

        # Failed checks
        failed = [r for r in report.results if not r.passed]
        if failed:
            lines.append("## Failed Checks")
            lines.append("")
            for r in sorted(failed, key=lambda r: (r.severity != Severity.CRITICAL, r.name)):
                lines.append(f"### `{r.name}` ({r.severity.value.upper()})")
                lines.append("")
                lines.append(r.message)
                lines.append("")
                if r.details:
                    lines.append("```json")
                    import json
                    lines.append(json.dumps(r.details, indent=2, default=str))
                    lines.append("```")
                    lines.append("")

        # Comparison vs previous
        if previous is not None:
            lines.append("## Comparison vs Previous Run")
            lines.append("")
            current_failing = {r.name for r in report.results if not r.passed}
            current_passing = {r.name for r in report.results if r.passed}
            regressions = current_failing & previous["passing"]
            fixes = current_passing & previous["failing"]
            new_checks = (current_passing | current_failing) - (
                previous["passing"] | previous["failing"]
            )
            if regressions:
                lines.append("### Regressions (was passing, now failing)")
                for name in sorted(regressions):
                    lines.append(f"- `{name}`")
                lines.append("")
            if fixes:
                lines.append("### Fixes (was failing, now passing)")
                for name in sorted(fixes):
                    lines.append(f"- `{name}`")
                lines.append("")
            if new_checks:
                lines.append("### New Checks Since Previous Run")
                for name in sorted(new_checks):
                    lines.append(f"- `{name}`")
                lines.append("")

        # All check results (one-line each, machine-parseable for next comparison)
        lines.append("## All Check Results")
        lines.append("")
        for r in sorted(report.results, key=lambda r: r.name):
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"- [{status}] `{r.name}` ({r.severity.value}, {r.duration_seconds:.2f}s)")
        lines.append("")

        return "\n".join(lines)
