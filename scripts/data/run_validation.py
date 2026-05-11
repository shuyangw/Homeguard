"""Run the data validation framework.

Examples:
    python scripts/data/run_validation.py --domain futures
    python scripts/data/run_validation.py --domain futures --layer 1
    python scripts/data/run_validation.py --domain futures --mode initial
    python scripts/data/run_validation.py --domain futures --check density_GC
    python scripts/data/run_validation.py --domain futures --external-yfinance
    python scripts/data/run_validation.py --domain futures --adaptation-f
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.validation.core.runner import ValidationRunner
from src.data.validation.core.reporter import MarkdownReporter
from src.data.validation.core.base import register_check
from src.utils.logger import get_logger

logger = get_logger(__name__)

VALID_DOMAINS = {"futures"}  # equities/crypto/fx/options to be added when implemented


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run data validation for a given domain.",
    )
    parser.add_argument("--domain", required=True, choices=sorted(VALID_DOMAINS))
    parser.add_argument("--layer", action="append", type=int, default=None,
                        help="Filter to layer(s); can be repeated")
    parser.add_argument("--check", default=None,
                        help="Substring filter on check name")
    parser.add_argument("--mode", default="quarterly",
                        choices=["initial", "quarterly"])
    parser.add_argument("--external-yfinance", action="store_true",
                        help="Opt-in for yfinance cross-check (best-effort)")
    parser.add_argument("--external-cme", action="store_true",
                        help="Opt-in for CME settlement scrape")
    parser.add_argument("--adaptation-f", action="store_true",
                        help="Register Adaptation F gating checks")
    parser.add_argument("--report-out", type=Path, default=None)
    parser.add_argument("--compare-to", type=Path, default=None,
                        help="Specific previous report path for regression diff")
    parser.add_argument("--skip-derivation", action="store_true",
                        help="Bypass derived-signal checks (when SR1/Micro Yield missing)")
    args = parser.parse_args()

    # Import the domain's checks (which auto-register)
    if args.domain == "futures":
        import src.data.validation.futures.checks  # noqa: F401
        if args.adaptation_f:
            from src.data.validation.futures.checks import adaptation_f
            register_check(adaptation_f.ChainDensityCheck)
            register_check(adaptation_f.IvRankComputabilityCheck)
            register_check(adaptation_f.IvSmileConsistencyCheck)

    flags = {
        "external_yfinance": args.external_yfinance,
        "external_cme": args.external_cme,
        "adaptation_f": args.adaptation_f,
        "skip_derivation": args.skip_derivation,
    }

    layer = args.layer if args.layer else None
    runner = ValidationRunner(
        domain=args.domain,
        layer=layer,
        check_filter=args.check,
        mode=args.mode,
        flags=flags,
    )
    logger.info(f"Running validation: domain={args.domain} mode={args.mode} layer={layer}")
    report = runner.run()

    # Determine report destination
    out = args.report_out
    if out is None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")
        out = PROJECT_ROOT / "docs" / "data" / "validation_reports" / f"{ts}_{args.domain}.md"

    # Auto-discover previous report for comparison if --compare-to not given
    previous = args.compare_to
    if previous is None:
        reports_dir = (PROJECT_ROOT / "docs" / "data" / "validation_reports")
        if reports_dir.exists():
            candidates = sorted([
                p for p in reports_dir.glob(f"*_{args.domain}.md")
                if p != out
            ])
            previous = candidates[-1] if candidates else None

    MarkdownReporter().write(report, out, previous_path=previous)
    logger.info(f"Wrote report to {out}")

    print(f"\nValidation Report Summary")
    print(f"  Domain:            {args.domain}")
    print(f"  Total checks:      {len(report.results)}")
    print(f"  Passed:            {report.passed_count}")
    print(f"  CRITICAL failures: {report.critical_failures}")
    print(f"  Warnings:          {report.warnings}")
    print(f"  Report:            {out}")

    return 1 if report.has_critical_failures else 0


if __name__ == "__main__":
    sys.exit(main())
