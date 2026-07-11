"""Gate 0.3 committed driver for #31 single-commodity calendar-spread mean
reversion (CL, NG, ZC, ZS, ZW).

Wraps `src.strategies.advanced.spread_calendar_strategy.run_calendar`
(already gate_convergence-deflated per root, persists returns.csv +
trades.csv + gate.json under output_dir) in RunStatus, appends one registry
row per root, and writes one readiness report per root. REJECT all
pre-Gate-0 (roll-masked). NG REJECT was provisional (PBO 0.320,
volume-rank F1/F2) -- the RollCalendar-based F1/F2 caveat-fix, if attempted,
is a SEPARATE follow-up, not part of this driver. Section 11 exit
diagnostics are reported HONESTLY via
`sp_retest_common.convergence_exit_summary`: `SpreadTrade` does not persist
an `exit_reason` field, so converge/structural-stop/time-stop counts cannot
be reconstructed from trades.csv alone (a code-review finding, tracked
separately -- not fabricated here). Consult
docs/methodology/backtesting.md Sections 1, 2, 3, 4, 9, 11, 12.
"""
from __future__ import annotations

import argparse
from datetime import date

from src.strategies.advanced.spread_calendar_strategy import STORABLES, run_calendar
from src.utils.run_status import RunStatus
from scripts.backtest_scripts.sp_retest_common import (
    append_registry_row, convergence_exit_summary, write_readiness_report)

_STRATEGY_NAME = "FuturesCalendarSpread"


def main() -> None:
    parser = argparse.ArgumentParser(description="#31 calendar-spread retest driver")
    parser.add_argument("--roots", nargs="+", default=list(STORABLES))
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default=str(date.today()))
    parser.add_argument("--output-root", default="output/backtests/futures/calendar")
    parser.add_argument("--report-root", default="docs/reports/futures")
    args = parser.parse_args()

    start_d = date.fromisoformat(args.start)
    end_d = date.fromisoformat(args.end)

    for root in args.roots:
        out_dir = f"{args.output_root}/{root}"
        with RunStatus(f"sp_retest_calendar_{root}", meta={"sleeve": "calendar", "root": root}) as st:
            result = run_calendar(root, start_d, end_d, out_dir)
            st.heartbeat(note=f"gate computed: oos_sharpe={result.get('oos_sharpe')}")

        run_id = append_registry_row(f"{_STRATEGY_NAME}_{root}", result,
                                      asset_class="futures", data_frequency="daily",
                                      universe_name=root)

        exit_summary = convergence_exit_summary(f"{out_dir}/trades.csv")
        extra = (f"## Section 11 exit-logic summary (SpreadTrade exits)\n\n{exit_summary}\n\n"
                 f"## Note\n\nNG's prior REJECT was provisional (roll-masked, volume-rank F1/F2 "
                 f"contract selection); a RollCalendar-based F1/F2 caveat-fix is a separate "
                 f"follow-up, not applied by this driver run.\n")
        write_readiness_report(f"{_STRATEGY_NAME}_{root}", f"Calendar Spread {root} (#31)",
                                result, f"{args.report_root}/CALENDAR_{root}_READINESS.md",
                                extra_notes=extra, run_id=run_id)
        print(f"[sp_retest_calendar:{root}] run_id={run_id} oos_sharpe={result.get('oos_sharpe')} "
              f"dsr={result.get('dsr')} pbo={result.get('pbo')} n_trades={result.get('n_trades')}")


if __name__ == "__main__":
    main()
