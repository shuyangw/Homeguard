"""Gate 0.3 committed driver for #35 yield-curve steepener/flattener (2s10s, 2s5s, 5s30s).

Wraps `src.strategies.advanced.spread_steepener_strategy.run_steepener`
(already gate_return_stream-deflated per segment, persists returns.csv +
gate.json under output_dir) in RunStatus and appends to
output/experiments.duckdb. Tier 3: ARCHITECTURALLY UNGRADEABLE by the
walk-forward gate for all three segments (2YY history from ~2021, 3yr
z-window leaves ~1.5yr < the 48-month walk-forward minimum; 5YY degraded
~440 rows) -- n_windows=0 is expected, not a bug. This driver exists for the
durable record, NOT to force a PASS/FAIL/WEAK verdict; report INCONCLUSIVE
and note the ZT/ZN DV01 fallback as a future rebuild path once yield-future
history matures. Consult docs/methodology/backtesting.md Sections 1, 2, 3, 9.
"""
from __future__ import annotations

import argparse
from datetime import date

from src.strategies.advanced.spread_steepener_strategy import SEGMENTS, run_steepener
from src.utils.run_status import RunStatus
from scripts.backtest_scripts.sp_retest_common import append_registry_row, write_readiness_report

_STRATEGY_NAME = "FuturesYieldSteepener"


def main() -> None:
    parser = argparse.ArgumentParser(description="#35 yield-curve steepener retest driver (Tier 3, ungradeable)")
    parser.add_argument("--segments", nargs="+", default=list(SEGMENTS.keys()))
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=str(date.today()))
    parser.add_argument("--output-root", default="output/backtests/futures/steepener")
    parser.add_argument("--report-root", default="docs/reports/futures")
    args = parser.parse_args()

    start_d = date.fromisoformat(args.start)
    end_d = date.fromisoformat(args.end)

    for segment in args.segments:
        out_dir = f"{args.output_root}/{segment}"
        with RunStatus(f"sp_retest_steepener_{segment}", meta={"sleeve": "steepener", "segment": segment}) as st:
            result = run_steepener(segment, start_d, end_d, out_dir)
            st.heartbeat(note=f"gate computed: n_windows={result.get('n_windows')} "
                              f"(0 expected -- ungradeable by design)")

        run_id = append_registry_row(f"{_STRATEGY_NAME}_{segment}", result,
                                      asset_class="futures", data_frequency="daily",
                                      universe_name=segment,
                                      notes="Tier 3 ungradeable (n_windows=0 expected)")

        extra = ("## Tier 3: architecturally ungradeable\n\n"
                 "n_windows=0 expected for this segment -- yield-future history is too "
                 "short relative to the 48-month walk-forward minimum (2YY from ~2021; "
                 "5YY degraded to ~440 usable rows). PSR/DSR/PBO are NaN by construction. "
                 "A ZT/ZN DV01-based fallback is a possible future rebuild once the "
                 "yield-future history matures -- not attempted by this driver.\n")
        write_readiness_report(f"{_STRATEGY_NAME}_{segment}", f"Yield Steepener {segment} (#35, Tier 3)",
                                result, f"{args.report_root}/STEEPENER_{segment}_READINESS.md",
                                extra_notes=extra, run_id=run_id)
        print(f"[sp_retest_steepener:{segment}] run_id={run_id} n_windows={result.get('n_windows')} "
              f"oos_sharpe={result.get('oos_sharpe')}")


if __name__ == "__main__":
    main()
