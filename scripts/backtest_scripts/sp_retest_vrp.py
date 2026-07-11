"""Gate 0.3 committed driver for #28 VRP-sized short-VX1 (vol risk premium).

Wraps `src.backtesting.vol.vrp_strategy.run_vrp` (already gate_return_stream-
deflated, persists returns.csv + gate.json, includes the #26 re-expression
check) in RunStatus, appends to output/experiments.duckdb, and writes the
readiness report. Already known FAIL + re-expression of #26 (corr 0.479,
marginal Sharpe 0.015) -- re-run for the durable record.
Consult docs/methodology/backtesting.md Sections 1, 2, 4, 9, 12.
"""
from __future__ import annotations

import argparse
from datetime import date

from src.backtesting.vol.vrp_strategy import run_vrp
from src.utils.run_status import RunStatus
from scripts.backtest_scripts.sp_retest_common import append_registry_row, write_readiness_report

_STRATEGY_NAME = "FuturesVrpShortVx1"


def main() -> None:
    parser = argparse.ArgumentParser(description="#28 VRP short-VX1 retest driver")
    parser.add_argument("--root", default="ES", help="Root whose ATM IV/HAR-RV drives the VRP signal")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=str(date.today()))
    parser.add_argument("--output-dir", default="output/backtests/futures/vrp")
    parser.add_argument("--report", default="docs/reports/futures/VRP_READINESS.md")
    args = parser.parse_args()

    start_d = date.fromisoformat(args.start)
    end_d = date.fromisoformat(args.end)

    with RunStatus("sp_retest_vrp", meta={"sleeve": "vrp", "root": args.root}) as st:
        result = run_vrp(args.root, start_d, end_d, args.output_dir)
        st.heartbeat(note=f"gate computed: oos_sharpe={result['oos_sharpe']:.4f}")

    run_id = append_registry_row(_STRATEGY_NAME, result, asset_class="futures", data_frequency="daily")

    extra = ("## Re-expression check vs #26 VIX roll-down\n\n"
             f"- corr: {result.get('reexpr_corr')}\n"
             f"- marginal_sharpe (residual after regressing on #26): {result.get('reexpr_marginal_sharpe')}\n")
    write_readiness_report(_STRATEGY_NAME, "VRP Short-VX1 (#28)", result,
                            args.report, extra_notes=extra, run_id=run_id)
    print(f"[sp_retest_vrp] run_id={run_id} oos_sharpe={result['oos_sharpe']:.4f} "
          f"dsr={result['dsr']:.6f} pbo={result['pbo']}")


if __name__ == "__main__":
    main()
