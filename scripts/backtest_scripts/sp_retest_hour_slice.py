"""Gate 0.3 committed driver for #21 NY-Fed hour-slice (long ES+NQ, 02:00-05:00 ET).

Wraps `src.strategies.advanced.overnight_drift_strategy.run_hour_slice`
(already gate_session_stream-deflated for both 1x and 1.5x cost legs,
persists returns.csv + gate.json under output/backtests/session/hour_slice/)
in RunStatus, appends to output/experiments.duckdb, and writes the readiness
report. Was REJECT (-0.023 pre-Gate-0) -- re-gate under the honest, growing
deflation. Consult docs/methodology/backtesting.md Sections 1, 2, 3, 4, 9, 12.
"""
from __future__ import annotations

import argparse

from src.strategies.advanced.overnight_drift_strategy import run_hour_slice
from src.utils.run_status import RunStatus
from scripts.backtest_scripts.sp_retest_common import append_registry_row, write_readiness_report

_STRATEGY_NAME = "FuturesHourSlice"


def main() -> None:
    parser = argparse.ArgumentParser(description="#21 NY-Fed hour-slice retest driver")
    parser.add_argument("--roots", nargs="+", default=["ES", "NQ"])
    parser.add_argument("--report", default="docs/reports/futures/HOUR_SLICE_READINESS.md")
    args = parser.parse_args()

    with RunStatus("sp_retest_hour_slice", meta={"sleeve": "hour_slice", "roots": args.roots}) as st:
        result = run_hour_slice(roots=tuple(args.roots))
        gate_1x = result["gate_1x"]
        gate_15 = result["gate_1_5x"]
        st.heartbeat(note=f"gate computed: oos_sharpe_1x={gate_1x['oos_sharpe']:.4f}")

    run_id = append_registry_row(_STRATEGY_NAME, gate_1x, asset_class="futures", data_frequency="daily",
                                  notes=f"1.5x-cost gate: {gate_15}")

    extra = f"## 1.5x cost-sensitivity gate\n\n- oos_sharpe_1.5x: {gate_15['oos_sharpe']:.4f}\n- n_trades: {result['n_trades']}\n"
    write_readiness_report(_STRATEGY_NAME, "NY-Fed Hour-Slice (#21)", gate_1x,
                            args.report, extra_notes=extra,
                            sharpe_1_5x=gate_15["oos_sharpe"], run_id=run_id)
    print(f"[sp_retest_hour_slice] run_id={run_id} oos_sharpe_1x={gate_1x['oos_sharpe']:.4f} "
          f"oos_sharpe_1.5x={gate_15['oos_sharpe']:.4f} dsr={gate_1x['dsr']:.6f} pbo={gate_1x['pbo']}")


if __name__ == "__main__":
    main()
