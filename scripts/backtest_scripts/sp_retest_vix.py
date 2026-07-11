"""Gate 0.3 committed driver for #26/#27 VIX roll-down (short VX1 in contango).

Wraps `src.backtesting.vix.vix_rolldown_eval.run_vix_rolldown` (already
gate_return_stream-deflated, persists returns.csv + gate.json) in RunStatus,
appends to output/experiments.duckdb, and writes the readiness report.
Already known FAIL (DSR ~8.9e-06 pre-Gate-0, PBO 0.613, max DD -81.1%) --
re-run here for the durable, honestly-deflated record plus the subperiod/tail
audit. Consult docs/methodology/backtesting.md Sections 1, 2, 4, 9, 12.
"""
from __future__ import annotations

import argparse

import polars as pl

from src.backtesting.vix.vix_rolldown_eval import rolldown_returns, run_vix_rolldown, subperiod_audit
from src.settings import get_local_storage_dir
from src.utils.run_status import RunStatus
from scripts.backtest_scripts.sp_retest_common import append_registry_row, write_readiness_report

_STRATEGY_NAME = "FuturesVixRolldown"


def main() -> None:
    parser = argparse.ArgumentParser(description="#26/#27 VIX roll-down retest driver")
    parser.add_argument("--output-dir", default="output/backtests/futures/vix_rolldown")
    parser.add_argument("--report", default="docs/reports/futures/VIX_ROLLDOWN_READINESS.md")
    args = parser.parse_args()

    curve = pl.read_parquet(
        get_local_storage_dir() / "alt_data" / "vix" / "vx_curve.parquet"
    ).to_pandas()

    with RunStatus("sp_retest_vix", meta={"sleeve": "vix_rolldown"}) as st:
        result = run_vix_rolldown(curve, args.output_dir)
        audit = subperiod_audit(rolldown_returns(curve))
        st.heartbeat(note=f"gate computed: oos_sharpe={result['oos_sharpe']:.4f}")

    run_id = append_registry_row(_STRATEGY_NAME, result, asset_class="futures", data_frequency="daily")

    extra = "## Subperiod / tail audit (reporting only, not a gate)\n\n" + "\n".join(
        f"- {k}: {v}" for k, v in audit.items()
    )
    write_readiness_report(_STRATEGY_NAME, "VIX Roll-Down (#26/#27)", result,
                            args.report, extra_notes=extra, run_id=run_id)
    print(f"[sp_retest_vix] run_id={run_id} oos_sharpe={result['oos_sharpe']:.4f} "
          f"dsr={result['dsr']:.6f} pbo={result['pbo']}")


if __name__ == "__main__":
    main()
