"""Gate 0.3 committed driver for #39 pre-FOMC drift (long ES+NQ into the 14:00 ET statement).

Wraps `src.strategies.advanced.prefomc_strategy.run_prefomc` (persists
returns.csv + gate.json under output/backtests/session/prefomc/, plus the
pre/post-2015 decay split) in RunStatus and appends to
output/experiments.duckdb. Tier 3: ARCHITECTURALLY UNGRADEABLE by the
walk-forward gate (n_windows=0 -- ~8 events/yr never fill a 12-month/
10-sample OOS window). This driver exists for the durable record and the
decay diagnostic, NOT to force a PASS/FAIL/WEAK verdict -- do not treat its
`_verdict` output as authoritative; report INCONCLUSIVE and move on per the
strategy-lead TODO. Consult docs/methodology/backtesting.md Sections 1, 3, 9.
"""
from __future__ import annotations

import argparse

from src.strategies.advanced.prefomc_strategy import run_prefomc
from src.utils.run_status import RunStatus
from scripts.backtest_scripts.sp_retest_common import append_registry_row, write_readiness_report

_STRATEGY_NAME = "FuturesPreFomcDrift"


def main() -> None:
    parser = argparse.ArgumentParser(description="#39 pre-FOMC drift retest driver (Tier 3, ungradeable)")
    parser.add_argument("--roots", nargs="+", default=["ES", "NQ"])
    parser.add_argument("--report", default="docs/reports/futures/PREFOMC_READINESS.md")
    args = parser.parse_args()

    with RunStatus("sp_retest_prefomc", meta={"sleeve": "prefomc", "roots": args.roots}) as st:
        result = run_prefomc(roots=tuple(args.roots))
        gate = result["gate"]
        decay = result["decay"]
        st.heartbeat(note=f"gate computed: n_windows={gate.get('n_windows')} "
                          f"(0 expected -- ungradeable by design)")

    run_id = append_registry_row(_STRATEGY_NAME, gate, asset_class="futures", data_frequency="daily",
                                  notes=f"Tier 3 ungradeable (n_windows=0 expected); decay={decay}")

    extra = ("## Tier 3: architecturally ungradeable\n\n"
             "n_windows=0 -- ~8 FOMC events/year never fill a 12-month/10-sample OOS "
             "window, so PSR/DSR/PBO are NaN by construction, not a data or code defect. "
             "The decay split below is small-n descriptive evidence only, not a "
             "statistical gate.\n\n"
             "## Pre/post-2015 decay split (Ma-Zhang 2020 decay hypothesis; not a sign flip)\n\n"
             f"- pre_2015_sharpe: {decay['pre_2015_sharpe']} (n={decay['pre_n']})\n"
             f"- post_2015_sharpe: {decay['post_2015_sharpe']} (n={decay['post_n']})\n"
             f"- n_trades: {result['n_trades']}\n")
    write_readiness_report(_STRATEGY_NAME, "Pre-FOMC Drift (#39, Tier 3)", gate,
                            args.report, extra_notes=extra, run_id=run_id)
    print(f"[sp_retest_prefomc] run_id={run_id} n_windows={gate.get('n_windows')} "
          f"pre2015={decay['pre_2015_sharpe']} post2015={decay['post_2015_sharpe']}")


if __name__ == "__main__":
    main()
