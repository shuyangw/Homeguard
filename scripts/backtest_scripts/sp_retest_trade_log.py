"""Gate 0.5 trade-log persistence driver for Tier 1 carver strategies.

The Tier 1 walk-forward gate (`run_carver_walkforward.py`) runs many small
per-window backtests with `register=False` and no trade-log persistence (a
validation-harness internal, per strategy-pipeline rules -- that's allowed to
suppress logging). But methodology Section 12.0 requires the PRIMARY,
representative backtest for each strategy to persist a simulated-trade log.
This driver runs ONE full-range, full-universe `run_futures_backtest` per
config with `log_trades=True`, writing
output/backtests/futures/<strategy>/<start>_to_<end>/{trades,equity,margin_utilization}.csv,
registers the run, and prints the trade_log_dir for the readiness report.

Consult docs/methodology/backtesting.md Sections 1, 2, 4, 9, 12 before use.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.backtesting.engine.futures_backtest import run_futures_backtest
from src.utils.run_status import RunStatus


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 0.5 trade-log persistence run")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    strat_name = cfg.get("strategy", {}).get("name", "unknown")

    with RunStatus(f"sp_retest_trade_log_{strat_name}", meta={"config": args.config}) as st:
        result = run_futures_backtest(cfg, register=True, log_trades=True, validate_prereg=True)
        st.heartbeat(note=f"trade log written to {result.get('trade_log_dir')}")

    print(f"[sp_retest_trade_log] strategy={strat_name} run_id={result.get('run_id')} "
          f"trade_log_dir={result.get('trade_log_dir')} n_days={result.get('n_days')} "
          f"metrics={result.get('metrics')}")


if __name__ == "__main__":
    main()
