"""
RAMP-CSP Walk-Forward Backtest Runner.

Runs in-sample and out-of-sample backtests for the RAMP-CSP strategy,
generates performance reports, and validates against success criteria.

Usage:
    python scripts/backtest/run_ramp_csp_backtest.py
    python scripts/backtest/run_ramp_csp_backtest.py --config config/strategies/ramp_csp.yaml
    python scripts/backtest/run_ramp_csp_backtest.py --oos-only
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
from datetime import date

import pandas as pd

from src.strategies.options.csp.ramp_integration import CSPBacktestRunner, load_csp_config
from src.strategies.options.csp.metrics import compute_csp_metrics, compute_sharpe, compute_max_drawdown
from src.utils.logger import get_logger

logger = get_logger()


def run_period(runner, start, end, label):
    """Run backtest for a period and report results."""
    logger.info("=" * 80)
    logger.info(f"  {label}: {start} to {end}")
    logger.info("=" * 80)

    result = runner.run(start_date=start, end_date=end)
    metrics = compute_csp_metrics(result.closed_trades)

    sharpe = compute_sharpe(result.equity_curve) if result.equity_curve is not None else 0.0
    max_dd = compute_max_drawdown(result.equity_curve) if result.equity_curve is not None else 0.0

    total_return = 0.0
    if result.equity_curve is not None and len(result.equity_curve) > 0:
        total_return = (
            result.equity_curve.iloc[-1] / result.equity_curve.iloc[0] - 1
        )

    logger.info(f"\n--- {label} Results ---")
    logger.info(f"  Total Return:    {total_return:.2%}")
    logger.info(f"  Sharpe Ratio:    {sharpe:.3f}")
    logger.info(f"  Max Drawdown:    {max_dd:.2%}")
    logger.info(f"  Total Trades:    {metrics['total_trades']}")
    logger.info(f"  Win Rate:        {metrics['win_rate']:.1%}")
    logger.info(f"  Avg ROC/trade:   {metrics['avg_return_on_collateral']:.2%}")
    logger.info(f"  Avg Hold Days:   {metrics['avg_holding_days']:.1f}")
    logger.info(f"  Total P&L:       ${metrics['total_pnl']:,.2f}")

    return result, metrics, sharpe, max_dd


def main():
    parser = argparse.ArgumentParser(description="RAMP-CSP Walk-Forward Backtest")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--oos-only", action="store_true", help="Skip in-sample, run OOS only")
    args = parser.parse_args()

    config = load_csp_config(args.config)
    dates_config = config.get("dates", {})
    validation = config.get("validation", {})

    is_start = date.fromisoformat(dates_config.get("in_sample_start", "2022-01-01"))
    is_end = date.fromisoformat(dates_config.get("in_sample_end", "2023-06-30"))
    oos_start = date.fromisoformat(dates_config.get("out_of_sample_start", "2023-07-01"))
    oos_end = date.fromisoformat(dates_config.get("out_of_sample_end", "2024-12-31"))

    logger.info("RAMP-CSP Walk-Forward Backtest")
    logger.info(f"  IS:  {is_start} to {is_end}")
    logger.info(f"  OOS: {oos_start} to {oos_end}")

    runner = CSPBacktestRunner(config=config)

    if not args.oos_only:
        is_result, is_metrics, is_sharpe, is_dd = run_period(
            runner, is_start, is_end, "IN-SAMPLE"
        )

    # Fresh runner for OOS (no state leakage)
    oos_runner = CSPBacktestRunner(config=config)
    oos_result, oos_metrics, oos_sharpe, oos_dd = run_period(
        oos_runner, oos_start, oos_end, "OUT-OF-SAMPLE"
    )

    # Validate against success criteria
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION AGAINST SUCCESS CRITERIA")
    logger.info("=" * 80)

    checks = [
        ("Sharpe >= 0.5", oos_sharpe >= validation.get("min_sharpe", 0.5), f"{oos_sharpe:.3f}"),
        ("Max DD < 10%", oos_dd < validation.get("max_drawdown", 0.10), f"{oos_dd:.2%}"),
        ("Win Rate >= 60%", oos_metrics["win_rate"] >= validation.get("min_win_rate", 0.60),
         f"{oos_metrics['win_rate']:.1%}"),
        ("Avg ROC >= 1%",
         oos_metrics["avg_return_on_collateral"] >= validation.get("min_return_on_collateral", 0.01),
         f"{oos_metrics['avg_return_on_collateral']:.2%}"),
    ]

    all_pass = True
    for name, passed, value in checks:
        status = "PASS" if passed else "FAIL"
        logger.info(f"  [{status}] {name}: {value}")
        if not passed:
            all_pass = False

    if all_pass:
        logger.info("\n  [+] All validation criteria met!")
    else:
        logger.info("\n  [-] Some criteria not met. Review results.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
