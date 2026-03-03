"""
RAMP-CSP Comprehensive Backtest Analysis.

Runs parameter sensitivity analysis on the OOS period only (2023-07 to 2024-12)
to assess robustness. Varies one parameter at a time from defaults.

Anti-overfitting approach:
- All sensitivity tests run on OOS period only (never IS)
- Parameters varied one at a time (no multi-param grid search)
- Results compared to default config baseline
- Flag any parameter set that improves OOS Sharpe by more than 50% (suspect)

Usage:
    python scripts/backtest/ramp_csp_comprehensive.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import json
import copy
from datetime import date, datetime

import pandas as pd

from src.strategies.options.csp.ramp_integration import CSPBacktestRunner, load_csp_config
from src.strategies.options.csp.metrics import compute_csp_metrics, compute_sharpe, compute_max_drawdown
from src.utils.logger import get_logger

logger = get_logger()


def compute_cagr(equity_curve: pd.Series) -> float:
    if equity_curve is None or len(equity_curve) < 2:
        return 0.0
    total_days = (equity_curve.index[-1] - equity_curve.index[0]).days
    if total_days <= 0:
        return 0.0
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    return float(total_return ** (365.25 / total_days) - 1)


def run_single_config(config, start_date, end_date, label):
    """Run a single backtest config and return summary dict."""
    logger.info(f"\n{'='*60}")
    logger.info(f"  {label}")
    logger.info(f"  {start_date} to {end_date}")
    logger.info(f"{'='*60}")

    runner = CSPBacktestRunner(config=config)
    result = runner.run(start_date=start_date, end_date=end_date)
    metrics = compute_csp_metrics(result.closed_trades)

    sharpe = compute_sharpe(result.equity_curve)
    max_dd = compute_max_drawdown(result.equity_curve)
    cagr = compute_cagr(result.equity_curve)
    total_return = 0.0
    if result.equity_curve is not None and len(result.equity_curve) > 0:
        total_return = float(result.equity_curve.iloc[-1] / result.equity_curve.iloc[0] - 1)

    summary = {
        "label": label,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "total_trades": metrics["total_trades"],
        "win_rate": metrics["win_rate"],
        "avg_return_on_collateral": metrics["avg_return_on_collateral"],
        "avg_holding_days": metrics["avg_holding_days"],
        "total_pnl": metrics["total_pnl"],
        "avg_premium": metrics["avg_premium"],
        "pnl_by_exit_reason": metrics["pnl_by_exit_reason"],
        "count_by_exit_reason": metrics["count_by_exit_reason"],
    }

    logger.info(f"  Return: {total_return:.2%}, Sharpe: {sharpe:.3f}, "
                f"MaxDD: {max_dd:.2%}, Trades: {metrics['total_trades']}, "
                f"WinRate: {metrics['win_rate']:.1%}")

    return summary


def main():
    config = load_csp_config()
    dates_cfg = config.get("dates", {})

    oos_start = date.fromisoformat(dates_cfg.get("out_of_sample_start", "2023-07-01"))
    oos_end = date.fromisoformat(dates_cfg.get("out_of_sample_end", "2024-12-31"))

    all_results = []

    # --- 1. Default config OOS baseline ---
    logger.info("\n" + "#" * 70)
    logger.info("  BASELINE: Default Configuration (OOS)")
    logger.info("#" * 70)
    baseline = run_single_config(config, oos_start, oos_end, "DEFAULT (baseline)")
    all_results.append(baseline)

    # --- 2. Parameter sensitivity: delta range ---
    logger.info("\n" + "#" * 70)
    logger.info("  SENSITIVITY: Delta Range")
    logger.info("#" * 70)

    delta_configs = [
        ({"target_delta_min": -0.40, "target_delta_max": -0.30}, "Delta [-0.40, -0.30] (deeper OTM)"),
        ({"target_delta_min": -0.30, "target_delta_max": -0.20}, "Delta [-0.30, -0.20] (shallower OTM)"),
    ]
    for delta_override, label in delta_configs:
        cfg = copy.deepcopy(config)
        cfg["contract_selection"].update(delta_override)
        result = run_single_config(cfg, oos_start, oos_end, label)
        all_results.append(result)

    # --- 3. Parameter sensitivity: DTE range ---
    logger.info("\n" + "#" * 70)
    logger.info("  SENSITIVITY: DTE Range")
    logger.info("#" * 70)

    dte_configs = [
        ({"min_dte": 14, "max_dte": 28}, "DTE [14, 28] (shorter)"),
        ({"min_dte": 28, "max_dte": 45}, "DTE [28, 45] (longer)"),
    ]
    for dte_override, label in dte_configs:
        cfg = copy.deepcopy(config)
        cfg["contract_selection"].update(dte_override)
        result = run_single_config(cfg, oos_start, oos_end, label)
        all_results.append(result)

    # --- 4. Parameter sensitivity: profit target ---
    logger.info("\n" + "#" * 70)
    logger.info("  SENSITIVITY: Profit Target")
    logger.info("#" * 70)

    pt_configs = [
        ({"profit_target_pct": 0.40}, "Profit Target 40%"),
        ({"profit_target_pct": 0.65}, "Profit Target 65%"),
        ({"profit_target_pct": 0.75}, "Profit Target 75%"),
    ]
    for pt_override, label in pt_configs:
        cfg = copy.deepcopy(config)
        cfg["strategy"].update(pt_override)
        result = run_single_config(cfg, oos_start, oos_end, label)
        all_results.append(result)

    # --- 5. Parameter sensitivity: max positions ---
    logger.info("\n" + "#" * 70)
    logger.info("  SENSITIVITY: Max Positions")
    logger.info("#" * 70)

    mp_configs = [
        ({"max_positions": 3}, "Max Positions 3"),
        ({"max_positions": 7}, "Max Positions 7"),
    ]
    for mp_override, label in mp_configs:
        cfg = copy.deepcopy(config)
        cfg["strategy"].update(mp_override)
        result = run_single_config(cfg, oos_start, oos_end, label)
        all_results.append(result)

    # --- 6. Parameter sensitivity: loss limit ---
    logger.info("\n" + "#" * 70)
    logger.info("  SENSITIVITY: Loss Limit")
    logger.info("#" * 70)

    ll_configs = [
        ({"loss_limit_multiple": 1.5}, "Loss Limit 1.5x"),
        ({"loss_limit_multiple": 3.0}, "Loss Limit 3.0x"),
    ]
    for ll_override, label in ll_configs:
        cfg = copy.deepcopy(config)
        cfg["strategy"].update(ll_override)
        result = run_single_config(cfg, oos_start, oos_end, label)
        all_results.append(result)

    # --- Save results to JSON ---
    output_dir = project_root / "docs" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "20260303_RAMP_CSP_SENSITIVITY_RESULTS.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {json_path}")

    # --- Print summary table ---
    logger.info("\n" + "=" * 100)
    logger.info("PARAMETER SENSITIVITY SUMMARY (OOS: 2023-07 to 2024-12)")
    logger.info("=" * 100)
    header = f"{'Config':<35} {'Return':>8} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>7} {'WinRate':>8} {'AvgROC':>8}"
    logger.info(header)
    logger.info("-" * 100)

    baseline_sharpe = baseline["sharpe"]
    for r in all_results:
        line = (
            f"{r['label']:<35} "
            f"{r['total_return']:>7.2%} "
            f"{r['sharpe']:>8.3f} "
            f"{r['max_drawdown']:>7.2%} "
            f"{r['total_trades']:>7d} "
            f"{r['win_rate']:>7.1%} "
            f"{r['avg_return_on_collateral']:>7.2%}"
        )
        # Flag suspect improvements
        if r["label"] != "DEFAULT (baseline)" and baseline_sharpe > 0:
            improvement = (r["sharpe"] - baseline_sharpe) / baseline_sharpe
            if improvement > 0.50:
                line += "  [!] SUSPECT"
        logger.info(line)

    # --- Anti-overfitting checks ---
    logger.info("\n" + "=" * 70)
    logger.info("ANTI-OVERFITTING DIAGNOSTICS")
    logger.info("=" * 70)

    sharpe_values = [r["sharpe"] for r in all_results]
    sharpe_range = max(sharpe_values) - min(sharpe_values)
    logger.info(f"  Sharpe range across configs: {sharpe_range:.3f}")
    if sharpe_range > 1.0:
        logger.info("  [!] HIGH variance -- results sensitive to params (overfitting risk)")
    elif sharpe_range < 0.3:
        logger.info("  [+] LOW variance -- results robust to param changes")
    else:
        logger.info("  [*] MODERATE variance -- some parameter sensitivity")

    trade_counts = [r["total_trades"] for r in all_results if r["total_trades"] > 0]
    if trade_counts:
        avg_trades = sum(trade_counts) / len(trade_counts)
        logger.info(f"  Avg trades across configs: {avg_trades:.0f}")
        if avg_trades < 20:
            logger.info("  [!] LOW trade count -- insufficient for statistical significance")
        elif avg_trades < 50:
            logger.info("  [*] MODERATE trade count -- results directionally useful")
        else:
            logger.info("  [+] SUFFICIENT trade count for statistical analysis")

    return 0


if __name__ == "__main__":
    sys.exit(main())
