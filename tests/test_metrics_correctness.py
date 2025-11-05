"""
Test that multi-symbol metrics calculations are correct.

Validates that the performance optimization didn't break metric correctness.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from backtesting.engine.multi_asset_portfolio import MultiAssetPortfolio
from backtesting.engine.multi_symbol_metrics import MultiSymbolMetrics
from utils import logger


def create_test_portfolio() -> MultiAssetPortfolio:
    """
    Create a deterministic test portfolio with known characteristics.
    """
    logger.info("Creating test portfolio with known characteristics...")

    # Create deterministic data
    timestamps = pd.date_range('2024-01-01 09:30', periods=1000, freq='1min')
    symbols = ['AAPL', 'MSFT']

    # Create price data with known patterns
    prices_data = {
        'AAPL': [100 + i * 0.1 for i in range(1000)],  # Steady uptrend
        'MSFT': [200 - i * 0.05 for i in range(1000)]  # Steady downtrend
    }
    prices = pd.DataFrame(prices_data, index=timestamps)

    # Create dummy signals
    entries = pd.DataFrame(False, index=timestamps, columns=symbols)
    exits = pd.DataFrame(False, index=timestamps, columns=symbols)

    # Create portfolio
    portfolio = MultiAssetPortfolio(
        symbols=symbols,
        prices=prices,
        entries=entries,
        exits=exits,
        init_cash=100000,
        fees=0.001,
        position_sizing_method='equal_weight'
    )

    # Manually set up known portfolio state
    portfolio.equity_curve = [100000 + i * 50 for i in range(1000)]  # Steady growth
    portfolio.equity_timestamps = list(timestamps)

    # Cash history - linearly decreasing as we deploy capital
    portfolio.cash_history = [(timestamps[i], 100000 - i * 30) for i in range(1000)]

    # Position count history - hold 2 positions throughout
    portfolio.position_count_history = [(timestamps[i], 2) for i in range(0, 1000, 100)]

    # Equal weights throughout
    portfolio.symbol_weights_history = [(timestamps[i], {'AAPL': 0.5, 'MSFT': 0.5}) for i in range(0, 1000, 100)]

    # Closed positions with known P&L
    portfolio.closed_positions = [
        {
            'symbol': 'AAPL',
            'entry_timestamp': timestamps[0],
            'exit_timestamp': timestamps[100],
            'pnl': 500.0,
            'pnl_pct': 5.0,
            'hold_duration_days': 1.0
        },
        {
            'symbol': 'AAPL',
            'entry_timestamp': timestamps[200],
            'exit_timestamp': timestamps[300],
            'pnl': 300.0,
            'pnl_pct': 3.0,
            'hold_duration_days': 1.0
        },
        {
            'symbol': 'MSFT',
            'entry_timestamp': timestamps[100],
            'exit_timestamp': timestamps[200],
            'pnl': -200.0,
            'pnl_pct': -2.0,
            'hold_duration_days': 1.0
        },
        {
            'symbol': 'MSFT',
            'entry_timestamp': timestamps[300],
            'exit_timestamp': timestamps[400],
            'pnl': -100.0,
            'pnl_pct': -1.0,
            'hold_duration_days': 1.0
        }
    ]

    logger.success("Created test portfolio")
    logger.info(f"  Equity curve: {len(portfolio.equity_curve)} points")
    logger.info(f"  Cash history: {len(portfolio.cash_history)} entries")
    logger.info(f"  Closed positions: {len(portfolio.closed_positions)} trades")

    return portfolio


def test_composition_metrics_correctness():
    """Test that composition metrics calculate correct values."""

    logger.header("TESTING COMPOSITION METRICS CORRECTNESS")
    logger.blank()

    portfolio = create_test_portfolio()

    logger.info("Calculating composition metrics...")
    metrics = MultiSymbolMetrics.calculate_portfolio_composition_metrics(portfolio)

    logger.blank()
    logger.info("Composition Metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")

    logger.blank()
    logger.separator()

    # Validate expected values
    logger.info("Validating metrics...")
    logger.blank()

    # Position count should be 2.0 (constant throughout)
    assert metrics['Avg Position Count'] == 2.0, f"Expected Avg Position Count=2.0, got {metrics['Avg Position Count']}"
    assert metrics['Max Position Count'] == 2, f"Expected Max Position Count=2, got {metrics['Max Position Count']}"
    assert metrics['Min Position Count'] == 2, f"Expected Min Position Count=2, got {metrics['Min Position Count']}"
    logger.success("✓ Position count metrics correct")

    # Capital utilization should increase as cash deploys
    # Starting: (100000-100000)/100000 = 0%
    # Ending: (149950-70100)/149950 ≈ 53.2%
    # Average should be between these
    assert 0 <= metrics['Avg Capital Utilization [%]'] <= 100, \
        f"Capital utilization out of range: {metrics['Avg Capital Utilization [%]']}"
    assert metrics['Max Capital Utilization [%]'] > metrics['Avg Capital Utilization [%]'], \
        "Max utilization should be > avg"
    logger.success("✓ Capital utilization metrics in valid range")

    # Concentration (Herfindahl) for equal weights: 0.5^2 + 0.5^2 = 0.5
    assert abs(metrics['Avg Concentration (Herfindahl)'] - 0.5) < 0.01, \
        f"Expected concentration≈0.5 for equal weights, got {metrics['Avg Concentration (Herfindahl)']}"
    assert abs(metrics['Max Concentration'] - 0.5) < 0.01, \
        f"Expected max concentration≈0.5, got {metrics['Max Concentration']}"
    logger.success("✓ Concentration metrics correct")

    logger.blank()
    logger.separator()
    logger.success("ALL COMPOSITION METRICS VALIDATED")
    logger.blank()


def test_attribution_metrics_correctness():
    """Test that attribution metrics calculate correct P&L."""

    logger.header("TESTING ATTRIBUTION METRICS CORRECTNESS")
    logger.blank()

    portfolio = create_test_portfolio()

    logger.info("Calculating attribution metrics...")
    metrics = MultiSymbolMetrics.calculate_symbol_attribution_metrics(portfolio)

    logger.blank()
    logger.info("Attribution Metrics:")
    logger.info(f"  Best Symbol: {metrics.get('best_symbol')}")
    logger.info(f"  Worst Symbol: {metrics.get('worst_symbol')}")
    logger.info(f"  Total P&L: {metrics.get('total_pnl'):.2f}")

    logger.blank()
    logger.info("Per-Symbol Stats:")
    for symbol, stats in metrics['per_symbol'].items():
        logger.info(f"  {symbol}:")
        logger.info(f"    Total P&L: {stats['Total P&L']:.2f}")
        logger.info(f"    Total Trades: {stats['Total Trades']}")
        logger.info(f"    Win Rate: {stats['Win Rate [%]']:.2f}%")
        logger.info(f"    Contribution: {stats['Contribution [%]']:.2f}%")

    logger.blank()
    logger.separator()
    logger.info("Validating metrics...")
    logger.blank()

    # AAPL: 500 + 300 = 800
    aapl_pnl = metrics['per_symbol']['AAPL']['Total P&L']
    assert abs(aapl_pnl - 800.0) < 0.01, f"Expected AAPL P&L=800, got {aapl_pnl}"
    logger.success(f"✓ AAPL P&L correct: {aapl_pnl:.2f}")

    # MSFT: -200 + -100 = -300
    msft_pnl = metrics['per_symbol']['MSFT']['Total P&L']
    assert abs(msft_pnl - (-300.0)) < 0.01, f"Expected MSFT P&L=-300, got {msft_pnl}"
    logger.success(f"✓ MSFT P&L correct: {msft_pnl:.2f}")

    # Total: 800 - 300 = 500
    total_pnl = metrics['total_pnl']
    assert abs(total_pnl - 500.0) < 0.01, f"Expected total P&L=500, got {total_pnl}"
    logger.success(f"✓ Total P&L correct: {total_pnl:.2f}")

    # AAPL should be best (positive P&L)
    assert metrics['best_symbol'] == 'AAPL', f"Expected best=AAPL, got {metrics['best_symbol']}"
    logger.success("✓ Best symbol correct: AAPL")

    # MSFT should be worst (negative P&L)
    assert metrics['worst_symbol'] == 'MSFT', f"Expected worst=MSFT, got {metrics['worst_symbol']}"
    logger.success("✓ Worst symbol correct: MSFT")

    # Trade counts
    assert metrics['per_symbol']['AAPL']['Total Trades'] == 2, "Expected 2 AAPL trades"
    assert metrics['per_symbol']['MSFT']['Total Trades'] == 2, "Expected 2 MSFT trades"
    logger.success("✓ Trade counts correct")

    # Win rates: AAPL = 2/2 = 100%, MSFT = 0/2 = 0%
    aapl_wr = metrics['per_symbol']['AAPL']['Win Rate [%]']
    msft_wr = metrics['per_symbol']['MSFT']['Win Rate [%]']
    assert abs(aapl_wr - 100.0) < 0.01, f"Expected AAPL win rate=100%, got {aapl_wr}"
    assert abs(msft_wr - 0.0) < 0.01, f"Expected MSFT win rate=0%, got {msft_wr}"
    logger.success("✓ Win rates correct")

    # Contributions: AAPL=800/500=160%, MSFT=-300/500=-60%
    aapl_contrib = metrics['per_symbol']['AAPL']['Contribution [%]']
    msft_contrib = metrics['per_symbol']['MSFT']['Contribution [%]']
    assert abs(aapl_contrib - 160.0) < 0.01, f"Expected AAPL contribution=160%, got {aapl_contrib}"
    assert abs(msft_contrib - (-60.0)) < 0.01, f"Expected MSFT contribution=-60%, got {msft_contrib}"
    logger.success("✓ Contribution percentages correct")

    logger.blank()
    logger.separator()
    logger.success("ALL ATTRIBUTION METRICS VALIDATED")
    logger.blank()


def test_trade_analysis_metrics_correctness():
    """Test that trade analysis metrics are correct."""

    logger.header("TESTING TRADE ANALYSIS METRICS CORRECTNESS")
    logger.blank()

    portfolio = create_test_portfolio()

    logger.info("Calculating trade analysis metrics...")
    metrics = MultiSymbolMetrics.calculate_trade_analysis_metrics(portfolio)

    logger.blank()
    logger.info("Trade Analysis Metrics:")
    logger.info(f"  Trades per symbol: {metrics['trades_per_symbol']}")
    logger.info(f"  Profit factors: {metrics['profit_factor_per_symbol']}")
    logger.info(f"  Expectancy: {metrics['expectancy_per_symbol']}")

    logger.blank()
    logger.separator()
    logger.info("Validating metrics...")
    logger.blank()

    # Trades per symbol
    assert metrics['trades_per_symbol']['AAPL'] == 2, "Expected 2 AAPL trades"
    assert metrics['trades_per_symbol']['MSFT'] == 2, "Expected 2 MSFT trades"
    logger.success("✓ Trades per symbol correct")

    # Profit factor = Total Profit / |Total Loss|
    # AAPL: 800 / 0 = undefined (no losses) -> should be 0 in code
    # MSFT: 0 / 300 = 0
    aapl_pf = metrics['profit_factor_per_symbol']['AAPL']
    msft_pf = metrics['profit_factor_per_symbol']['MSFT']
    assert aapl_pf == 0, f"Expected AAPL profit factor=0 (no losses), got {aapl_pf}"
    assert msft_pf == 0, f"Expected MSFT profit factor=0 (no wins), got {msft_pf}"
    logger.success("✓ Profit factors correct")

    # Expectancy = (WinRate × AvgWin) - (LossRate × AvgLoss)
    # AAPL: (1.0 × 400) - (0.0 × 0) = 400
    # MSFT: (0.0 × 0) - (1.0 × 150) = -150
    aapl_exp = metrics['expectancy_per_symbol']['AAPL']
    msft_exp = metrics['expectancy_per_symbol']['MSFT']
    assert abs(aapl_exp - 400.0) < 0.01, f"Expected AAPL expectancy=400, got {aapl_exp}"
    assert abs(msft_exp - (-150.0)) < 0.01, f"Expected MSFT expectancy=-150, got {msft_exp}"
    logger.success("✓ Expectancy values correct")

    logger.blank()
    logger.separator()
    logger.success("ALL TRADE ANALYSIS METRICS VALIDATED")
    logger.blank()


def test_capital_utilization_calculation():
    """Test the specific capital utilization calculation that was optimized."""

    logger.header("TESTING CAPITAL UTILIZATION CALCULATION")
    logger.blank()

    portfolio = create_test_portfolio()

    # Manually calculate expected capital utilization for first few points
    logger.info("Manual verification of capital utilization calculation...")
    logger.blank()

    expected_utilizations = []
    for i in range(min(5, len(portfolio.cash_history))):
        timestamp = portfolio.equity_timestamps[i]
        cash = portfolio.cash_history[i][1]
        portfolio_value = portfolio.equity_curve[i]

        deployed = portfolio_value - cash
        utilization_pct = (deployed / portfolio_value) * 100
        expected_utilizations.append(utilization_pct)

        logger.info(f"Point {i}:")
        logger.info(f"  Portfolio Value: {portfolio_value:.2f}")
        logger.info(f"  Cash: {cash:.2f}")
        logger.info(f"  Deployed: {deployed:.2f}")
        logger.info(f"  Utilization: {utilization_pct:.2f}%")

    logger.blank()
    logger.separator()

    # Now get metrics and verify the average makes sense
    metrics = MultiSymbolMetrics.calculate_portfolio_composition_metrics(portfolio)

    avg_util = metrics['Avg Capital Utilization [%]']
    max_util = metrics['Max Capital Utilization [%]']

    logger.info(f"Calculated Avg Utilization: {avg_util:.2f}%")
    logger.info(f"Calculated Max Utilization: {max_util:.2f}%")
    logger.blank()

    # Sanity checks
    assert avg_util >= 0, "Utilization cannot be negative"
    assert avg_util <= 100, "Utilization cannot exceed 100%"
    assert max_util >= avg_util, "Max should be >= average"

    # For our test data:
    # Start: (100000-100000)/100000 = 0%
    # Point 1: (100050-99970)/100050 = 80/100050 = 0.08%
    # Point 999: (149950-70030)/149950 = 79920/149950 = 53.3%
    # Average should be between 0% and 53.3%
    assert 0 <= avg_util <= 60, f"Average utilization {avg_util:.2f}% outside expected range"
    logger.success(f"✓ Capital utilization in expected range: {avg_util:.2f}%")

    # The max should be near the end (highest utilization)
    assert 50 <= max_util <= 60, f"Max utilization {max_util:.2f}% outside expected range"
    logger.success(f"✓ Max utilization in expected range: {max_util:.2f}%")

    logger.blank()
    logger.separator()
    logger.success("CAPITAL UTILIZATION CALCULATION VALIDATED")
    logger.blank()


if __name__ == '__main__':
    try:
        test_composition_metrics_correctness()
        test_attribution_metrics_correctness()
        test_trade_analysis_metrics_correctness()
        test_capital_utilization_calculation()

        logger.blank()
        logger.separator()
        logger.header("ALL CORRECTNESS TESTS PASSED")
        logger.success("✓ Performance optimization did NOT break metric calculations")
        logger.success("✓ All metrics are calculating correct values")
        logger.blank()
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise
