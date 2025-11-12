"""
Multi-Pair Portfolio Backtest Runner

Runs the top 5 optimized pairs simultaneously with optimal parameters.
Target: Achieve portfolio Sharpe >= 0.85

Author: Homeguard Team
Date: November 11, 2025
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import logger
from src.backtesting.engine.multi_pair_portfolio import (
    MultiPairPortfolio,
    PairConfig,
    PortfolioRiskLimits
)


def create_portfolio_config():
    """Create optimized portfolio configuration."""

    # Top 5 pairs with optimal parameters from optimization
    pairs = [
        PairConfig(
            name='XLY/UVXY',
            symbol1='XLY',
            symbol2='UVXY',
            weight=0.25,
            expected_sharpe=0.735,
            params={
                'entry_zscore': 2.75,
                'exit_zscore': 0.25,
                'stop_loss_zscore': 3.5,
                'zscore_window': 25,
                'pair_selection_window': 252,
                'cointegration_pvalue': 0.05,
                'hedge_ratio_method': 'ols'
            }
        ),
        PairConfig(
            name='XLI/UVXY',
            symbol1='XLI',
            symbol2='UVXY',
            weight=0.25,
            expected_sharpe=0.710,
            params={
                'entry_zscore': 2.75,
                'exit_zscore': 0.25,
                'stop_loss_zscore': 3.5,
                'zscore_window': 25,
                'pair_selection_window': 252,
                'cointegration_pvalue': 0.05,
                'hedge_ratio_method': 'ols'
            }
        ),
        PairConfig(
            name='DIA/UVXY',
            symbol1='DIA',
            symbol2='UVXY',
            weight=0.20,
            expected_sharpe=0.677,
            params={
                'entry_zscore': 2.5,
                'exit_zscore': 0.25,
                'stop_loss_zscore': 3.5,
                'zscore_window': 25,
                'pair_selection_window': 252,
                'cointegration_pvalue': 0.05,
                'hedge_ratio_method': 'ols'
            }
        ),
        PairConfig(
            name='VTI/UVXY',
            symbol1='VTI',
            symbol2='UVXY',
            weight=0.15,
            expected_sharpe=0.679,
            params={
                'entry_zscore': 2.5,
                'exit_zscore': 0.25,
                'stop_loss_zscore': 3.5,
                'zscore_window': 25,
                'pair_selection_window': 252,
                'cointegration_pvalue': 0.05,
                'hedge_ratio_method': 'ols'
            }
        ),
        PairConfig(
            name='XLK/UVXY',
            symbol1='XLK',
            symbol2='UVXY',
            weight=0.15,
            expected_sharpe=0.687,
            params={
                'entry_zscore': 2.75,
                'exit_zscore': 0.25,
                'stop_loss_zscore': 3.5,
                'zscore_window': 25,
                'pair_selection_window': 252,
                'cointegration_pvalue': 0.05,
                'hedge_ratio_method': 'ols'
            }
        )
    ]

    # Portfolio risk limits
    risk_limits = PortfolioRiskLimits(
        max_portfolio_drawdown=0.20,
        max_pair_drawdown=0.15,
        max_leverage=1.5,
        min_pair_sharpe=0.5,
        max_correlation=0.7
    )

    return pairs, risk_limits


def main():
    """Main execution."""

    # Configuration
    start_date = '2023-01-01'
    end_date = '2024-11-11'
    initial_capital = 100000
    fees = 0.0001  # 0.01%
    slippage = 0.001  # 0.10%

    # Create portfolio configuration
    pairs, risk_limits = create_portfolio_config()

    # Initialize portfolio
    portfolio = MultiPairPortfolio(
        pairs=pairs,
        risk_limits=risk_limits,
        initial_capital=initial_capital,
        fees=fees,
        slippage=slippage
    )

    # Run backtest
    results = portfolio.run_backtest(
        start_date=start_date,
        end_date=end_date,
        market_hours_only=True
    )

    # Save results
    portfolio.save_results()

    # Return metrics for programmatic access
    return results['portfolio_metrics']


if __name__ == "__main__":
    metrics = main()