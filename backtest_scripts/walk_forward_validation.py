"""
Walk-Forward Validation for Multi-Pair Portfolio

Tests strategy robustness by:
1. Running portfolio on 2023 data (training period)
2. Running portfolio on 2024 data (testing period)
3. Comparing performance degradation

Target: < 30% Sharpe degradation between periods

Author: Homeguard Team
Date: November 11, 2025
"""

import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import logger
from src.config import get_backtest_results_dir
from src.backtesting.engine.multi_pair_portfolio import (
    MultiPairPortfolio,
    PairConfig,
    PortfolioRiskLimits
)


def create_portfolio_config():
    """Create optimized portfolio configuration (from full-period optimization)."""

    # Top 5 pairs with optimal parameters from full-period optimization
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


def run_period(period_name, start_date, end_date):
    """Run backtest for a specific period."""

    logger.info(f"\n{'='*80}")
    logger.info(f"Running {period_name} Period: {start_date} to {end_date}")
    logger.info(f"{'='*80}\n")

    # Configuration
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
    try:
        results = portfolio.run_backtest(
            start_date=start_date,
            end_date=end_date,
            market_hours_only=True
        )

        # Extract metrics
        metrics = results.get('portfolio_metrics', {})

        logger.info(f"\n{period_name} Period Results:")
        logger.info(f"  Sharpe Ratio: {metrics.get('sharpe', 0):.4f}")
        logger.info(f"  Total Return: {metrics.get('return', 0)*100:.2f}%")
        logger.info(f"  Max Drawdown: {metrics.get('max_dd', 0)*100:.2f}%")
        logger.info(f"  Total Trades: {metrics.get('trades', 0)}")
        logger.info(f"  Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")

        return metrics

    except Exception as e:
        logger.error(f"Error running {period_name} period: {e}")
        import traceback
        traceback.print_exc()
        return None


def calculate_degradation(train_metrics, test_metrics):
    """Calculate performance degradation from train to test."""

    if not train_metrics or not test_metrics:
        logger.error("Missing metrics for degradation calculation")
        return None

    train_sharpe = train_metrics.get('sharpe', 0)
    test_sharpe = test_metrics.get('sharpe', 0)

    if train_sharpe == 0:
        logger.warning("Train Sharpe is zero, cannot calculate degradation")
        return None

    degradation_pct = ((train_sharpe - test_sharpe) / train_sharpe) * 100

    return {
        'train_sharpe': train_sharpe,
        'test_sharpe': test_sharpe,
        'degradation_pct': degradation_pct,
        'degradation_abs': train_sharpe - test_sharpe,
        'train_return': train_metrics.get('return', 0),
        'test_return': test_metrics.get('return', 0),
        'train_max_dd': train_metrics.get('max_dd', 0),
        'test_max_dd': test_metrics.get('max_dd', 0),
        'train_trades': train_metrics.get('trades', 0),
        'test_trades': test_metrics.get('test_trades', 0),
        'train_win_rate': train_metrics.get('win_rate', 0),
        'test_win_rate': test_metrics.get('win_rate', 0)
    }


def main():
    """Main execution."""

    logger.info("\n" + "="*80)
    logger.info("WALK-FORWARD VALIDATION - MULTI-PAIR PORTFOLIO")
    logger.info("="*80)
    logger.info("\nObjective: Verify strategy robustness across time periods")
    logger.info("Target: < 30% Sharpe degradation from train to test")
    logger.info("\nMethodology:")
    logger.info("  - Train Period: 2023-01-01 to 2023-12-31 (12 months)")
    logger.info("  - Test Period: 2024-01-01 to 2024-11-11 (10.4 months)")
    logger.info("  - Using full-period optimized parameters on both periods")
    logger.info("")

    # Run training period
    train_metrics = run_period(
        period_name="TRAINING",
        start_date='2023-01-01',
        end_date='2023-12-31'
    )

    # Run testing period
    test_metrics = run_period(
        period_name="TESTING",
        start_date='2024-01-01',
        end_date='2024-11-11'
    )

    # Calculate degradation
    if train_metrics and test_metrics:
        logger.info("\n" + "="*80)
        logger.info("WALK-FORWARD VALIDATION RESULTS")
        logger.info("="*80)

        degradation = calculate_degradation(train_metrics, test_metrics)

        if degradation:
            logger.info(f"\nSharpe Ratio:")
            logger.info(f"  Training (2023):  {degradation['train_sharpe']:.4f}")
            logger.info(f"  Testing (2024):   {degradation['test_sharpe']:.4f}")
            logger.info(f"  Degradation:      {degradation['degradation_abs']:.4f} ({degradation['degradation_pct']:.1f}%)")

            logger.info(f"\nTotal Return:")
            logger.info(f"  Training (2023):  {degradation['train_return']*100:.2f}%")
            logger.info(f"  Testing (2024):   {degradation['test_return']*100:.2f}%")

            logger.info(f"\nMax Drawdown:")
            logger.info(f"  Training (2023):  {degradation['train_max_dd']*100:.2f}%")
            logger.info(f"  Testing (2024):   {degradation['test_max_dd']*100:.2f}%")

            logger.info(f"\nTrading Activity:")
            logger.info(f"  Training Trades:  {degradation['train_trades']}")
            logger.info(f"  Testing Trades:   {degradation['test_trades']}")
            logger.info(f"  Training Win Rate: {degradation['train_win_rate']*100:.1f}%")
            logger.info(f"  Testing Win Rate:  {degradation['test_win_rate']*100:.1f}%")

            # Pass/Fail Assessment
            logger.info("\n" + "="*80)
            logger.info("VALIDATION ASSESSMENT")
            logger.info("="*80)

            target_degradation = 30.0  # 30% maximum allowed degradation

            if degradation['degradation_pct'] < 0:
                logger.info(f"\n✅ EXCELLENT: Test period OUTPERFORMED training!")
                logger.info(f"   Improvement: {abs(degradation['degradation_pct']):.1f}%")
                logger.info(f"   Status: PASSED (Strategy is robust)")
            elif degradation['degradation_pct'] <= target_degradation:
                logger.info(f"\n✅ PASSED: Degradation {degradation['degradation_pct']:.1f}% < {target_degradation}%")
                logger.info(f"   Status: Strategy is robust across periods")
            else:
                logger.info(f"\n❌ FAILED: Degradation {degradation['degradation_pct']:.1f}% > {target_degradation}%")
                logger.info(f"   Status: Strategy may be overfitted to training period")

            # Production readiness
            if test_metrics.get('sharpe', 0) >= 0.8:
                logger.info(f"\n✅ PRODUCTION READY: Test Sharpe {test_metrics['sharpe']:.4f} >= 0.8")
            elif test_metrics.get('sharpe', 0) >= 0.6:
                logger.info(f"\n⚠️  BORDERLINE: Test Sharpe {test_metrics['sharpe']:.4f} is 0.6-0.8")
                logger.info(f"   Recommendation: Consider paper trading before live deployment")
            else:
                logger.info(f"\n❌ NOT READY: Test Sharpe {test_metrics['sharpe']:.4f} < 0.6")
                logger.info(f"   Recommendation: Further optimization or strategy enhancement needed")

            # Save results
            results_df = pd.DataFrame([{
                'Period': 'Training (2023)',
                'Sharpe': degradation['train_sharpe'],
                'Return': f"{degradation['train_return']*100:.2f}%",
                'Max_DD': f"{degradation['train_max_dd']*100:.2f}%",
                'Trades': degradation['train_trades'],
                'Win_Rate': f"{degradation['train_win_rate']*100:.1f}%"
            }, {
                'Period': 'Testing (2024)',
                'Sharpe': degradation['test_sharpe'],
                'Return': f"{degradation['test_return']*100:.2f}%",
                'Max_DD': f"{degradation['test_max_dd']*100:.2f}%",
                'Trades': degradation['test_trades'],
                'Win_Rate': f"{degradation['test_win_rate']*100:.1f}%"
            }, {
                'Period': 'Degradation',
                'Sharpe': f"{degradation['degradation_pct']:.1f}%",
                'Return': '-',
                'Max_DD': '-',
                'Trades': '-',
                'Win_Rate': '-'
            }])

            output_path = get_backtest_results_dir() / 'walk_forward_validation_results.csv'
            results_df.to_csv(output_path, index=False)
            logger.info(f"\n📊 Results saved to: {output_path}")

            return degradation

    else:
        logger.error("\n❌ Walk-forward validation FAILED - unable to compute metrics")
        return None


if __name__ == "__main__":
    results = main()
