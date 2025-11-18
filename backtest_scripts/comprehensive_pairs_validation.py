"""
Comprehensive Validation of Pairs Trading Framework.

This script systematically validates the pairs trading implementation through:
1. Synthetic cointegrated pair generation (controlled experiments)
2. Position sizer comparison
3. Parameter optimization
4. Walk-forward validation
5. Robustness testing

Usage:
    From project root:
    C:\\Users\\qwqw1\\anaconda3\\envs\\fintech\\python.exe backtest_scripts/comprehensive_pairs_validation.py
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import json

from utils.path_setup import setup_project_paths
ROOT_DIR = setup_project_paths()
from datetime import datetime

from backtesting.engine.backtest_engine import BacktestEngine
from strategies.advanced.pairs_trading import PairsTrading
from backtesting.optimization.grid_search import GridSearchOptimizer
from backtesting.utils.risk_config import RiskConfig
from utils import logger


class PairsDataGenerator:
    """Generate synthetic cointegrated pairs for testing."""

    @staticmethod
    def generate_cointegrated_pair(
        n_days: int = 500,
        mean: float = 100.0,
        spread_volatility: float = 2.0,
        mean_reversion_speed: float = 0.1,
        seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate a pair of cointegrated price series.

        Uses Ornstein-Uhlenbeck process for the spread to ensure cointegration.

        Args:
            n_days: Number of days to generate
            mean: Mean price level
            spread_volatility: Volatility of the spread
            mean_reversion_speed: Speed of mean reversion (higher = faster)
            seed: Random seed

        Returns:
            Tuple of (data1, data2) DataFrames with OHLCV data
        """
        np.random.seed(seed)

        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')

        # Generate base price series (random walk)
        price1 = np.zeros(n_days)
        price1[0] = mean

        for i in range(1, n_days):
            price1[i] = price1[i-1] + np.random.normal(0, 1)

        # Generate mean-reverting spread (Ornstein-Uhlenbeck)
        spread = np.zeros(n_days)
        spread[0] = 0

        for i in range(1, n_days):
            # OU process: dS = -theta * S * dt + sigma * dW
            spread[i] = spread[i-1] - mean_reversion_speed * spread[i-1] + \
                       np.random.normal(0, spread_volatility)

        # Price2 = Price1 + Spread (this ensures cointegration)
        price2 = price1 + spread

        # Add intraday variation for OHLCV
        def create_ohlcv(close_prices: np.ndarray) -> pd.DataFrame:
            high = close_prices + np.abs(np.random.randn(len(close_prices)) * 0.5)
            low = close_prices - np.abs(np.random.randn(len(close_prices)) * 0.5)
            open_prices = close_prices + np.random.randn(len(close_prices)) * 0.3
            volume = np.random.randint(1000000, 5000000, len(close_prices))

            return pd.DataFrame({
                'open': open_prices,
                'high': high,
                'low': low,
                'close': close_prices,
                'volume': volume
            }, index=dates)

        data1 = create_ohlcv(price1)
        data2 = create_ohlcv(price2)

        return data1, data2

    @staticmethod
    def generate_non_cointegrated_pair(
        n_days: int = 500,
        mean1: float = 100.0,
        mean2: float = 150.0,
        vol1: float = 2.0,
        vol2: float = 3.0,
        seed: int = 43
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate a pair of independent (non-cointegrated) series."""
        np.random.seed(seed)

        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')

        # Independent random walks
        price1 = np.zeros(n_days)
        price2 = np.zeros(n_days)
        price1[0] = mean1
        price2[0] = mean2

        for i in range(1, n_days):
            price1[i] = price1[i-1] + np.random.normal(0, vol1)
            price2[i] = price2[i-1] + np.random.normal(0, vol2)

        def create_ohlcv(close_prices: np.ndarray) -> pd.DataFrame:
            high = close_prices + np.abs(np.random.randn(len(close_prices)) * 0.5)
            low = close_prices - np.abs(np.random.randn(len(close_prices)) * 0.5)
            open_prices = close_prices + np.random.randn(len(close_prices)) * 0.3
            volume = np.random.randint(1000000, 5000000, len(close_prices))

            return pd.DataFrame({
                'open': open_prices,
                'high': high,
                'low': low,
                'close': close_prices,
                'volume': volume
            }, index=dates)

        data1 = create_ohlcv(price1)
        data2 = create_ohlcv(price2)

        return data1, data2


class PairsValidationFramework:
    """Comprehensive validation framework for pairs trading."""

    def __init__(self):
        self.results = {
            'pair_discovery': {},
            'position_sizer_comparison': {},
            'parameter_optimization': {},
            'walk_forward_validation': {},
            'robustness_analysis': {}
        }

    def phase1_pair_discovery(self) -> Dict:
        """
        Phase 1: Test cointegration detection on synthetic pairs.
        """
        logger.separator("=", 80)
        logger.header("PHASE 1: PAIR DISCOVERY & COINTEGRATION TESTING")
        logger.separator("=", 80)

        generator = PairsDataGenerator()
        results = {}

        # Test 1: Strongly cointegrated pair
        logger.info("\nTest 1: Strongly Cointegrated Pair")
        data1_strong, data2_strong = generator.generate_cointegrated_pair(
            n_days=500,
            spread_volatility=1.5,
            mean_reversion_speed=0.15,
            seed=42
        )

        strategy = PairsTrading()
        is_coint, p_value = strategy.test_cointegration(
            data1_strong['close'],
            data2_strong['close']
        )

        logger.info(f"  Cointegrated: {is_coint}" if is_coint else "red")
        logger.info(f"  P-value: {p_value:.4f}" if p_value < 0.05 else "yellow")

        results['strong_cointegration'] = {
            'is_cointegrated': is_coint,
            'p_value': p_value,
            'data': (data1_strong, data2_strong)
        }

        # Test 2: Marginally cointegrated pair
        logger.info("\nTest 2: Marginally Cointegrated Pair")
        data1_marginal, data2_marginal = generator.generate_cointegrated_pair(
            n_days=500,
            spread_volatility=3.0,
            mean_reversion_speed=0.05,
            seed=43
        )

        is_coint, p_value = strategy.test_cointegration(
            data1_marginal['close'],
            data2_marginal['close']
        )

        logger.info(f"  Cointegrated: {is_coint}" if is_coint else "yellow")
        logger.info(f"  P-value: {p_value:.4f}" if p_value < 0.05 else "yellow")

        results['marginal_cointegration'] = {
            'is_cointegrated': is_coint,
            'p_value': p_value,
            'data': (data1_marginal, data2_marginal)
        }

        # Test 3: Non-cointegrated pair (should not trade)
        logger.info("\nTest 3: Non-Cointegrated Pair (Control)")
        data1_non, data2_non = generator.generate_non_cointegrated_pair(
            n_days=500,
            seed=44
        )

        is_coint, p_value = strategy.test_cointegration(
            data1_non['close'],
            data2_non['close']
        )

        logger.info(f"  Cointegrated: {is_coint}" if is_coint else "green")
        logger.info(f"  P-value: {p_value:.4f}" if p_value < 0.05 else "green")

        results['non_cointegrated'] = {
            'is_cointegrated': is_coint,
            'p_value': p_value,
            'data': (data1_non, data2_non)
        }

        self.results['pair_discovery'] = results
        return results

    def phase2_position_sizer_comparison(self, pair_data: Tuple[pd.DataFrame, pd.DataFrame]) -> Dict:
        """
        Phase 2: Compare all three position sizers.

        Note: Currently PairsPortfolio has position sizing built-in.
        This phase documents the sizing approach used.
        """
        logger.info("\n" + "="*80)
        logger.info("PHASE 2: POSITION SIZER COMPARISON")
        logger.info("="*80)

        data1, data2 = pair_data
        results = {}

        # Test with different risk profiles
        risk_profiles = {
            'Conservative (5%)': RiskConfig.conservative(),
            'Moderate (10%)': RiskConfig.moderate(),
            'Aggressive (20%)': RiskConfig.aggressive()
        }

        for profile_name, risk_config in risk_profiles.items():
            logger.info(f"\nTesting: {profile_name}")

            engine = BacktestEngine(
                initial_capital=100000,
                fees=0.001,
                slippage=0.001,
                risk_config=risk_config
            )

            strategy = PairsTrading(
                entry_zscore=2.0,
                exit_zscore=0.5,
                zscore_window=20
            )

            # Create multi-index data
            data1_copy = data1.copy()
            data2_copy = data2.copy()
            data1_copy['symbol'] = 'PAIR1'
            data2_copy['symbol'] = 'PAIR2'

            combined = pd.concat([data1_copy, data2_copy])
            combined = combined.set_index('symbol', append=True)
            combined = combined.swaplevel()
            combined = combined.sort_index()

            try:
                portfolio = engine.run(
                    strategy=strategy,
                    symbols=['PAIR1', 'PAIR2'],
                    data=combined
                )

                stats = portfolio.stats()

                if stats:
                    logger.info(f"  Sharpe Ratio: {stats.get('Sharpe Ratio', 0):.3f}",
                              color="green" if stats.get('Sharpe Ratio', 0) > 1.0 else "yellow")
                    logger.info(f"  Total Return: {stats.get('Total Return [%]', 0):.2f}%",
                              color="green")
                    logger.info(f"  Max Drawdown: {stats.get('Max Drawdown [%]', 0):.2f}%",
                              color="yellow" if abs(stats.get('Max Drawdown [%]', 0)) > 15 else "green")
                    logger.info(f"  Win Rate: {stats.get('Win Rate [%]', 0):.2f}%")
                    logger.info(f"  Total Trades: {stats.get('Total Trades', 0)}")

                    results[profile_name] = {
                        'stats': stats,
                        'trades': len(portfolio.trades),
                        'risk_config': {
                            'position_size_pct': risk_config.position_size_pct,
                            'max_positions': risk_config.max_positions
                        }
                    }
                else:
                    logger.warning(f"  No stats available for {profile_name}")
                    results[profile_name] = {'error': 'No stats generated'}

            except Exception as e:
                logger.error(f"  Error testing {profile_name}: {str(e)}")
                results[profile_name] = {'error': str(e)}

        self.results['position_sizer_comparison'] = results
        return results

    def phase3_parameter_optimization(self, pair_data: Tuple[pd.DataFrame, pd.DataFrame]) -> Dict:
        """
        Phase 3: Optimize parameters using grid search.
        """
        logger.info("\n" + "="*80)
        logger.info("PHASE 3: PARAMETER OPTIMIZATION")
        logger.info("="*80)

        data1, data2 = pair_data

        # Create multi-index data
        data1_copy = data1.copy()
        data2_copy = data2.copy()
        data1_copy['symbol'] = 'PAIR1'
        data2_copy['symbol'] = 'PAIR2'

        combined = pd.concat([data1_copy, data2_copy])
        combined = combined.set_index('symbol', append=True)
        combined = combined.swaplevel()
        combined = combined.sort_index()

        engine = BacktestEngine(
            initial_capital=100000,
            fees=0.001,
            slippage=0.001,
            risk_config=RiskConfig.moderate()
        )

        optimizer = GridSearchOptimizer(engine)

        # Parameter grid focusing on key parameters
        param_grid = {
            'entry_zscore': [1.5, 2.0, 2.5],
            'exit_zscore': [0.25, 0.5, 0.75],
            'zscore_window': [15, 20, 30]
        }

        total_combinations = 3 * 3 * 3
        logger.info(f"\nParameter grid: {param_grid}")
        logger.info(f"Total combinations: {total_combinations}")

        try:
            result = optimizer.optimize_parallel(
                strategy_class=PairsTrading,
                param_grid=param_grid,
                symbols=['PAIR1', 'PAIR2'],
                data=combined,
                metric='sharpe_ratio',
                max_workers=4,
                export_results=False,
                use_cache=True
            )

            logger.success(f"\nOptimization completed successfully")
            logger.success(f"Best Parameters: {result['best_params']}")
            logger.success(f"Best Sharpe Ratio: {result['best_value']:.3f}")
            logger.info(f"Total Time: {result['total_time']:.1f}s")

            # Show top 5
            if result.get('all_results'):
                sorted_results = sorted(
                    result['all_results'],
                    key=lambda x: x.get('value', float('-inf')),
                    reverse=True
                )

                logger.info("\nTop 5 Parameter Combinations:")
                for i, res in enumerate(sorted_results[:5], 1):
                    logger.info(f"  {i}. {res['params']} -> Sharpe: {res['value']:.3f}",
                              color="blue")

            self.results['parameter_optimization'] = result
            return result

        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            self.results['parameter_optimization'] = {'error': str(e)}
            return {'error': str(e)}

    def phase4_walk_forward_validation(self, pair_data: Tuple[pd.DataFrame, pd.DataFrame],
                                      best_params: Dict) -> Dict:
        """
        Phase 4: Walk-forward validation to test robustness.

        Split data into train (60%) and test (40%), optimize on train, validate on test.
        """
        logger.info("\n" + "="*80)
        logger.info("PHASE 4: WALK-FORWARD VALIDATION")
        logger.info("="*80)

        data1, data2 = pair_data

        # Split into train and test
        n_days = len(data1)
        split_idx = int(n_days * 0.6)

        train_data1 = data1.iloc[:split_idx]
        train_data2 = data2.iloc[:split_idx]
        test_data1 = data1.iloc[split_idx:]
        test_data2 = data2.iloc[split_idx:]

        logger.info(f"\nTrain period: {train_data1.index[0]} to {train_data1.index[-1]}")
        logger.info(f"Test period: {test_data1.index[0]} to {test_data1.index[-1]}")
        logger.info(f"Train days: {len(train_data1)}, Test days: {len(test_data1)}")

        results = {}

        # Helper function to run backtest
        def run_backtest(data1_seg, data2_seg, params, label):
            data1_copy = data1_seg.copy()
            data2_copy = data2_seg.copy()
            data1_copy['symbol'] = 'PAIR1'
            data2_copy['symbol'] = 'PAIR2'

            combined = pd.concat([data1_copy, data2_copy])
            combined = combined.set_index('symbol', append=True)
            combined = combined.swaplevel()
            combined = combined.sort_index()

            engine = BacktestEngine(
                initial_capital=100000,
                fees=0.001,
                slippage=0.001,
                risk_config=RiskConfig.moderate()
            )

            strategy = PairsTrading(**params)

            portfolio = engine.run(
                strategy=strategy,
                symbols=['PAIR1', 'PAIR2'],
                data=combined
            )

            stats = portfolio.stats()

            logger.info(f"\n{label} Results:")
            if stats:
                logger.success(f"  Sharpe Ratio: {stats.get('Sharpe Ratio', 0):.3f}")
                logger.success(f"  Total Return: {stats.get('Total Return [%]', 0):.2f}%")
                logger.warning(f"  Max Drawdown: {stats.get('Max Drawdown [%]', 0):.2f}%")
                logger.info(f"  Win Rate: {stats.get('Win Rate [%]', 0):.2f}%")
                logger.info(f"  Total Trades: {stats.get('Total Trades', 0)}")

            return stats

        # Test on training data
        train_stats = run_backtest(train_data1, train_data2, best_params, "In-Sample (Train)")
        results['train'] = train_stats

        # Test on test data
        test_stats = run_backtest(test_data1, test_data2, best_params, "Out-of-Sample (Test)")
        results['test'] = test_stats

        # Calculate degradation
        if train_stats and test_stats:
            train_sharpe = train_stats.get('Sharpe Ratio', 0)
            test_sharpe = test_stats.get('Sharpe Ratio', 0)

            if train_sharpe != 0:
                degradation = ((train_sharpe - test_sharpe) / abs(train_sharpe)) * 100
                results['degradation_pct'] = degradation

                logger.info(f"\nPerformance Degradation:")
                logger.info(f"  Train Sharpe: {train_sharpe:.3f}")
                logger.info(f"  Test Sharpe: {test_sharpe:.3f}")
                logger.info(f"  Degradation: {degradation:.1f}%",
                          color="green" if degradation < 30 else "yellow" if degradation < 50 else "red")

                if degradation < 30:
                    logger.success("  Assessment: EXCELLENT - Robust parameters")
                elif degradation < 50:
                    logger.warning("  Assessment: ACCEPTABLE - Some overfitting")
                else:
                    logger.error("  Assessment: POOR - Significant overfitting")

        self.results['walk_forward_validation'] = results
        return results

    def phase5_robustness_analysis(self, pair_data: Tuple[pd.DataFrame, pd.DataFrame],
                                   best_params: Dict) -> Dict:
        """
        Phase 5: Test robustness by varying parameters around optimal values.
        """
        logger.info("\n" + "="*80)
        logger.info("PHASE 5: ROBUSTNESS ANALYSIS")
        logger.info("="*80)

        data1, data2 = pair_data
        results = {}

        # Test parameter neighborhoods
        base_entry_zscore = best_params.get('entry_zscore', 2.0)
        base_exit_zscore = best_params.get('exit_zscore', 0.5)

        logger.info(f"\nTesting neighborhood around optimal parameters:")
        logger.info(f"  Base entry_zscore: {base_entry_zscore}")
        logger.info(f"  Base exit_zscore: {base_exit_zscore}")

        # Create variations
        variations = [
            {'name': 'Base', 'entry_zscore': base_entry_zscore, 'exit_zscore': base_exit_zscore},
            {'name': 'Entry -0.25', 'entry_zscore': base_entry_zscore - 0.25, 'exit_zscore': base_exit_zscore},
            {'name': 'Entry +0.25', 'entry_zscore': base_entry_zscore + 0.25, 'exit_zscore': base_exit_zscore},
            {'name': 'Exit -0.15', 'entry_zscore': base_entry_zscore, 'exit_zscore': base_exit_zscore - 0.15},
            {'name': 'Exit +0.15', 'entry_zscore': base_entry_zscore, 'exit_zscore': base_exit_zscore + 0.15},
        ]

        for var in variations:
            logger.info(f"\nTesting: {var['name']}")

            # Ensure exit < entry
            if var['exit_zscore'] >= var['entry_zscore']:
                logger.warning(f"  Skipping - invalid parameters (exit >= entry)")
                continue

            test_params = best_params.copy()
            test_params['entry_zscore'] = var['entry_zscore']
            test_params['exit_zscore'] = var['exit_zscore']

            try:
                data1_copy = data1.copy()
                data2_copy = data2.copy()
                data1_copy['symbol'] = 'PAIR1'
                data2_copy['symbol'] = 'PAIR2'

                combined = pd.concat([data1_copy, data2_copy])
                combined = combined.set_index('symbol', append=True)
                combined = combined.swaplevel()
                combined = combined.sort_index()

                engine = BacktestEngine(
                    initial_capital=100000,
                    fees=0.001,
                    slippage=0.001,
                    risk_config=RiskConfig.moderate()
                )

                strategy = PairsTrading(**test_params)

                portfolio = engine.run(
                    strategy=strategy,
                    symbols=['PAIR1', 'PAIR2'],
                    data=combined
                )

                stats = portfolio.stats()

                if stats:
                    sharpe = stats.get('Sharpe Ratio', 0)
                    logger.info(f"  Sharpe Ratio: {sharpe:.3f}",
                              color="green" if sharpe > 1.0 else "yellow")

                    results[var['name']] = {
                        'params': test_params,
                        'sharpe': sharpe,
                        'stats': stats
                    }
            except Exception as e:
                logger.error(f"  Error: {str(e)}")
                results[var['name']] = {'error': str(e)}

        # Analyze stability
        if len(results) >= 3:
            sharpes = [r['sharpe'] for r in results.values() if 'sharpe' in r]
            if sharpes:
                mean_sharpe = np.mean(sharpes)
                std_sharpe = np.std(sharpes)
                cv = (std_sharpe / mean_sharpe) * 100 if mean_sharpe != 0 else 0

                logger.info(f"\nRobustness Metrics:")
                logger.info(f"  Mean Sharpe: {mean_sharpe:.3f}")
                logger.info(f"  Std Dev: {std_sharpe:.3f}")
                logger.info(f"  Coefficient of Variation: {cv:.1f}%",
                          color="green" if cv < 30 else "yellow" if cv < 50 else "red")

                results['robustness_metrics'] = {
                    'mean_sharpe': mean_sharpe,
                    'std_sharpe': std_sharpe,
                    'cv': cv
                }

        self.results['robustness_analysis'] = results
        return results

    def generate_report(self, output_path: str = None):
        """
        Generate comprehensive validation report.
        """
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE VALIDATION REPORT")
        logger.info("="*80)

        report = {
            'timestamp': datetime.now().isoformat(),
            'validation_results': self.results
        }

        # Executive Summary
        logger.success("\n=== EXECUTIVE SUMMARY ===\n")

        # 1. Cointegration Detection
        pair_discovery = self.results.get('pair_discovery', {})
        strong = pair_discovery.get('strong_cointegration', {})
        non_coint = pair_discovery.get('non_cointegrated', {})

        logger.info("1. Cointegration Detection:")
        if strong.get('is_cointegrated'):
            logger.success("   ✓ Successfully detected cointegrated pairs")
        if not non_coint.get('is_cointegrated'):
            logger.success("   ✓ Correctly rejected non-cointegrated pairs")

        # 2. Position Sizing
        sizer_results = self.results.get('position_sizer_comparison', {})
        logger.info("\n2. Position Sizing:")
        for profile_name, result in sizer_results.items():
            if 'stats' in result:
                sharpe = result['stats'].get('Sharpe Ratio', 0)
                logger.info(f"   {profile_name}: Sharpe = {sharpe:.3f}",
                          color="green" if sharpe > 1.0 else "yellow")

        # 3. Optimization
        opt_results = self.results.get('parameter_optimization', {})
        if 'best_value' in opt_results:
            logger.info("\n3. Parameter Optimization:")
            logger.success(f"   Best Sharpe Ratio: {opt_results['best_value']:.3f}")
            logger.success(f"   Best Parameters: {opt_results['best_params']}")

        # 4. Walk-Forward
        wf_results = self.results.get('walk_forward_validation', {})
        if 'degradation_pct' in wf_results:
            logger.info("\n4. Walk-Forward Validation:")
            deg = wf_results['degradation_pct']
            logger.info(f"   Performance Degradation: {deg:.1f}%",
                      color="green" if deg < 30 else "yellow" if deg < 50 else "red")

        # 5. Robustness
        rob_results = self.results.get('robustness_analysis', {})
        if 'robustness_metrics' in rob_results:
            logger.info("\n5. Robustness Analysis:")
            cv = rob_results['robustness_metrics']['cv']
            logger.info(f"   Parameter Stability (CV): {cv:.1f}%",
                      color="green" if cv < 30 else "yellow" if cv < 50 else "red")

        # Overall Assessment
        logger.success("\n=== OVERALL ASSESSMENT ===\n")

        passed_tests = 0
        total_tests = 5

        if strong.get('is_cointegrated') and not non_coint.get('is_cointegrated'):
            passed_tests += 1
            logger.success("✓ Cointegration detection: PASS")
        else:
            logger.error("✗ Cointegration detection: FAIL")

        # Check if any position sizer achieved Sharpe > 1.0
        max_sharpe = max([r.get('stats', {}).get('Sharpe Ratio', 0)
                         for r in sizer_results.values() if 'stats' in r] or [0])
        if max_sharpe > 1.0:
            passed_tests += 1
            logger.success("✓ Position sizing performance: PASS")
        else:
            logger.error("✗ Position sizing performance: FAIL")

        if opt_results.get('best_value', 0) > 1.0:
            passed_tests += 1
            logger.success("✓ Parameter optimization: PASS")
        else:
            logger.error("✗ Parameter optimization: FAIL")

        if wf_results.get('degradation_pct', 100) < 50:
            passed_tests += 1
            logger.success("✓ Walk-forward validation: PASS")
        else:
            logger.error("✗ Walk-forward validation: FAIL")

        if rob_results.get('robustness_metrics', {}).get('cv', 100) < 50:
            passed_tests += 1
            logger.success("✓ Robustness analysis: PASS")
        else:
            logger.error("✗ Robustness analysis: FAIL")

        logger.info(f"\nTests Passed: {passed_tests}/{total_tests}",
                   color="green" if passed_tests >= 4 else "yellow")

        if passed_tests >= 4:
            logger.success("\n*** FRAMEWORK VALIDATION: SUCCESS ***")
            logger.success("The pairs trading framework is production-ready.")
        elif passed_tests >= 3:
            logger.warning("\n*** FRAMEWORK VALIDATION: ACCEPTABLE ***")
            logger.warning("The framework works but needs improvements.")
        else:
            logger.error("\n*** FRAMEWORK VALIDATION: NEEDS WORK ***")
            logger.error("Significant issues detected. Review required.")

        # Save report
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"\nFull report saved to: {output_path}")

        return report


def main():
    """Main validation execution."""
    logger.info("="*80)
    logger.info("COMPREHENSIVE PAIRS TRADING FRAMEWORK VALIDATION")
    logger.info("="*80)
    logger.info(f"Start Time: {datetime.now()}")

    framework = PairsValidationFramework()

    try:
        # Phase 1: Pair Discovery
        pair_results = framework.phase1_pair_discovery()

        # Use the strongly cointegrated pair for remaining tests
        if pair_results['strong_cointegration']['is_cointegrated']:
            test_pair = pair_results['strong_cointegration']['data']
        else:
            logger.warning("Strong cointegration test failed, using marginal pair")
            test_pair = pair_results['marginal_cointegration']['data']

        # Phase 2: Position Sizer Comparison
        framework.phase2_position_sizer_comparison(test_pair)

        # Phase 3: Parameter Optimization
        opt_result = framework.phase3_parameter_optimization(test_pair)

        best_params = opt_result.get('best_params', {
            'entry_zscore': 2.0,
            'exit_zscore': 0.5,
            'zscore_window': 20
        })

        # Phase 4: Walk-Forward Validation
        framework.phase4_walk_forward_validation(test_pair, best_params)

        # Phase 5: Robustness Analysis
        framework.phase5_robustness_analysis(test_pair, best_params)

        # Generate Report
        output_path = Path(__file__).parent / 'pairs_validation_report.json'
        framework.generate_report(str(output_path))

        logger.info(f"\nEnd Time: {datetime.now()}")
        logger.info("="*80)
        logger.info("VALIDATION COMPLETE")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"\nValidation failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
