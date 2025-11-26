"""
Comprehensive Validation of Pairs Trading Framework - Version 2.

This version directly uses PairsPortfolio for testing with synthetic data,
bypassing the BacktestEngine to have full control over inputs.

Usage:
    From project root:
    C:\\Users\\qwqw1\\anaconda3\\envs\\fintech\\python.exe backtest_scripts/comprehensive_pairs_validation_v2.py
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

from strategies.advanced.pairs_trading import PairsTrading
from backtesting.engine.pairs_portfolio import PairsPortfolio
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
            'baseline_performance': {},
            'parameter_sensitivity': {},
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
        logger.blank()
        logger.info("Test 1: Strongly Cointegrated Pair")
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

        if is_coint:
            logger.success(f"  Cointegrated: {is_coint}")
        else:
            logger.error(f"  Cointegrated: {is_coint}")

        if p_value < 0.05:
            logger.success(f"  P-value: {p_value:.4f}")
        else:
            logger.warning(f"  P-value: {p_value:.4f}")

        results['strong_cointegration'] = {
            'is_cointegrated': is_coint,
            'p_value': p_value,
            'data': (data1_strong, data2_strong)
        }

        # Test 2: Marginally cointegrated pair
        logger.blank()
        logger.info("Test 2: Marginally Cointegrated Pair")
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

        if is_coint:
            logger.success(f"  Cointegrated: {is_coint}")
        else:
            logger.warning(f"  Cointegrated: {is_coint}")

        if p_value < 0.05:
            logger.success(f"  P-value: {p_value:.4f}")
        else:
            logger.warning(f"  P-value: {p_value:.4f}")

        results['marginal_cointegration'] = {
            'is_cointegrated': is_coint,
            'p_value': p_value,
            'data': (data1_marginal, data2_marginal)
        }

        # Test 3: Non-cointegrated pair (should not trade)
        logger.blank()
        logger.info("Test 3: Non-Cointegrated Pair (Control)")
        data1_non, data2_non = generator.generate_non_cointegrated_pair(
            n_days=500,
            seed=44
        )

        is_coint, p_value = strategy.test_cointegration(
            data1_non['close'],
            data2_non['close']
        )

        if not is_coint:
            logger.success(f"  Cointegrated: {is_coint} (correctly rejected)")
        else:
            logger.error(f"  Cointegrated: {is_coint} (should be False)")

        if p_value >= 0.05:
            logger.success(f"  P-value: {p_value:.4f}")
        else:
            logger.error(f"  P-value: {p_value:.4f}")

        results['non_cointegrated'] = {
            'is_cointegrated': is_coint,
            'p_value': p_value,
            'data': (data1_non, data2_non)
        }

        self.results['pair_discovery'] = results
        return results

    def phase2_baseline_performance(self, pair_data: Tuple[pd.DataFrame, pd.DataFrame]) -> Dict:
        """
        Phase 2: Test baseline performance with different risk profiles.
        """
        logger.blank()
        logger.separator("=", 80)
        logger.header("PHASE 2: BASELINE PERFORMANCE TESTING")
        logger.separator("=", 80)

        data1, data2 = pair_data
        results = {}

        # Generate signals once
        strategy = PairsTrading(
            entry_zscore=2.0,
            exit_zscore=0.5,
            zscore_window=20
        )

        signals_dict = strategy.generate_pairs_signals(data1, data2, 'PAIR1', 'PAIR2')

        # Extract signals for symbol1 (strategy generates swapped signals)
        long_entries1, long_exits1, short_entries1, short_exits1 = signals_dict['PAIR1']

        # Test with different risk profiles
        risk_profiles = {
            'Conservative (5%)': RiskConfig.conservative(),
            'Moderate (10%)': RiskConfig.moderate(),
            'Aggressive (20%)': RiskConfig.aggressive()
        }

        for profile_name, risk_config in risk_profiles.items():
            logger.blank()
            logger.info(f"Testing: {profile_name}")

            try:
                # PairsPortfolio expects:
                # - entries = long spread entries (short sym1, long sym2)
                # - short_entries = short spread entries (long sym1, short sym2)
                # Our strategy returns these swapped, so:
                # - short_entries1 -> long spread
                # - long_entries1 -> short spread

                portfolio = PairsPortfolio(
                    symbols=('PAIR1', 'PAIR2'),
                    prices1=data1['close'],
                    prices2=data2['close'],
                    entries=short_entries1,  # Long spread entries
                    exits=short_exits1,      # Long spread exits
                    short_entries=long_entries1,  # Short spread entries
                    short_exits=long_exits1,      # Short spread exits
                    init_cash=100000,
                    fees=0.001,
                    slippage=0.001,
                    freq='1D',
                    market_hours_only=False,  # Disable for daily data
                    risk_config=risk_config,
                    price_data1=data1,
                    price_data2=data2
                )

                stats = portfolio.stats()

                if stats:
                    sharpe = stats.get('Sharpe Ratio', 0)
                    returns = stats.get('Total Return [%]', 0)
                    drawdown = stats.get('Max Drawdown [%]', 0)
                    win_rate = stats.get('Win Rate [%]', 0)
                    trades = stats.get('Total Trades', 0)

                    if sharpe > 1.0:
                        logger.success(f"  Sharpe Ratio: {sharpe:.3f}")
                    else:
                        logger.warning(f"  Sharpe Ratio: {sharpe:.3f}")

                    logger.info(f"  Total Return: {returns:.2f}%")

                    if abs(drawdown) > 15:
                        logger.warning(f"  Max Drawdown: {drawdown:.2f}%")
                    else:
                        logger.success(f"  Max Drawdown: {drawdown:.2f}%")

                    logger.info(f"  Win Rate: {win_rate:.2f}%")
                    logger.info(f"  Total Trades: {trades}")

                    results[profile_name] = {
                        'stats': stats,
                        'trades': trades,
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

        self.results['baseline_performance'] = results
        return results

    def phase3_parameter_sensitivity(self, pair_data: Tuple[pd.DataFrame, pd.DataFrame]) -> Dict:
        """
        Phase 3: Test parameter sensitivity (manual grid search).
        """
        logger.blank()
        logger.separator("=", 80)
        logger.header("PHASE 3: PARAMETER SENSITIVITY ANALYSIS")
        logger.separator("=", 80)

        data1, data2 = pair_data
        results = {}

        # Parameter grid
        param_grid = {
            'entry_zscore': [1.5, 2.0, 2.5],
            'exit_zscore': [0.25, 0.5, 0.75],
            'zscore_window': [15, 20, 30]
        }

        total_combinations = len(param_grid['entry_zscore']) * len(param_grid['exit_zscore']) * len(param_grid['zscore_window'])
        logger.blank()
        logger.info(f"Parameter grid: {param_grid}")
        logger.info(f"Total combinations: {total_combinations}")

        best_sharpe = float('-inf')
        best_params = {}
        all_results = []

        risk_config = RiskConfig.moderate()

        for entry_z in param_grid['entry_zscore']:
            for exit_z in param_grid['exit_zscore']:
                for zscore_win in param_grid['zscore_window']:
                    # Skip invalid combinations
                    if exit_z >= entry_z:
                        continue

                    params = {
                        'entry_zscore': entry_z,
                        'exit_zscore': exit_z,
                        'zscore_window': zscore_win
                    }

                    try:
                        strategy = PairsTrading(**params)
                        signals_dict = strategy.generate_pairs_signals(data1, data2, 'PAIR1', 'PAIR2')

                        long_entries1, long_exits1, short_entries1, short_exits1 = signals_dict['PAIR1']

                        portfolio = PairsPortfolio(
                            symbols=('PAIR1', 'PAIR2'),
                            prices1=data1['close'],
                            prices2=data2['close'],
                            entries=short_entries1,
                            exits=short_exits1,
                            short_entries=long_entries1,
                            short_exits=long_exits1,
                            init_cash=100000,
                            fees=0.001,
                            slippage=0.001,
                            freq='1D',
                            market_hours_only=False,
                            risk_config=risk_config,
                            price_data1=data1,
                            price_data2=data2
                        )

                        stats = portfolio.stats()

                        if stats:
                            sharpe = stats.get('Sharpe Ratio', 0)
                            all_results.append({
                                'params': params,
                                'sharpe': sharpe,
                                'stats': stats
                            })

                            if sharpe > best_sharpe:
                                best_sharpe = sharpe
                                best_params = params.copy()

                    except Exception as e:
                        logger.warning(f"  Error with params {params}: {str(e)}")

        # Show results
        logger.blank()
        if all_results:
            logger.success(f"Optimization completed successfully")
            logger.success(f"Best Parameters: {best_params}")
            logger.success(f"Best Sharpe Ratio: {best_sharpe:.3f}")

            # Sort and show top 5
            sorted_results = sorted(all_results, key=lambda x: x['sharpe'], reverse=True)
            logger.blank()
            logger.info("Top 5 Parameter Combinations:")
            for i, res in enumerate(sorted_results[:5], 1):
                logger.info(f"  {i}. {res['params']} -> Sharpe: {res['sharpe']:.3f}")

            results = {
                'best_params': best_params,
                'best_sharpe': best_sharpe,
                'all_results': all_results
            }
        else:
            logger.error("No valid results from parameter search")
            results = {'error': 'No valid results'}

        self.results['parameter_sensitivity'] = results
        return results

    def phase4_walk_forward_validation(self, pair_data: Tuple[pd.DataFrame, pd.DataFrame],
                                      best_params: Dict) -> Dict:
        """
        Phase 4: Walk-forward validation to test robustness.
        """
        logger.blank()
        logger.separator("=", 80)
        logger.header("PHASE 4: WALK-FORWARD VALIDATION")
        logger.separator("=", 80)

        data1, data2 = pair_data

        # Split into train (60%) and test (40%)
        n_days = len(data1)
        split_idx = int(n_days * 0.6)

        train_data1 = data1.iloc[:split_idx]
        train_data2 = data2.iloc[:split_idx]
        test_data1 = data1.iloc[split_idx:]
        test_data2 = data2.iloc[split_idx:]

        logger.blank()
        logger.info(f"Train period: {train_data1.index[0]} to {train_data1.index[-1]}")
        logger.info(f"Test period: {test_data1.index[0]} to {test_data1.index[-1]}")
        logger.info(f"Train days: {len(train_data1)}, Test days: {len(test_data1)}")

        results = {}
        risk_config = RiskConfig.moderate()

        # Test on training data
        logger.blank()
        logger.info("In-Sample (Train) Results:")
        try:
            strategy = PairsTrading(**best_params)
            signals_dict = strategy.generate_pairs_signals(train_data1, train_data2, 'PAIR1', 'PAIR2')
            long_entries1, long_exits1, short_entries1, short_exits1 = signals_dict['PAIR1']

            train_portfolio = PairsPortfolio(
                symbols=('PAIR1', 'PAIR2'),
                prices1=train_data1['close'],
                prices2=train_data2['close'],
                entries=short_entries1,
                exits=short_exits1,
                short_entries=long_entries1,
                short_exits=long_exits1,
                init_cash=100000,
                fees=0.001,
                slippage=0.001,
                freq='1D',
                market_hours_only=False,
                risk_config=risk_config,
                price_data1=train_data1,
                price_data2=train_data2
            )

            train_stats = train_portfolio.stats()
            if train_stats:
                logger.success(f"  Sharpe Ratio: {train_stats.get('Sharpe Ratio', 0):.3f}")
                logger.info(f"  Total Return: {train_stats.get('Total Return [%]', 0):.2f}%")
                logger.info(f"  Max Drawdown: {train_stats.get('Max Drawdown [%]', 0):.2f}%")
                logger.info(f"  Total Trades: {train_stats.get('Total Trades', 0)}")
                results['train'] = train_stats
        except Exception as e:
            logger.error(f"  Error in train period: {str(e)}")
            results['train'] = None

        # Test on test data
        logger.blank()
        logger.info("Out-of-Sample (Test) Results:")
        try:
            strategy = PairsTrading(**best_params)
            signals_dict = strategy.generate_pairs_signals(test_data1, test_data2, 'PAIR1', 'PAIR2')
            long_entries1, long_exits1, short_entries1, short_exits1 = signals_dict['PAIR1']

            test_portfolio = PairsPortfolio(
                symbols=('PAIR1', 'PAIR2'),
                prices1=test_data1['close'],
                prices2=test_data2['close'],
                entries=short_entries1,
                exits=short_exits1,
                short_entries=long_entries1,
                short_exits=long_exits1,
                init_cash=100000,
                fees=0.001,
                slippage=0.001,
                freq='1D',
                market_hours_only=False,
                risk_config=risk_config,
                price_data1=test_data1,
                price_data2=test_data2
            )

            test_stats = test_portfolio.stats()
            if test_stats:
                logger.success(f"  Sharpe Ratio: {test_stats.get('Sharpe Ratio', 0):.3f}")
                logger.info(f"  Total Return: {test_stats.get('Total Return [%]', 0):.2f}%")
                logger.info(f"  Max Drawdown: {test_stats.get('Max Drawdown [%]', 0):.2f}%")
                logger.info(f"  Total Trades: {test_stats.get('Total Trades', 0)}")
                results['test'] = test_stats
        except Exception as e:
            logger.error(f"  Error in test period: {str(e)}")
            results['test'] = None

        # Calculate degradation
        if results.get('train') and results.get('test'):
            train_sharpe = results['train'].get('Sharpe Ratio', 0)
            test_sharpe = results['test'].get('Sharpe Ratio', 0)

            if train_sharpe != 0:
                degradation = ((train_sharpe - test_sharpe) / abs(train_sharpe)) * 100
                results['degradation_pct'] = degradation

                logger.blank()
                logger.info("Performance Degradation:")
                logger.info(f"  Train Sharpe: {train_sharpe:.3f}")
                logger.info(f"  Test Sharpe: {test_sharpe:.3f}")

                if degradation < 30:
                    logger.success(f"  Degradation: {degradation:.1f}% - EXCELLENT")
                elif degradation < 50:
                    logger.warning(f"  Degradation: {degradation:.1f}% - ACCEPTABLE")
                else:
                    logger.error(f"  Degradation: {degradation:.1f}% - POOR")

        self.results['walk_forward_validation'] = results
        return results

    def phase5_robustness_analysis(self, pair_data: Tuple[pd.DataFrame, pd.DataFrame],
                                   best_params: Dict) -> Dict:
        """
        Phase 5: Test robustness by varying parameters around optimal values.
        """
        logger.blank()
        logger.separator("=", 80)
        logger.header("PHASE 5: ROBUSTNESS ANALYSIS")
        logger.separator("=", 80)

        data1, data2 = pair_data
        results = {}

        base_entry_zscore = best_params.get('entry_zscore', 2.0)
        base_exit_zscore = best_params.get('exit_zscore', 0.5)

        logger.blank()
        logger.info(f"Testing neighborhood around optimal parameters:")
        logger.info(f"  Base entry_zscore: {base_entry_zscore}")
        logger.info(f"  Base exit_zscore: {base_exit_zscore}")

        variations = [
            {'name': 'Base', 'entry_zscore': base_entry_zscore, 'exit_zscore': base_exit_zscore},
            {'name': 'Entry -0.25', 'entry_zscore': base_entry_zscore - 0.25, 'exit_zscore': base_exit_zscore},
            {'name': 'Entry +0.25', 'entry_zscore': base_entry_zscore + 0.25, 'exit_zscore': base_exit_zscore},
            {'name': 'Exit -0.15', 'entry_zscore': base_entry_zscore, 'exit_zscore': base_exit_zscore - 0.15},
            {'name': 'Exit +0.15', 'entry_zscore': base_entry_zscore, 'exit_zscore': base_exit_zscore + 0.15},
        ]

        risk_config = RiskConfig.moderate()

        for var in variations:
            logger.blank()
            logger.info(f"Testing: {var['name']}")

            if var['exit_zscore'] >= var['entry_zscore']:
                logger.warning(f"  Skipping - invalid parameters")
                continue

            test_params = best_params.copy()
            test_params['entry_zscore'] = var['entry_zscore']
            test_params['exit_zscore'] = var['exit_zscore']

            try:
                strategy = PairsTrading(**test_params)
                signals_dict = strategy.generate_pairs_signals(data1, data2, 'PAIR1', 'PAIR2')
                long_entries1, long_exits1, short_entries1, short_exits1 = signals_dict['PAIR1']

                portfolio = PairsPortfolio(
                    symbols=('PAIR1', 'PAIR2'),
                    prices1=data1['close'],
                    prices2=data2['close'],
                    entries=short_entries1,
                    exits=short_exits1,
                    short_entries=long_entries1,
                    short_exits=long_exits1,
                    init_cash=100000,
                    fees=0.001,
                    slippage=0.001,
                    freq='1D',
                    market_hours_only=False,
                    risk_config=risk_config,
                    price_data1=data1,
                    price_data2=data2
                )

                stats = portfolio.stats()

                if stats:
                    sharpe = stats.get('Sharpe Ratio', 0)
                    if sharpe > 1.0:
                        logger.success(f"  Sharpe Ratio: {sharpe:.3f}")
                    else:
                        logger.warning(f"  Sharpe Ratio: {sharpe:.3f}")

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

                logger.blank()
                logger.info("Robustness Metrics:")
                logger.info(f"  Mean Sharpe: {mean_sharpe:.3f}")
                logger.info(f"  Std Dev: {std_sharpe:.3f}")

                if cv < 30:
                    logger.success(f"  Coefficient of Variation: {cv:.1f}% - ROBUST")
                elif cv < 50:
                    logger.warning(f"  Coefficient of Variation: {cv:.1f}% - ACCEPTABLE")
                else:
                    logger.error(f"  Coefficient of Variation: {cv:.1f}% - UNSTABLE")

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
        logger.blank()
        logger.separator("=", 80)
        logger.header("COMPREHENSIVE VALIDATION REPORT")
        logger.separator("=", 80)

        report = {
            'timestamp': datetime.now().isoformat(),
            'validation_results': self.results
        }

        # Executive Summary
        logger.blank()
        logger.success("=== EXECUTIVE SUMMARY ===")

        # 1. Cointegration Detection
        pair_discovery = self.results.get('pair_discovery', {})
        strong = pair_discovery.get('strong_cointegration', {})
        non_coint = pair_discovery.get('non_cointegrated', {})

        logger.blank()
        logger.info("1. Cointegration Detection:")
        if strong.get('is_cointegrated'):
            logger.success("   [+] Successfully detected cointegrated pairs")
        if not non_coint.get('is_cointegrated'):
            logger.success("   [+] Correctly rejected non-cointegrated pairs")

        # 2. Baseline Performance
        baseline_results = self.results.get('baseline_performance', {})
        logger.blank()
        logger.info("2. Baseline Performance:")
        for profile_name, result in baseline_results.items():
            if 'stats' in result:
                sharpe = result['stats'].get('Sharpe Ratio', 0)
                if sharpe > 1.0:
                    logger.success(f"   {profile_name}: Sharpe = {sharpe:.3f}")
                else:
                    logger.warning(f"   {profile_name}: Sharpe = {sharpe:.3f}")

        # 3. Parameter Sensitivity
        sensitivity_results = self.results.get('parameter_sensitivity', {})
        if 'best_sharpe' in sensitivity_results:
            logger.blank()
            logger.info("3. Parameter Sensitivity:")
            logger.success(f"   Best Sharpe Ratio: {sensitivity_results['best_sharpe']:.3f}")
            logger.success(f"   Best Parameters: {sensitivity_results['best_params']}")

        # 4. Walk-Forward
        wf_results = self.results.get('walk_forward_validation', {})
        if 'degradation_pct' in wf_results:
            logger.blank()
            logger.info("4. Walk-Forward Validation:")
            deg = wf_results['degradation_pct']
            if deg < 30:
                logger.success(f"   Performance Degradation: {deg:.1f}%")
            elif deg < 50:
                logger.warning(f"   Performance Degradation: {deg:.1f}%")
            else:
                logger.error(f"   Performance Degradation: {deg:.1f}%")

        # 5. Robustness
        rob_results = self.results.get('robustness_analysis', {})
        if 'robustness_metrics' in rob_results:
            logger.blank()
            logger.info("5. Robustness Analysis:")
            cv = rob_results['robustness_metrics']['cv']
            if cv < 30:
                logger.success(f"   Parameter Stability (CV): {cv:.1f}%")
            elif cv < 50:
                logger.warning(f"   Parameter Stability (CV): {cv:.1f}%")
            else:
                logger.error(f"   Parameter Stability (CV): {cv:.1f}%")

        # Overall Assessment
        logger.blank()
        logger.success("=== OVERALL ASSESSMENT ===")
        logger.blank()

        passed_tests = 0
        total_tests = 5

        if strong.get('is_cointegrated') and not non_coint.get('is_cointegrated'):
            passed_tests += 1
            logger.success("[+] Cointegration detection: PASS")
        else:
            logger.error("[X] Cointegration detection: FAIL")

        max_sharpe = max([r.get('stats', {}).get('Sharpe Ratio', 0)
                         for r in baseline_results.values() if 'stats' in r] or [0])
        if max_sharpe > 1.0:
            passed_tests += 1
            logger.success("[+] Baseline performance: PASS")
        else:
            logger.error("[X] Baseline performance: FAIL")

        if sensitivity_results.get('best_sharpe', 0) > 1.0:
            passed_tests += 1
            logger.success("[+] Parameter sensitivity: PASS")
        else:
            logger.error("[X] Parameter sensitivity: FAIL")

        if wf_results.get('degradation_pct', 100) < 50:
            passed_tests += 1
            logger.success("[+] Walk-forward validation: PASS")
        else:
            logger.error("[X] Walk-forward validation: FAIL")

        if rob_results.get('robustness_metrics', {}).get('cv', 100) < 50:
            passed_tests += 1
            logger.success("[+] Robustness analysis: PASS")
        else:
            logger.error("[X] Robustness analysis: FAIL")

        logger.blank()
        if passed_tests >= 4:
            logger.success(f"Tests Passed: {passed_tests}/{total_tests}")
        else:
            logger.warning(f"Tests Passed: {passed_tests}/{total_tests}")

        logger.blank()
        if passed_tests >= 4:
            logger.success("*** FRAMEWORK VALIDATION: SUCCESS ***")
            logger.success("The pairs trading framework is production-ready.")
        elif passed_tests >= 3:
            logger.warning("*** FRAMEWORK VALIDATION: ACCEPTABLE ***")
            logger.warning("The framework works but needs improvements.")
        else:
            logger.error("*** FRAMEWORK VALIDATION: NEEDS WORK ***")
            logger.error("Significant issues detected. Review required.")

        # Save report
        if output_path:
            # Remove data objects for JSON serialization
            report_clean = {
                'timestamp': report['timestamp'],
                'validation_results': {}
            }

            for phase, phase_results in report['validation_results'].items():
                if isinstance(phase_results, dict):
                    report_clean['validation_results'][phase] = {}
                    for key, value in phase_results.items():
                        if isinstance(value, dict) and 'data' in value:
                            # Remove data tuples
                            clean_value = {k: v for k, v in value.items() if k != 'data'}
                            report_clean['validation_results'][phase][key] = clean_value
                        else:
                            report_clean['validation_results'][phase][key] = value

            with open(output_path, 'w') as f:
                json.dump(report_clean, f, indent=2, default=str)
            logger.blank()
            logger.info(f"Full report saved to: {output_path}")

        return report


def main():
    """Main validation execution."""
    logger.separator("=", 80)
    logger.header("COMPREHENSIVE PAIRS TRADING FRAMEWORK VALIDATION")
    logger.separator("=", 80)
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

        # Phase 2: Baseline Performance
        framework.phase2_baseline_performance(test_pair)

        # Phase 3: Parameter Sensitivity
        sensitivity_result = framework.phase3_parameter_sensitivity(test_pair)

        best_params = sensitivity_result.get('best_params', {
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

        logger.blank()
        logger.info(f"End Time: {datetime.now()}")
        logger.separator("=", 80)
        logger.header("VALIDATION COMPLETE")
        logger.separator("=", 80)

    except Exception as e:
        logger.error(f"\nValidation failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
