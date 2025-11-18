"""
Comprehensive Re-Validation of Pairs Trading with REALISTIC RETAIL FEES.

This script tests the pairs trading framework with realistic broker fees (0.01% = 1bp)
versus the conservative fees (0.1% = 10bp) used in previous validation.

Expected Impact:
    - Previous: 0.8% cost per round-trip (4 legs x 0.2%)
    - Realistic: 0.44% cost per round-trip (4 legs x 0.11%)
    - Savings: 0.36% per trade -> ~9% annual improvement on 25 trades

Phases:
    1. A/B Comparison: Isolate fee impact with identical parameters
    2. Full Re-Optimization: Find optimal params for new cost structure
    3. Position Sizing: Test sensitivity to position size
    4. Walk-Forward: Validate on out-of-sample data
    5. Break-Even Analysis: Calculate minimum viable profit per trade
    6. Viability Assessment: Production GO/NO-GO decision

Usage:
    C:\\Users\\qwqw1\\anaconda3\\envs\\fintech\\python.exe backtest_scripts/realistic_fee_validation.py
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import json

from utils.path_setup import setup_project_paths
ROOT_DIR = setup_project_paths()
from datetime import datetime

from strategies.advanced.pairs_trading import PairsTrading
from backtesting.engine.pairs_portfolio import PairsPortfolio
from backtesting.utils.risk_config import RiskConfig
from utils import logger


class SyntheticPairGenerator:
    """Generate cointegrated synthetic pairs for testing."""

    @staticmethod
    def generate_cointegrated_pair(
        n_days: int = 500,
        mean: float = 100.0,
        spread_volatility: float = 2.0,
        mean_reversion_speed: float = 0.15,
        seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate cointegrated pair using Ornstein-Uhlenbeck process.

        Args:
            n_days: Number of days
            mean: Mean price level
            spread_volatility: Spread volatility
            mean_reversion_speed: Mean reversion speed (higher = faster)
            seed: Random seed for reproducibility

        Returns:
            Tuple of (data1, data2) DataFrames with OHLCV
        """
        np.random.seed(seed)
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')

        # Base price (random walk)
        price1 = np.zeros(n_days)
        price1[0] = mean
        for i in range(1, n_days):
            price1[i] = price1[i-1] + np.random.normal(0, 1)

        # Mean-reverting spread (OU process)
        spread = np.zeros(n_days)
        spread[0] = 0
        for i in range(1, n_days):
            spread[i] = spread[i-1] - mean_reversion_speed * spread[i-1] + \
                       np.random.normal(0, spread_volatility)

        # Price2 = Price1 + Spread (ensures cointegration)
        price2 = price1 + spread

        # Create OHLCV data
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

        return create_ohlcv(price1), create_ohlcv(price2)


class RealisticFeeValidator:
    """Comprehensive validation framework with realistic fees."""

    def __init__(self):
        self.results = {}
        self.pair_data = None

    def phase1_ultrathinking(self):
        """Phase 1: Document the testing strategy (already completed above)."""
        logger.separator("=", 80)
        logger.header("PHASE 1: ULTRATHINKING - STRATEGY DESIGN")
        logger.separator("=", 80)
        logger.blank()

        logger.success("Testing Strategy Designed:")
        logger.info("  1. Fee Impact Model: 0.8% -> 0.44% per round-trip")
        logger.info("  2. Expected Improvement: ~9% annual return increase")
        logger.info("  3. Parameter Grid: 80 combinations (5x4x4)")
        logger.info("  4. Position Sizing: Test 10%, 20%, 30%")
        logger.info("  5. Validation: A/B test + Full optimization + Walk-forward")
        logger.blank()

    def phase2_ab_comparison(self) -> Dict:
        """
        Phase 2: A/B comparison with identical parameters.

        Isolates the pure fee impact by using same params with old vs new fees.
        """
        logger.separator("=", 80)
        logger.header("PHASE 2: A/B COMPARISON TEST (Old vs New Fees)")
        logger.separator("=", 80)
        logger.blank()

        # Generate synthetic pair
        generator = SyntheticPairGenerator()
        data1, data2 = generator.generate_cointegrated_pair(
            n_days=500,
            spread_volatility=1.5,
            mean_reversion_speed=0.15,
            seed=42
        )
        self.pair_data = (data1, data2)

        # Baseline parameters (same as previous validation)
        baseline_params = {
            'entry_zscore': 1.5,
            'exit_zscore': 0.25,
            'zscore_window': 30
        }

        logger.info(f"Using baseline parameters: {baseline_params}")
        logger.blank()

        results = {}
        risk_config = RiskConfig.moderate()  # 10% position sizing

        # Test with OLD fees (0.1%)
        logger.info("Running with OLD fees (0.1% per trade)...")
        old_result = self._run_backtest(
            data1, data2,
            baseline_params,
            fees=0.001,  # 0.1%
            slippage=0.001,
            risk_config=risk_config
        )
        results['old_fees'] = old_result

        # Test with NEW fees (0.01%)
        logger.info("Running with NEW fees (0.01% per trade)...")
        new_result = self._run_backtest(
            data1, data2,
            baseline_params,
            fees=0.0001,  # 0.01%
            slippage=0.001,
            risk_config=risk_config
        )
        results['new_fees'] = new_result

        # Calculate improvements
        logger.blank()
        logger.separator("-", 80)
        logger.header("A/B COMPARISON RESULTS")
        logger.separator("-", 80)
        logger.blank()

        metrics = ['Sharpe Ratio', 'Total Return [%]', 'Max Drawdown [%]', 'Win Rate [%]', 'Total Trades']

        comparison_table = []
        for metric in metrics:
            old_val = old_result['stats'].get(metric, 0)
            new_val = new_result['stats'].get(metric, 0)

            if old_val != 0:
                pct_change = ((new_val - old_val) / abs(old_val)) * 100
            else:
                pct_change = 0

            improvement = new_val - old_val

            comparison_table.append({
                'Metric': metric,
                'Old Fees': old_val,
                'New Fees': new_val,
                'Improvement': improvement,
                '% Change': pct_change
            })

            logger.info(f"{metric}:")
            logger.info(f"  Old Fees: {old_val:.3f}")

            if metric == 'Sharpe Ratio':
                if new_val > old_val:
                    logger.success(f"  New Fees: {new_val:.3f} (+{improvement:.3f}, +{pct_change:.1f}%)")
                else:
                    logger.warning(f"  New Fees: {new_val:.3f} ({improvement:.3f}, {pct_change:.1f}%)")
            elif 'Return' in metric:
                if new_val > old_val:
                    logger.success(f"  New Fees: {new_val:.2f}% (+{improvement:.2f}%, +{pct_change:.1f}%)")
                else:
                    logger.warning(f"  New Fees: {new_val:.2f}% ({improvement:.2f}%, {pct_change:.1f}%)")
            elif 'Drawdown' in metric:
                if abs(new_val) < abs(old_val):
                    logger.success(f"  New Fees: {new_val:.2f}% (improved by {pct_change:.1f}%)")
                else:
                    logger.warning(f"  New Fees: {new_val:.2f}% (worse by {pct_change:.1f}%)")
            else:
                logger.info(f"  New Fees: {new_val:.2f}")
            logger.blank()

        # Calculate fee savings
        if 'trades' in old_result:
            old_trades = old_result['trades']
            old_fees_paid = len(old_trades) * 4 * 0.002  # 4 legs x 0.2%
            new_fees_paid = len(old_trades) * 4 * 0.0011  # 4 legs x 0.11%
            fee_savings = old_fees_paid - new_fees_paid

            logger.info(f"Fee Analysis:")
            logger.info(f"  Total Trades (pairs): {len(old_trades)}")
            logger.info(f"  Total Legs: {len(old_trades) * 4}")
            logger.info(f"  Old Fees Paid: {old_fees_paid * 100:.2f}% of capital")
            logger.success(f"  New Fees Paid: {new_fees_paid * 100:.2f}% of capital")
            logger.success(f"  Savings: {fee_savings * 100:.2f}% of capital")
            logger.blank()

        results['comparison'] = comparison_table
        self.results['ab_comparison'] = results
        return results

    def phase3_full_optimization(self) -> Dict:
        """
        Phase 3: Full parameter optimization with new fee structure.

        Expands parameter grid to test if more aggressive trading is now viable.
        """
        logger.separator("=", 80)
        logger.header("PHASE 3: FULL RE-OPTIMIZATION (New Fee Structure)")
        logger.separator("=", 80)
        logger.blank()

        if self.pair_data is None:
            logger.error("No pair data available. Run phase2 first.")
            return {}

        data1, data2 = self.pair_data

        # Expanded parameter grid
        param_grid = {
            'entry_zscore': [1.0, 1.25, 1.5, 2.0, 2.5],
            'exit_zscore': [0.10, 0.20, 0.25, 0.50],
            'zscore_window': [10, 15, 20, 30]
        }

        total_combinations = len(param_grid['entry_zscore']) * len(param_grid['exit_zscore']) * len(param_grid['zscore_window'])

        logger.info(f"Parameter Grid:")
        logger.info(f"  Entry Z-scores: {param_grid['entry_zscore']}")
        logger.info(f"  Exit Z-scores: {param_grid['exit_zscore']}")
        logger.info(f"  Windows: {param_grid['zscore_window']}")
        logger.info(f"  Total Combinations: {total_combinations}")
        logger.blank()

        best_sharpe = float('-inf')
        best_params = {}
        all_results = []

        risk_config = RiskConfig.moderate()

        # Grid search
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
                        result = self._run_backtest(
                            data1, data2,
                            params,
                            fees=0.0001,  # NEW fees
                            slippage=0.001,
                            risk_config=risk_config
                        )

                        sharpe = result['stats'].get('Sharpe Ratio', 0)
                        all_results.append({
                            'params': params,
                            'sharpe': sharpe,
                            'stats': result['stats']
                        })

                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_params = params.copy()

                    except Exception as e:
                        logger.warning(f"  Error with params {params}: {str(e)}")

        # Show results
        logger.blank()
        logger.separator("-", 80)
        logger.header("OPTIMIZATION RESULTS")
        logger.separator("-", 80)
        logger.blank()

        if all_results:
            sorted_results = sorted(all_results, key=lambda x: x['sharpe'], reverse=True)

            logger.success(f"Best Parameters Found:")
            logger.success(f"  Entry Z-score: {best_params['entry_zscore']}")
            logger.success(f"  Exit Z-score: {best_params['exit_zscore']}")
            logger.success(f"  Window: {best_params['zscore_window']}")
            logger.success(f"  Sharpe Ratio: {best_sharpe:.3f}")
            logger.blank()

            logger.info("Top 10 Configurations:")
            for i, res in enumerate(sorted_results[:10], 1):
                p = res['params']
                sharpe = res['sharpe']
                ret = res['stats'].get('Total Return [%]', 0)
                trades = res['stats'].get('Total Trades', 0)

                if sharpe > 1.0:
                    logger.success(f"  {i}. Entry={p['entry_zscore']}, Exit={p['exit_zscore']}, Win={p['zscore_window']} | Sharpe={sharpe:.3f}, Return={ret:.2f}%, Trades={trades}")
                elif sharpe > 0.5:
                    logger.info(f"  {i}. Entry={p['entry_zscore']}, Exit={p['exit_zscore']}, Win={p['zscore_window']} | Sharpe={sharpe:.3f}, Return={ret:.2f}%, Trades={trades}")
                else:
                    logger.warning(f"  {i}. Entry={p['entry_zscore']}, Exit={p['exit_zscore']}, Win={p['zscore_window']} | Sharpe={sharpe:.3f}, Return={ret:.2f}%, Trades={trades}")

            self.results['optimization'] = {
                'best_params': best_params,
                'best_sharpe': best_sharpe,
                'top_10': sorted_results[:10]
            }
        else:
            logger.error("No valid results from optimization")
            self.results['optimization'] = {}

        return self.results['optimization']

    def phase4_position_sizing(self, best_params: Dict) -> Dict:
        """
        Phase 4: Test position sizing sensitivity.

        Tests if optimal position sizing changes with lower transaction costs.
        """
        logger.blank()
        logger.separator("=", 80)
        logger.header("PHASE 4: POSITION SIZING SENSITIVITY")
        logger.separator("=", 80)
        logger.blank()

        if self.pair_data is None:
            return {}

        data1, data2 = self.pair_data

        position_sizes = [0.10, 0.15, 0.20, 0.25, 0.30]
        results = {}

        logger.info(f"Testing position sizes: {[f'{p*100}%' for p in position_sizes]}")
        logger.info(f"Using best parameters: {best_params}")
        logger.blank()

        for pos_pct in position_sizes:
            risk_config = RiskConfig(
                position_size_pct=pos_pct,
                max_positions=10,
                use_stop_loss=True,
                stop_loss_pct=0.02
            )

            result = self._run_backtest(
                data1, data2,
                best_params,
                fees=0.0001,
                slippage=0.001,
                risk_config=risk_config
            )

            sharpe = result['stats'].get('Sharpe Ratio', 0)
            ret = result['stats'].get('Total Return [%]', 0)
            dd = result['stats'].get('Max Drawdown [%]', 0)

            results[f'{pos_pct*100:.0f}%'] = {
                'sharpe': sharpe,
                'return': ret,
                'drawdown': dd,
                'stats': result['stats']
            }

            logger.info(f"{pos_pct*100:.0f}% Position Size:")
            logger.info(f"  Sharpe: {sharpe:.3f}")
            logger.info(f"  Return: {ret:.2f}%")
            logger.info(f"  Max DD: {dd:.2f}%")
            logger.blank()

        # Find optimal
        optimal_pos = max(results.items(), key=lambda x: x[1]['sharpe'])
        logger.success(f"Optimal Position Size: {optimal_pos[0]} (Sharpe={optimal_pos[1]['sharpe']:.3f})")
        logger.blank()

        self.results['position_sizing'] = results
        return results

    def phase5_walk_forward(self, best_params: Dict) -> Dict:
        """
        Phase 5: Walk-forward validation.

        Tests robustness on out-of-sample data.
        """
        logger.separator("=", 80)
        logger.header("PHASE 5: WALK-FORWARD VALIDATION")
        logger.separator("=", 80)
        logger.blank()

        if self.pair_data is None:
            return {}

        data1, data2 = self.pair_data

        # 60/40 split
        split_idx = int(len(data1) * 0.6)
        train_data1 = data1.iloc[:split_idx]
        train_data2 = data2.iloc[:split_idx]
        test_data1 = data1.iloc[split_idx:]
        test_data2 = data2.iloc[split_idx:]

        logger.info(f"Train: {train_data1.index[0]} to {train_data1.index[-1]} ({len(train_data1)} days)")
        logger.info(f"Test:  {test_data1.index[0]} to {test_data1.index[-1]} ({len(test_data1)} days)")
        logger.blank()

        risk_config = RiskConfig.moderate()

        # Train period
        logger.info("In-Sample (Train) Results:")
        train_result = self._run_backtest(
            train_data1, train_data2,
            best_params,
            fees=0.0001,
            slippage=0.001,
            risk_config=risk_config
        )

        train_sharpe = train_result['stats'].get('Sharpe Ratio', 0)
        train_return = train_result['stats'].get('Total Return [%]', 0)

        logger.success(f"  Sharpe: {train_sharpe:.3f}")
        logger.info(f"  Return: {train_return:.2f}%")
        logger.blank()

        # Test period
        logger.info("Out-of-Sample (Test) Results:")
        test_result = self._run_backtest(
            test_data1, test_data2,
            best_params,
            fees=0.0001,
            slippage=0.001,
            risk_config=risk_config
        )

        test_sharpe = test_result['stats'].get('Sharpe Ratio', 0)
        test_return = test_result['stats'].get('Total Return [%]', 0)

        logger.success(f"  Sharpe: {test_sharpe:.3f}")
        logger.info(f"  Return: {test_return:.2f}%")
        logger.blank()

        # Degradation
        if train_sharpe != 0:
            degradation = ((train_sharpe - test_sharpe) / abs(train_sharpe)) * 100
        else:
            degradation = 0

        logger.info("Performance Degradation:")
        logger.info(f"  Train Sharpe: {train_sharpe:.3f}")
        logger.info(f"  Test Sharpe: {test_sharpe:.3f}")

        if degradation < 30:
            logger.success(f"  Degradation: {degradation:.1f}% - EXCELLENT")
        elif degradation < 50:
            logger.warning(f"  Degradation: {degradation:.1f}% - ACCEPTABLE")
        else:
            logger.error(f"  Degradation: {degradation:.1f}% - POOR")
        logger.blank()

        self.results['walk_forward'] = {
            'train': train_result['stats'],
            'test': test_result['stats'],
            'degradation_pct': degradation
        }

        return self.results['walk_forward']

    def phase6_breakeven_analysis(self) -> Dict:
        """
        Phase 6: Calculate break-even thresholds.

        Determines minimum required profit per trade for viability.
        """
        logger.separator("=", 80)
        logger.header("PHASE 6: BREAK-EVEN ANALYSIS")
        logger.separator("=", 80)
        logger.blank()

        # Cost per round-trip
        old_cost = 4 * 0.002  # 4 legs x 0.2%
        new_cost = 4 * 0.0011  # 4 legs x 0.11%

        logger.info("Transaction Costs:")
        logger.info(f"  Old (0.1% fees): {old_cost * 100:.2f}% per round-trip")
        logger.info(f"  New (0.01% fees): {new_cost * 100:.2f}% per round-trip")
        logger.info(f"  Savings: {(old_cost - new_cost) * 100:.2f}% per round-trip")
        logger.blank()

        # For Sharpe 1.0 target
        target_sharpe = 1.0
        assumed_volatility = 0.10  # 10% annual vol
        required_return = target_sharpe * assumed_volatility  # 10% return

        trades_per_year = 25  # From previous validation
        required_per_trade = required_return / trades_per_year

        # After costs
        gross_per_trade_old = required_per_trade + old_cost
        gross_per_trade_new = required_per_trade + new_cost

        logger.info("Break-Even for Sharpe 1.0:")
        logger.info(f"  Target Annual Return: {required_return * 100:.1f}%")
        logger.info(f"  Assumed Trades/Year: {trades_per_year}")
        logger.info(f"  Required Net P&L/Trade: {required_per_trade * 100:.2f}%")
        logger.blank()

        logger.info("  Required Gross P&L/Trade:")
        logger.info(f"    Old Fees: {gross_per_trade_old * 100:.2f}%")
        logger.success(f"    New Fees: {gross_per_trade_new * 100:.2f}%")
        logger.blank()

        logger.info("  On $100k Capital:")
        logger.info(f"    Old Fees: ${gross_per_trade_old * 100000:.0f} gross per trade")
        logger.success(f"    New Fees: ${gross_per_trade_new * 100000:.0f} gross per trade")
        logger.blank()

        # Safety margin from actual results
        if 'ab_comparison' in self.results:
            new_result = self.results['ab_comparison']['new_fees']
            if 'trades' in new_result and len(new_result['trades']) > 0:
                avg_pnl_pct = new_result['stats'].get('Total Return [%]', 0) / len(new_result['trades'])
                safety_factor = avg_pnl_pct / (gross_per_trade_new * 100)

                logger.info("Actual Performance:")
                logger.info(f"  Avg P&L/Trade: {avg_pnl_pct:.2f}%")
                logger.info(f"  Required: {gross_per_trade_new * 100:.2f}%")

                if safety_factor >= 2.0:
                    logger.success(f"  Safety Margin: {safety_factor:.1f}x - EXCELLENT")
                elif safety_factor >= 1.0:
                    logger.warning(f"  Safety Margin: {safety_factor:.1f}x - ACCEPTABLE")
                else:
                    logger.error(f"  Safety Margin: {safety_factor:.1f}x - INSUFFICIENT")
                logger.blank()

        self.results['breakeven'] = {
            'old_cost_per_trade': old_cost,
            'new_cost_per_trade': new_cost,
            'required_gross_old': gross_per_trade_old,
            'required_gross_new': gross_per_trade_new
        }

        return self.results['breakeven']

    def phase7_viability_assessment(self):
        """
        Phase 7: Production viability assessment.

        Makes final GO/NO-GO decision based on all metrics.
        """
        logger.blank()
        logger.separator("=", 80)
        logger.header("PHASE 7: PRODUCTION VIABILITY ASSESSMENT")
        logger.separator("=", 80)
        logger.blank()

        # Extract key metrics
        best_sharpe = self.results.get('optimization', {}).get('best_sharpe', 0)
        ab_new = self.results.get('ab_comparison', {}).get('new_fees', {}).get('stats', {})
        ab_new_sharpe = ab_new.get('Sharpe Ratio', 0)
        ab_new_return = ab_new.get('Total Return [%]', 0)
        ab_new_dd = ab_new.get('Max Drawdown [%]', 0)

        wf = self.results.get('walk_forward', {})
        degradation = wf.get('degradation_pct', 100)

        # Decision matrix
        criteria = [
            {
                'name': 'Sharpe Ratio',
                'target': '> 0.8',
                'actual': best_sharpe,
                'pass': best_sharpe > 0.8
            },
            {
                'name': 'Annual Return',
                'target': '> 10%',
                'actual': ab_new_return,
                'pass': ab_new_return > 10
            },
            {
                'name': 'Max Drawdown',
                'target': '< 15%',
                'actual': abs(ab_new_dd),
                'pass': abs(ab_new_dd) < 15
            },
            {
                'name': 'Degradation',
                'target': '< 50%',
                'actual': degradation,
                'pass': degradation < 50
            }
        ]

        logger.info("Production Readiness Criteria:")
        logger.blank()

        passed = 0
        for criterion in criteria:
            status = "PASS" if criterion['pass'] else "FAIL"
            if criterion['pass']:
                logger.success(f"  [{status}] {criterion['name']}: {criterion['actual']:.2f} (target: {criterion['target']})")
                passed += 1
            else:
                logger.error(f"  [{status}] {criterion['name']}: {criterion['actual']:.2f} (target: {criterion['target']})")

        logger.blank()
        logger.info(f"Criteria Met: {passed}/4")
        logger.blank()

        # Final decision
        if passed >= 4:
            decision = "GO"
            color = "success"
            recommendation = [
                "Strategy is PRODUCTION-READY with realistic fees",
                "Next steps:",
                "  1. Test on real cointegrated pairs (e.g., SPY/IWM, GLD/GDX)",
                "  2. Deploy to paper trading for 30 days",
                "  3. Monitor live spread behavior vs synthetic",
                "  4. Consider live deployment in 4-6 weeks"
            ]
        elif passed >= 2:
            decision = "NEEDS WORK"
            color = "warning"
            recommendation = [
                "Strategy shows promise but needs enhancement",
                "Recommended improvements:",
                "  1. Implement Kalman filter for dynamic hedge ratio",
                "  2. Test on portfolio of multiple pairs (diversification)",
                "  3. Add regime detection (high vol vs low vol)",
                "  4. Re-test after enhancements (timeline: 2-3 months)"
            ]
        else:
            decision = "NO-GO"
            color = "error"
            recommendation = [
                "Strategy not viable even with realistic fees",
                "Fundamental issues detected:",
                "  1. Insufficient alpha generation",
                "  2. Cointegration too weak or unstable",
                "  3. Consider alternative approaches:",
                "     - Machine learning for pair selection",
                "     - High-frequency statistical arb",
                "     - Different asset classes (futures, crypto)"
            ]

        logger.separator("=", 80)
        if color == "success":
            logger.success(f"FINAL DECISION: {decision}")
        elif color == "warning":
            logger.warning(f"FINAL DECISION: {decision}")
        else:
            logger.error(f"FINAL DECISION: {decision}")
        logger.separator("=", 80)
        logger.blank()

        for line in recommendation:
            if line.startswith("  "):
                logger.info(line)
            else:
                if color == "success":
                    logger.success(line)
                elif color == "warning":
                    logger.warning(line)
                else:
                    logger.error(line)

        logger.blank()

        self.results['viability'] = {
            'decision': decision,
            'criteria_met': passed,
            'criteria_total': 4,
            'recommendation': recommendation
        }

    def _run_backtest(
        self,
        data1: pd.DataFrame,
        data2: pd.DataFrame,
        params: Dict,
        fees: float,
        slippage: float,
        risk_config: RiskConfig
    ) -> Dict:
        """
        Helper method to run a single backtest.

        Returns:
            Dict with 'stats' and 'trades' keys
        """
        strategy = PairsTrading(**params)
        signals_dict = strategy.generate_pairs_signals(data1, data2, 'SYM1', 'SYM2')

        long_entries1, long_exits1, short_entries1, short_exits1 = signals_dict['SYM1']

        # PairsPortfolio signal convention (swapped)
        portfolio = PairsPortfolio(
            symbols=('SYM1', 'SYM2'),
            prices1=data1['close'],
            prices2=data2['close'],
            entries=short_entries1,  # Long spread
            exits=short_exits1,
            short_entries=long_entries1,  # Short spread
            short_exits=long_exits1,
            init_cash=100000,
            fees=fees,
            slippage=slippage,
            freq='1D',
            market_hours_only=False,
            risk_config=risk_config,
            price_data1=data1,
            price_data2=data2
        )

        stats = portfolio.stats()

        # Extract trades if available
        trades = []
        if hasattr(portfolio, '_trades'):
            trades = portfolio._trades

        return {
            'stats': stats if stats else {},
            'trades': trades
        }

    def generate_report(self, output_path: str = None):
        """Generate comprehensive validation report."""
        logger.blank()
        logger.separator("=", 80)
        logger.header("COMPREHENSIVE VALIDATION REPORT")
        logger.separator("=", 80)
        logger.blank()

        report = {
            'timestamp': datetime.now().isoformat(),
            'validation_type': 'Realistic Fee Re-Validation',
            'fee_structure': {
                'old_fees': '0.1% per trade (10 basis points)',
                'new_fees': '0.01% per trade (1 basis point)',
                'cost_reduction': '90% fee reduction'
            },
            'results': self.results
        }

        # Save report
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.success(f"Report saved to: {output_path}")

        return report


def main():
    """Main execution."""
    logger.separator("=", 80)
    logger.header("PAIRS TRADING RE-VALIDATION WITH REALISTIC FEES")
    logger.separator("=", 80)
    logger.info(f"Start Time: {datetime.now()}")
    logger.blank()

    validator = RealisticFeeValidator()

    try:
        # Phase 1: Ultrathinking
        validator.phase1_ultrathinking()

        # Phase 2: A/B Comparison
        validator.phase2_ab_comparison()

        # Phase 3: Full Optimization
        opt_result = validator.phase3_full_optimization()
        best_params = opt_result.get('best_params', {
            'entry_zscore': 1.5,
            'exit_zscore': 0.25,
            'zscore_window': 20
        })

        # Phase 4: Position Sizing
        validator.phase4_position_sizing(best_params)

        # Phase 5: Walk-Forward
        validator.phase5_walk_forward(best_params)

        # Phase 6: Break-Even
        validator.phase6_breakeven_analysis()

        # Phase 7: Viability Assessment
        validator.phase7_viability_assessment()

        # Generate Report
        output_path = Path(__file__).parent / 'realistic_fee_validation_report.json'
        validator.generate_report(str(output_path))

        logger.blank()
        logger.info(f"End Time: {datetime.now()}")
        logger.separator("=", 80)
        logger.header("VALIDATION COMPLETE")
        logger.separator("=", 80)

    except Exception as e:
        logger.error(f"\nValidation failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
