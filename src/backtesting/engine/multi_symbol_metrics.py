"""
Multi-symbol portfolio metrics calculator.

Calculates comprehensive metrics specific to multi-asset portfolios including:
- Portfolio composition and allocation
- Symbol attribution and contribution
- Diversification and correlation
- Rebalancing effectiveness
- Trade analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from backtesting.engine.multi_asset_portfolio import MultiAssetPortfolio


class MultiSymbolMetrics:
    """
    Calculate comprehensive metrics for multi-symbol portfolios.

    Analyzes portfolio-level dynamics that are unique to multi-asset portfolios
    such as allocation evolution, symbol contribution, and diversification benefits.
    """

    @staticmethod
    def calculate_portfolio_composition_metrics(portfolio: 'MultiAssetPortfolio') -> Dict[str, Any]:
        """
        Calculate metrics related to portfolio composition and capital allocation.

        Args:
            portfolio: MultiAssetPortfolio instance

        Returns:
            Dictionary with composition metrics
        """
        if not portfolio.position_count_history:
            return {}

        # Extract position counts over time
        position_counts = [count for _, count in portfolio.position_count_history]

        # Calculate capital utilization over time
        # PERFORMANCE FIX: Create O(1) lookup dict instead of O(n) .index() calls
        # For 98K timestamps, this changes O(n²) = 9.6B operations to O(n) = 98K operations
        timestamp_to_idx = {ts: idx for idx, ts in enumerate(portfolio.equity_timestamps)}

        capital_utilization = []
        for timestamp, cash in portfolio.cash_history:
            portfolio_idx = timestamp_to_idx.get(timestamp)
            if portfolio_idx is None:
                continue

            try:
                portfolio_value = portfolio.equity_curve[portfolio_idx]
                if portfolio_value > 0:
                    deployed = portfolio_value - cash
                    utilization_pct = (deployed / portfolio_value) * 100
                    capital_utilization.append(utilization_pct)
            except IndexError:
                # Index out of range - skip
                continue

        # Calculate concentration (Herfindahl Index)
        concentration_scores = []
        for timestamp, weights in portfolio.symbol_weights_history:
            if weights:
                # Herfindahl Index: sum of squared weights
                herfindahl = sum(w**2 for w in weights.values())
                concentration_scores.append(herfindahl)

        return {
            'Avg Position Count': np.mean(position_counts) if position_counts else 0,
            'Max Position Count': max(position_counts) if position_counts else 0,
            'Min Position Count': min(position_counts) if position_counts else 0,
            'Avg Capital Utilization [%]': np.mean(capital_utilization) if capital_utilization else 0,
            'Max Capital Utilization [%]': max(capital_utilization) if capital_utilization else 0,
            'Avg Concentration (Herfindahl)': np.mean(concentration_scores) if concentration_scores else 0,
            'Max Concentration': max(concentration_scores) if concentration_scores else 0,
        }

    @staticmethod
    def calculate_symbol_attribution_metrics(portfolio: 'MultiAssetPortfolio') -> Dict[str, Any]:
        """
        Calculate per-symbol contribution and attribution metrics.

        Args:
            portfolio: MultiAssetPortfolio instance

        Returns:
            Dictionary with per-symbol attribution metrics
        """
        if not portfolio.closed_positions:
            return {}

        # Group trades by symbol
        symbol_stats = {}
        for symbol in portfolio.symbols:
            symbol_trades = [t for t in portfolio.closed_positions if t['symbol'] == symbol]

            if not symbol_trades:
                symbol_stats[symbol] = {
                    'Total P&L': 0,
                    'Total Trades': 0,
                    'Win Rate [%]': 0,
                    'Avg Hold Duration [days]': 0,
                    'Sharpe Ratio': 0,
                    'Total Return [%]': 0,
                }
                continue

            # Calculate P&L stats
            total_pnl = sum(t['pnl'] for t in symbol_trades)
            winning_trades = [t for t in symbol_trades if t['pnl'] > 0]
            losing_trades = [t for t in symbol_trades if t['pnl'] <= 0]
            win_rate = (len(winning_trades) / len(symbol_trades) * 100) if symbol_trades else 0

            # Calculate average hold duration
            hold_durations = [t['hold_duration_days'] for t in symbol_trades if 'hold_duration_days' in t]
            avg_hold_duration = np.mean(hold_durations) if hold_durations else 0

            # Calculate return percentage
            initial_capital_per_symbol = portfolio.init_cash / len(portfolio.symbols)
            return_pct = (total_pnl / initial_capital_per_symbol) * 100

            # Calculate Sharpe (approximate from trade returns)
            if symbol_trades:
                trade_returns = [t['pnl_pct'] / 100 for t in symbol_trades]
                sharpe = (np.mean(trade_returns) / np.std(trade_returns)) * np.sqrt(len(symbol_trades)) if np.std(trade_returns) > 0 else 0
            else:
                sharpe = 0

            symbol_stats[symbol] = {
                'Total P&L': total_pnl,
                'Total Trades': len(symbol_trades),
                'Win Rate [%]': win_rate,
                'Avg Hold Duration [days]': avg_hold_duration,
                'Sharpe Ratio': sharpe,
                'Total Return [%]': return_pct,
                'Avg P&L per Trade': total_pnl / len(symbol_trades) if symbol_trades else 0,
                'Best Trade': max((t['pnl'] for t in symbol_trades), default=0),
                'Worst Trade': min((t['pnl'] for t in symbol_trades), default=0),
            }

        # Calculate contribution percentages
        total_pnl = sum(stats['Total P&L'] for stats in symbol_stats.values())
        for symbol, stats in symbol_stats.items():
            if total_pnl != 0:
                stats['Contribution [%]'] = (stats['Total P&L'] / total_pnl) * 100
            else:
                stats['Contribution [%]'] = 0

        return {
            'per_symbol': symbol_stats,
            'best_symbol': max(symbol_stats.items(), key=lambda x: x[1]['Total P&L'])[0] if symbol_stats else None,
            'worst_symbol': min(symbol_stats.items(), key=lambda x: x[1]['Total P&L'])[0] if symbol_stats else None,
            'total_pnl': total_pnl,
        }

    @staticmethod
    def calculate_diversification_metrics(portfolio: 'MultiAssetPortfolio') -> Dict[str, Any]:
        """
        Calculate diversification and correlation metrics.

        Args:
            portfolio: MultiAssetPortfolio instance

        Returns:
            Dictionary with diversification metrics
        """
        # Get correlation matrix
        corr_matrix = portfolio.get_correlation_matrix()

        if corr_matrix.empty:
            return {}

        # Calculate average pairwise correlation
        # Extract upper triangle (exclude diagonal)
        corr_values = []
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                corr_values.append(corr_matrix.iloc[i, j])

        avg_correlation = np.mean(corr_values) if corr_values else 0

        # Calculate portfolio stats
        stats = portfolio.stats()
        if stats is None:
            return {'Average Correlation': avg_correlation}

        # Calculate Sortino Ratio (downside deviation)
        returns = portfolio.returns()
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        annual_return = stats.get('Annual Return [%]', 0)
        sortino = (annual_return / downside_std) if downside_std > 0 else 0

        # Calculate Calmar Ratio (CAGR / Max Drawdown)
        max_dd = abs(stats.get('Max Drawdown [%]', 0))
        calmar = (annual_return / max_dd) if max_dd > 0 else 0

        return {
            'Average Correlation': avg_correlation,
            'Max Correlation': max(corr_values) if corr_values else 0,
            'Min Correlation': min(corr_values) if corr_values else 0,
            'Sortino Ratio': sortino,
            'Calmar Ratio': calmar,
        }

    @staticmethod
    def calculate_rebalancing_metrics(portfolio: 'MultiAssetPortfolio') -> Dict[str, Any]:
        """
        Calculate rebalancing effectiveness metrics.

        Args:
            portfolio: MultiAssetPortfolio instance

        Returns:
            Dictionary with rebalancing metrics
        """
        if not portfolio.rebalancing_events:
            return {
                'Rebalancing Event Count': 0,
                'Avg Rebalancing Cost': 0,
                'Position Turnover [%]': 0,
            }

        rebalance_count = len(portfolio.rebalancing_events)

        # Calculate average rebalancing cost
        # (would need to track transaction costs per rebalance - TODO in future)
        avg_cost = 0  # Placeholder

        # Calculate position turnover
        # (trades per month)
        if portfolio.closed_positions:
            start_date = portfolio.equity_timestamps[0]
            end_date = portfolio.equity_timestamps[-1]
            months = (end_date - start_date).days / 30.44
            if months > 0:
                turnover = len(portfolio.closed_positions) / months
            else:
                turnover = 0
        else:
            turnover = 0

        return {
            'Rebalancing Event Count': rebalance_count,
            'Avg Rebalancing Cost': avg_cost,
            'Position Turnover [trades/month]': turnover,
        }

    @staticmethod
    def calculate_trade_analysis_metrics(portfolio: 'MultiAssetPortfolio') -> Dict[str, Any]:
        """
        Calculate trade-level analysis metrics.

        Args:
            portfolio: MultiAssetPortfolio instance

        Returns:
            Dictionary with trade analysis metrics
        """
        if not portfolio.closed_positions:
            return {}

        # Group by symbol
        trades_per_symbol = {}
        for trade in portfolio.closed_positions:
            symbol = trade['symbol']
            if symbol not in trades_per_symbol:
                trades_per_symbol[symbol] = []
            trades_per_symbol[symbol].append(trade)

        # Calculate profit factor per symbol
        profit_factors = {}
        expectancy = {}

        for symbol, trades in trades_per_symbol.items():
            winning = [t for t in trades if t['pnl'] > 0]
            losing = [t for t in trades if t['pnl'] <= 0]

            total_profit = sum(t['pnl'] for t in winning)
            total_loss = abs(sum(t['pnl'] for t in losing))

            profit_factor = (total_profit / total_loss) if total_loss > 0 else 0
            profit_factors[symbol] = profit_factor

            # Expectancy = (Win Rate × Avg Win) - (Loss Rate × Avg Loss)
            if trades:
                win_rate = len(winning) / len(trades)
                loss_rate = len(losing) / len(trades)
                avg_win = (total_profit / len(winning)) if winning else 0
                avg_loss = (total_loss / len(losing)) if losing else 0
                expectancy[symbol] = (win_rate * avg_win) - (loss_rate * avg_loss)
            else:
                expectancy[symbol] = 0

        return {
            'trades_per_symbol': {symbol: len(trades) for symbol, trades in trades_per_symbol.items()},
            'profit_factor_per_symbol': profit_factors,
            'expectancy_per_symbol': expectancy,
        }

    @staticmethod
    def calculate_all_metrics(portfolio: 'MultiAssetPortfolio') -> Dict[str, Any]:
        """
        Calculate all multi-symbol portfolio metrics.

        Args:
            portfolio: MultiAssetPortfolio instance

        Returns:
            Dictionary with all calculated metrics organized by category
        """
        import time
        from utils import logger

        results = {}
        total_start = time.time()

        # Composition metrics
        start = time.time()
        results['composition'] = MultiSymbolMetrics.calculate_portfolio_composition_metrics(portfolio)
        logger.metric(f"  - Composition metrics: {time.time() - start:.2f}s")

        # Attribution metrics
        start = time.time()
        results['attribution'] = MultiSymbolMetrics.calculate_symbol_attribution_metrics(portfolio)
        logger.metric(f"  - Attribution metrics: {time.time() - start:.2f}s")

        # Diversification metrics
        start = time.time()
        results['diversification'] = MultiSymbolMetrics.calculate_diversification_metrics(portfolio)
        logger.metric(f"  - Diversification metrics: {time.time() - start:.2f}s")

        # Rebalancing metrics
        start = time.time()
        results['rebalancing'] = MultiSymbolMetrics.calculate_rebalancing_metrics(portfolio)
        logger.metric(f"  - Rebalancing metrics: {time.time() - start:.2f}s")

        # Trade analysis metrics
        start = time.time()
        results['trade_analysis'] = MultiSymbolMetrics.calculate_trade_analysis_metrics(portfolio)
        logger.metric(f"  - Trade analysis metrics: {time.time() - start:.2f}s")

        logger.metric(f"  - Total metrics calculation: {time.time() - total_start:.2f}s")

        return results
