"""
Trade Inspector utility for analyzing individual trades.

Extracts and formats trade data from backtest results for detailed inspection.
"""

import pandas as pd
from typing import Dict, List, Any, Optional


class TradeInspector:
    """
    Inspector for analyzing individual trades from backtest results.

    Provides methods to:
    - Extract trade details
    - Calculate per-trade metrics
    - Filter and sort trades
    """

    @staticmethod
    def extract_trades_from_portfolio(portfolio) -> Optional[pd.DataFrame]:
        """
        Extract trade records from VectorBT portfolio.

        Args:
            portfolio: VectorBT Portfolio object

        Returns:
            DataFrame with trade details or None if no trades
        """
        try:
            if hasattr(portfolio, 'trades') and hasattr(portfolio.trades, 'records_readable'):
                trades_df = portfolio.trades.records_readable  # type: ignore[attr-defined]

                if trades_df is not None and not trades_df.empty:
                    return trades_df

            return None

        except Exception:
            return None

    @staticmethod
    def format_trade_summary(trades_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Format trades into summary dictionaries for display.

        Args:
            trades_df: DataFrame from extract_trades_from_portfolio

        Returns:
            List of trade summary dictionaries
        """
        if trades_df is None or trades_df.empty:
            return []

        trade_summaries = []

        for idx, trade in trades_df.iterrows():
            summary = {
                'id': int(idx) if isinstance(idx, (int, float)) else idx,
                'entry_date': str(trade.get('Entry Timestamp', trade.get('Entry Index', 'N/A'))),
                'exit_date': str(trade.get('Exit Timestamp', trade.get('Exit Index', 'N/A'))),
                'entry_price': float(trade.get('Avg Entry Price', 0)),
                'exit_price': float(trade.get('Avg Exit Price', 0)),
                'size': float(trade.get('Size', 0)),
                'pnl': float(trade.get('PnL', 0)),
                'pnl_pct': float(trade.get('Return', 0)) * 100 if 'Return' in trade else 0,
                'duration': int(trade.get('Duration', 0)),
                'direction': str(trade.get('Direction', 'Long')),
                'status': str(trade.get('Status', 'Closed'))
            }

            trade_summaries.append(summary)

        return trade_summaries

    @staticmethod
    def filter_winning_trades(trade_summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter to only winning trades (PnL > 0)."""
        return [t for t in trade_summaries if t.get('pnl', 0) > 0]

    @staticmethod
    def filter_losing_trades(trade_summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter to only losing trades (PnL < 0)."""
        return [t for t in trade_summaries if t.get('pnl', 0) < 0]

    @staticmethod
    def get_largest_winner(trade_summaries: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Get the trade with largest profit."""
        if not trade_summaries:
            return None
        return max(trade_summaries, key=lambda t: t.get('pnl', 0))

    @staticmethod
    def get_largest_loser(trade_summaries: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Get the trade with largest loss."""
        if not trade_summaries:
            return None
        return min(trade_summaries, key=lambda t: t.get('pnl', 0))

    @staticmethod
    def calculate_trade_statistics(trade_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate statistics across all trades.

        Returns:
            Dictionary with trade statistics
        """
        if not trade_summaries:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_winner': 0.0,
                'avg_loser': 0.0,
                'avg_duration': 0.0,
                'total_pnl': 0.0
            }

        winners = TradeInspector.filter_winning_trades(trade_summaries)
        losers = TradeInspector.filter_losing_trades(trade_summaries)

        stats = {
            'total_trades': len(trade_summaries),
            'winning_trades': len(winners),
            'losing_trades': len(losers),
            'win_rate': (len(winners) / len(trade_summaries) * 100) if trade_summaries else 0,
            'avg_winner': sum(t['pnl'] for t in winners) / len(winners) if winners else 0,
            'avg_loser': sum(t['pnl'] for t in losers) / len(losers) if losers else 0,
            'avg_duration': sum(t['duration'] for t in trade_summaries) / len(trade_summaries) if trade_summaries else 0,
            'total_pnl': sum(t['pnl'] for t in trade_summaries)
        }

        return stats
