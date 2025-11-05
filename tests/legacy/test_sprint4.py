"""
Tests for Sprint 4: Analysis Tools

Sprint 4 Features:
- E1: Trade Inspector
- E2: Data Inspector
- E3: Signal Analyzer
"""

import pytest
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gui.utils.trade_inspector import TradeInspector
from gui.utils.data_inspector import DataInspector
from gui.utils.signal_analyzer import SignalAnalyzer


class TestTradeInspector:
    """Test trade inspector functionality."""

    def create_mock_trades_df(self, num_trades=5):
        """Create a mock trades DataFrame for testing."""
        trades = []
        for i in range(num_trades):
            pnl = 100 * (i - 2)  # Mix of winners and losers
            trade = {
                'Entry Timestamp': f'2023-01-{i+1:02d}',
                'Exit Timestamp': f'2023-01-{i+2:02d}',
                'Avg Entry Price': 100 + i,
                'Avg Exit Price': 100 + i + (pnl / 10),
                'Size': 10,
                'PnL': pnl,
                'Return': pnl / 1000,
                'Duration': i + 1,
                'Direction': 'Long',
                'Status': 'Closed'
            }
            trades.append(trade)

        return pd.DataFrame(trades)

    def test_format_trade_summary(self):
        """Test formatting trades into summaries."""
        trades_df = self.create_mock_trades_df(5)
        summaries = TradeInspector.format_trade_summary(trades_df)

        assert len(summaries) == 5
        assert all('id' in s for s in summaries)
        assert all('pnl' in s for s in summaries)
        assert all('entry_price' in s for s in summaries)

    def test_format_trade_summary_empty(self):
        """Test formatting empty trades."""
        empty_df = pd.DataFrame()
        summaries = TradeInspector.format_trade_summary(empty_df)
        assert summaries == []

    def test_filter_winning_trades(self):
        """Test filtering to winning trades only."""
        trades_df = self.create_mock_trades_df(5)
        summaries = TradeInspector.format_trade_summary(trades_df)

        winners = TradeInspector.filter_winning_trades(summaries)

        # Trades 3 and 4 should be winners (PnL > 0)
        assert len(winners) == 2
        assert all(t['pnl'] > 0 for t in winners)

    def test_filter_losing_trades(self):
        """Test filtering to losing trades only."""
        trades_df = self.create_mock_trades_df(5)
        summaries = TradeInspector.format_trade_summary(trades_df)

        losers = TradeInspector.filter_losing_trades(summaries)

        # Trades 0 and 1 should be losers (PnL < 0)
        assert len(losers) == 2
        assert all(t['pnl'] < 0 for t in losers)

    def test_get_largest_winner(self):
        """Test finding the largest winning trade."""
        trades_df = self.create_mock_trades_df(5)
        summaries = TradeInspector.format_trade_summary(trades_df)

        largest_winner = TradeInspector.get_largest_winner(summaries)

        assert largest_winner is not None
        assert largest_winner['pnl'] == 200  # Trade 4 has PnL of 200

    def test_get_largest_loser(self):
        """Test finding the largest losing trade."""
        trades_df = self.create_mock_trades_df(5)
        summaries = TradeInspector.format_trade_summary(trades_df)

        largest_loser = TradeInspector.get_largest_loser(summaries)

        assert largest_loser is not None
        assert largest_loser['pnl'] == -200  # Trade 0 has PnL of -200

    def test_calculate_trade_statistics(self):
        """Test calculating trade statistics."""
        trades_df = self.create_mock_trades_df(5)
        summaries = TradeInspector.format_trade_summary(trades_df)

        stats = TradeInspector.calculate_trade_statistics(summaries)

        assert stats['total_trades'] == 5
        assert stats['winning_trades'] == 2
        assert stats['losing_trades'] == 2
        assert stats['win_rate'] == 40.0  # 2/5 = 40%
        assert 'avg_winner' in stats
        assert 'avg_loser' in stats
        assert 'avg_duration' in stats

    def test_calculate_trade_statistics_empty(self):
        """Test statistics with no trades."""
        stats = TradeInspector.calculate_trade_statistics([])

        assert stats['total_trades'] == 0
        assert stats['winning_trades'] == 0
        assert stats['losing_trades'] == 0
        assert stats['win_rate'] == 0.0


class TestDataInspector:
    """Test data inspector functionality."""

    def create_mock_price_data(self, num_rows=100):
        """Create mock OHLCV price data."""
        dates = pd.date_range('2023-01-01', periods=num_rows, freq='D')
        data = {
            'open': np.random.uniform(95, 105, num_rows),
            'high': np.random.uniform(100, 110, num_rows),
            'low': np.random.uniform(90, 100, num_rows),
            'close': np.random.uniform(95, 105, num_rows),
            'volume': np.random.uniform(1000000, 5000000, num_rows)
        }
        df = pd.DataFrame(data, index=dates)
        return df

    def test_summarize_data(self):
        """Test data summarization."""
        df = self.create_mock_price_data(100)
        summary = DataInspector.summarize_data(df, 'AAPL')

        assert summary['symbol'] == 'AAPL'
        assert summary['rows'] == 100
        assert summary['trading_days'] == 100
        assert 'start_date' in summary
        assert 'end_date' in summary
        assert 'price_min' in summary
        assert 'price_max' in summary

    def test_summarize_empty_data(self):
        """Test summarizing empty DataFrame."""
        empty_df = pd.DataFrame()
        summary = DataInspector.summarize_data(empty_df, 'TEST')

        assert summary['symbol'] == 'TEST'
        assert summary['rows'] == 0
        assert summary['start_date'] == 'N/A'

    def test_detect_data_issues_clean(self):
        """Test detecting issues in clean data."""
        df = self.create_mock_price_data(100)
        issues = DataInspector.detect_data_issues(df)

        # Should find no issues in clean data
        assert "No data quality issues detected" in issues

    def test_detect_data_issues_missing_values(self):
        """Test detecting missing values."""
        df = self.create_mock_price_data(100)
        # Introduce missing values
        df.loc[df.index[5], 'close'] = np.nan

        issues = DataInspector.detect_data_issues(df)

        assert any("missing values" in issue for issue in issues)

    def test_detect_data_issues_limited_data(self):
        """Test detecting limited data."""
        df = self.create_mock_price_data(30)  # Less than 50 bars
        issues = DataInspector.detect_data_issues(df)

        assert any("Limited data" in issue for issue in issues)

    def test_calculate_returns_summary(self):
        """Test calculating returns statistics."""
        df = self.create_mock_price_data(100)
        returns_stats = DataInspector.calculate_returns_summary(df)

        assert 'total_return' in returns_stats
        assert 'daily_return_mean' in returns_stats
        assert 'daily_return_std' in returns_stats
        assert 'best_day' in returns_stats
        assert 'worst_day' in returns_stats

    def test_calculate_returns_empty_data(self):
        """Test returns calculation with empty data."""
        empty_df = pd.DataFrame()
        returns_stats = DataInspector.calculate_returns_summary(empty_df)

        assert returns_stats['total_return'] == 0.0
        assert returns_stats['daily_return_mean'] == 0.0


class TestSignalAnalyzer:
    """Test signal analyzer functionality."""

    def create_mock_signals(self, num_bars=100, entry_freq=0.1, exit_freq=0.1):
        """Create mock entry/exit signals."""
        dates = pd.date_range('2023-01-01', periods=num_bars, freq='D')

        entries = pd.Series(
            np.random.random(num_bars) < entry_freq,
            index=dates
        )

        exits = pd.Series(
            np.random.random(num_bars) < exit_freq,
            index=dates
        )

        prices = pd.Series(
            np.random.uniform(95, 105, num_bars),
            index=dates
        )

        return entries, exits, prices

    def test_analyze_signals(self):
        """Test signal analysis."""
        entries, exits, prices = self.create_mock_signals(100, 0.1, 0.1)

        analysis = SignalAnalyzer.analyze_signals(entries, exits, prices)

        assert 'total_entries' in analysis
        assert 'total_exits' in analysis
        assert 'signal_ratio' in analysis
        assert analysis['total_entries'] >= 0
        assert analysis['total_exits'] >= 0

    def test_analyze_signals_empty(self):
        """Test analyzing with no signals."""
        analysis = SignalAnalyzer.analyze_signals(None, None, None)

        assert analysis['total_entries'] == 0
        assert analysis['total_exits'] == 0

    def test_calculate_signal_frequency(self):
        """Test signal frequency calculation."""
        entries, _, _ = self.create_mock_signals(100, 0.2, 0.2)

        freq_stats = SignalAnalyzer.calculate_signal_frequency(entries)

        assert 'signals_per_period' in freq_stats
        assert 'days_between_signals' in freq_stats
        assert freq_stats['signals_per_period'] >= 0

    def test_get_signal_summary(self):
        """Test getting human-readable signal summary."""
        entries, exits, prices = self.create_mock_signals(100, 0.1, 0.1)

        summary = SignalAnalyzer.get_signal_summary(entries, exits, prices)

        assert isinstance(summary, str)
        assert "Entry Signals" in summary
        assert "Exit Signals" in summary

    def test_detect_signal_issues_no_signals(self):
        """Test detecting no signals issue."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        entries = pd.Series([False] * 100, index=dates)
        exits = pd.Series([False] * 100, index=dates)

        issues = SignalAnalyzer.detect_signal_issues(entries, exits)

        assert any("no entry or exit signals" in issue.lower() for issue in issues)

    def test_detect_signal_issues_imbalanced(self):
        """Test detecting imbalanced signals."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')

        # Many entries, few exits
        entries = pd.Series([True] * 50 + [False] * 50, index=dates)
        exits = pd.Series([True] * 5 + [False] * 95, index=dates)

        issues = SignalAnalyzer.detect_signal_issues(entries, exits)

        assert any("imbalance" in issue.lower() for issue in issues)

    def test_detect_signal_issues_very_frequent(self):
        """Test detecting overly frequent signals."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')

        # Signals on 60% of bars
        entries = pd.Series([True] * 60 + [False] * 40, index=dates)
        exits = pd.Series([True] * 60 + [False] * 40, index=dates)

        issues = SignalAnalyzer.detect_signal_issues(entries, exits)

        assert any("frequent" in issue.lower() for issue in issues)

    def test_detect_signal_issues_very_rare(self):
        """Test detecting rare signals."""
        dates = pd.date_range('2023-01-01', periods=1000, freq='D')

        # Signals on < 1% of bars
        entries = pd.Series([True] * 5 + [False] * 995, index=dates)
        exits = pd.Series([True] * 5 + [False] * 995, index=dates)

        issues = SignalAnalyzer.detect_signal_issues(entries, exits)

        assert any("rare" in issue.lower() for issue in issues)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
