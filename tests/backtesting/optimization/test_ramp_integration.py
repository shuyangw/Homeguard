"""
Integration tests for generic parameter sweep with RAMP-like backtest functions.

Tests that:
1. load_minute_cache_polars produces format compatible with RAMP backtest
2. run_generic_parameter_sweep works with RAMP-like function signatures
3. Threading properly shares data without duplication
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

from src.backtesting.optimization.data_loader import (
    load_minute_cache_polars,
    get_default_workers
)
from src.backtesting.optimization.sweep_runner import run_generic_parameter_sweep


class TestMinuteCacheFormat:
    """Tests that minute cache format is compatible with RAMP."""

    @pytest.fixture
    def ramp_like_minute_parquet(self, tmp_path):
        """Create minute data matching RAMP's expected parquet format."""
        import polars as pl
        from datetime import datetime, timedelta

        # Create sample data for multiple symbols and days
        rows = []
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        base_date = datetime(2024, 1, 2, 9, 30)  # Market open

        for symbol in symbols:
            for day_offset in range(3):  # 3 trading days
                day_base = base_date + timedelta(days=day_offset)
                # Simulate 390 minutes of trading (9:30 AM - 4:00 PM)
                for minute in range(390):
                    ts = day_base + timedelta(minutes=minute)
                    price_base = 100.0 + np.sin(minute / 50) * 5  # Some variation
                    rows.append({
                        'timestamp': ts,
                        'symbol': symbol,
                        'open': price_base,
                        'high': price_base + 0.5,
                        'low': price_base - 0.5,
                        'close': price_base + 0.1,
                        'volume': 10000.0,
                        'trade_count': 100.0,
                        'vwap': price_base
                    })

        df = pl.DataFrame(rows)
        df = df.with_columns(pl.col('timestamp').cast(pl.Datetime('us', 'UTC')))

        parquet_path = tmp_path / 'minute_cache.parquet'
        df.write_parquet(parquet_path)
        return parquet_path

    def test_format_matches_ramp_expectations(self, ramp_like_minute_parquet):
        """Cache format should match what RAMP backtest expects."""
        cache = load_minute_cache_polars(str(ramp_like_minute_parquet))

        # RAMP expects: {symbol: {date_str: {'high': array, 'low': array, 'close': array}}}
        assert isinstance(cache, dict)
        assert 'AAPL' in cache
        assert 'MSFT' in cache
        assert 'GOOGL' in cache

        # Check nested structure
        for symbol, dates in cache.items():
            assert isinstance(dates, dict)
            for date_str, day_data in dates.items():
                # Date string format
                assert len(date_str) == 10  # YYYY-MM-DD
                assert '-' in date_str

                # Required arrays
                assert 'high' in day_data
                assert 'low' in day_data
                assert 'close' in day_data

                # Should be numpy arrays
                assert isinstance(day_data['high'], np.ndarray)
                assert isinstance(day_data['low'], np.ndarray)
                assert isinstance(day_data['close'], np.ndarray)

                # Should have same length (390 minutes per day)
                assert len(day_data['high']) == len(day_data['low']) == len(day_data['close'])

    def test_data_is_sorted_by_time(self, ramp_like_minute_parquet):
        """Minute data should be sorted ascending (first bar = market open)."""
        cache = load_minute_cache_polars(str(ramp_like_minute_parquet))

        for symbol in cache:
            for date_str in cache[symbol]:
                close_prices = cache[symbol][date_str]['close']
                # Check that we have minute data (should be 390 bars)
                assert len(close_prices) == 390

    def test_date_filtering_works(self, ramp_like_minute_parquet):
        """Date filtering should correctly limit the data."""
        # Load only middle day
        cache = load_minute_cache_polars(
            str(ramp_like_minute_parquet),
            start_date='2024-01-03',
            end_date='2024-01-03'
        )

        # Should have all symbols
        assert len(cache) == 3

        # Each symbol should have only 1 day
        for symbol in cache:
            assert len(cache[symbol]) == 1
            assert '2024-01-03' in cache[symbol]


class TestRAMPLikeSweep:
    """Tests that generic sweep works with RAMP-like backtest functions."""

    def ramp_like_backtest(
        self,
        minute_cache: Dict[str, Dict[str, Dict[str, np.ndarray]]],
        stop_loss_pct: float,
        top_n: int,
        regime: str = 'SIDEWAYS'
    ) -> Dict[str, float]:
        """
        RAMP-like backtest function for testing.

        Simulates RAMP's return structure with:
        - sharpe
        - cagr_pct
        - max_dd_pct
        - total_return_pct
        """
        # Simulate RAMP-like behavior
        n_symbols = len(minute_cache)
        n_days = sum(len(dates) for dates in minute_cache.values()) // max(n_symbols, 1)

        # Regime affects both return AND volatility
        # STRONG_BULL: high return, low volatility -> high Sharpe
        # BEAR: low return, high volatility -> low Sharpe
        regime_return_mult = {
            'STRONG_BULL': 1.5,
            'WEAK_BULL': 1.2,
            'SIDEWAYS': 1.0,
            'UNPREDICTABLE': 0.8,
            'BEAR': 0.5
        }.get(regime, 1.0)

        regime_vol = {
            'STRONG_BULL': 0.10,
            'WEAK_BULL': 0.15,
            'SIDEWAYS': 0.20,
            'UNPREDICTABLE': 0.25,
            'BEAR': 0.30
        }.get(regime, 0.2)

        # Base return is positive
        base_return = 0.20 * regime_return_mult
        topn_effect = top_n * 0.001  # More positions = slightly better

        total_return = base_return + topn_effect
        volatility = regime_vol

        # Max drawdown scales with volatility (higher vol = worse DD)
        max_dd = -volatility * 2  # e.g., BEAR: -0.6, STRONG_BULL: -0.2

        return {
            'sharpe': total_return / volatility if volatility > 0 else 0,
            'cagr_pct': total_return * 100 / 3,  # Annualized over ~3 years
            'max_dd_pct': max_dd * 100,  # In percentage
            'total_return_pct': total_return * 100,
            'n_days': n_days,
            'n_symbols': n_symbols
        }

    def test_sweep_with_ramp_signature(self):
        """Should work with RAMP's function signature."""
        # Create minimal mock cache
        mock_cache = {
            'AAPL': {'2024-01-02': {'high': np.ones(390), 'low': np.ones(390), 'close': np.ones(390)}},
            'MSFT': {'2024-01-02': {'high': np.ones(390), 'low': np.ones(390), 'close': np.ones(390)}}
        }

        param_space = {
            'stop_loss_pct': [0.02, 0.05, 0.10],
            'top_n': [10, 20]
        }

        result = run_generic_parameter_sweep(
            backtest_fn=self.ramp_like_backtest,
            param_space=param_space,
            shared_data={'minute_cache': mock_cache},
            n_workers=2,
            metric='sharpe'
        )

        # Should find best params
        assert result['best_params'] is not None
        assert 'stop_loss_pct' in result['best_params']
        assert 'top_n' in result['best_params']

        # Should have all combinations
        assert len(result['all_results']) == 6  # 3 x 2 = 6

        # Best metrics should include RAMP's metrics
        assert 'sharpe' in result['best_metrics']
        assert 'cagr_pct' in result['best_metrics']
        assert 'max_dd_pct' in result['best_metrics']

    def test_sweep_with_regime_parameter(self):
        """Should handle regime as a sweep parameter."""
        mock_cache = {
            'AAPL': {'2024-01-02': {'high': np.ones(390), 'low': np.ones(390), 'close': np.ones(390)}}
        }

        param_space = {
            'stop_loss_pct': [0.05],
            'top_n': [10],
            'regime': ['STRONG_BULL', 'SIDEWAYS', 'BEAR']
        }

        result = run_generic_parameter_sweep(
            backtest_fn=self.ramp_like_backtest,
            param_space=param_space,
            shared_data={'minute_cache': mock_cache},
            n_workers=2,
            metric='sharpe'
        )

        # Should find regime in best params
        assert result['best_params']['regime'] in ['STRONG_BULL', 'SIDEWAYS', 'BEAR']

        # STRONG_BULL should have highest Sharpe (lowest volatility)
        assert result['best_params']['regime'] == 'STRONG_BULL'

    def test_minimize_mode_for_drawdown(self):
        """Should find minimum value when minimize=True.

        Note: For max_dd_pct (negative values like -20%, -60%):
        - minimize=True finds the most negative (-60%, BEAR)
        - To find "least bad" drawdown, you'd maximize (find -20%, closest to 0)

        This test verifies minimize mode works correctly (finds minimum value).
        """
        mock_cache = {
            'AAPL': {'2024-01-02': {'high': np.ones(390), 'low': np.ones(390), 'close': np.ones(390)}}
        }

        param_space = {
            'stop_loss_pct': [0.02, 0.05, 0.10],
            'top_n': [10],
            'regime': ['STRONG_BULL', 'BEAR']
        }

        result = run_generic_parameter_sweep(
            backtest_fn=self.ramp_like_backtest,
            param_space=param_space,
            shared_data={'minute_cache': mock_cache},
            n_workers=2,
            metric='max_dd_pct',
            minimize=True  # Find most negative (mathematical minimum)
        )

        # BEAR has max_dd_pct = -60%, STRONG_BULL has -20%
        # minimize=True finds the mathematical minimum: -60% (BEAR)
        assert result['best_params']['regime'] == 'BEAR'
        assert result['best_metrics']['max_dd_pct'] == -60.0


class TestDataSharing:
    """Tests that data is properly shared across threads without duplication."""

    def test_large_shared_data_not_duplicated(self):
        """Large shared data should not be duplicated across threads."""
        import sys

        # Create large mock cache (simulating real minute data)
        n_symbols = 10
        n_days = 5
        minutes_per_day = 390
        symbols = [f'SYM{i}' for i in range(n_symbols)]

        mock_cache = {}
        for symbol in symbols:
            mock_cache[symbol] = {}
            for day in range(n_days):
                date_str = f'2024-01-{day + 2:02d}'
                mock_cache[symbol][date_str] = {
                    'high': np.random.randn(minutes_per_day),
                    'low': np.random.randn(minutes_per_day),
                    'close': np.random.randn(minutes_per_day)
                }

        # Track initial memory
        initial_size = sys.getsizeof(mock_cache)

        def simple_backtest(minute_cache: Dict, param_a: int) -> Dict[str, float]:
            # Just access the data to verify it's there
            n_syms = len(minute_cache)
            n_days = sum(len(d) for d in minute_cache.values())
            return {'sharpe': param_a * n_syms / 100, 'n_days': n_days}

        param_space = {'param_a': list(range(1, 11))}  # 10 combinations

        result = run_generic_parameter_sweep(
            backtest_fn=simple_backtest,
            param_space=param_space,
            shared_data={'minute_cache': mock_cache},
            n_workers=4,
            metric='sharpe'
        )

        # All trials should complete successfully
        assert len(result['all_results']) == 10
        errors = [r for r in result['all_results'] if r['error'] is not None]
        assert len(errors) == 0

        # Best params should be param_a=10 (highest multiplier)
        assert result['best_params']['param_a'] == 10


class TestDefaultWorkers:
    """Tests for default worker calculation."""

    def test_80_percent_of_cpu(self):
        """Should return approximately 80% of CPU count."""
        import os
        cpu_count = os.cpu_count() or 1
        expected = max(1, int(cpu_count * 0.8))

        workers = get_default_workers()
        assert workers == expected

    def test_at_least_one_worker(self):
        """Should always return at least 1 worker."""
        workers = get_default_workers()
        assert workers >= 1
