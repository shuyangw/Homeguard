"""
Performance benchmarks for backtesting engine components.

These tests measure and verify performance characteristics rather than correctness.
Run with: pytest tests/backtesting/engine/test_performance.py -v
"""

import pytest
import time
import numpy as np
import pandas as pd
import polars as pl
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.backtesting.utils.performance import (
    PerformanceMonitor,
    TimingResult,
    benchmark,
    compare_implementations,
    get_memory_usage_mb,
    estimate_dataframe_memory,
    memory_tracked
)


class TestPerformanceMonitor:
    """Test PerformanceMonitor class."""

    def test_timing_context_manager(self):
        """Test time() context manager records duration."""
        monitor = PerformanceMonitor()

        with monitor.time("test_operation"):
            time.sleep(0.01)  # 10ms

        assert len(monitor.timings) == 1
        assert monitor.timings[0].name == "test_operation"
        assert monitor.timings[0].duration_ms >= 10  # At least 10ms

    def test_disabled_monitor_has_zero_overhead(self):
        """Test disabled monitor adds minimal overhead."""
        monitor = PerformanceMonitor(enabled=False)

        start = time.perf_counter()
        for _ in range(1000):
            with monitor.time("iteration"):
                pass
        duration_ms = (time.perf_counter() - start) * 1000

        assert len(monitor.timings) == 0
        assert duration_ms < 50  # Should be very fast when disabled

    def test_timing_breakdown(self):
        """Test get_timing_breakdown aggregates correctly."""
        monitor = PerformanceMonitor()

        with monitor.time("section_a"):
            time.sleep(0.005)
        with monitor.time("section_b"):
            time.sleep(0.010)
        with monitor.time("section_a"):
            time.sleep(0.005)

        breakdown = monitor.get_timing_breakdown()

        assert "section_a" in breakdown
        assert "section_b" in breakdown
        assert breakdown["section_a"] >= 10  # Two 5ms sections
        assert breakdown["section_b"] >= 10  # One 10ms section

    def test_function_decorator(self):
        """Test time_function decorator."""
        monitor = PerformanceMonitor()

        @monitor.time_function("decorated_func")
        def slow_function():
            time.sleep(0.01)
            return 42

        result = slow_function()

        assert result == 42
        assert len(monitor.timings) == 1
        assert monitor.timings[0].name == "decorated_func"


class TestBenchmark:
    """Test benchmark function."""

    def test_benchmark_measures_iterations(self):
        """Test benchmark runs correct number of iterations."""
        call_count = 0

        def counting_func():
            nonlocal call_count
            call_count += 1

        result = benchmark(counting_func, iterations=5, warmup=2)

        assert call_count == 7  # 2 warmup + 5 timed
        assert result.iterations == 5
        assert result.avg_ms == result.duration_ms / 5

    def test_compare_implementations(self):
        """Test comparing multiple implementations."""
        def fast_func():
            pass

        def slow_func():
            time.sleep(0.001)

        results = compare_implementations(
            {"fast": fast_func, "slow": slow_func},
            iterations=3,
            warmup=1
        )

        assert "fast" in results
        assert "slow" in results
        assert results["slow"].duration_ms > results["fast"].duration_ms


class TestMemoryUtilities:
    """Test memory monitoring utilities."""

    def test_estimate_polars_dataframe_memory(self):
        """Test memory estimation for Polars DataFrame."""
        df = pl.DataFrame({
            "a": np.random.randn(10000),
            "b": np.random.randn(10000),
            "c": np.random.randn(10000),
        })

        size_mb = estimate_dataframe_memory(df)
        assert size_mb > 0  # Should be > 0 for non-empty DataFrame

    def test_estimate_pandas_dataframe_memory(self):
        """Test memory estimation for pandas DataFrame."""
        df = pd.DataFrame({
            "a": np.random.randn(10000),
            "b": np.random.randn(10000),
            "c": np.random.randn(10000),
        })

        size_mb = estimate_dataframe_memory(df)
        assert size_mb > 0

    def test_memory_tracked_context(self):
        """Test memory_tracked context manager."""
        with memory_tracked("test") as result:
            # Allocate some memory
            data = np.zeros((1000, 1000))  # ~8MB

        assert "before" in result
        assert "after" in result
        assert "delta" in result
        assert "label" in result
        # Note: delta may not be accurate due to GC, just verify structure


class TestTimingResult:
    """Test TimingResult dataclass."""

    def test_repr_single_iteration(self):
        """Test string representation for single iteration."""
        result = TimingResult(name="test", duration_ms=100.5, iterations=1)
        repr_str = repr(result)

        assert "test" in repr_str
        assert "100.50ms" in repr_str

    def test_repr_multiple_iterations(self):
        """Test string representation for multiple iterations."""
        result = TimingResult(name="test", duration_ms=100.0, iterations=10)
        repr_str = repr(result)

        assert "test" in repr_str
        assert "100.00ms total" in repr_str
        assert "10.00ms avg" in repr_str
        assert "10 iterations" in repr_str

    def test_avg_ms_calculation(self):
        """Test average calculation."""
        result = TimingResult(name="test", duration_ms=100.0, iterations=4)
        assert result.avg_ms == 25.0


class TestPolarsVsPandasPerformance:
    """Compare Polars vs pandas for common operations."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for benchmarks."""
        n = 100000
        np.random.seed(42)
        return {
            "timestamp": pd.date_range("2023-01-01", periods=n, freq="T"),
            "open": np.random.randn(n).cumsum() + 100,
            "high": np.random.randn(n).cumsum() + 101,
            "low": np.random.randn(n).cumsum() + 99,
            "close": np.random.randn(n).cumsum() + 100,
            "volume": np.random.randint(1000, 100000, n).astype(float),
        }

    def test_filtering_performance(self, sample_data):
        """Test filtering performance Polars vs pandas."""
        df_pd = pd.DataFrame(sample_data)
        df_pl = pl.DataFrame(sample_data)

        # pandas filtering
        def filter_pandas():
            return df_pd[df_pd["close"] > df_pd["open"]]

        # Polars filtering
        def filter_polars():
            return df_pl.filter(pl.col("close") > pl.col("open"))

        results = compare_implementations(
            {"pandas": filter_pandas, "polars": filter_polars},
            iterations=10,
            warmup=2
        )

        # Polars should be comparable or faster
        # We don't enforce this strictly as performance varies
        assert results["polars"].duration_ms < results["pandas"].duration_ms * 2

    def test_aggregation_performance(self, sample_data):
        """Test aggregation performance Polars vs pandas."""
        df_pd = pd.DataFrame(sample_data)
        df_pl = pl.DataFrame(sample_data)

        # Add date column for grouping
        df_pd["date"] = df_pd["timestamp"].dt.date
        df_pl = df_pl.with_columns(pl.col("timestamp").dt.date().alias("date"))

        # pandas aggregation
        def agg_pandas():
            return df_pd.groupby("date").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum"
            })

        # Polars aggregation
        def agg_polars():
            return df_pl.group_by("date").agg([
                pl.col("open").first(),
                pl.col("high").max(),
                pl.col("low").min(),
                pl.col("close").last(),
                pl.col("volume").sum()
            ])

        results = compare_implementations(
            {"pandas": agg_pandas, "polars": agg_polars},
            iterations=10,
            warmup=2
        )

        # Just verify both complete successfully
        assert results["pandas"].duration_ms > 0
        assert results["polars"].duration_ms > 0

    def test_returns_calculation_performance(self, sample_data):
        """Test returns calculation performance."""
        df_pd = pd.DataFrame(sample_data)
        df_pl = pl.DataFrame(sample_data)

        # pandas returns
        def returns_pandas():
            return df_pd["close"].pct_change().fillna(0)

        # Polars returns
        def returns_polars():
            return df_pl.select(
                pl.col("close").pct_change().fill_null(0).alias("returns")
            )

        results = compare_implementations(
            {"pandas": returns_pandas, "polars": returns_polars},
            iterations=20,
            warmup=3
        )

        # Both should complete quickly
        assert results["pandas"].avg_ms < 100  # Less than 100ms average
        assert results["polars"].avg_ms < 100


class TestLazyEvaluationBenefits:
    """Test that lazy evaluation provides performance benefits."""

    def test_lazy_filter_chain_optimization(self):
        """Test that chained filters benefit from lazy evaluation."""
        n = 100000
        df = pl.DataFrame({
            "a": np.random.randn(n),
            "b": np.random.randn(n),
            "c": np.random.randn(n),
        })

        # Eager: each operation materializes
        def eager_chain():
            result = df.filter(pl.col("a") > 0)
            result = result.filter(pl.col("b") > 0)
            result = result.filter(pl.col("c") > 0)
            return result

        # Lazy: operations fused
        def lazy_chain():
            return (
                df.lazy()
                .filter(pl.col("a") > 0)
                .filter(pl.col("b") > 0)
                .filter(pl.col("c") > 0)
                .collect()
            )

        results = compare_implementations(
            {"eager": eager_chain, "lazy": lazy_chain},
            iterations=10,
            warmup=2
        )

        # Lazy should be faster due to predicate pushdown
        # Allow some tolerance for variability
        assert results["lazy"].duration_ms <= results["eager"].duration_ms * 1.5

    def test_lazy_projection_optimization(self):
        """Test that column selection benefits from lazy evaluation."""
        n = 100000
        df = pl.DataFrame({
            "a": np.random.randn(n),
            "b": np.random.randn(n),
            "c": np.random.randn(n),
            "d": np.random.randn(n),
            "e": np.random.randn(n),
        })

        # Eager: loads all columns first
        def eager_select():
            result = df.filter(pl.col("a") > 0)
            return result.select(["a", "b"])

        # Lazy: only loads needed columns
        def lazy_select():
            return (
                df.lazy()
                .filter(pl.col("a") > 0)
                .select(["a", "b"])
                .collect()
            )

        results = compare_implementations(
            {"eager": eager_select, "lazy": lazy_select},
            iterations=10,
            warmup=2
        )

        # Both should work correctly
        assert results["eager"].duration_ms > 0
        assert results["lazy"].duration_ms > 0
