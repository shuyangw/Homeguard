"""
Performance monitoring and benchmarking utilities for backtesting.

This module provides tools for:
- Timing code sections
- Memory usage monitoring
- Performance benchmarking
- Lazy evaluation verification
"""

import gc
import time
import functools
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.utils import logger


@dataclass
class TimingResult:
    """Result of a timing measurement."""
    name: str
    duration_ms: float
    iterations: int = 1

    @property
    def avg_ms(self) -> float:
        """Average duration per iteration in milliseconds."""
        return self.duration_ms / self.iterations

    def __repr__(self) -> str:
        if self.iterations > 1:
            return f"{self.name}: {self.duration_ms:.2f}ms total ({self.avg_ms:.2f}ms avg over {self.iterations} iterations)"
        return f"{self.name}: {self.duration_ms:.2f}ms"


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""
    label: str
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    timestamp: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        return f"{self.label}: RSS={self.rss_mb:.1f}MB, VMS={self.vms_mb:.1f}MB"


class PerformanceMonitor:
    """
    Monitor performance metrics during backtest execution.

    Usage:
        monitor = PerformanceMonitor()

        with monitor.time("data_loading"):
            data = loader.load_symbol("AAPL", ...)

        with monitor.time("simulation"):
            portfolio = engine.run(...)

        monitor.print_summary()
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize performance monitor.

        Args:
            enabled: If False, all monitoring is no-op (zero overhead)
        """
        self.enabled = enabled
        self.timings: List[TimingResult] = []
        self.memory_snapshots: List[MemorySnapshot] = []
        self._start_time = time.perf_counter()

    @contextmanager
    def time(self, name: str):
        """
        Context manager for timing a code section.

        Args:
            name: Label for this timing measurement
        """
        if not self.enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            self.timings.append(TimingResult(name=name, duration_ms=duration_ms))

    def time_function(self, name: Optional[str] = None) -> Callable:
        """
        Decorator for timing a function.

        Args:
            name: Optional label (defaults to function name)
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                label = name or func.__name__
                with self.time(label):
                    return func(*args, **kwargs)
            return wrapper
        return decorator

    def snapshot_memory(self, label: str) -> Optional[MemorySnapshot]:
        """
        Take a memory usage snapshot.

        Args:
            label: Label for this snapshot

        Returns:
            MemorySnapshot or None if psutil not available
        """
        if not self.enabled:
            return None

        try:
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            snapshot = MemorySnapshot(
                label=label,
                rss_mb=mem_info.rss / (1024 * 1024),
                vms_mb=mem_info.vms / (1024 * 1024)
            )
            self.memory_snapshots.append(snapshot)
            return snapshot
        except ImportError:
            return None

    def force_gc(self) -> int:
        """
        Force garbage collection and return bytes freed.

        Returns:
            Number of objects collected
        """
        if not self.enabled:
            return 0
        return gc.collect()

    def get_total_time_ms(self) -> float:
        """Get total elapsed time since monitor creation."""
        return (time.perf_counter() - self._start_time) * 1000

    def get_timing_breakdown(self) -> Dict[str, float]:
        """
        Get breakdown of time spent in each timed section.

        Returns:
            Dict mapping section names to total milliseconds
        """
        breakdown: Dict[str, float] = {}
        for timing in self.timings:
            if timing.name in breakdown:
                breakdown[timing.name] += timing.duration_ms
            else:
                breakdown[timing.name] = timing.duration_ms
        return breakdown

    def print_summary(self, min_pct: float = 1.0):
        """
        Print performance summary.

        Args:
            min_pct: Minimum percentage of total time to include in summary
        """
        if not self.enabled or not self.timings:
            return

        total_ms = self.get_total_time_ms()
        breakdown = self.get_timing_breakdown()

        logger.info("\n" + "=" * 60)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total time: {total_ms:.2f}ms ({total_ms/1000:.2f}s)")
        logger.info("")

        # Sort by time descending
        sorted_items = sorted(breakdown.items(), key=lambda x: x[1], reverse=True)

        logger.info("Timing Breakdown:")
        for name, ms in sorted_items:
            pct = (ms / total_ms) * 100 if total_ms > 0 else 0
            if pct >= min_pct:
                logger.info(f"  {name}: {ms:.2f}ms ({pct:.1f}%)")

        # Memory snapshots
        if self.memory_snapshots:
            logger.info("")
            logger.info("Memory Snapshots:")
            for snap in self.memory_snapshots:
                logger.info(f"  {snap}")

        logger.info("=" * 60)


def benchmark(func: Callable, iterations: int = 10, warmup: int = 2) -> TimingResult:
    """
    Benchmark a function over multiple iterations.

    Args:
        func: Function to benchmark (no arguments)
        iterations: Number of timed iterations
        warmup: Number of warmup iterations (not timed)

    Returns:
        TimingResult with total and average times
    """
    # Warmup
    for _ in range(warmup):
        func()

    # Force GC before timing
    gc.collect()

    # Timed runs
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    total_ms = (time.perf_counter() - start) * 1000

    return TimingResult(
        name=getattr(func, '__name__', 'benchmark'),
        duration_ms=total_ms,
        iterations=iterations
    )


def compare_implementations(
    implementations: Dict[str, Callable],
    iterations: int = 10,
    warmup: int = 2
) -> Dict[str, TimingResult]:
    """
    Compare multiple implementations of the same functionality.

    Args:
        implementations: Dict mapping names to callables
        iterations: Number of timed iterations per implementation
        warmup: Number of warmup iterations

    Returns:
        Dict mapping names to TimingResults
    """
    results = {}
    for name, func in implementations.items():
        results[name] = benchmark(func, iterations, warmup)
    return results


def get_memory_usage_mb() -> Optional[float]:
    """
    Get current process memory usage in MB.

    Returns:
        Memory usage in MB or None if psutil not available
    """
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return None


def estimate_dataframe_memory(df: Any) -> float:
    """
    Estimate memory usage of a DataFrame in MB.

    Supports both pandas and polars DataFrames.

    Args:
        df: pandas or polars DataFrame

    Returns:
        Estimated memory usage in MB
    """
    try:
        # Polars DataFrame
        if hasattr(df, 'estimated_size'):
            return df.estimated_size() / (1024 * 1024)
        # pandas DataFrame
        if hasattr(df, 'memory_usage'):
            return df.memory_usage(deep=True).sum() / (1024 * 1024)
    except Exception:
        pass
    return 0.0


@contextmanager
def memory_tracked(label: str):
    """
    Context manager that tracks memory usage before and after.

    Args:
        label: Label for the tracked section

    Yields:
        Dict with 'before', 'after', and 'delta' keys (values in MB)
    """
    result = {'before': 0.0, 'after': 0.0, 'delta': 0.0, 'label': label}

    before = get_memory_usage_mb()
    if before is not None:
        result['before'] = before

    try:
        yield result
    finally:
        after = get_memory_usage_mb()
        if after is not None:
            result['after'] = after
            result['delta'] = after - result['before']
            if abs(result['delta']) > 10:  # Only log significant changes
                logger.debug(f"Memory [{label}]: {result['before']:.1f}MB -> {result['after']:.1f}MB (delta: {result['delta']:+.1f}MB)")
