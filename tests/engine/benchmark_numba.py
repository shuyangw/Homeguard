"""Benchmark Numba vs Python portfolio simulation performance."""
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import time
from src.backtesting.engine.portfolio_simulator import Portfolio
from src.backtesting.utils.risk_config import RiskConfig

def generate_test_data(n_bars: int, seed: int = 42):
    """Generate realistic price and signal data."""
    np.random.seed(seed)
    dates = pd.date_range('2020-01-01 09:30:00', periods=n_bars, freq='min', tz='US/Eastern')

    # Random walk price series
    returns = np.random.randn(n_bars) * 0.001  # 0.1% per bar volatility
    prices = 100 * (1 + returns).cumprod()
    price_series = pd.Series(prices, index=dates)

    # Generate signals (roughly 5% of bars have signals)
    entries = pd.Series(np.random.random(n_bars) < 0.025, index=dates)
    exits = pd.Series(np.random.random(n_bars) < 0.025, index=dates)

    return price_series, entries, exits

def benchmark_single(n_bars: int, n_runs: int = 5, warmup: bool = True):
    """Benchmark a single data size."""
    price, entries, exits = generate_test_data(n_bars)
    risk_config = RiskConfig(use_stop_loss=False)

    # Warmup Numba (JIT compilation)
    if warmup:
        Portfolio(price, entries, exits, 10000, 0.001, 0.0005,
                  risk_config=risk_config, market_hours_only=False, use_numba=True)

    # Benchmark Python
    python_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        Portfolio(price, entries, exits, 10000, 0.001, 0.0005,
                  risk_config=risk_config, market_hours_only=False, use_numba=False)
        python_times.append(time.perf_counter() - start)

    # Benchmark Numba
    numba_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        Portfolio(price, entries, exits, 10000, 0.001, 0.0005,
                  risk_config=risk_config, market_hours_only=False, use_numba=True)
        numba_times.append(time.perf_counter() - start)

    python_avg = np.mean(python_times)
    numba_avg = np.mean(numba_times)
    speedup = python_avg / numba_avg

    return python_avg, numba_avg, speedup

def main():
    print("=" * 70)
    print("NUMBA vs PYTHON PORTFOLIO SIMULATION BENCHMARK")
    print("=" * 70)

    # Test different data sizes
    test_sizes = [
        (1_000, "1K bars (~2 trading hours)"),
        (10_000, "10K bars (~1 week intraday)"),
        (50_000, "50K bars (~1 month intraday)"),
        (100_000, "100K bars (~2 months intraday)"),
        (390_000, "390K bars (~1 year @ 1-min)"),
    ]

    print(f"\n{'Bars':<12} {'Python (s)':<12} {'Numba (s)':<12} {'Speedup':<10} Description")
    print("-" * 70)

    results = []
    for n_bars, description in test_sizes:
        try:
            python_time, numba_time, speedup = benchmark_single(n_bars, n_runs=3)
            print(f"{n_bars:<12,} {python_time:<12.4f} {numba_time:<12.4f} {speedup:<10.1f}x {description}")
            results.append((n_bars, python_time, numba_time, speedup))
        except Exception as e:
            print(f"{n_bars:<12,} ERROR: {e}")

    print("-" * 70)

    # Summary
    if results:
        avg_speedup = np.mean([r[3] for r in results])
        print(f"\nAverage speedup: {avg_speedup:.1f}x")

        # Estimate optimization time savings
        print("\n" + "=" * 70)
        print("OPTIMIZATION SCENARIO ESTIMATES")
        print("=" * 70)

        # Use the 390K bars result for yearly data
        if len(results) >= 5:
            py_yearly, nb_yearly = results[4][1], results[4][2]
        else:
            py_yearly, nb_yearly = results[-1][1], results[-1][2]

        scenarios = [
            ("Single symbol, 1 year", 1, 1),
            ("500 symbols, 1 year", 500, 1),
            ("500 symbols, 7 years", 500, 7),
            ("100 param sweep, 500 symbols", 500 * 100, 1),
        ]

        print(f"\n{'Scenario':<35} {'Python':<15} {'Numba':<15} {'Saved'}")
        print("-" * 70)

        for name, multiplier, years in scenarios:
            py_total = py_yearly * multiplier * years
            nb_total = nb_yearly * multiplier * years
            saved = py_total - nb_total

            def format_time(seconds):
                if seconds < 60:
                    return f"{seconds:.1f}s"
                elif seconds < 3600:
                    return f"{seconds/60:.1f}min"
                else:
                    return f"{seconds/3600:.1f}h"

            print(f"{name:<35} {format_time(py_total):<15} {format_time(nb_total):<15} {format_time(saved)}")

if __name__ == '__main__':
    main()
