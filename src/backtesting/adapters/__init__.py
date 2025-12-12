"""
Backtest Adapters.

Adapters that connect pure strategy implementations to the backtesting engine.

Each adapter:
1. Extends the backtest engine's Strategy base class
2. Wraps a pure StrategySignals implementation
3. Translates between backtest engine format and pure strategy format

Usage:
    ```python
    from src.backtesting.adapters.ma_backtest_adapter import MABacktestAdapter

    # Create backtest adapter
    strategy = MABacktestAdapter({
        'fast_period': 50,
        'slow_period': 200,
        'symbols': ['AAPL', 'MSFT']
    })

    # Run backtest
    results = run_backtest(portfolio, strategy, historical_data)
    ```

NautilusTrader Adapters:
    ```python
    from src.backtesting.adapters.nautilus import (
        NAUTILUS_AVAILABLE,
        NautilusStrategyConfig,
        BarAccumulator,
        HomeguardOrderFactory,
        homeguard_to_nautilus_catalog,
    )

    if NAUTILUS_AVAILABLE:
        from src.backtesting.adapters.nautilus import (
            NautilusStrategyAdapter,
            RAMPAdapter,
            OMRAdapter,
            ORBAdapter,
        )
    ```
"""

# Import nautilus module (optional dependency)
from src.backtesting.adapters.nautilus import NAUTILUS_AVAILABLE

__all__ = ['NAUTILUS_AVAILABLE']
