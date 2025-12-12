"""
NautilusTrader Adapter Module.

Provides adapters to run Homeguard strategies within NautilusTrader's
backtesting engine while maintaining compatibility with existing code.

NautilusTrader is an optional dependency - all code should check
NAUTILUS_AVAILABLE before using NautilusTrader-specific features.

Usage:
    from src.backtesting.adapters.nautilus import NAUTILUS_AVAILABLE

    if NAUTILUS_AVAILABLE:
        from src.backtesting.adapters.nautilus import (
            NautilusStrategyConfig,
            BarAccumulator,
            HomeguardOrderFactory,
            NautilusStrategyAdapter,
            homeguard_to_nautilus_catalog,
        )
"""

# Check if NautilusTrader is available
try:
    from nautilus_trader.trading.strategy import Strategy as NautilusStrategy
    NAUTILUS_AVAILABLE = True
except ImportError:
    NAUTILUS_AVAILABLE = False

# Always export config and utilities (no NautilusTrader dependency)
from src.backtesting.adapters.nautilus.config import (
    NautilusStrategyConfig,
    DataConversionConfig,
)
from src.backtesting.adapters.nautilus.bar_accumulator import BarAccumulator
from src.backtesting.adapters.nautilus.order_factory import HomeguardOrderFactory
from src.backtesting.adapters.nautilus.data_converter import (
    load_homeguard_bars,
    homeguard_to_nautilus_catalog,
    create_backtest_data_configs,
)

# Conditionally export adapters (require NautilusTrader)
if NAUTILUS_AVAILABLE:
    from src.backtesting.adapters.nautilus.base_adapter import NautilusStrategyAdapter
    from src.backtesting.adapters.nautilus.strategy_adapters import (
        StrategySignalsAdapter,
        BaseStrategyAdapter,
        RAMPAdapter,
        OMRAdapter,
        ORBAdapter,
    )
    __all__ = [
        'NAUTILUS_AVAILABLE',
        'NautilusStrategyConfig',
        'DataConversionConfig',
        'BarAccumulator',
        'HomeguardOrderFactory',
        'NautilusStrategyAdapter',
        'StrategySignalsAdapter',
        'BaseStrategyAdapter',
        'RAMPAdapter',
        'OMRAdapter',
        'ORBAdapter',
        'load_homeguard_bars',
        'homeguard_to_nautilus_catalog',
        'create_backtest_data_configs',
    ]
else:
    __all__ = [
        'NAUTILUS_AVAILABLE',
        'NautilusStrategyConfig',
        'DataConversionConfig',
        'BarAccumulator',
        'HomeguardOrderFactory',
        'load_homeguard_bars',
        'homeguard_to_nautilus_catalog',
        'create_backtest_data_configs',
    ]
