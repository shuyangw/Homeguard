# Implementation Example: Decoupled Architecture

## Complete Working Example

This document shows a **complete, working implementation** of the decoupled architecture for a Moving Average Crossover strategy.

## File Structure

```
src/
├── strategies/                          # PURE STRATEGY LOGIC
│   ├── core/
│   │   ├── base_strategy.py            # [NEW] Abstract interface
│   │   └── signal.py                   # [NEW] Signal data structure
│   ├── implementations/
│   │   └── moving_average/
│   │       └── ma_crossover_signals.py # [NEW] Pure MA logic
│   └── universe/
│       └── equity_universe.py          # [NEW] Symbol lists
│
├── backtesting/                         # BACKTEST ADAPTERS
│   └── adapters/
│       └── ma_backtest_adapter.py      # [NEW] Backtest adapter
│
└── trading/                             # LIVE TRADING ADAPTERS
    └── adapters/
        └── ma_live_adapter.py          # [NEW] Live trading adapter
```

## 1. Core Abstract Interface

### src/strategies/core/signal.py
```python
"""Signal data structure - used by all strategies."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict


@dataclass
class Signal:
    """
    Pure signal data structure.

    No dependencies on backtesting or live trading infrastructure.
    """
    timestamp: datetime
    symbol: str
    direction: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0.0 to 1.0
    price: float  # Signal price
    metadata: Dict  # Strategy-specific data (indicators, reasoning, etc.)

    def __str__(self):
        return (f"Signal({self.symbol} {self.direction} @ ${self.price:.2f}, "
                f"confidence={self.confidence:.1%})")
```

### src/strategies/core/base_strategy.py
```python
"""
Abstract base class for pure strategy implementations.

No dependencies on:
- Backtesting engine (Portfolio, data access methods)
- Live trading infrastructure (BrokerInterface, ExecutionEngine)
"""

from abc import ABC, abstractmethod
from typing import Dict, List
from datetime import datetime
import pandas as pd

from src.strategies.core.signal import Signal


class StrategySignals(ABC):
    """
    Pure signal generation strategy.

    Subclasses implement signal logic without any knowledge of
    how signals will be executed (backtest vs live trading).
    """

    @abstractmethod
    def generate_signals(
        self,
        market_data: Dict[str, pd.DataFrame],
        timestamp: datetime
    ) -> List[Signal]:
        """
        Generate trading signals based on market data.

        Args:
            market_data: Dict of symbol -> DataFrame with columns:
                         ['open', 'high', 'low', 'close', 'volume']
                         Index must be DatetimeIndex
            timestamp: Current timestamp for signal generation

        Returns:
            List of Signal objects
        """
        pass

    @abstractmethod
    def get_required_lookback(self) -> int:
        """
        Return number of periods needed for indicator calculation.

        For example:
        - MA crossover (50/200): return 200
        - RSI (14): return 14
        """
        pass

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate that DataFrame has required structure.

        Can be overridden by subclasses for custom validation.
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']

        if not all(col in df.columns for col in required_columns):
            return False

        if not isinstance(df.index, pd.DatetimeIndex):
            return False

        if len(df) < self.get_required_lookback():
            return False

        return True
```

## 2. Pure Strategy Implementation

### src/strategies/implementations/moving_average/ma_crossover_signals.py
```python
"""
Pure Moving Average Crossover signal generation.

No dependencies on backtesting or live trading infrastructure.
"""

from typing import Dict, List
from datetime import datetime
import pandas as pd
import numpy as np

from src.strategies.core.base_strategy import StrategySignals
from src.strategies.core.signal import Signal


class MACrossoverSignals(StrategySignals):
    """
    Generate signals based on moving average crossovers.

    Golden cross: fast MA crosses above slow MA -> BUY
    Death cross: fast MA crosses below slow MA -> SELL
    """

    def __init__(self, fast_period: int = 50, slow_period: int = 200):
        """
        Initialize MA crossover strategy.

        Args:
            fast_period: Fast MA period (default 50)
            slow_period: Slow MA period (default 200)
        """
        if fast_period >= slow_period:
            raise ValueError("Fast period must be less than slow period")

        self.fast_period = fast_period
        self.slow_period = slow_period

    def get_required_lookback(self) -> int:
        """Need enough data for slow MA calculation."""
        return self.slow_period + 1  # +1 for crossover detection

    def generate_signals(
        self,
        market_data: Dict[str, pd.DataFrame],
        timestamp: datetime
    ) -> List[Signal]:
        """
        Generate MA crossover signals.

        Args:
            market_data: Dict of symbol -> OHLCV DataFrame
            timestamp: Current timestamp

        Returns:
            List of signals (one per symbol if crossover detected)
        """
        signals = []

        for symbol, df in market_data.items():
            # Validate data
            if not self.validate_data(df):
                continue

            # Calculate moving averages
            df = df.copy()
            df['fast_ma'] = df['close'].rolling(self.fast_period).mean()
            df['slow_ma'] = df['close'].rolling(self.slow_period).mean()

            # Drop NaN values from MA calculation
            df = df.dropna()

            if len(df) < 2:
                continue  # Need at least 2 periods for crossover

            # Get current and previous values
            current = df.iloc[-1]
            previous = df.iloc[-2]

            # Detect crossovers
            signal_direction = None
            confidence = 0.0

            # Golden cross (bullish)
            if (current['fast_ma'] > current['slow_ma'] and
                previous['fast_ma'] <= previous['slow_ma']):

                signal_direction = 'BUY'

                # Calculate confidence based on gap between MAs
                ma_gap = (current['fast_ma'] - current['slow_ma']) / current['slow_ma']
                confidence = min(0.5 + ma_gap * 10, 1.0)  # 0.5 to 1.0

            # Death cross (bearish)
            elif (current['fast_ma'] < current['slow_ma'] and
                  previous['fast_ma'] >= previous['slow_ma']):

                signal_direction = 'SELL'

                # Calculate confidence based on gap between MAs
                ma_gap = (current['slow_ma'] - current['fast_ma']) / current['slow_ma']
                confidence = min(0.5 + ma_gap * 10, 1.0)  # 0.5 to 1.0

            # Create signal if crossover detected
            if signal_direction:
                signals.append(Signal(
                    timestamp=timestamp,
                    symbol=symbol,
                    direction=signal_direction,
                    confidence=confidence,
                    price=current['close'],
                    metadata={
                        'fast_ma': current['fast_ma'],
                        'slow_ma': current['slow_ma'],
                        'fast_period': self.fast_period,
                        'slow_period': self.slow_period,
                        'strategy': 'MA_Crossover',
                        'ma_gap_pct': abs(current['fast_ma'] - current['slow_ma']) / current['slow_ma']
                    }
                ))

        return signals
```

## 3. Backtest Adapter

### src/backtesting/adapters/ma_backtest_adapter.py
```python
"""
Adapter that connects pure MA signal logic to backtesting engine.

This adapter bridges the gap between:
- Pure strategy (MACrossoverSignals)
- Backtesting engine (Portfolio, Strategy base class)
"""

from typing import Dict
from datetime import datetime
import pandas as pd

from src.engine.base_strategy import Strategy  # Backtest engine base class
from src.strategies.implementations.moving_average.ma_crossover_signals import MACrossoverSignals
from src.strategies.core.signal import Signal


class MABacktestAdapter(Strategy):
    """
    Backtest adapter for MA crossover strategy.

    Translates between:
    - Backtest engine expectations (generate_signals method with specific signature)
    - Pure strategy interface (StrategySignals)
    """

    def __init__(self, params: Dict):
        """
        Initialize backtest adapter.

        Args:
            params: Strategy parameters from backtest config
                    {
                        'fast_period': 50,
                        'slow_period': 200,
                        ...
                    }
        """
        super().__init__(params)

        # Create pure signal generator
        self.signal_generator = MACrossoverSignals(
            fast_period=params.get('fast_period', 50),
            slow_period=params.get('slow_period', 200)
        )

    def generate_signals(self, data: pd.DataFrame, timestamp: datetime) -> Dict:
        """
        Backtest engine interface.

        This method is called by the backtesting engine.
        We adapt it to call our pure strategy.

        Args:
            data: Backtest engine's data structure
            timestamp: Current simulation timestamp

        Returns:
            Dict of signals in backtest engine format:
            {
                'AAPL': {'action': 'BUY', 'quantity': 100, ...},
                'MSFT': {'action': 'SELL', 'quantity': 50, ...}
            }
        """
        # Step 1: Convert backtest data to pure strategy format
        market_data = self._extract_market_data(data)

        # Step 2: Get pure signals
        pure_signals = self.signal_generator.generate_signals(market_data, timestamp)

        # Step 3: Convert pure signals to backtest engine format
        backtest_signals = self._convert_to_backtest_format(pure_signals)

        return backtest_signals

    def _extract_market_data(self, backtest_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Convert backtest engine data to pure strategy format.

        Args:
            backtest_data: MultiIndex DataFrame from backtest engine
                          Index: (symbol, timestamp)
                          Columns: ['open', 'high', 'low', 'close', 'volume']

        Returns:
            Dict of symbol -> DataFrame for pure strategy
        """
        market_data = {}

        if isinstance(backtest_data.index, pd.MultiIndex):
            # Extract each symbol's data
            for symbol in backtest_data.index.get_level_values(0).unique():
                symbol_data = backtest_data.xs(symbol, level=0, drop_level=True)
                market_data[symbol] = symbol_data
        else:
            # Single symbol case
            symbol = self.params.get('symbol', 'SPY')
            market_data[symbol] = backtest_data

        return market_data

    def _convert_to_backtest_format(self, pure_signals: List[Signal]) -> Dict:
        """
        Convert pure signals to backtest engine format.

        Args:
            pure_signals: List of Signal objects from pure strategy

        Returns:
            Dict in backtest engine format:
            {
                'AAPL': {
                    'action': 'BUY',
                    'quantity': 100,
                    'price': 150.25,
                    'metadata': {...}
                }
            }
        """
        backtest_signals = {}

        for signal in pure_signals:
            # Calculate position size based on backtest config
            position_size = self._calculate_position_size(signal)

            backtest_signals[signal.symbol] = {
                'action': signal.direction,
                'quantity': position_size,
                'price': signal.price,
                'confidence': signal.confidence,
                'metadata': signal.metadata
            }

        return backtest_signals

    def _calculate_position_size(self, signal: Signal) -> int:
        """
        Calculate position size based on backtest parameters.

        This uses backtest-specific logic (portfolio value, position sizing method).
        """
        # Get portfolio value from backtest engine
        portfolio_value = self.portfolio.get_total_value()

        # Get position size percentage from params
        position_pct = self.params.get('position_size_pct', 0.10)

        # Calculate dollar amount
        position_value = portfolio_value * position_pct

        # Convert to shares
        shares = int(position_value / signal.price)

        return shares
```

## 4. Live Trading Adapter

### src/trading/adapters/ma_live_adapter.py
```python
"""
Adapter that connects pure MA signal logic to live trading infrastructure.

This adapter bridges the gap between:
- Pure strategy (MACrossoverSignals)
- Live trading infrastructure (TradingBot, BrokerInterface)
"""

from typing import Dict, List
from datetime import datetime
import pandas as pd

from src.trading.strategies.base_strategy import TradingStrategy  # Live trading base class
from src.trading.brokers.broker_interface import BrokerInterface, OrderSide
from src.strategies.implementations.moving_average.ma_crossover_signals import MACrossoverSignals


class MALiveAdapter(TradingStrategy):
    """
    Live trading adapter for MA crossover strategy.

    Translates between:
    - TradingBot expectations (generate_entry_signals, generate_exit_signals)
    - Pure strategy interface (StrategySignals)
    """

    def __init__(self, config: Dict):
        """
        Initialize live trading adapter.

        Args:
            config: Strategy configuration
                    {
                        'symbols': ['AAPL', 'MSFT', 'GOOGL'],
                        'fast_period': 50,
                        'slow_period': 200,
                        'position_size_pct': 0.10
                    }
        """
        self.symbols = config.get('symbols', [])
        self.position_size_pct = config.get('position_size_pct', 0.10)

        # Create pure signal generator
        self.signal_generator = MACrossoverSignals(
            fast_period=config.get('fast_period', 50),
            slow_period=config.get('slow_period', 200)
        )

        self.is_trained = True  # MA doesn't need training

    def train(self, historical_data: Dict[str, pd.DataFrame]) -> None:
        """MA strategy doesn't require training."""
        pass

    def generate_entry_signals(
        self,
        current_data: Dict[str, pd.DataFrame],
        broker: BrokerInterface
    ) -> List[Dict]:
        """
        Generate entry signals for live trading.

        This is called by TradingBot at scheduled times.

        Args:
            current_data: Dict of symbol -> recent OHLCV DataFrame
            broker: Broker interface for fetching quotes

        Returns:
            List of entry signal dicts:
            [
                {
                    'symbol': 'AAPL',
                    'side': OrderSide.BUY,
                    'entry_price': 150.25,
                    'confidence': 0.75,
                    ...
                }
            ]
        """
        # Get pure signals
        timestamp = datetime.now()
        pure_signals = self.signal_generator.generate_signals(current_data, timestamp)

        # Convert to live trading format
        live_signals = []

        for signal in pure_signals:
            # Only process BUY signals for entry
            if signal.direction != 'BUY':
                continue

            try:
                # Get current quote from broker
                quote = broker.get_latest_quote(signal.symbol)

                live_signals.append({
                    'symbol': signal.symbol,
                    'side': OrderSide.BUY,
                    'entry_price': quote['ask'],  # Use ask price for buying
                    'confidence': signal.confidence,
                    'signal_price': signal.price,  # Price when signal generated
                    'metadata': signal.metadata
                })

            except Exception as e:
                logger.error(f"Failed to get quote for {signal.symbol}: {e}")
                continue

        return live_signals

    def generate_exit_signals(
        self,
        broker: BrokerInterface
    ) -> List[Dict]:
        """
        Generate exit signals for live trading.

        This is called by TradingBot to close positions.

        Args:
            broker: Broker interface

        Returns:
            List of exit signal dicts
        """
        # Get current positions
        positions = broker.get_positions()

        if not positions:
            return []

        # Fetch current data for positions
        symbols = [p['symbol'] for p in positions]
        current_data = self._fetch_position_data(symbols, broker)

        # Get pure signals
        timestamp = datetime.now()
        pure_signals = self.signal_generator.generate_signals(current_data, timestamp)

        # Convert SELL signals to exits
        exit_signals = []

        for signal in pure_signals:
            if signal.direction != 'SELL':
                continue

            # Find matching position
            position = next((p for p in positions if p['symbol'] == signal.symbol), None)

            if position:
                try:
                    quote = broker.get_latest_quote(signal.symbol)

                    exit_signals.append({
                        'symbol': signal.symbol,
                        'side': OrderSide.SELL,
                        'quantity': abs(position['quantity']),
                        'exit_price': quote['bid'],  # Use bid price for selling
                        'entry_price': position.get('avg_entry_price'),
                        'metadata': signal.metadata
                    })

                except Exception as e:
                    logger.error(f"Failed to generate exit for {signal.symbol}: {e}")
                    continue

        return exit_signals

    def get_data_requirements(self) -> Dict:
        """Return data requirements for MA strategy."""
        lookback = self.signal_generator.get_required_lookback()

        return {
            'daily_data': [(s, '1Day', lookback) for s in self.symbols],
            'intraday_data': []  # MA uses daily data only
        }

    def _fetch_position_data(
        self,
        symbols: List[str],
        broker: BrokerInterface
    ) -> Dict[str, pd.DataFrame]:
        """Fetch recent data for open positions."""
        from datetime import timedelta

        data = {}
        lookback = self.signal_generator.get_required_lookback()
        end = datetime.now()
        start = end - timedelta(days=lookback + 50)  # Extra buffer for weekends

        for symbol in symbols:
            try:
                bars = broker.get_bars(
                    symbols=[symbol],
                    timeframe='1Day',
                    start=start,
                    end=end
                )

                if bars is not None and not bars.empty:
                    data[symbol] = bars

            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {e}")

        return data
```

## 5. Usage Examples

### Backtesting Usage:
```python
# backtest_scripts/ma_crossover_backtest.py
from src.backtesting.adapters.ma_backtest_adapter import MABacktestAdapter
from src.engine.portfolio import Portfolio

# Configure strategy
strategy_params = {
    'fast_period': 50,
    'slow_period': 200,
    'position_size_pct': 0.10
}

# Create backtest adapter
strategy = MABacktestAdapter(strategy_params)

# Run backtest
portfolio = Portfolio(initial_capital=100000)
results = run_backtest(portfolio, strategy, historical_data)
```

### Live Trading Usage:
```python
# scripts/trading/run_ma_live.py
from src.trading.adapters.ma_live_adapter import MALiveAdapter
from src.trading.core.trading_bot import TradingBot

# Configure strategy
strategy_config = {
    'symbols': ['AAPL', 'MSFT', 'GOOGL', 'NVDA'],
    'fast_period': 50,
    'slow_period': 200,
    'position_size_pct': 0.10
}

# Create live trading adapter
strategy = MALiveAdapter(strategy_config)

# Create trading bot with this strategy
bot = TradingBot(
    broker_config_path='config/trading/broker_alpaca.yaml',
    strategy=strategy
)

# Start live trading
bot.start_trading()
```

## Summary

This example demonstrates the **complete decoupling**:

1. **Pure Strategy** (`ma_crossover_signals.py`):
   - No imports from backtesting or live trading
   - Only depends on pandas and numpy
   - 100% reusable

2. **Backtest Adapter** (`ma_backtest_adapter.py`):
   - Bridges pure strategy ↔ backtest engine
   - Handles position sizing using backtest portfolio
   - Translates data formats

3. **Live Trading Adapter** (`ma_live_adapter.py`):
   - Bridges pure strategy ↔ live trading bot
   - Handles broker integration
   - Manages real-time quotes

**Same strategy logic**, **two different execution environments**!
