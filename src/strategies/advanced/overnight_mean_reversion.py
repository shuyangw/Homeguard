"""
Overnight Mean Reversion Strategy Implementation.

This strategy exploits predictable overnight mean reversion patterns in leveraged ETFs
using regime-based signal generation and Bayesian probability models.

Strategy Overview:
1. Entry: Buy leveraged ETFs at 3:50 PM EST
2. Exit: Sell at market open (9:31 AM) next day
3. Selection: Based on Bayesian probabilities from 10 years of data
4. Adaptation: Different signals for different market regimes
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, time, timedelta
from pathlib import Path

from src.strategies.base import BaseStrategy
from src.strategies.advanced.market_regime_detector import MarketRegimeDetector
from src.strategies.advanced.bayesian_reversion_model import BayesianReversionModel
from src.strategies.advanced.overnight_signal_generator import OvernightReversionSignals
from src.utils.logger import logger


class OvernightMeanReversionStrategy(BaseStrategy):
    """
    Main strategy implementation for Homeguard framework.

    Integrates:
    - Market regime detection (5 regimes)
    - Bayesian probability model (10 years historical patterns)
    - Signal generation at 3:50 PM
    - Position management (overnight holding only)
    """

    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize the overnight mean reversion strategy.

        Args:
            params: Strategy parameters including:
                - min_probability: Minimum win rate (default 0.55)
                - min_expected_return: Minimum expected return (default 0.002)
                - max_positions: Maximum positions (default 5)
                - position_size: Base position size (default 0.20)
                - data_dir: Directory with historical data
        """
        super().__init__(params)

        # Extract parameters
        self.min_probability = params.get('min_probability', 0.55)
        self.min_expected_return = params.get('min_expected_return', 0.002)
        self.max_positions = params.get('max_positions', 5)
        self.position_size = params.get('position_size', 0.20)
        self.data_dir = params.get('data_dir', 'data/leveraged_etfs')

        # Initialize components
        self.regime_detector = MarketRegimeDetector()
        self.bayesian_model = BayesianReversionModel()
        self.signal_generator = None  # Will be initialized after training

        # Track positions
        self.positions = {}
        self.pending_exits = []

        # Performance tracking
        self.trade_history = []
        self.regime_history = []

        logger.info("Initialized Overnight Mean Reversion Strategy")
        logger.info(f"  Min Probability: {self.min_probability:.1%}")
        logger.info(f"  Min Expected Return: {self.min_expected_return:.3%}")
        logger.info(f"  Max Positions: {self.max_positions}")

    def train_models(self, historical_data: Dict[str, pd.DataFrame]):
        """
        Train the Bayesian model and prepare for trading.

        Args:
            historical_data: Dict of symbol -> DataFrame with historical minute data
        """
        logger.info("\n" + "="*80)
        logger.info("TRAINING OVERNIGHT MEAN REVERSION MODELS")
        logger.info("="*80)

        # Ensure we have SPY and VIX for regime detection
        if 'SPY' not in historical_data or 'VIX' not in historical_data:
            raise ValueError("SPY and VIX data required for regime detection")

        # Convert minute data to daily for regime detection
        spy_daily = self._resample_to_daily(historical_data['SPY'])
        vix_daily = self._resample_to_daily(historical_data['VIX'])

        logger.info("\n[Step 1] Training Bayesian probability model...")

        # Train Bayesian model
        self.bayesian_model.train(
            historical_data,
            self.regime_detector,
            spy_daily,
            vix_daily
        )

        logger.info("\n[Step 2] Initializing signal generator...")

        # Initialize signal generator with trained models
        self.signal_generator = OvernightReversionSignals(
            self.regime_detector,
            self.bayesian_model,
            min_probability=self.min_probability,
            min_expected_return=self.min_expected_return,
            max_positions=self.max_positions
        )

        logger.success("\n✓ Models trained successfully!")

        # Display training statistics
        self._display_training_stats()

    def _resample_to_daily(self, minute_data: pd.DataFrame) -> pd.DataFrame:
        """Resample minute data to daily OHLCV."""
        daily = minute_data.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        return daily.dropna()

    def _display_training_stats(self):
        """Display training statistics."""
        stats = self.bayesian_model.training_stats

        logger.info("\nTraining Statistics:")
        logger.info("-"*40)
        logger.info(f"Total patterns analyzed: {stats.get('total_patterns', 0):,}")
        logger.info(f"Unique setups found: {stats.get('total_setups', 0)}")
        logger.info(f"Average win rate: {stats.get('avg_win_rate', 0):.1%}")
        logger.info(f"Average expected return: {stats.get('avg_expected_return', 0):.3%}")
        logger.info(f"Symbols trained: {stats.get('symbols_trained', 0)}")

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals at 3:50 PM.

        Args:
            data: Current market data

        Returns:
            DataFrame with trading signals
        """
        current_time = data.index[-1]

        # Only generate signals at 3:50 PM
        if current_time.time() != time(15, 50):
            return pd.DataFrame()

        # Check if we should exit positions first
        self._check_exits(data, current_time)

        # Prepare market data dict
        market_data = self._prepare_market_data(data)

        # Generate signals
        signals = self.signal_generator.generate_signals(market_data, current_time)

        if not signals:
            return pd.DataFrame()

        # Convert to DataFrame for Homeguard
        signals_df = pd.DataFrame(signals)
        signals_df['timestamp'] = current_time
        signals_df['action'] = 'BUY'  # Always buy at 3:50 PM

        # Track positions
        for _, signal in signals_df.iterrows():
            self.positions[signal['symbol']] = {
                'entry_time': current_time,
                'entry_price': signal['current_price'],
                'expected_return': signal['expected_return'],
                'position_size': signal['position_size'],
                'regime': signal['regime']
            }
            self.pending_exits.append(signal['symbol'])

        # Log signals
        logger.info(f"\nGenerated {len(signals)} signals at {current_time}")
        for _, signal in signals_df.iterrows():
            logger.info(
                f"  {signal['symbol']}: "
                f"P={signal['probability']:.1%}, "
                f"E[R]={signal['expected_return']:.3%}, "
                f"Size={signal['position_size']:.1%}"
            )

        return signals_df[['timestamp', 'symbol', 'action', 'position_size']]

    def _prepare_market_data(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Prepare market data dict from DataFrame."""
        market_data = {}

        # Extract individual symbols from multi-index DataFrame
        if isinstance(data.columns, pd.MultiIndex):
            symbols = data.columns.get_level_values(1).unique()
            for symbol in symbols:
                market_data[symbol] = data.xs(symbol, axis=1, level=1)
        else:
            # Assume single symbol data
            market_data['SPY'] = data

        return market_data

    def should_exit(self, position: Dict, current_data: pd.DataFrame) -> bool:
        """
        Check if position should be exited.

        Exits all positions at market open (9:31 AM).

        Args:
            position: Position information
            current_data: Current market data

        Returns:
            True if should exit, False otherwise
        """
        current_time = current_data.index[-1]

        # Exit at 9:31 AM (1 minute after market open)
        if current_time.time() >= time(9, 31) and current_time.time() < time(10, 0):
            return True

        # Also exit if held overnight and it's next day
        if 'entry_time' in position:
            entry_date = position['entry_time'].date()
            current_date = current_time.date()

            if current_date > entry_date and current_time.time() >= time(9, 31):
                return True

        return False

    def _check_exits(self, data: pd.DataFrame, current_time: datetime):
        """Check and execute pending exits."""
        if current_time.time() >= time(9, 31) and self.pending_exits:
            logger.info(f"Exiting {len(self.pending_exits)} overnight positions")

            for symbol in self.pending_exits:
                if symbol in self.positions:
                    position = self.positions[symbol]

                    # Calculate actual return
                    exit_price = self._get_current_price(data, symbol)
                    entry_price = position['entry_price']
                    actual_return = (exit_price - entry_price) / entry_price

                    # Record trade
                    self.trade_history.append({
                        'symbol': symbol,
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'expected_return': position['expected_return'],
                        'actual_return': actual_return,
                        'position_size': position['position_size'],
                        'regime': position['regime'],
                        'profitable': actual_return > 0
                    })

                    # Remove from positions
                    del self.positions[symbol]

                    logger.info(
                        f"  Exited {symbol}: "
                        f"Return={actual_return:.3%} "
                        f"(Expected={position['expected_return']:.3%})"
                    )

            self.pending_exits.clear()

    def _get_current_price(self, data: pd.DataFrame, symbol: str) -> float:
        """Get current price for symbol."""
        if isinstance(data.columns, pd.MultiIndex):
            return data.xs(symbol, axis=1, level=1)['close'].iloc[-1]
        else:
            return data['close'].iloc[-1]

    def get_performance_stats(self) -> Dict:
        """
        Calculate performance statistics.

        Returns:
            Dictionary with performance metrics
        """
        if not self.trade_history:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_return': 0,
                'total_return': 0,
                'sharpe_ratio': 0
            }

        trades_df = pd.DataFrame(self.trade_history)

        # Calculate metrics
        total_trades = len(trades_df)
        win_rate = trades_df['profitable'].mean()
        avg_return = trades_df['actual_return'].mean()
        total_return = (trades_df['actual_return'] * trades_df['position_size']).sum()

        # Sharpe ratio
        returns = trades_df['actual_return']
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe = 0

        # Regime breakdown
        regime_stats = trades_df.groupby('regime').agg({
            'profitable': 'mean',
            'actual_return': 'mean'
        })

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'regime_stats': regime_stats.to_dict()
        }

    def plot_performance(self):
        """Plot strategy performance."""
        if not self.trade_history:
            logger.warning("No trades to plot")
            return

        import matplotlib.pyplot as plt

        trades_df = pd.DataFrame(self.trade_history)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Overnight Mean Reversion Performance', fontsize=16)

        # 1. Cumulative returns
        ax = axes[0, 0]
        trades_df['weighted_return'] = trades_df['actual_return'] * trades_df['position_size']
        cumulative = (1 + trades_df['weighted_return']).cumprod()
        cumulative.plot(ax=ax, color='green', linewidth=2)
        ax.set_title('Cumulative Returns')
        ax.set_xlabel('Trade Number')
        ax.set_ylabel('Cumulative Return')
        ax.grid(True, alpha=0.3)

        # 2. Win rate by regime
        ax = axes[0, 1]
        regime_wins = trades_df.groupby('regime')['profitable'].mean()
        regime_wins.plot(kind='bar', ax=ax, color='blue', alpha=0.7)
        ax.set_title('Win Rate by Regime')
        ax.set_ylabel('Win Rate')
        ax.axhline(y=0.5, color='r', linestyle='--', label='50%')
        ax.legend()

        # 3. Expected vs Actual returns
        ax = axes[1, 0]
        ax.scatter(trades_df['expected_return'], trades_df['actual_return'],
                  alpha=0.5, s=30)
        ax.plot([-0.01, 0.01], [-0.01, 0.01], 'r--', label='Perfect Prediction')
        ax.set_xlabel('Expected Return')
        ax.set_ylabel('Actual Return')
        ax.set_title('Expected vs Actual Returns')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Return distribution
        ax = axes[1, 1]
        trades_df['actual_return'].hist(bins=30, ax=ax, color='green', alpha=0.7)
        ax.axvline(x=0, color='r', linestyle='--', label='Zero')
        ax.axvline(x=trades_df['actual_return'].mean(), color='b',
                  linestyle='--', label=f"Mean: {trades_df['actual_return'].mean():.3%}")
        ax.set_xlabel('Return')
        ax.set_ylabel('Frequency')
        ax.set_title('Return Distribution')
        ax.legend()

        plt.tight_layout()
        plt.show()

        # Print summary
        stats = self.get_performance_stats()
        print("\nPerformance Summary")
        print("="*50)
        print(f"Total Trades: {stats['total_trades']}")
        print(f"Win Rate: {stats['win_rate']:.1%}")
        print(f"Average Return: {stats['avg_return']:.3%}")
        print(f"Total Return: {stats['total_return']:.1%}")
        print(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}")


def test_overnight_strategy():
    """Test the overnight mean reversion strategy."""

    logger.info("Testing Overnight Mean Reversion Strategy")
    logger.info("="*60)

    # Initialize strategy
    params = {
        'min_probability': 0.55,
        'min_expected_return': 0.002,
        'max_positions': 5,
        'position_size': 0.20
    }

    strategy = OvernightMeanReversionStrategy(params)

    logger.info("\n✓ Strategy initialized successfully")
    logger.info("\nNext steps:")
    logger.info("1. Download leveraged ETF data using download_leveraged_etfs.py")
    logger.info("2. Load historical data into strategy")
    logger.info("3. Train models using train_models()")
    logger.info("4. Backtest or run live trading")

    return strategy


if __name__ == "__main__":
    test_overnight_strategy()
