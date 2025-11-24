"""
Signal Generator for Overnight Mean Reversion Strategy.

Generates trading signals at 3:50 PM EST based on:
- Current market regime
- Intraday price movements
- Bayesian reversion probabilities
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, time
from src.utils.logger import logger
from src.strategies.advanced.market_regime_detector import MarketRegimeDetector
from src.strategies.advanced.bayesian_reversion_model import BayesianReversionModel


class OvernightReversionSignals:
    """
    Generates trading signals for overnight mean reversion.

    Signals are generated at 3:50 PM EST based on regime and probabilities.

    NOTE: Symbol list is now injected via constructor (not hardcoded).
    For backward compatibility, defaults to leveraged 3x ETFs if not specified.
    """

    def __init__(
        self,
        regime_detector: MarketRegimeDetector,
        bayesian_model: BayesianReversionModel,
        symbols: Optional[List[str]] = None,
        min_probability: float = 0.55,
        min_expected_return: float = 0.002,
        max_positions: int = 5,
        skip_bear_regime: bool = True
    ):
        """
        Initialize signal generator.

        Args:
            regime_detector: Trained market regime detector
            bayesian_model: Trained Bayesian probability model
            symbols: List of symbols to generate signals for.
                    If None, defaults to leveraged 3x ETFs from ETFUniverse.
            min_probability: Minimum win rate to generate signal
            min_expected_return: Minimum expected return to generate signal
            max_positions: Maximum number of positions to take
            skip_bear_regime: If True, skip all trades during BEAR regime (default: True)
                            CRITICAL: Backtests show BEAR regime has negative edge (-1.31 Sharpe)
                            and causes 100% of catastrophic drawdowns.
        """
        self.regime_detector = regime_detector
        self.bayesian_model = bayesian_model
        self.min_probability = min_probability
        self.min_expected_return = min_expected_return
        self.max_positions = max_positions
        self.skip_bear_regime = skip_bear_regime

        # Symbol list - injected or default from ETFUniverse
        if symbols is None:
            # Backward compatibility: use leveraged 3x ETFs as default
            from src.strategies.universe import ETFUniverse
            self.symbols = ETFUniverse.LEVERAGED_3X
            logger.info(f"Using default symbol universe: {len(self.symbols)} leveraged 3x ETFs")
        else:
            self.symbols = symbols
            logger.info(f"Using custom symbol universe: {len(self.symbols)} symbols")

        # Log BEAR regime filter status
        if self.skip_bear_regime:
            logger.info("BEAR regime filter ENABLED - will skip trades in bear markets")
        else:
            logger.warning("BEAR regime filter DISABLED - trades allowed in all regimes")

    def generate_signals(
        self,
        market_data: Dict[str, pd.DataFrame],
        timestamp: datetime
    ) -> List[Dict]:
        """
        Generate trading signals at 3:50 PM.

        Args:
            market_data: Dict of symbol -> intraday data
            timestamp: Current timestamp

        Returns:
            List of trading signals sorted by strength
        """
        # NOTE: Time validation removed - caller (bot/scheduler) is responsible for
        # calling at the correct time. This allows flexible scheduling windows.
        # Previously: if timestamp.hour != 15 or timestamp.minute != 50: return []

        # Clear signal generation start marker
        start_msg = f"=== SIGNAL GENERATION START: {timestamp.strftime('%Y-%m-%d %H:%M:%S')} ==="
        logger.info(start_msg)
        print(f"\n{start_msg}", flush=True)

        # Get current market regime
        spy_data = market_data.get('SPY')
        vix_data = market_data.get('VIX')

        if spy_data is None or vix_data is None:
            logger.error("Missing SPY or VIX data for regime detection")
            return []

        regime, regime_confidence = self.regime_detector.classify_regime(
            spy_data, vix_data, timestamp
        )

        # Get VIX level for logging
        current_vix = float(vix_data['close'].iloc[-1])
        current_spy = float(spy_data['close'].iloc[-1])

        # Log regime detection results (INFO level, always visible)
        regime_msg = f"REGIME DETECTION: {regime} (confidence: {regime_confidence:.2f}) | SPY: ${current_spy:.2f} | VIX: {current_vix:.2f}"
        logger.info(regime_msg)
        # Force immediate console output for regime detection
        print(f"[REGIME] {regime_msg}", flush=True)

        # CRITICAL: Skip trading in BEAR regime (negative edge, causes catastrophic losses)
        if self.skip_bear_regime and regime == 'BEAR':
            bear_msg = (
                "BEAR REGIME - SKIPPING ALL TRADES "
                f"(BEAR has -1.31 Sharpe, SPY=${current_spy:.2f}, VIX={current_vix:.2f})"
            )
            logger.warning(bear_msg)
            # Force immediate console output for critical trade blocking
            print(f"[WARNING] {bear_msg}", flush=True)
            return []

        # Get regime-specific parameters
        regime_params = self.regime_detector.get_regime_parameters(regime)

        # Adjust filters based on regime
        regime_min_probability = max(
            self.min_probability,
            regime_params.get('min_win_rate', 0.55)
        )
        regime_min_return = max(
            self.min_expected_return,
            regime_params.get('min_expected_return', 0.002)
        )
        regime_max_positions = min(
            self.max_positions,
            regime_params.get('max_positions', 5)
        )

        # Generate signals for each symbol
        signals = []
        symbols_evaluated = 0
        symbols_in_data = 0

        for symbol in self.symbols:
            if symbol not in market_data:
                continue

            symbols_in_data += 1
            symbols_evaluated += 1

            signal = self._evaluate_symbol(
                symbol, market_data[symbol], regime,
                regime_confidence, regime_min_probability,
                regime_min_return
            )

            if signal:
                signals.append(signal)

        # Log evaluation summary
        signals_before_limit = len(signals)
        logger.info(
            f"Signal evaluation: {symbols_evaluated} symbols checked, "
            f"{signals_before_limit} passed filters "
            f"(min prob: {regime_min_probability:.0%}, min return: {regime_min_return:.2%})"
        )

        # Sort by signal strength and take top positions
        signals.sort(key=lambda x: x['signal_strength'], reverse=True)
        signals = signals[:regime_max_positions]

        # Add position sizing
        signals = self._add_position_sizing(signals, regime_params)

        # Final summary
        summary_msg = (
            f"Generated {len(signals)} signals for overnight holding "
            f"({signals_before_limit} candidates -> top {len(signals)} by signal strength)"
        )
        logger.info(summary_msg)
        print(f"[SIGNALS] {summary_msg}", flush=True)

        return signals

    def _evaluate_symbol(
        self,
        symbol: str,
        data: pd.DataFrame,
        regime: str,
        regime_confidence: float,
        min_probability: float,
        min_expected_return: float
    ) -> Optional[Dict]:
        """
        Evaluate a single symbol for trading signal.

        Args:
            symbol: Trading symbol
            data: Intraday data for symbol
            regime: Current market regime
            regime_confidence: Confidence in regime classification
            min_probability: Minimum win rate threshold
            min_expected_return: Minimum expected return threshold

        Returns:
            Signal dictionary or None
        """
        try:
            # Calculate intraday return (open to 3:50 PM)
            today_data = data[data.index.date == data.index[-1].date()]

            # Get open price (first bar after 9:30 AM)
            open_data = today_data.between_time(time(9, 30), time(9, 31))
            if open_data.empty:
                return None

            open_price = open_data['open'].iloc[0]

            # Get current price (3:50 PM)
            current_data = today_data.between_time(time(15, 50), time(15, 50))
            if current_data.empty:
                return None

            current_price = current_data['close'].iloc[-1]

            # Calculate intraday return
            intraday_return = (current_price - open_price) / open_price

            # Get probability from Bayesian model
            prob_data = self.bayesian_model.get_reversion_probability(
                symbol, regime, intraday_return
            )

            if prob_data is None:
                return None

            # Apply filters
            if (prob_data['probability'] < min_probability or
                prob_data['expected_return'] < min_expected_return or
                prob_data['sample_size'] < 30):
                return None

            # Calculate signal strength
            signal_strength = self._calculate_signal_strength(
                prob_data, regime_confidence, intraday_return
            )

            # Determine direction based on intraday move
            # Large down move -> expect bounce (BUY)
            # Large up move -> expect pullback (SHORT)
            if abs(intraday_return) < 0.01:
                # Flat day, skip
                return None

            direction = 'BUY' if intraday_return < 0 else 'SHORT'

            # For leveraged inverse ETFs, reverse the logic
            if symbol in ['SQQQ', 'SPXU', 'SDOW', 'TZA', 'SOXS', 'FAZ', 'LABD', 'TECS', 'QID', 'SDS']:
                direction = 'SHORT' if direction == 'BUY' else 'BUY'

            return {
                'symbol': symbol,
                'regime': regime,
                'direction': direction,
                'entry_time': '15:50',
                'exit_time': '09:31',
                'intraday_return': intraday_return,
                'probability': prob_data['probability'],
                'expected_return': prob_data['expected_return'],
                'sharpe': prob_data.get('sharpe', 0),
                'signal_strength': signal_strength,
                'sample_size': prob_data['sample_size'],
                'confidence': prob_data['confidence'],
                'current_price': current_price
            }

        except Exception as e:
            logger.debug(f"Error evaluating {symbol}: {e}")
            return None

    def _calculate_signal_strength(
        self,
        prob_data: Dict,
        regime_confidence: float,
        intraday_return: float
    ) -> float:
        """
        Calculate signal strength score (0-1).

        Components:
        - Win rate probability (40% weight)
        - Expected return magnitude (30% weight)
        - Regime confidence (20% weight)
        - Intraday move extremeness (10% weight)
        """
        # Win rate component (0-1)
        win_rate_score = min((prob_data['probability'] - 0.5) * 2, 1.0)

        # Expected return component (0-1)
        # Normalize expected return (0.5% = score of 1.0)
        return_score = min(prob_data['expected_return'] / 0.005, 1.0)

        # Regime confidence component
        regime_score = regime_confidence

        # Intraday move extremeness
        # More extreme moves get higher scores
        move_score = min(abs(intraday_return) / 0.05, 1.0)

        # Weighted combination
        signal_strength = (
            win_rate_score * 0.40 +
            return_score * 0.30 +
            regime_score * 0.20 +
            move_score * 0.10
        )

        return signal_strength

    def _add_position_sizing(
        self,
        signals: List[Dict],
        regime_params: Dict
    ) -> List[Dict]:
        """
        Add position sizing to signals based on regime.

        Args:
            signals: List of signals
            regime_params: Regime-specific parameters

        Returns:
            Signals with position_size field added
        """
        if not signals:
            return signals

        # Base position size (equal weight)
        base_size = 1.0 / len(signals)

        # Apply regime multiplier
        regime_multiplier = regime_params.get('position_size_multiplier', 1.0)

        for signal in signals:
            # Adjust size based on signal strength
            strength_adjustment = 0.8 + (0.4 * signal['signal_strength'])

            # Calculate final position size
            position_size = base_size * regime_multiplier * strength_adjustment

            # Cap at 25% max per position
            signal['position_size'] = min(position_size, 0.25)

        # Normalize to ensure total doesn't exceed 100%
        total_size = sum(s['position_size'] for s in signals)
        if total_size > 1.0:
            for signal in signals:
                signal['position_size'] /= total_size

        return signals

    def backtest_signals(
        self,
        historical_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Backtest signal generation on historical data.

        Args:
            historical_data: Dict of symbol -> historical data
            start_date: Start of backtest period
            end_date: End of backtest period

        Returns:
            DataFrame with backtest results
        """
        results = []
        trading_days = pd.bdate_range(start_date, end_date)

        for date in trading_days:
            # Set time to 3:50 PM
            timestamp = datetime.combine(date.date(), time(15, 50))

            # Get data up to this point
            current_data = {}
            for symbol, data in historical_data.items():
                mask = data.index <= timestamp
                current_data[symbol] = data[mask]

            # Generate signals
            signals = self.generate_signals(current_data, timestamp)

            # Calculate actual overnight returns
            for signal in signals:
                symbol = signal['symbol']
                if symbol not in historical_data:
                    continue

                # Get next open price
                next_day = date + pd.Timedelta(days=1)
                next_open_data = historical_data[symbol][
                    (historical_data[symbol].index.date == next_day.date()) &
                    (historical_data[symbol].index.time == time(9, 30))
                ]

                if next_open_data.empty:
                    continue

                next_open = next_open_data['open'].iloc[0]
                entry_price = signal['current_price']

                # Calculate actual return
                if signal['direction'] == 'BUY':
                    actual_return = (next_open - entry_price) / entry_price
                else:  # SHORT
                    actual_return = (entry_price - next_open) / entry_price

                results.append({
                    'date': date,
                    'symbol': symbol,
                    'regime': signal['regime'],
                    'direction': signal['direction'],
                    'intraday_return': signal['intraday_return'],
                    'expected_return': signal['expected_return'],
                    'actual_return': actual_return,
                    'probability': signal['probability'],
                    'signal_strength': signal['signal_strength'],
                    'position_size': signal['position_size'],
                    'profitable': actual_return > 0
                })

        return pd.DataFrame(results)


def create_signal_dashboard(backtest_results: pd.DataFrame):
    """Create dashboard to visualize signal performance."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Overnight Mean Reversion Signal Analysis', fontsize=16)

    # 1. Win rate by regime
    ax = axes[0, 0]
    regime_wins = backtest_results.groupby('regime')['profitable'].mean()
    regime_wins.plot(kind='bar', ax=ax, color='green', alpha=0.7)
    ax.set_title('Win Rate by Regime')
    ax.set_ylabel('Win Rate')
    ax.axhline(y=0.5, color='r', linestyle='--', label='50%')
    ax.legend()

    # 2. Average return by regime
    ax = axes[0, 1]
    regime_returns = backtest_results.groupby('regime')['actual_return'].mean()
    regime_returns.plot(kind='bar', ax=ax, color='blue', alpha=0.7)
    ax.set_title('Avg Return by Regime')
    ax.set_ylabel('Return')
    ax.axhline(y=0, color='r', linestyle='--')

    # 3. Signal count by regime
    ax = axes[0, 2]
    regime_counts = backtest_results['regime'].value_counts()
    regime_counts.plot(kind='bar', ax=ax, color='orange', alpha=0.7)
    ax.set_title('Signal Count by Regime')
    ax.set_ylabel('Number of Signals')

    # 4. Expected vs Actual returns
    ax = axes[1, 0]
    ax.scatter(backtest_results['expected_return'],
              backtest_results['actual_return'],
              alpha=0.5)
    ax.plot([0, 0.01], [0, 0.01], 'r--', label='Perfect Prediction')
    ax.set_xlabel('Expected Return')
    ax.set_ylabel('Actual Return')
    ax.set_title('Expected vs Actual Returns')
    ax.legend()

    # 5. Cumulative returns
    ax = axes[1, 1]
    backtest_results['weighted_return'] = (
        backtest_results['actual_return'] *
        backtest_results['position_size']
    )
    cumulative = (1 + backtest_results.groupby('date')['weighted_return'].sum()).cumprod()
    cumulative.plot(ax=ax, color='green')
    ax.set_title('Cumulative Strategy Returns')
    ax.set_ylabel('Cumulative Return')
    ax.grid(True, alpha=0.3)

    # 6. Win rate by signal strength
    ax = axes[1, 2]
    bins = pd.cut(backtest_results['signal_strength'], bins=5)
    strength_wins = backtest_results.groupby(bins)['profitable'].mean()
    strength_wins.plot(kind='bar', ax=ax, color='purple', alpha=0.7)
    ax.set_title('Win Rate by Signal Strength')
    ax.set_xlabel('Signal Strength Bin')
    ax.set_ylabel('Win Rate')
    ax.axhline(y=0.5, color='r', linestyle='--', label='50%')

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\nSignal Performance Summary")
    print("="*60)
    print(f"Total Signals: {len(backtest_results)}")
    print(f"Overall Win Rate: {backtest_results['profitable'].mean():.1%}")
    print(f"Average Return: {backtest_results['actual_return'].mean():.3%}")
    print(f"Sharpe Ratio: {backtest_results['actual_return'].mean() / backtest_results['actual_return'].std() * np.sqrt(252):.2f}")

    print("\nTop Performing Symbols:")
    symbol_perf = backtest_results.groupby('symbol').agg({
        'actual_return': 'mean',
        'profitable': 'mean'
    }).sort_values('actual_return', ascending=False).head(10)
    print(symbol_perf)


if __name__ == "__main__":
    logger.info("Signal Generator Module")
    logger.info("="*60)
    logger.info("This module generates overnight mean reversion signals")
    logger.info("Requires trained regime detector and Bayesian model")
    logger.info("")
    logger.info("To use:")
    logger.info("1. Download leveraged ETF data")
    logger.info("2. Train regime detector")
    logger.info("3. Train Bayesian model")
    logger.info("4. Initialize signal generator")
    logger.info("5. Call generate_signals() at 3:50 PM")