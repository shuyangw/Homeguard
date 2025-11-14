"""
Bayesian Probability Model for Overnight Mean Reversion.

Calculates probability of profitable overnight reversion based on:
- Market regime
- Intraday move magnitude and direction
- Historical patterns over 10 years
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, time, timedelta
import pickle
from pathlib import Path
from src.utils.logger import logger
from src.strategies.advanced.market_regime_detector import MarketRegimeDetector
from src.config import get_models_dir


class BayesianReversionModel:
    """
    Bayesian model for calculating overnight reversion probabilities.

    The model calculates P(profitable_overnight | regime, intraday_move)
    based on historical patterns.
    """

    # Define intraday move buckets for categorization
    MOVE_BUCKETS = [
        {'min': -1.0, 'max': -0.05, 'label': 'large_down'},
        {'min': -0.05, 'max': -0.03, 'label': 'medium_down'},
        {'min': -0.03, 'max': -0.01, 'label': 'small_down'},
        {'min': -0.01, 'max': 0.01, 'label': 'flat'},
        {'min': 0.01, 'max': 0.03, 'label': 'small_up'},
        {'min': 0.03, 'max': 0.05, 'label': 'medium_up'},
        {'min': 0.05, 'max': 1.0, 'label': 'large_up'}
    ]

    # Minimum sample size for statistical significance
    MIN_SAMPLE_SIZE = 30

    def __init__(self, lookback_years: int = 10, data_frequency: str = 'minute'):
        """
        Initialize the Bayesian reversion model.

        Args:
            lookback_years: Number of years of historical data to use
            data_frequency: 'minute' for live trading, 'daily' for backtesting
        """
        self.lookback_years = lookback_years
        self.data_frequency = data_frequency
        self.regime_probabilities = {}
        self.trained = False
        self.training_stats = {}
        self.model_path = get_models_dir() / 'bayesian_reversion_model.pkl'

    def train(
        self,
        historical_data: Dict[str, pd.DataFrame],
        regime_detector: MarketRegimeDetector,
        spy_data: pd.DataFrame,
        vix_data: pd.DataFrame
    ):
        """
        Train the model on historical data.

        Args:
            historical_data: Dict of symbol -> DataFrame (minute or daily data)
            regime_detector: Trained regime detector
            spy_data: SPY daily data for regime detection
            vix_data: VIX daily data for regime detection
        """
        logger.info("Training Bayesian Reversion Model")
        logger.info(f"Data frequency: {self.data_frequency}")
        logger.info("="*60)

        # Initialize probability tables
        for symbol in historical_data.keys():
            self.regime_probabilities[symbol] = {
                'STRONG_BULL': {},
                'WEAK_BULL': {},
                'SIDEWAYS': {},
                'UNPREDICTABLE': {},
                'BEAR': {}
            }

        # Process each symbol
        total_patterns = 0
        for symbol, data in historical_data.items():
            logger.info(f"Processing {symbol}...")

            # Calculate daily metrics (handles both minute and daily data)
            if self.data_frequency == 'daily':
                daily_data = self._calculate_daily_metrics_from_daily(data)
            else:
                daily_data = self._calculate_daily_metrics(data)

            # Add regime labels
            daily_data = self._add_regime_labels(
                daily_data, regime_detector, spy_data, vix_data
            )

            # Calculate probabilities for each regime and move bucket
            symbol_patterns = 0
            for regime in self.regime_probabilities[symbol].keys():
                for bucket in self.MOVE_BUCKETS:
                    prob_data = self._calculate_reversion_probability(
                        daily_data, regime, bucket
                    )

                    if prob_data is not None:
                        self.regime_probabilities[symbol][regime][bucket['label']] = prob_data
                        symbol_patterns += prob_data['sample_size']

            logger.info(f"  Found {symbol_patterns:,} patterns for {symbol}")
            total_patterns += symbol_patterns

        # Calculate training statistics
        self._calculate_training_stats()

        self.trained = True
        logger.success(f"Training complete! Analyzed {total_patterns:,} total patterns")

        # Save model
        self.save_model()

    def _calculate_daily_metrics(self, minute_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily metrics from minute data.

        Metrics include:
        - Intraday return (open to 3:50 PM)
        - Overnight return (3:50 PM to next open)
        """
        daily_metrics = []

        # Group by date
        grouped = minute_data.groupby(minute_data.index.date)

        for date, group in grouped:
            try:
                # Get market open (9:30 AM)
                open_time = time(9, 30)
                open_data = group.between_time(open_time, open_time)

                if open_data.empty:
                    continue

                open_price = open_data['open'].iloc[0]

                # Get 3:50 PM price
                close_time = time(15, 50)
                close_data = group.between_time(close_time, close_time)

                if close_data.empty:
                    continue

                close_price = close_data['close'].iloc[-1]

                # Calculate intraday return
                intraday_return = (close_price - open_price) / open_price

                # Get next day's open
                next_date = date + timedelta(days=1)
                next_open = None

                # Look for next trading day (may not be next calendar day)
                for i in range(1, 5):  # Check up to 4 days ahead
                    check_date = date + timedelta(days=i)
                    if check_date in grouped.groups:
                        next_group = grouped.get_group(check_date)
                        next_open_data = next_group.between_time(open_time, open_time)

                        if not next_open_data.empty:
                            next_open = next_open_data['open'].iloc[0]
                            break

                if next_open is None:
                    continue

                # Calculate overnight return
                overnight_return = (next_open - close_price) / close_price

                daily_metrics.append({
                    'date': pd.Timestamp(date),
                    'open': open_price,
                    'close_3_50': close_price,
                    'next_open': next_open,
                    'intraday_return': intraday_return,
                    'overnight_return': overnight_return
                })

            except Exception as e:
                # Skip error (logger.debug not available)
                continue

        # Return DataFrame with date index, handle empty case
        if not daily_metrics:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=['open', 'close_3_50', 'next_open', 'intraday_return', 'overnight_return'])

        return pd.DataFrame(daily_metrics).set_index('date')

    def _calculate_daily_metrics_from_daily(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily metrics from daily OHLC data (for backtesting).

        For daily data, we use:
        - intraday_return = (Close - Open) / Open
        - overnight_return = (Next Open - Close) / Close

        Note: This uses market close (4:00 PM) instead of 3:50 PM.
        """
        metrics = []

        for i in range(len(daily_data) - 1):
            try:
                row = daily_data.iloc[i]
                next_row = daily_data.iloc[i + 1]

                # Calculate returns (use lowercase column names)
                intraday_return = (row['close'] - row['open']) / row['open']
                overnight_return = (next_row['open'] - row['close']) / row['close']

                metrics.append({
                    'date': row.name,
                    'open': row['open'],
                    'close_3_50': row['close'],  # Actually 4:00 PM but kept for consistency
                    'next_open': next_row['open'],
                    'intraday_return': intraday_return,
                    'overnight_return': overnight_return
                })

            except Exception as e:
                # Skip error (logger.debug not available)
                continue

        if not metrics:
            # Return empty DataFrame with correct structure
            return pd.DataFrame(columns=['open', 'close_3_50', 'next_open', 'intraday_return', 'overnight_return'])

        return pd.DataFrame(metrics).set_index('date')

    def _add_regime_labels(
        self,
        daily_data: pd.DataFrame,
        regime_detector: MarketRegimeDetector,
        spy_data: pd.DataFrame,
        vix_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Add regime labels to daily data."""

        regimes = []
        confidences = []

        for date in daily_data.index:
            # Get data up to this date
            spy_subset = spy_data[spy_data.index <= date]
            vix_subset = vix_data[vix_data.index <= date]

            if len(spy_subset) < 200 or len(vix_subset) < 252:
                regimes.append('SIDEWAYS')
                confidences.append(0.5)
                continue

            # Classify regime
            regime, confidence = regime_detector.classify_regime(
                spy_subset, vix_subset, date
            )

            regimes.append(regime)
            confidences.append(confidence)

        daily_data['regime'] = regimes
        daily_data['regime_confidence'] = confidences

        return daily_data

    def _calculate_reversion_probability(
        self,
        data: pd.DataFrame,
        regime: str,
        move_bucket: Dict
    ) -> Optional[Dict]:
        """
        Calculate probability of profitable overnight reversion.

        Returns:
            Dictionary with probability metrics or None if insufficient data
        """
        # Filter data for regime
        regime_data = data[data['regime'] == regime]

        # Filter for move bucket
        bucket_data = regime_data[
            (regime_data['intraday_return'] >= move_bucket['min']) &
            (regime_data['intraday_return'] < move_bucket['max'])
        ]

        # Check minimum sample size
        if len(bucket_data) < self.MIN_SAMPLE_SIZE:
            return None

        # Calculate metrics
        overnight_returns = bucket_data['overnight_return']

        # Profitable if overnight return > 0.1%
        profitable = overnight_returns > 0.001
        win_rate = profitable.sum() / len(profitable)

        # Expected return
        expected_return = overnight_returns.mean()
        return_std = overnight_returns.std()

        # Sharpe ratio (annualized)
        if return_std > 0:
            sharpe = (expected_return / return_std) * np.sqrt(252)
        else:
            sharpe = 0

        # Calculate confidence based on sample size
        sample_size = len(bucket_data)
        confidence = min(sample_size / 100, 1.0)

        # Calculate percentiles
        percentiles = overnight_returns.quantile([0.05, 0.25, 0.5, 0.75, 0.95])

        return {
            'probability': win_rate,
            'expected_return': expected_return,
            'return_std': return_std,
            'sample_size': sample_size,
            'confidence': confidence,
            'sharpe': sharpe,
            'percentile_5': percentiles[0.05],
            'percentile_25': percentiles[0.25],
            'median': percentiles[0.5],
            'percentile_75': percentiles[0.75],
            'percentile_95': percentiles[0.95]
        }

    def get_reversion_probability(
        self,
        symbol: str,
        regime: str,
        intraday_return: float
    ) -> Optional[Dict]:
        """
        Get reversion probability for given conditions.

        Args:
            symbol: Trading symbol
            regime: Current market regime
            intraday_return: Today's intraday return

        Returns:
            Probability data or None if not available
        """
        if not self.trained:
            raise ValueError("Model not trained yet")

        if symbol not in self.regime_probabilities:
            return None

        # Find appropriate move bucket
        move_bucket = None
        for bucket in self.MOVE_BUCKETS:
            if bucket['min'] <= intraday_return < bucket['max']:
                move_bucket = bucket['label']
                break

        if move_bucket is None:
            return None

        return self.regime_probabilities[symbol][regime].get(move_bucket)

    def _calculate_training_stats(self):
        """Calculate and log training statistics."""

        total_patterns = 0
        total_setups = 0
        avg_win_rate = []
        avg_expected_return = []

        for symbol, regimes in self.regime_probabilities.items():
            for regime, buckets in regimes.items():
                for bucket_label, prob_data in buckets.items():
                    if prob_data:
                        total_patterns += prob_data['sample_size']
                        total_setups += 1
                        avg_win_rate.append(prob_data['probability'])
                        avg_expected_return.append(prob_data['expected_return'])

        self.training_stats = {
            'total_patterns': total_patterns,
            'total_setups': total_setups,
            'avg_win_rate': np.mean(avg_win_rate) if avg_win_rate else 0,
            'avg_expected_return': np.mean(avg_expected_return) if avg_expected_return else 0,
            'symbols_trained': len(self.regime_probabilities)
        }

        logger.info("\nTraining Statistics:")
        logger.info("-"*40)
        logger.info(f"Total patterns: {total_patterns:,}")
        logger.info(f"Unique setups: {total_setups}")
        logger.info(f"Average win rate: {self.training_stats['avg_win_rate']:.1%}")
        logger.info(f"Average expected return: {self.training_stats['avg_expected_return']:.3%}")

    def save_model(self):
        """Save trained model to disk."""
        if not self.trained:
            raise ValueError("Cannot save untrained model")

        # Create models directory if it doesn't exist
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'regime_probabilities': self.regime_probabilities,
            'training_stats': self.training_stats,
            'trained': self.trained,
            'lookback_years': self.lookback_years,
            'timestamp': datetime.now()
        }

        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.success(f"Model saved to {self.model_path}")

    def load_model(self):
        """Load trained model from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.regime_probabilities = model_data['regime_probabilities']
        self.training_stats = model_data['training_stats']
        self.trained = model_data['trained']
        self.lookback_years = model_data['lookback_years']

        logger.success(
            f"Model loaded from {self.model_path} "
            f"(trained on {model_data['timestamp'].strftime('%Y-%m-%d')})"
        )

    def get_best_setups(
        self,
        current_regime: str,
        min_probability: float = 0.60,
        min_expected_return: float = 0.002
    ) -> List[Dict]:
        """
        Get best trading setups for current regime.

        Args:
            current_regime: Current market regime
            min_probability: Minimum win rate required
            min_expected_return: Minimum expected return required

        Returns:
            List of best setups sorted by expected return
        """
        best_setups = []

        for symbol, regimes in self.regime_probabilities.items():
            if current_regime not in regimes:
                continue

            for bucket_label, prob_data in regimes[current_regime].items():
                if prob_data is None:
                    continue

                if (prob_data['probability'] >= min_probability and
                    prob_data['expected_return'] >= min_expected_return):

                    best_setups.append({
                        'symbol': symbol,
                        'move_bucket': bucket_label,
                        'probability': prob_data['probability'],
                        'expected_return': prob_data['expected_return'],
                        'sharpe': prob_data['sharpe'],
                        'sample_size': prob_data['sample_size']
                    })

        # Sort by expected return
        best_setups.sort(key=lambda x: x['expected_return'], reverse=True)

        return best_setups

    def analyze_regime_performance(self) -> pd.DataFrame:
        """
        Analyze performance across different regimes.

        Returns:
            DataFrame with regime performance statistics
        """
        regime_stats = []

        for regime in ['STRONG_BULL', 'WEAK_BULL', 'SIDEWAYS', 'UNPREDICTABLE', 'BEAR']:
            regime_patterns = 0
            regime_win_rates = []
            regime_returns = []

            for symbol, regimes in self.regime_probabilities.items():
                for bucket_label, prob_data in regimes[regime].items():
                    if prob_data:
                        regime_patterns += prob_data['sample_size']
                        regime_win_rates.append(prob_data['probability'])
                        regime_returns.append(prob_data['expected_return'])

            if regime_win_rates:
                regime_stats.append({
                    'regime': regime,
                    'total_patterns': regime_patterns,
                    'avg_win_rate': np.mean(regime_win_rates),
                    'avg_expected_return': np.mean(regime_returns),
                    'std_expected_return': np.std(regime_returns)
                })

        return pd.DataFrame(regime_stats).set_index('regime')


def test_bayesian_model():
    """Test the Bayesian reversion model with sample data."""

    logger.info("Testing Bayesian Reversion Model")
    logger.info("="*60)

    # Initialize model
    model = BayesianReversionModel()

    # Create sample training data
    logger.info("Creating sample training data...")

    # This is a placeholder - in production, load actual minute data
    sample_data = {
        'TQQQ': pd.DataFrame({
            'open': np.random.randn(1000) * 2 + 100,
            'close': np.random.randn(1000) * 2 + 100
        }, index=pd.date_range('2023-01-01', periods=1000, freq='T'))
    }

    # Note: In production, you would train with actual data
    logger.info("Model requires actual historical data for training")
    logger.info("Use download_leveraged_etfs.py to get data first")

    # Show example of getting probabilities (after training)
    if model.trained:
        prob_data = model.get_reversion_probability(
            symbol='TQQQ',
            regime='WEAK_BULL',
            intraday_return=-0.025
        )

        if prob_data:
            logger.info("\nExample Probability Data:")
            logger.info(f"  Win Rate: {prob_data['probability']:.1%}")
            logger.info(f"  Expected Return: {prob_data['expected_return']:.3%}")
            logger.info(f"  Sample Size: {prob_data['sample_size']}")
    else:
        logger.warning("Model not trained - train with historical data first")

    return model


if __name__ == "__main__":
    test_bayesian_model()