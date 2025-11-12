"""
Market Regime Detector for Overnight Mean Reversion Strategy.

Classifies market into 5 regimes based on momentum and volatility indicators:
- STRONG_BULL: High momentum, low volatility
- WEAK_BULL: Moderate upward momentum, elevated volatility
- SIDEWAYS: Neutral momentum, moderate volatility
- UNPREDICTABLE: Rapidly shifting momentum, irregular volatility
- BEAR: Strong negative momentum, high volatility
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from datetime import datetime, timedelta
from src.utils.logger import logger


class MarketRegimeDetector:
    """
    Classifies market into one of 5 regimes based on momentum and volatility.

    Regimes are determined by:
    - SPY position relative to moving averages (20, 50, 200 SMA)
    - VIX percentile rank (1-year lookback)
    - Recent realized volatility
    - Momentum slope (20-day rate of change)
    """

    REGIMES = {
        'STRONG_BULL': 1,
        'WEAK_BULL': 2,
        'SIDEWAYS': 3,
        'UNPREDICTABLE': 4,
        'BEAR': 5
    }

    # Regime characteristics for classification
    REGIME_CRITERIA = {
        'STRONG_BULL': {
            'momentum_slope_min': 0.02,      # Strong positive momentum
            'vix_percentile_max': 30,        # Low volatility
            'above_mas': ['20', '50', '200'], # Above all moving averages
            'volatility_regime': 'low'
        },
        'WEAK_BULL': {
            'momentum_slope_min': 0.0,       # Positive but weakening
            'momentum_slope_max': 0.02,
            'vix_percentile_max': 50,        # Moderate volatility
            'above_mas': ['20', '50'],       # Above short-term MAs
            'volatility_regime': 'moderate'
        },
        'SIDEWAYS': {
            'momentum_slope_min': -0.01,     # Flat momentum
            'momentum_slope_max': 0.01,
            'vix_percentile_min': 30,
            'vix_percentile_max': 60,
            'volatility_regime': 'moderate'
        },
        'UNPREDICTABLE': {
            'vix_percentile_min': 60,        # High volatility
            'volatility_spike': True,        # Recent vol spike
            'volatility_regime': 'high'
        },
        'BEAR': {
            'momentum_slope_max': -0.02,     # Strong negative momentum
            'vix_percentile_min': 70,        # High volatility
            'below_mas': ['20', '50', '200'], # Below all moving averages
            'volatility_regime': 'high'
        }
    }

    def __init__(self, lookback_window: int = 252):
        """
        Initialize the market regime detector.

        Args:
            lookback_window: Number of trading days for VIX percentile calculation (default 252 = 1 year)
        """
        self.lookback_window = lookback_window
        self.momentum_periods = [20, 50, 200]  # SMA periods for SPY
        self.volatility_window = 20            # Window for realized volatility
        self.momentum_slope_window = 20        # Window for momentum slope calculation

    def classify_regime(
        self,
        spy_data: pd.DataFrame,
        vix_data: pd.DataFrame,
        timestamp: datetime
    ) -> Tuple[str, float]:
        """
        Classify current market regime.

        Args:
            spy_data: SPY price data with OHLCV columns
            vix_data: VIX data with close prices
            timestamp: Current timestamp for classification

        Returns:
            Tuple of (regime_name, confidence_score)
        """
        # Ensure we have enough data
        if len(spy_data) < 200 or len(vix_data) < self.lookback_window:
            logger.warning("Insufficient data for regime classification")
            return 'SIDEWAYS', 0.5

        # Calculate indicators
        indicators = self._calculate_indicators(spy_data, vix_data, timestamp)

        # Score each regime
        regime_scores = {}
        for regime, criteria in self.REGIME_CRITERIA.items():
            score = self._score_regime(indicators, criteria)
            regime_scores[regime] = score

        # Select regime with highest score
        best_regime = max(regime_scores, key=regime_scores.get)
        confidence = regime_scores[best_regime]

        # Log regime classification
        logger.debug(
            f"Regime: {best_regime} (confidence: {confidence:.2f}) | "
            f"Momentum: {indicators['momentum_slope']:.3f} | "
            f"VIX percentile: {indicators['vix_percentile']:.1f}%"
        )

        return best_regime, confidence

    def _calculate_indicators(
        self,
        spy_data: pd.DataFrame,
        vix_data: pd.DataFrame,
        timestamp: datetime
    ) -> Dict:
        """Calculate all indicators needed for regime classification."""

        # Calculate moving averages
        sma_20 = spy_data['close'].rolling(20).mean()
        sma_50 = spy_data['close'].rolling(50).mean()
        sma_200 = spy_data['close'].rolling(200).mean()

        # Current price and position relative to MAs
        current_price = spy_data['close'].iloc[-1]
        above_20 = current_price > sma_20.iloc[-1]
        above_50 = current_price > sma_50.iloc[-1]
        above_200 = current_price > sma_200.iloc[-1]

        # Momentum slope (20-day rate of change)
        momentum_slope = (sma_20.iloc[-1] - sma_20.iloc[-20]) / sma_20.iloc[-20]

        # VIX percentile rank
        current_vix = vix_data['close'].iloc[-1]
        vix_percentile = self._calculate_vix_percentile(vix_data, current_vix)

        # Realized volatility (20-day)
        returns = spy_data['close'].pct_change()
        realized_vol = returns.rolling(self.volatility_window).std().iloc[-1] * np.sqrt(252)

        # Volatility spike detection (VIX > 1.5x 20-day average)
        vix_20_avg = vix_data['close'].rolling(20).mean().iloc[-1]
        volatility_spike = current_vix > (vix_20_avg * 1.5)

        # MA slopes for trend strength
        sma_20_slope = (sma_20.iloc[-1] - sma_20.iloc[-5]) / sma_20.iloc[-5]
        sma_50_slope = (sma_50.iloc[-1] - sma_50.iloc[-10]) / sma_50.iloc[-10]

        return {
            'current_price': current_price,
            'sma_20': sma_20.iloc[-1],
            'sma_50': sma_50.iloc[-1],
            'sma_200': sma_200.iloc[-1],
            'above_20': above_20,
            'above_50': above_50,
            'above_200': above_200,
            'momentum_slope': momentum_slope,
            'vix': current_vix,
            'vix_percentile': vix_percentile,
            'realized_vol': realized_vol,
            'volatility_spike': volatility_spike,
            'sma_20_slope': sma_20_slope,
            'sma_50_slope': sma_50_slope
        }

    def _calculate_vix_percentile(
        self,
        vix_data: pd.DataFrame,
        current_vix: float
    ) -> float:
        """
        Calculate VIX percentile rank over lookback window.

        Returns:
            Percentile rank (0-100)
        """
        lookback_data = vix_data['close'].iloc[-self.lookback_window:]
        percentile = (lookback_data < current_vix).sum() / len(lookback_data) * 100
        return percentile

    def _score_regime(
        self,
        indicators: Dict,
        criteria: Dict
    ) -> float:
        """
        Score how well current indicators match regime criteria.

        Returns:
            Confidence score (0-1)
        """
        score = 0.0
        criteria_count = 0

        # Check momentum slope
        if 'momentum_slope_min' in criteria:
            if indicators['momentum_slope'] >= criteria['momentum_slope_min']:
                score += 1.0
            criteria_count += 1

        if 'momentum_slope_max' in criteria:
            if indicators['momentum_slope'] <= criteria['momentum_slope_max']:
                score += 1.0
            criteria_count += 1

        # Check VIX percentile
        if 'vix_percentile_min' in criteria:
            if indicators['vix_percentile'] >= criteria['vix_percentile_min']:
                score += 1.0
            criteria_count += 1

        if 'vix_percentile_max' in criteria:
            if indicators['vix_percentile'] <= criteria['vix_percentile_max']:
                score += 1.0
            criteria_count += 1

        # Check moving average positions
        if 'above_mas' in criteria:
            ma_score = 0
            for ma in criteria['above_mas']:
                if indicators[f'above_{ma}']:
                    ma_score += 1
            score += ma_score / len(criteria['above_mas'])
            criteria_count += 1

        if 'below_mas' in criteria:
            ma_score = 0
            for ma in criteria['below_mas']:
                if not indicators[f'above_{ma}']:
                    ma_score += 1
            score += ma_score / len(criteria['below_mas'])
            criteria_count += 1

        # Check volatility spike
        if 'volatility_spike' in criteria:
            if indicators['volatility_spike'] == criteria['volatility_spike']:
                score += 1.0
            criteria_count += 1

        # Calculate final confidence score
        if criteria_count > 0:
            confidence = score / criteria_count
        else:
            confidence = 0.5

        return confidence

    def get_regime_parameters(self, regime: str) -> Dict:
        """
        Get trading parameters for a specific regime.

        Args:
            regime: Name of the regime

        Returns:
            Dictionary of regime-specific trading parameters
        """
        regime_params = {
            'STRONG_BULL': {
                'max_positions': 3,
                'position_size_multiplier': 1.0,
                'min_win_rate': 0.60,
                'min_expected_return': 0.0025,
                'trade_frequency': 'moderate'
            },
            'WEAK_BULL': {
                'max_positions': 5,
                'position_size_multiplier': 1.0,
                'min_win_rate': 0.55,
                'min_expected_return': 0.002,
                'trade_frequency': 'high'
            },
            'SIDEWAYS': {
                'max_positions': 4,
                'position_size_multiplier': 0.9,
                'min_win_rate': 0.55,
                'min_expected_return': 0.002,
                'trade_frequency': 'moderate'
            },
            'UNPREDICTABLE': {
                'max_positions': 2,
                'position_size_multiplier': 0.6,
                'min_win_rate': 0.65,
                'min_expected_return': 0.003,
                'trade_frequency': 'low'
            },
            'BEAR': {
                'max_positions': 5,
                'position_size_multiplier': 1.1,
                'min_win_rate': 0.55,
                'min_expected_return': 0.0025,
                'trade_frequency': 'high'
            }
        }

        return regime_params.get(regime, regime_params['SIDEWAYS'])

    def analyze_regime_history(
        self,
        spy_data: pd.DataFrame,
        vix_data: pd.DataFrame,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Analyze historical regime classifications over a period.

        Args:
            spy_data: SPY historical data
            vix_data: VIX historical data
            start_date: Start of analysis period
            end_date: End of analysis period

        Returns:
            DataFrame with daily regime classifications
        """
        results = []

        # Filter data to date range
        mask = (spy_data.index >= start_date) & (spy_data.index <= end_date)
        analysis_dates = spy_data[mask].index

        for date in analysis_dates:
            # Get data up to this date
            spy_subset = spy_data[spy_data.index <= date]
            vix_subset = vix_data[vix_data.index <= date]

            # Classify regime
            regime, confidence = self.classify_regime(spy_subset, vix_subset, date)

            results.append({
                'date': date,
                'regime': regime,
                'confidence': confidence,
                'spy_close': spy_subset['close'].iloc[-1],
                'vix': vix_subset['close'].iloc[-1]
            })

        return pd.DataFrame(results).set_index('date')

    def plot_regime_transitions(
        self,
        regime_history: pd.DataFrame,
        spy_data: pd.DataFrame
    ):
        """
        Plot SPY price with regime overlays.

        Args:
            regime_history: DataFrame from analyze_regime_history
            spy_data: SPY price data
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

        # Define colors for each regime
        regime_colors = {
            'STRONG_BULL': 'darkgreen',
            'WEAK_BULL': 'lightgreen',
            'SIDEWAYS': 'gray',
            'UNPREDICTABLE': 'orange',
            'BEAR': 'red'
        }

        # Plot SPY price
        ax1.plot(spy_data.index, spy_data['close'], label='SPY', color='black', linewidth=1)

        # Overlay regime colors
        for regime, color in regime_colors.items():
            regime_mask = regime_history['regime'] == regime
            regime_periods = regime_history[regime_mask]

            for idx in regime_periods.index:
                ax1.axvspan(idx, idx + pd.Timedelta(days=1),
                           alpha=0.3, color=color, label=regime)

        # Clean up legend (remove duplicates)
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys(), loc='upper left')

        ax1.set_ylabel('SPY Price ($)')
        ax1.set_title('Market Regimes and SPY Price')
        ax1.grid(True, alpha=0.3)

        # Plot regime confidence
        ax2.plot(regime_history.index, regime_history['confidence'],
                color='blue', label='Regime Confidence')
        ax2.fill_between(regime_history.index, 0, regime_history['confidence'],
                         alpha=0.3, color='blue')

        ax2.set_ylabel('Confidence Score')
        ax2.set_xlabel('Date')
        ax2.set_title('Regime Classification Confidence')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print regime statistics
        print("\nRegime Distribution:")
        print("-" * 40)
        regime_counts = regime_history['regime'].value_counts()
        for regime, count in regime_counts.items():
            pct = count / len(regime_history) * 100
            print(f"{regime:15} {count:5} days ({pct:5.1f}%)")

        print(f"\nAverage Confidence: {regime_history['confidence'].mean():.3f}")


def test_regime_detector():
    """Test the regime detector with sample data."""
    import yfinance as yf

    logger.info("Testing Market Regime Detector")
    logger.info("="*60)

    # Download test data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)  # 2 years of data

    logger.info("Downloading test data...")
    spy = yf.download('SPY', start=start_date, end=end_date, interval='1d')
    vix = yf.download('^VIX', start=start_date, end=end_date, interval='1d')

    # Rename columns to lowercase
    spy.columns = [col.lower() for col in spy.columns]
    vix.columns = [col.lower() for col in vix.columns]

    # Initialize detector
    detector = MarketRegimeDetector()

    # Test current regime
    current_regime, confidence = detector.classify_regime(spy, vix, datetime.now())
    logger.info(f"\nCurrent Market Regime: {current_regime}")
    logger.info(f"Confidence: {confidence:.2%}")

    # Get regime parameters
    params = detector.get_regime_parameters(current_regime)
    logger.info(f"\nRegime Trading Parameters:")
    for key, value in params.items():
        logger.info(f"  {key}: {value}")

    # Analyze last 6 months
    analysis_start = end_date - timedelta(days=180)
    logger.info(f"\nAnalyzing regime history from {analysis_start.date()} to {end_date.date()}...")

    regime_history = detector.analyze_regime_history(
        spy, vix, analysis_start, end_date
    )

    # Plot results
    detector.plot_regime_transitions(regime_history, spy)

    return detector, regime_history


if __name__ == "__main__":
    test_regime_detector()