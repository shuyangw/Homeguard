"""
Regime detection for market condition analysis.

Automatically detects different market regimes:
- Trend regimes: Bull, Bear, Sideways
- Volatility regimes: High, Low
- Drawdown regimes: Drawdown, Recovery, Calm
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np


class RegimeLabel(Enum):
    """Market regime labels."""
    # Trend regimes
    BULL = "Bull Market"
    BEAR = "Bear Market"
    SIDEWAYS = "Sideways"

    # Volatility regimes
    HIGH_VOL = "High Volatility"
    LOW_VOL = "Low Volatility"

    # Drawdown regimes
    DRAWDOWN = "Drawdown"
    RECOVERY = "Recovery"
    CALM = "Calm"


@dataclass
class RegimePeriod:
    """Single regime period."""
    start_date: str
    end_date: str
    regime: RegimeLabel
    metric_value: float  # The metric that determined this regime


class TrendDetector:
    """
    Detects trend-based regimes (Bull/Bear/Sideways).

    Uses simple moving average and price comparison to classify trends.
    """

    def __init__(self, lookback_days: int = 60, threshold_pct: float = 5.0):
        """
        Initialize trend detector.

        Args:
            lookback_days: Days to look back for trend calculation
            threshold_pct: Minimum % move to classify as bull/bear (otherwise sideways)
        """
        self.lookback_days = lookback_days
        self.threshold_pct = threshold_pct

    def detect(self, prices: pd.Series) -> List[RegimePeriod]:
        """
        Detect trend regimes from price series.

        Args:
            prices: Price series (typically close prices)

        Returns:
            List of RegimePeriod objects
        """
        if len(prices) < self.lookback_days:
            return []

        # Calculate rolling returns over lookback period
        rolling_returns = prices.pct_change(self.lookback_days) * 100

        # Classify each period
        regimes = []
        current_regime = None
        regime_start = None

        for date, return_val in rolling_returns.items():
            if pd.isna(return_val):
                continue

            # Determine regime
            if return_val > self.threshold_pct:
                regime = RegimeLabel.BULL
            elif return_val < -self.threshold_pct:
                regime = RegimeLabel.BEAR
            else:
                regime = RegimeLabel.SIDEWAYS

            # Track regime changes
            if regime != current_regime:
                # Save previous regime if it exists
                if current_regime is not None and regime_start is not None:
                    regimes.append(RegimePeriod(
                        start_date=regime_start.strftime('%Y-%m-%d'),
                        end_date=date.strftime('%Y-%m-%d'),
                        regime=current_regime,
                        metric_value=return_val
                    ))

                current_regime = regime
                regime_start = date

        # Add final regime
        if current_regime is not None and regime_start is not None:
            regimes.append(RegimePeriod(
                start_date=regime_start.strftime('%Y-%m-%d'),
                end_date=prices.index[-1].strftime('%Y-%m-%d'),
                regime=current_regime,
                metric_value=rolling_returns.iloc[-1]
            ))

        return regimes


class VolatilityDetector:
    """
    Detects volatility-based regimes (High/Low Volatility).

    Uses rolling standard deviation of returns.
    """

    def __init__(self, lookback_days: int = 20, percentile: float = 70):
        """
        Initialize volatility detector.

        Args:
            lookback_days: Days to calculate volatility over
            percentile: Percentile threshold (above = high vol, below = low vol)
        """
        self.lookback_days = lookback_days
        self.percentile = percentile

    def detect(self, prices: pd.Series) -> List[RegimePeriod]:
        """
        Detect volatility regimes from price series.

        Args:
            prices: Price series (typically close prices)

        Returns:
            List of RegimePeriod objects
        """
        if len(prices) < self.lookback_days:
            return []

        # Calculate rolling volatility (annualized)
        returns = prices.pct_change()
        rolling_vol = returns.rolling(window=self.lookback_days).std() * np.sqrt(252) * 100

        # Determine threshold
        threshold = np.percentile(rolling_vol.dropna(), self.percentile)

        # Classify each period
        regimes = []
        current_regime = None
        regime_start = None

        for date, vol in rolling_vol.items():
            if pd.isna(vol):
                continue

            # Determine regime
            regime = RegimeLabel.HIGH_VOL if vol > threshold else RegimeLabel.LOW_VOL

            # Track regime changes
            if regime != current_regime:
                # Save previous regime if it exists
                if current_regime is not None and regime_start is not None:
                    regimes.append(RegimePeriod(
                        start_date=regime_start.strftime('%Y-%m-%d'),
                        end_date=date.strftime('%Y-%m-%d'),
                        regime=current_regime,
                        metric_value=vol
                    ))

                current_regime = regime
                regime_start = date

        # Add final regime
        if current_regime is not None and regime_start is not None:
            regimes.append(RegimePeriod(
                start_date=regime_start.strftime('%Y-%m-%d'),
                end_date=prices.index[-1].strftime('%Y-%m-%d'),
                regime=current_regime,
                metric_value=rolling_vol.iloc[-1]
            ))

        return regimes


class DrawdownDetector:
    """
    Detects drawdown-based regimes (Drawdown/Recovery/Calm).

    Identifies when the market is experiencing drawdowns or recovering.
    """

    def __init__(self, drawdown_threshold: float = 10.0):
        """
        Initialize drawdown detector.

        Args:
            drawdown_threshold: Minimum drawdown % to classify as drawdown regime
        """
        self.drawdown_threshold = drawdown_threshold

    def detect(self, prices: pd.Series) -> List[RegimePeriod]:
        """
        Detect drawdown regimes from price series.

        Args:
            prices: Price series (typically close prices)

        Returns:
            List of RegimePeriod objects
        """
        if len(prices) < 2:
            return []

        # Calculate running maximum (high water mark)
        running_max = prices.expanding().max()

        # Calculate drawdown from high water mark
        drawdown = ((prices - running_max) / running_max) * 100

        # Classify each period
        regimes = []
        current_regime = None
        regime_start = None
        prev_drawdown = 0

        for date, dd in drawdown.items():
            if pd.isna(dd):
                continue

            # Determine regime
            if dd < -self.drawdown_threshold:
                # In drawdown
                if prev_drawdown < dd:
                    # Drawdown getting worse
                    regime = RegimeLabel.DRAWDOWN
                else:
                    # Recovering from drawdown
                    regime = RegimeLabel.RECOVERY
            else:
                # Near high water mark
                regime = RegimeLabel.CALM

            # Track regime changes
            if regime != current_regime:
                # Save previous regime if it exists
                if current_regime is not None and regime_start is not None:
                    regimes.append(RegimePeriod(
                        start_date=regime_start.strftime('%Y-%m-%d'),
                        end_date=date.strftime('%Y-%m-%d'),
                        regime=current_regime,
                        metric_value=dd
                    ))

                current_regime = regime
                regime_start = date

            prev_drawdown = dd

        # Add final regime
        if current_regime is not None and regime_start is not None:
            regimes.append(RegimePeriod(
                start_date=regime_start.strftime('%Y-%m-%d'),
                end_date=prices.index[-1].strftime('%Y-%m-%d'),
                regime=current_regime,
                metric_value=drawdown.iloc[-1]
            ))

        return regimes


def detect_all_regimes(prices: pd.Series) -> Dict[str, List[RegimePeriod]]:
    """
    Run all regime detectors on price series.

    Args:
        prices: Price series (typically close prices)

    Returns:
        Dictionary mapping detector name to list of RegimePeriod objects
    """
    trend = TrendDetector()
    volatility = VolatilityDetector()
    drawdown = DrawdownDetector()

    return {
        'trend': trend.detect(prices),
        'volatility': volatility.detect(prices),
        'drawdown': drawdown.detect(prices)
    }
