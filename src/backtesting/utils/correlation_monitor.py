"""
Correlation Monitor

Tracks correlations between pairs to ensure portfolio diversification.

Author: Homeguard Team
Date: November 11, 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from src.utils import logger


@dataclass
class CorrelationWarning:
    """Warning about high correlation between pairs."""
    pair1: str
    pair2: str
    correlation: float
    severity: str  # 'low', 'medium', 'high', 'critical'


class CorrelationMonitor:
    """
    Monitor correlations between trading pairs for diversification.

    Tracks rolling correlations and alerts when pairs become too correlated,
    which reduces diversification benefits.
    """

    def __init__(
        self,
        lookback_days: int = 60,
        warning_threshold: float = 0.70,
        critical_threshold: float = 0.85
    ):
        """
        Initialize correlation monitor.

        Args:
            lookback_days: Days to use for correlation calculation
            warning_threshold: Correlation level to trigger warnings
            critical_threshold: Correlation level for critical alerts
        """
        self.lookback_days = lookback_days
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

        self.returns = {}
        self.correlation_matrix = None
        self.warnings = []

    def add_pair_returns(self, pair_name: str, returns: pd.Series):
        """
        Add return series for a pair.

        Args:
            pair_name: Name of the pair (e.g., 'XLY/UVXY')
            returns: Series of returns
        """
        self.returns[pair_name] = returns

    def update_correlations(self) -> pd.DataFrame:
        """
        Update correlation matrix for all pairs.

        Returns:
            DataFrame with correlation matrix
        """
        if len(self.returns) < 2:
            logger.warning("Need at least 2 pairs to calculate correlations")
            return None

        # Create returns dataframe
        returns_df = pd.DataFrame(self.returns)

        # Calculate rolling correlation
        if len(returns_df) > self.lookback_days:
            returns_df = returns_df.tail(self.lookback_days)

        self.correlation_matrix = returns_df.corr()

        return self.correlation_matrix

    def check_diversification(self) -> List[CorrelationWarning]:
        """
        Check for high correlations that reduce diversification.

        Returns:
            List of correlation warnings
        """
        if self.correlation_matrix is None:
            self.update_correlations()

        if self.correlation_matrix is None:
            return []

        self.warnings = []

        # Check all pair combinations
        pairs = self.correlation_matrix.index.tolist()
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                corr = abs(self.correlation_matrix.iloc[i, j])

                if corr >= self.critical_threshold:
                    severity = 'critical'
                elif corr >= self.warning_threshold:
                    severity = 'high'
                else:
                    continue

                warning = CorrelationWarning(
                    pair1=pairs[i],
                    pair2=pairs[j],
                    correlation=self.correlation_matrix.iloc[i, j],
                    severity=severity
                )
                self.warnings.append(warning)

        return self.warnings

    def display_correlation_report(self):
        """Display correlation analysis report."""
        if self.correlation_matrix is None:
            logger.warning("No correlation data available")
            return

        logger.info("\n" + "="*80)
        logger.info("CORRELATION ANALYSIS")
        logger.info("="*80)

        logger.info(f"\nCorrelation Matrix (lookback: {self.lookback_days} days):")
        logger.info("\n" + str(self.correlation_matrix.round(3)))

        # Check for warnings
        warnings = self.check_diversification()

        if not warnings:
            logger.success("\n[OK] Good diversification - no high correlations detected")
        else:
            logger.warning(f"\n[WARNING] Found {len(warnings)} correlation warnings:")
            for w in warnings:
                if w.severity == 'critical':
                    logger.error(f"  [CRITICAL] {w.pair1} vs {w.pair2}: {w.correlation:.3f}")
                else:
                    logger.warning(f"  [HIGH] {w.pair1} vs {w.pair2}: {w.correlation:.3f}")

        # Calculate average correlation
        n = len(self.correlation_matrix)
        if n > 1:
            # Get upper triangle (excluding diagonal)
            mask = np.triu(np.ones((n, n)), k=1).astype(bool)
            avg_corr = abs(self.correlation_matrix.values[mask]).mean()

            logger.info(f"\n[AVERAGE CORRELATION] {avg_corr:.3f}")

            if avg_corr < 0.3:
                logger.success("  [EXCELLENT] Diversification (< 0.3)")
            elif avg_corr < 0.5:
                logger.info("  [GOOD] Diversification (0.3-0.5)")
            elif avg_corr < 0.7:
                logger.warning("  [MODERATE] Diversification (0.5-0.7)")
            else:
                logger.error("  [POOR] Diversification (>= 0.7)")

    def get_diversification_score(self) -> float:
        """
        Calculate diversification score (0-1, higher is better).

        Returns:
            Diversification score
        """
        if self.correlation_matrix is None or len(self.correlation_matrix) < 2:
            return 1.0

        # Calculate average absolute correlation
        n = len(self.correlation_matrix)
        mask = np.triu(np.ones((n, n)), k=1).astype(bool)
        avg_corr = abs(self.correlation_matrix.values[mask]).mean()

        # Convert to diversification score (inverse of correlation)
        # 0 correlation = 1.0 score
        # 1 correlation = 0.0 score
        score = 1.0 - avg_corr

        return score

    def calculate_diversification_benefit(self) -> float:
        """
        Estimate Sharpe ratio improvement from diversification.

        Based on correlation:
        - 0.0 correlation: ~40% Sharpe boost
        - 0.3 correlation: ~25% boost
        - 0.5 correlation: ~15% boost
        - 0.7 correlation: ~5% boost
        - 1.0 correlation: 0% boost

        Returns:
            Multiplicative factor for Sharpe ratio (e.g., 1.15 = 15% boost)
        """
        if self.correlation_matrix is None or len(self.correlation_matrix) < 2:
            return 1.0

        # Get average correlation
        n = len(self.correlation_matrix)
        mask = np.triu(np.ones((n, n)), k=1).astype(bool)
        avg_corr = abs(self.correlation_matrix.values[mask]).mean()

        # Calculate benefit using empirical formula
        # Benefit decreases linearly from 40% at 0 corr to 0% at 1.0 corr
        max_benefit = 0.40
        benefit = max_benefit * (1.0 - avg_corr)

        return 1.0 + benefit