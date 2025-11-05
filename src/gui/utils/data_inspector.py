"""
Data Inspector utility for previewing loaded price data.

Provides summaries and statistics about the price data used in backtests.
"""

import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime


class DataInspector:
    """
    Inspector for analyzing price data used in backtests.

    Provides methods to:
    - Summarize data availability
    - Calculate data quality metrics
    - Identify gaps and issues
    """

    @staticmethod
    def summarize_data(df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Create summary statistics for price data.

        Args:
            df: Price DataFrame with OHLCV columns
            symbol: Symbol name

        Returns:
            Dictionary with data summary
        """
        if df is None or df.empty:
            return {
                'symbol': symbol,
                'rows': 0,
                'start_date': 'N/A',
                'end_date': 'N/A',
                'trading_days': 0,
                'missing_data': 0,
                'price_range': 'N/A'
            }

        summary = {
            'symbol': symbol,
            'rows': len(df),
            'start_date': str(df.index[0]) if len(df) > 0 else 'N/A',
            'end_date': str(df.index[-1]) if len(df) > 0 else 'N/A',
            'trading_days': len(df),
            'columns': list(df.columns),
        }

        # Price range
        if 'close' in df.columns:
            close_col = df['close']
            summary['price_min'] = float(close_col.min())
            summary['price_max'] = float(close_col.max())
            summary['price_mean'] = float(close_col.mean())
            summary['price_range'] = f"${summary['price_min']:.2f} - ${summary['price_max']:.2f}"

        # Missing data check
        summary['missing_data'] = int(df.isna().sum().sum())

        # Volume stats if available
        if 'volume' in df.columns:
            volume_col = df['volume']
            summary['avg_volume'] = float(volume_col.mean())
            summary['total_volume'] = float(volume_col.sum())

        return summary

    @staticmethod
    def detect_data_issues(df: pd.DataFrame) -> List[str]:
        """
        Detect potential data quality issues.

        Args:
            df: Price DataFrame

        Returns:
            List of issue descriptions
        """
        issues = []

        if df is None or df.empty:
            issues.append("No data available")
            return issues

        # Check for missing values
        missing_count = df.isna().sum().sum()
        if missing_count > 0:
            issues.append(f"Found {missing_count} missing values")

        # Check for zero or negative prices
        if 'close' in df.columns:
            if (df['close'] <= 0).any():
                issues.append("Found zero or negative prices")

        # Check for zero volume
        if 'volume' in df.columns:
            zero_volume_count = (df['volume'] == 0).sum()
            if zero_volume_count > 0:
                issues.append(f"Found {zero_volume_count} bars with zero volume")

        # Check data span
        if len(df) < 50:
            issues.append(f"Limited data: only {len(df)} bars available")

        if not issues:
            issues.append("No data quality issues detected")

        return issues

    @staticmethod
    def calculate_returns_summary(df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate returns statistics from price data.

        Args:
            df: Price DataFrame with 'close' column

        Returns:
            Dictionary with returns statistics
        """
        if df is None or df.empty or 'close' not in df.columns:
            return {
                'total_return': 0.0,
                'daily_return_mean': 0.0,
                'daily_return_std': 0.0,
                'best_day': 0.0,
                'worst_day': 0.0
            }

        close = df['close']
        returns = close.pct_change().dropna()

        total_return = ((close.iloc[-1] / close.iloc[0]) - 1) * 100 if len(close) > 0 else 0

        stats = {
            'total_return': float(total_return),
            'daily_return_mean': float(returns.mean() * 100),
            'daily_return_std': float(returns.std() * 100),
            'best_day': float(returns.max() * 100) if len(returns) > 0 else 0,
            'worst_day': float(returns.min() * 100) if len(returns) > 0 else 0
        }

        return stats
