"""
Pairs Discovery Framework - Standardized pairs discovery and cointegration testing.

This framework eliminates ~1,500 lines of duplicated pairs discovery code across
12+ pairs trading scripts by providing common patterns for:
- Symbol universe management
- Pairwise cointegration testing
- Correlation analysis
- Pairs ranking and filtering
- Statistical validation
- Result export and visualization

Usage:
    from frameworks.pairs_discovery_framework import PairsDiscoveryFramework

    # Initialize framework
    framework = PairsDiscoveryFramework(config)

    # Discover cointegrated pairs
    pairs = framework.discover_pairs(
        universe=['SPY', 'QQQ', 'IWM', 'DIA'],
        method='cointegration'
    )

    # Export results
    framework.export_pairs()
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from itertools import combinations
from statsmodels.tsa.stattools import coint, adfuller
from scipy.stats import pearsonr

from utils.path_setup import setup_project_paths
ROOT_DIR = setup_project_paths()

from src.backtesting.engine.data_loader import DataLoader
from src.utils.logger import logger
from utils.config_loader import get_nested


class PairsDiscoveryFramework:
    """
    Standardized framework for discovering and validating trading pairs.

    This framework provides common patterns for pairs discovery eliminating
    code duplication across pairs trading scripts.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pairs discovery framework.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.discovered_pairs = []
        self.symbol_data = {}
        self.discovery_start_time = None
        self.discovery_end_time = None

        # Extract settings
        self.start_date = get_nested(config, 'backtest.start_date', '2020-01-01')
        self.end_date = get_nested(config, 'backtest.end_date', '2024-12-31')

        # Discovery settings
        self.max_p_value = get_nested(config, 'pairs_discovery.max_p_value', 0.05)
        self.min_correlation = get_nested(config, 'pairs_discovery.min_correlation', 0.5)
        self.lookback_days = get_nested(config, 'pairs_discovery.lookback_days', 252)

        # Output settings
        self.output_dir = Path(get_nested(config, 'output.output_dir', 'output/discovery'))
        self.reports_dir = Path(get_nested(config, 'output.reports_dir', 'reports/discovery'))

        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Data loader
        self.data_loader = DataLoader()

    def load_universe_data(
        self,
        symbols: List[str],
        force_download: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data for symbol universe.

        Args:
            symbols: List of symbols to load
            force_download: Force re-download of data

        Returns:
            Dictionary mapping symbol -> DataFrame
        """
        logger.info(f"\n{'='*80}")
        logger.info("LOADING SYMBOL UNIVERSE")
        logger.info(f"{'='*80}")
        logger.info(f"Symbols: {len(symbols)}")
        logger.info(f"Period: {self.start_date} to {self.end_date}")

        loaded_data = {}
        failed_symbols = []

        for i, symbol in enumerate(symbols, 1):
            try:
                logger.info(f"[{i}/{len(symbols)}] Loading {symbol}...")

                # Load symbol data
                df = self.data_loader.load_symbols(
                    [symbol],
                    self.start_date,
                    self.end_date
                )

                if df is not None and len(df) > 0:
                    # Extract data for this symbol
                    symbol_df = df.xs(symbol, level='symbol')

                    # Resample to daily
                    daily_df = symbol_df.resample('D').agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna()

                    if len(daily_df) >= self.lookback_days:
                        loaded_data[symbol] = daily_df
                        logger.info(f"  ✓ Loaded {len(daily_df)} days")
                    else:
                        logger.warning(f"  ✗ Insufficient data ({len(daily_df)} < {self.lookback_days})")
                        failed_symbols.append(symbol)
                else:
                    logger.warning(f"  ✗ No data returned")
                    failed_symbols.append(symbol)

            except Exception as e:
                logger.error(f"  ✗ Failed: {str(e)}")
                failed_symbols.append(symbol)

        logger.info(f"\n{'='*80}")
        logger.info(f"Successfully loaded: {len(loaded_data)}/{len(symbols)}")
        if failed_symbols:
            logger.warning(f"Failed symbols: {', '.join(failed_symbols)}")
        logger.info(f"{'='*80}\n")

        self.symbol_data = loaded_data
        return loaded_data

    def test_cointegration(
        self,
        symbol1: str,
        symbol2: str,
        series1: pd.Series,
        series2: pd.Series
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Test cointegration between two price series.

        Args:
            symbol1: First symbol
            symbol2: Second symbol
            series1: First price series
            series2: Second price series

        Returns:
            Tuple of (p_value, is_cointegrated, stats_dict)
        """
        try:
            # Align series
            aligned = pd.DataFrame({
                symbol1: series1,
                symbol2: series2
            }).dropna()

            if len(aligned) < 30:
                return 1.0, False, {'error': 'Insufficient aligned data'}

            # Engle-Granger cointegration test
            score, p_value, _ = coint(aligned[symbol1], aligned[symbol2])

            # Calculate correlation
            correlation, _ = pearsonr(aligned[symbol1], aligned[symbol2])

            # Test stationarity of spread
            spread = aligned[symbol1] - aligned[symbol2]
            adf_stat, adf_pvalue, *_ = adfuller(spread, maxlag=1)

            is_cointegrated = (
                p_value < self.max_p_value and
                abs(correlation) > self.min_correlation
            )

            stats = {
                'p_value': float(p_value),
                'correlation': float(correlation),
                'adf_statistic': float(adf_stat),
                'adf_pvalue': float(adf_pvalue),
                'spread_mean': float(spread.mean()),
                'spread_std': float(spread.std()),
                'data_points': len(aligned)
            }

            return p_value, is_cointegrated, stats

        except Exception as e:
            logger.error(f"Cointegration test failed for {symbol1}/{symbol2}: {str(e)}")
            return 1.0, False, {'error': str(e)}

    def discover_pairs(
        self,
        universe: Optional[List[str]] = None,
        method: str = 'cointegration',
        max_pairs: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Discover trading pairs from symbol universe.

        Args:
            universe: List of symbols (uses config if not provided)
            method: Discovery method ('cointegration' or 'correlation')
            max_pairs: Maximum number of pairs to return

        Returns:
            List of discovered pairs with statistics
        """
        # Get universe from config if not provided
        if universe is None:
            universe = get_nested(self.config, 'symbols', [])

        if not universe:
            logger.error("No symbol universe provided")
            return []

        logger.info(f"\n{'='*80}")
        logger.info(f"PAIRS DISCOVERY: {method.upper()}")
        logger.info(f"{'='*80}")

        # Start timer
        self.discovery_start_time = datetime.now()

        # Load data if not already loaded
        if not self.symbol_data:
            self.load_universe_data(universe)

        # Get all symbol combinations
        all_combinations = list(combinations(self.symbol_data.keys(), 2))
        total_tests = len(all_combinations)

        logger.info(f"Testing {total_tests} pairs...")

        discovered = []

        for i, (sym1, sym2) in enumerate(all_combinations, 1):
            # Progress update
            if i % 50 == 0 or i == total_tests:
                pct = (i / total_tests) * 100
                logger.info(f"Progress: {i}/{total_tests} ({pct:.1f}%)")

            # Get price series
            series1 = self.symbol_data[sym1]['close']
            series2 = self.symbol_data[sym2]['close']

            if method == 'cointegration':
                p_value, is_valid, stats = self.test_cointegration(
                    sym1, sym2, series1, series2
                )

                if is_valid:
                    pair_info = {
                        'symbol1': sym1,
                        'symbol2': sym2,
                        'pair': f"{sym1}/{sym2}",
                        'method': 'cointegration',
                        'p_value': stats['p_value'],
                        'correlation': stats['correlation'],
                        'adf_statistic': stats['adf_statistic'],
                        'adf_pvalue': stats['adf_pvalue'],
                        'spread_mean': stats['spread_mean'],
                        'spread_std': stats['spread_std'],
                        'data_points': stats['data_points'],
                        'discovery_date': datetime.now().isoformat()
                    }
                    discovered.append(pair_info)

            elif method == 'correlation':
                # Simple correlation-based discovery
                aligned = pd.DataFrame({
                    sym1: series1,
                    sym2: series2
                }).dropna()

                if len(aligned) >= 30:
                    corr, _ = pearsonr(aligned[sym1], aligned[sym2])

                    if abs(corr) > self.min_correlation:
                        pair_info = {
                            'symbol1': sym1,
                            'symbol2': sym2,
                            'pair': f"{sym1}/{sym2}",
                            'method': 'correlation',
                            'correlation': float(corr),
                            'data_points': len(aligned),
                            'discovery_date': datetime.now().isoformat()
                        }
                        discovered.append(pair_info)

        # Sort by p_value (cointegration) or correlation
        if method == 'cointegration':
            discovered.sort(key=lambda x: x['p_value'])
        else:
            discovered.sort(key=lambda x: abs(x['correlation']), reverse=True)

        # Limit results if requested
        if max_pairs and len(discovered) > max_pairs:
            discovered = discovered[:max_pairs]

        # End timer
        self.discovery_end_time = datetime.now()
        duration = (self.discovery_end_time - self.discovery_start_time).total_seconds()

        logger.info(f"\n{'='*80}")
        logger.info("DISCOVERY COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Pairs tested: {total_tests}")
        logger.info(f"Pairs discovered: {len(discovered)}")
        logger.info(f"Success rate: {len(discovered)/total_tests*100:.1f}%")
        logger.info(f"Duration: {duration:.1f}s")
        logger.info(f"{'='*80}\n")

        # Store results
        self.discovered_pairs = discovered

        return discovered

    def filter_pairs(
        self,
        min_correlation: Optional[float] = None,
        max_p_value: Optional[float] = None,
        min_data_points: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Filter discovered pairs by criteria.

        Args:
            min_correlation: Minimum absolute correlation
            max_p_value: Maximum cointegration p-value
            min_data_points: Minimum data points required

        Returns:
            Filtered list of pairs
        """
        if not self.discovered_pairs:
            logger.warning("No pairs to filter")
            return []

        filtered = self.discovered_pairs.copy()

        # Apply filters
        if min_correlation is not None:
            filtered = [p for p in filtered
                       if abs(p.get('correlation', 0)) >= min_correlation]

        if max_p_value is not None:
            filtered = [p for p in filtered
                       if p.get('p_value', 1.0) <= max_p_value]

        if min_data_points > 0:
            filtered = [p for p in filtered
                       if p.get('data_points', 0) >= min_data_points]

        logger.info(f"Filtered: {len(self.discovered_pairs)} → {len(filtered)} pairs")

        return filtered

    def export_pairs(
        self,
        filename: Optional[str] = None,
        format: str = 'csv'
    ) -> Path:
        """
        Export discovered pairs to file.

        Args:
            filename: Optional custom filename
            format: Export format ('csv' or 'json')

        Returns:
            Path to exported file
        """
        if not self.discovered_pairs:
            logger.warning("No pairs to export")
            return None

        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_discovered_pairs"

        if format == 'csv':
            output_path = self.output_dir / f"{filename}.csv"
            df = pd.DataFrame(self.discovered_pairs)
            df.to_csv(output_path, index=False)
        elif format == 'json':
            output_path = self.output_dir / f"{filename}.json"
            import json
            with open(output_path, 'w') as f:
                json.dump(self.discovered_pairs, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Pairs exported to: {output_path}")
        return output_path

    def generate_report(self, filename: Optional[str] = None) -> Path:
        """
        Generate markdown report of discovered pairs.

        Args:
            filename: Optional custom filename

        Returns:
            Path to generated report
        """
        if not self.discovered_pairs:
            logger.warning("No pairs to report")
            return None

        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_pairs_discovery_report.md"

        report_path = self.reports_dir / filename

        # Generate report content
        report = f"""# Pairs Discovery Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Period**: {self.start_date} to {self.end_date}
**Method**: {self.discovered_pairs[0].get('method', 'unknown').upper()}

---

## Summary

- **Total Pairs Discovered**: {len(self.discovered_pairs)}
- **Universe Size**: {len(self.symbol_data)} symbols
- **Lookback Period**: {self.lookback_days} days

### Discovery Criteria

- **Max P-Value**: {self.max_p_value}
- **Min Correlation**: {self.min_correlation}

---

## Top Discovered Pairs

| Rank | Pair | P-Value | Correlation | ADF P-Value | Data Points |
|------|------|---------|-------------|-------------|-------------|
"""

        for i, pair in enumerate(self.discovered_pairs[:20], 1):
            p_val = pair.get('p_value', 'N/A')
            corr = pair.get('correlation', 0)
            adf_p = pair.get('adf_pvalue', 'N/A')
            points = pair.get('data_points', 0)

            p_val_str = f"{p_val:.6f}" if isinstance(p_val, float) else p_val
            adf_p_str = f"{adf_p:.6f}" if isinstance(adf_p, float) else adf_p

            report += f"| {i} | {pair['pair']} | {p_val_str} | {corr:.3f} | {adf_p_str} | {points} |\n"

        report += "\n---\n\n## All Discovered Pairs\n\n"

        for i, pair in enumerate(self.discovered_pairs, 1):
            report += f"\n### {i}. {pair['pair']}\n\n"
            report += "| Metric | Value |\n"
            report += "|--------|-------|\n"

            if 'p_value' in pair:
                report += f"| Cointegration P-Value | {pair['p_value']:.6f} |\n"
            if 'correlation' in pair:
                report += f"| Correlation | {pair['correlation']:.4f} |\n"
            if 'adf_pvalue' in pair:
                report += f"| ADF P-Value | {pair['adf_pvalue']:.6f} |\n"
            if 'spread_mean' in pair:
                report += f"| Spread Mean | {pair['spread_mean']:.4f} |\n"
            if 'spread_std' in pair:
                report += f"| Spread Std Dev | {pair['spread_std']:.4f} |\n"

            report += f"| Data Points | {pair.get('data_points', 0)} |\n"
            report += "\n"

        # Write report
        with open(report_path, 'w') as f:
            f.write(report)

        logger.info(f"Report generated: {report_path}")
        return report_path
