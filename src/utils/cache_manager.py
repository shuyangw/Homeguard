"""
Cache manager for storing backtest results and configurations.

Provides persistent storage of backtest results and configurations in OS-appropriate
cache directories for quick retrieval without re-running backtests.
"""

import json
import pickle
import hashlib
import platform
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import pandas as pd


class CacheManager:
    """
    Manages caching of backtest results and configurations.

    Cache structure:
        cache_dir/
            configs/
                <config_hash>.json - Configuration metadata
            results/
                <config_hash>.pkl - Pickled results DataFrame
            portfolios/
                <config_hash>_<symbol>.pkl - Pickled portfolio objects
            metadata.json - Cache metadata and index
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize cache manager.

        Args:
            cache_dir: Custom cache directory (defaults to OS-appropriate location)
        """
        if cache_dir is None:
            cache_dir = self._get_default_cache_dir()

        self.cache_dir = Path(cache_dir)
        self.configs_dir = self.cache_dir / "configs"
        self.results_dir = self.cache_dir / "results"
        self.portfolios_dir = self.cache_dir / "portfolios"
        self.metadata_file = self.cache_dir / "metadata.json"

        # Create directories
        self._ensure_directories()

        # Load or initialize metadata
        self.metadata = self._load_metadata()

    def _get_default_cache_dir(self) -> Path:
        """
        Get OS-appropriate cache directory.

        Returns:
            Path to cache directory
        """
        system = platform.system()

        if system == 'Darwin':  # macOS
            base = Path.home() / "Library" / "Caches"
        elif system == 'Windows':
            base = Path.home() / "AppData" / "Local" / "Temp"
        else:  # Linux
            base = Path.home() / ".cache"

        return base / "Homeguard" / "backtests"

    def _ensure_directories(self):
        """Create cache directories if they don't exist."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        self.portfolios_dir.mkdir(exist_ok=True)

    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata or create new."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass

        return {
            'version': '1.0',
            'created': datetime.now().isoformat(),
            'entries': {}
        }

    def _save_metadata(self):
        """Save cache metadata to disk."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def _compute_config_hash(self, config: Dict[str, Any]) -> str:
        """
        Compute unique hash for configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Hash string
        """
        # Create normalized config for hashing
        normalized = {
            'strategy': config.get('strategy_class').__name__ if 'strategy_class' in config else '',
            'strategy_params': config.get('strategy_params', {}),
            'symbols': sorted(config.get('symbols', [])),
            'start_date': config.get('start_date', ''),
            'end_date': config.get('end_date', ''),
            'initial_capital': config.get('initial_capital', 0),
            'fees': config.get('fees', 0),
            'risk_profile': config.get('risk_profile', 'Moderate'),
            'portfolio_mode': config.get('portfolio_mode', 'Single-Symbol'),
            'position_sizing_method': config.get('position_sizing_method', 'equal_weight'),
            'rebalancing_frequency': config.get('rebalancing_frequency', 'never'),
            'rebalancing_threshold_pct': config.get('rebalancing_threshold_pct', 0.05)
        }

        # Convert to JSON string and hash
        config_str = json.dumps(normalized, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def cache_results(
        self,
        config: Dict[str, Any],
        results_df: pd.DataFrame,
        portfolios: Optional[Dict[str, Any]] = None,
        description: str = ""
    ) -> str:
        """
        Cache backtest results and configuration.

        Args:
            config: Configuration dictionary
            results_df: Results DataFrame
            portfolios: Optional dictionary of portfolio objects
            description: Optional description of the backtest

        Returns:
            Cache key (config hash)
        """
        config_hash = self._compute_config_hash(config)

        # Save configuration
        config_metadata = {
            'hash': config_hash,
            'timestamp': datetime.now().isoformat(),
            'description': description,
            'strategy': config.get('strategy_class').__name__ if 'strategy_class' in config else '',
            'strategy_params': config.get('strategy_params', {}),
            'symbols': config.get('symbols', []),
            'start_date': config.get('start_date', ''),
            'end_date': config.get('end_date', ''),
            'initial_capital': config.get('initial_capital', 0),
            'fees': config.get('fees', 0),
            'risk_profile': config.get('risk_profile', 'Moderate'),
            'portfolio_mode': config.get('portfolio_mode', 'Single-Symbol'),
            'position_sizing_method': config.get('position_sizing_method', 'equal_weight'),
            'rebalancing_frequency': config.get('rebalancing_frequency', 'never'),
            'rebalancing_threshold_pct': config.get('rebalancing_threshold_pct', 0.05),
            'generate_full_output': config.get('generate_full_output', True),
            'num_symbols': len(config.get('symbols', [])),
            'num_results': len(results_df) if results_df is not None else 0
        }

        config_file = self.configs_dir / f"{config_hash}.json"
        with open(config_file, 'w') as f:
            json.dump(config_metadata, f, indent=2)

        # Save results DataFrame
        if results_df is not None:
            results_file = self.results_dir / f"{config_hash}.pkl"
            results_df.to_pickle(results_file)

        # Save portfolio objects if provided
        if portfolios:
            for symbol, portfolio in portfolios.items():
                if portfolio is not None:
                    portfolio_file = self.portfolios_dir / f"{config_hash}_{symbol}.pkl"
                    with open(portfolio_file, 'wb') as f:
                        pickle.dump(portfolio, f)

        # Update metadata
        self.metadata['entries'][config_hash] = {
            'timestamp': config_metadata['timestamp'],
            'description': description,
            'strategy': config_metadata['strategy'],
            'symbols': config_metadata['symbols'],
            'date_range': f"{config_metadata['start_date']} to {config_metadata['end_date']}"
        }
        self._save_metadata()

        return config_hash

    def get_cached_results(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached results for configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Dictionary with 'config', 'results_df', and 'portfolios', or None if not cached
        """
        config_hash = self._compute_config_hash(config)
        return self.get_cached_results_by_hash(config_hash)

    def get_cached_results_by_hash(self, config_hash: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached results by hash.

        Args:
            config_hash: Configuration hash

        Returns:
            Dictionary with 'config', 'results_df', and 'portfolios', or None if not cached
        """
        config_file = self.configs_dir / f"{config_hash}.json"
        results_file = self.results_dir / f"{config_hash}.pkl"

        if not config_file.exists() or not results_file.exists():
            return None

        try:
            # Load configuration
            with open(config_file, 'r') as f:
                config = json.load(f)

            # Load results
            results_df = pd.read_pickle(results_file)

            # Load portfolios if they exist
            portfolios = {}
            for symbol in config.get('symbols', []):
                portfolio_file = self.portfolios_dir / f"{config_hash}_{symbol}.pkl"
                if portfolio_file.exists():
                    with open(portfolio_file, 'rb') as f:
                        portfolios[symbol] = pickle.load(f)

            return {
                'config': config,
                'results_df': results_df,
                'portfolios': portfolios if portfolios else None
            }

        except Exception:
            return None

    def list_cached_runs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        List recent cached backtest runs.

        Args:
            limit: Maximum number of runs to return

        Returns:
            List of run metadata dictionaries
        """
        runs = []

        for config_hash, entry in self.metadata['entries'].items():
            config_file = self.configs_dir / f"{config_hash}.json"
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)

                    runs.append({
                        'hash': config_hash,
                        'timestamp': config['timestamp'],
                        'description': entry.get('description', ''),
                        'strategy': config['strategy'],
                        'strategy_params': config.get('strategy_params', {}),
                        'symbols': config['symbols'],
                        'date_range': entry.get('date_range', ''),
                        'num_symbols': config.get('num_symbols', 0),
                        'num_results': config.get('num_results', 0),
                        'risk_profile': config.get('risk_profile', 'Moderate'),
                        'portfolio_mode': config.get('portfolio_mode', 'Single-Symbol'),
                        'position_sizing_method': config.get('position_sizing_method', 'equal_weight'),
                        'rebalancing_frequency': config.get('rebalancing_frequency', 'never')
                    })
                except Exception:
                    continue

        # Sort by timestamp (most recent first)
        runs.sort(key=lambda x: x['timestamp'], reverse=True)

        return runs[:limit]

    def clear_cache(self, older_than_days: Optional[int] = None) -> int:
        """
        Clear cached results.

        Args:
            older_than_days: Only clear cache older than N days (None = clear all)

        Returns:
            Number of cache entries cleared
        """
        cleared = 0
        cutoff_time = None

        if older_than_days is not None:
            cutoff_time = datetime.now() - timedelta(days=older_than_days)

        hashes_to_remove = []

        for config_hash, entry in list(self.metadata['entries'].items()):
            should_remove = False

            if cutoff_time is None:
                should_remove = True
            else:
                try:
                    entry_time = datetime.fromisoformat(entry['timestamp'])
                    if entry_time < cutoff_time:
                        should_remove = True
                except Exception:
                    should_remove = True

            if should_remove:
                # Remove files
                config_file = self.configs_dir / f"{config_hash}.json"
                results_file = self.results_dir / f"{config_hash}.pkl"

                if config_file.exists():
                    config_file.unlink()
                if results_file.exists():
                    results_file.unlink()

                # Remove portfolio files
                for portfolio_file in self.portfolios_dir.glob(f"{config_hash}_*.pkl"):
                    portfolio_file.unlink()

                hashes_to_remove.append(config_hash)
                cleared += 1

        # Update metadata
        for config_hash in hashes_to_remove:
            self.metadata['entries'].pop(config_hash, None)

        self._save_metadata()

        return cleared

    def get_cache_size(self) -> Dict[str, Any]:
        """
        Get cache size statistics.

        Returns:
            Dictionary with size information
        """
        total_size = 0
        file_count = 0

        for directory in [self.configs_dir, self.results_dir, self.portfolios_dir]:
            for file in directory.rglob('*'):
                if file.is_file():
                    total_size += file.stat().st_size
                    file_count += 1

        return {
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'file_count': file_count,
            'num_cached_runs': len(self.metadata['entries']),
            'cache_dir': str(self.cache_dir)
        }

    def is_cached(self, config: Dict[str, Any]) -> bool:
        """
        Check if results for configuration are cached.

        Args:
            config: Configuration dictionary

        Returns:
            True if cached, False otherwise
        """
        config_hash = self._compute_config_hash(config)
        config_file = self.configs_dir / f"{config_hash}.json"
        results_file = self.results_dir / f"{config_hash}.pkl"

        return config_file.exists() and results_file.exists()

    def get_last_run_settings(self) -> Optional[Dict[str, Any]]:
        """
        Get the settings from the most recent backtest run.

        Returns:
            Dictionary with backtest settings, or None if no runs found
        """
        runs = self.list_cached_runs(limit=1)

        if not runs:
            return None

        config_hash = runs[0]['hash']
        config_file = self.configs_dir / f"{config_hash}.json"

        if not config_file.exists():
            return None

        try:
            with open(config_file, 'r') as f:
                config = json.load(f)

            return {
                'strategy': config.get('strategy', ''),
                'strategy_params': config.get('strategy_params', {}),
                'symbols': config.get('symbols', []),
                'start_date': config.get('start_date', ''),
                'end_date': config.get('end_date', ''),
                'initial_capital': config.get('initial_capital', 100000),
                'fees': config.get('fees', 0.001),
                'risk_profile': config.get('risk_profile', 'Moderate'),
                'portfolio_mode': config.get('portfolio_mode', 'Single-Symbol'),
                'position_sizing_method': config.get('position_sizing_method', 'equal_weight'),
                'rebalancing_frequency': config.get('rebalancing_frequency', 'never'),
                'rebalancing_threshold_pct': config.get('rebalancing_threshold_pct', 0.05),
                'generate_full_output': config.get('generate_full_output', True),
                'timestamp': runs[0]['timestamp']
            }

        except Exception:
            return None
