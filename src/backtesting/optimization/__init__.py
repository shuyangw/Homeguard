"""
Parameter optimization module for backtesting strategies.

This module provides optimization capabilities including:
- Grid search optimization (exhaustive parameter testing)
- Random search optimization (random sampling - Phase 4a)
- Bayesian optimization (Gaussian Process - Phase 4b, requires scikit-optimize)
- Genetic algorithm optimization (evolutionary - Phase 4c)
- Universe sweep optimization (multi-symbol parameter testing)
- Walk-forward validation (rolling window optimization for robustness testing)
- Regime-aware optimization (optimize separately per market regime)
- Smart result caching (Phase 3)
- Generic parameter sweep for any backtest function (threading-based)
- Efficient data loading with Polars for large datasets
"""

from src.backtesting.optimization.grid_search import GridSearchOptimizer
from src.backtesting.optimization.random_search import RandomSearchOptimizer
from src.backtesting.optimization.genetic_optimizer import GeneticOptimizer
from src.backtesting.optimization.sweep_runner import SweepRunner, run_generic_parameter_sweep
from src.backtesting.optimization.walk_forward import WalkForwardOptimizer
from src.backtesting.optimization.regime_aware import RegimeAwareOptimizer
from src.backtesting.optimization.result_cache import ResultCache, CacheConfig
from src.backtesting.optimization.base_optimizer import BaseOptimizer
from src.backtesting.optimization.data_loader import (
    get_default_workers,
    load_minute_cache_polars,
    load_daily_data_polars,
    load_daily_data_pandas
)

# Conditional import of BayesianOptimizer (requires scikit-optimize)
try:
    from src.backtesting.optimization.bayesian_optimizer import BayesianOptimizer, is_bayesian_available
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    BayesianOptimizer = None
    is_bayesian_available = lambda: False

__all__ = [
    'GridSearchOptimizer',
    'RandomSearchOptimizer',
    'GeneticOptimizer',
    'SweepRunner',
    'run_generic_parameter_sweep',
    'WalkForwardOptimizer',
    'RegimeAwareOptimizer',
    'ResultCache',
    'CacheConfig',
    'BaseOptimizer',
    'get_default_workers',
    'load_minute_cache_polars',
    'load_daily_data_polars',
    'load_daily_data_pandas',
    'BAYESIAN_AVAILABLE',
    'is_bayesian_available'
]

# Add BayesianOptimizer to exports if available
if BAYESIAN_AVAILABLE:
    __all__.append('BayesianOptimizer')
