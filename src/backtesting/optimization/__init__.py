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
"""

from backtesting.optimization.grid_search import GridSearchOptimizer
from backtesting.optimization.random_search import RandomSearchOptimizer
from backtesting.optimization.genetic_optimizer import GeneticOptimizer
from backtesting.optimization.sweep_runner import SweepRunner
from backtesting.optimization.walk_forward import WalkForwardOptimizer
from backtesting.optimization.regime_aware import RegimeAwareOptimizer
from backtesting.optimization.result_cache import ResultCache, CacheConfig
from backtesting.optimization.base_optimizer import BaseOptimizer

# Conditional import of BayesianOptimizer (requires scikit-optimize)
try:
    from backtesting.optimization.bayesian_optimizer import BayesianOptimizer, is_bayesian_available
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
    'WalkForwardOptimizer',
    'RegimeAwareOptimizer',
    'ResultCache',
    'CacheConfig',
    'BaseOptimizer',
    'BAYESIAN_AVAILABLE',
    'is_bayesian_available'
]

# Add BayesianOptimizer to exports if available
if BAYESIAN_AVAILABLE:
    __all__.append('BayesianOptimizer')
