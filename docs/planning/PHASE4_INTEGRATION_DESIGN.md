# Phase 4: Advanced Optimization Methods - Integration Design

**Status**: Design Document (Not Yet Implemented)
**Date**: November 8, 2025

---

## Overview

Phase 4 adds advanced optimization methods (Random Search, Bayesian Optimization, Early Stopping, Genetic Algorithms) that integrate seamlessly with the existing grid search infrastructure.

**Key Design Principle**: **Don't replace GridSearch - enhance it with alternatives**

---

## Architecture: Strategy Pattern

### Base Class Hierarchy

```python
# src/backtesting/optimization/base_optimizer.py

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import pandas as pd

class BaseOptimizer(ABC):
    """
    Abstract base class for all optimization methods.

    Provides common infrastructure:
    - Result caching (Phase 3)
    - Parallel execution (Phase 1)
    - Progress tracking (Phase 2)
    - CSV export (Phase 2)
    - Statistics collection
    """

    def __init__(self, engine: 'BacktestEngine'):
        self.engine = engine
        self._prepare_engine_config()

    @abstractmethod
    def optimize(
        self,
        strategy_class: type,
        param_space: Dict[str, Any],  # Can be grid, ranges, or distributions
        symbols: Union[str, List[str]],
        start_date: str,
        end_date: str,
        metric: str = 'sharpe_ratio',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run optimization to find best parameters.

        Each subclass implements this with their algorithm.
        """
        pass

    # ========== Shared Infrastructure (from GridSearchOptimizer) ==========

    def _test_single_params(self, ...):
        """Test a single parameter combination (reused from Phase 1)."""
        pass

    def _prepare_engine_config(self):
        """Prepare engine config for multiprocessing (reused from Phase 1)."""
        pass

    def _export_results_to_csv(self, ...):
        """Export results to CSV (reused from Phase 2)."""
        pass

    def _export_sensitivity_analysis(self, ...):
        """Export parameter sensitivity (reused from Phase 2)."""
        pass
```

### Concrete Implementations

```python
# src/backtesting/optimization/grid_search.py

class GridSearchOptimizer(BaseOptimizer):
    """
    Exhaustive grid search (existing implementation).

    Tests ALL parameter combinations.
    Best for: Small parameter spaces, thorough exploration.
    """

    def optimize(self, strategy_class, param_grid, symbols, start_date, end_date,
                 metric='sharpe_ratio', max_workers=None, use_cache=True, **kwargs):
        # Existing implementation (Phases 1+2+3)
        # No changes needed!
        pass


# src/backtesting/optimization/random_search.py (NEW)

class RandomSearchOptimizer(BaseOptimizer):
    """
    Random sampling from parameter space.

    Tests N random combinations instead of all.
    Best for: Large parameter spaces, time-constrained optimization.

    Speedup: 10-100x faster than grid search for large spaces.
    """

    def optimize(self, strategy_class, param_ranges, symbols, start_date, end_date,
                 metric='sharpe_ratio', n_iterations=100, max_workers=None,
                 use_cache=True, **kwargs):
        """
        Optimize using random search.

        Args:
            param_ranges: Dict mapping param names to ranges/distributions
                Example: {
                    'fast_window': (5, 30),           # Uniform range
                    'slow_window': (40, 120),
                    'threshold': (0.01, 0.10, 'log')  # Log-uniform
                }
            n_iterations: Number of random samples to test
        """
        # Generate N random parameter combinations
        param_samples = self._generate_random_samples(param_ranges, n_iterations)

        # Test each sample (reuses parallel execution & caching from base class)
        results = self._test_parameters_parallel(
            param_samples,
            strategy_class,
            symbols,
            start_date,
            end_date,
            metric,
            max_workers,
            use_cache
        )

        return self._find_best_result(results, metric)


# src/backtesting/optimization/bayesian_optimizer.py (NEW)

class BayesianOptimizer(BaseOptimizer):
    """
    Bayesian optimization using Gaussian Processes.

    Intelligently selects next parameters to test based on previous results.
    Best for: Expensive objective functions, limited budget.

    Speedup: 5-20x fewer iterations than random/grid search.
    Requires: scikit-optimize library
    """

    def optimize(self, strategy_class, param_space, symbols, start_date, end_date,
                 metric='sharpe_ratio', n_iterations=50, n_initial_points=10,
                 acquisition_func='EI', max_workers=None, use_cache=True, **kwargs):
        """
        Optimize using Bayesian optimization.

        Args:
            param_space: List of parameter dimensions
                Example: [
                    Integer(5, 30, name='fast_window'),
                    Integer(40, 120, name='slow_window'),
                    Real(0.01, 0.10, prior='log-uniform', name='threshold')
                ]
            n_iterations: Total optimization iterations
            n_initial_points: Random initialization points
            acquisition_func: 'EI' (Expected Improvement), 'LCB', or 'PI'
        """
        from skopt import Optimizer as BayesOptimizer
        from skopt.space import Integer, Real, Categorical

        # Initialize Bayesian optimizer
        bayes_opt = BayesOptimizer(
            dimensions=param_space,
            n_initial_points=n_initial_points,
            acquisition_func=acquisition_func
        )

        best_value = float('-inf')
        best_params = None

        for iteration in range(n_iterations):
            # Ask optimizer for next point to evaluate
            next_params = bayes_opt.ask()

            # Test parameter combination (reuses caching!)
            result = self._test_single_parameter(
                next_params, strategy_class, symbols,
                start_date, end_date, metric, use_cache
            )

            # Tell optimizer the result
            bayes_opt.tell(next_params, -result['value'])  # Negative for maximization

            # Track progress
            if result['value'] > best_value:
                best_value = result['value']
                best_params = result['params']

            # Log progress with ETA
            self._log_bayesian_progress(iteration, n_iterations, best_value)

        return {
            'best_params': best_params,
            'best_value': best_value,
            'convergence_plot': bayes_opt.convergence_plot_data
        }


# src/backtesting/optimization/genetic_optimizer.py (NEW)

class GeneticOptimizer(BaseOptimizer):
    """
    Genetic algorithm optimization.

    Evolves population of parameter sets over generations.
    Best for: Complex, multi-modal objective functions.

    Features: Crossover, mutation, elitism.
    """

    def optimize(self, strategy_class, param_ranges, symbols, start_date, end_date,
                 metric='sharpe_ratio', population_size=50, n_generations=20,
                 mutation_rate=0.1, crossover_rate=0.7, elitism=0.2,
                 max_workers=None, use_cache=True, **kwargs):
        """
        Optimize using genetic algorithm.

        Args:
            population_size: Number of individuals per generation
            n_generations: Number of evolutionary generations
            mutation_rate: Probability of random mutation
            crossover_rate: Probability of crossover
            elitism: Fraction of top performers to keep unchanged
        """
        # Initialize random population
        population = self._initialize_population(param_ranges, population_size)

        for generation in range(n_generations):
            # Evaluate all individuals (uses parallel execution & caching!)
            fitness_scores = self._evaluate_population(
                population, strategy_class, symbols,
                start_date, end_date, metric, max_workers, use_cache
            )

            # Select parents
            parents = self._tournament_selection(population, fitness_scores)

            # Create next generation
            offspring = self._crossover(parents, crossover_rate)
            offspring = self._mutate(offspring, mutation_rate)

            # Apply elitism (keep best individuals)
            population = self._apply_elitism(population, offspring,
                                            fitness_scores, elitism)

            # Log progress
            best_fitness = max(fitness_scores)
            self._log_genetic_progress(generation, n_generations, best_fitness)

        # Return best individual
        best_idx = fitness_scores.index(max(fitness_scores))
        return {
            'best_params': population[best_idx],
            'best_value': fitness_scores[best_idx],
            'evolution_history': self.evolution_history
        }
```

---

## User API: How to Choose Optimization Method

### Option 1: Separate Classes (Explicit Choice)

```python
from backtesting.optimization import (
    GridSearchOptimizer,      # Existing (Phases 1-3)
    RandomSearchOptimizer,    # New
    BayesianOptimizer,        # New
    GeneticOptimizer          # New
)
from backtesting.engine import BacktestEngine

engine = BacktestEngine(initial_capital=100000, fees=0.001)

# ========== Grid Search (exhaustive) ==========
grid_optimizer = GridSearchOptimizer(engine)
result = grid_optimizer.optimize_parallel(
    strategy_class=MovingAverageCrossover,
    param_grid={
        'fast_window': [10, 15, 20, 25, 30],
        'slow_window': [50, 60, 70, 80, 90, 100]
    },
    symbols='AAPL',
    start_date='2023-01-01',
    end_date='2024-01-01'
)
# Tests: 5 * 6 = 30 combinations


# ========== Random Search (sampling) ==========
random_optimizer = RandomSearchOptimizer(engine)
result = random_optimizer.optimize(
    strategy_class=MovingAverageCrossover,
    param_ranges={
        'fast_window': (5, 30),      # Sample uniformly from [5, 30]
        'slow_window': (40, 120)     # Sample uniformly from [40, 120]
    },
    symbols='AAPL',
    start_date='2023-01-01',
    end_date='2024-01-01',
    n_iterations=30  # Test only 30 random combinations
)
# Tests: 30 combinations (same as grid, but sampled randomly)
# Can explore much larger space: (5-30) * (40-120) = 25*80 = 2000 possible combos


# ========== Bayesian Optimization (intelligent) ==========
from skopt.space import Integer

bayes_optimizer = BayesianOptimizer(engine)
result = bayes_optimizer.optimize(
    strategy_class=MovingAverageCrossover,
    param_space=[
        Integer(5, 30, name='fast_window'),
        Integer(40, 120, name='slow_window')
    ],
    symbols='AAPL',
    start_date='2023-01-01',
    end_date='2024-01-01',
    n_iterations=30,        # Only need 30 intelligent samples
    n_initial_points=10     # Start with 10 random points
)
# Tests: 30 combinations (intelligently chosen!)
# Often finds optimum 5-10x faster than random search


# ========== Genetic Algorithm (evolutionary) ==========
genetic_optimizer = GeneticOptimizer(engine)
result = genetic_optimizer.optimize(
    strategy_class=MovingAverageCrossover,
    param_ranges={
        'fast_window': (5, 30),
        'slow_window': (40, 120)
    },
    symbols='AAPL',
    start_date='2023-01-01',
    end_date='2024-01-01',
    population_size=15,
    n_generations=10
)
# Tests: 15 * 10 = 150 combinations (but converges quickly)
```

### Option 2: Unified Interface (Strategy Parameter)

```python
from backtesting.optimization import ParameterOptimizer

optimizer = ParameterOptimizer(engine)

# Same interface, different methods via 'method' parameter
result = optimizer.optimize(
    strategy_class=MovingAverageCrossover,
    param_space={
        'fast_window': (5, 30),
        'slow_window': (40, 120)
    },
    symbols='AAPL',
    start_date='2023-01-01',
    end_date='2024-01-01',
    method='bayesian',      # 'grid', 'random', 'bayesian', 'genetic'
    n_iterations=30
)
```

---

## Backward Compatibility

**100% Backward Compatible** - Existing code doesn't break:

```python
# ✅ OLD CODE (Phases 1-3) - STILL WORKS!
from backtesting.optimization import GridSearchOptimizer

optimizer = GridSearchOptimizer(engine)
result = optimizer.optimize_parallel(
    strategy_class=MovingAverageCrossover,
    param_grid={'fast_window': [10, 20], 'slow_window': [50, 100]},
    symbols='AAPL',
    start_date='2023-01-01',
    end_date='2024-01-01'
)
# No changes needed!


# ✅ NEW CODE (Phase 4) - New capabilities!
from backtesting.optimization import BayesianOptimizer

optimizer = BayesianOptimizer(engine)
result = optimizer.optimize(...)
```

---

## GUI Integration

Add dropdown to optimization dialog:

```
┌─────────────────────────────────────────────┐
│  Optimize Parameters                         │
├─────────────────────────────────────────────┤
│                                              │
│  Optimization Method:                        │
│  ┌──────────────────────────────────────┐   │
│  │ Grid Search (Exhaustive)         ▼  │   │  ← NEW DROPDOWN
│  └──────────────────────────────────────┘   │
│     └─ Options:                              │
│        • Grid Search (Exhaustive)            │
│        • Random Search (Fast)                │
│        • Bayesian Optimization (Smart)       │
│        • Genetic Algorithm (Evolutionary)    │
│                                              │
│  ┌─────────────────────────────────────────┐│
│  │ Grid Search Settings (default)          ││
│  │ • Tests all combinations                ││
│  │ • Best for small grids                  ││
│  └─────────────────────────────────────────┘│
│                                              │
│  ┌─────────────────────────────────────────┐│
│  │ Random Search Settings (when selected)  ││
│  │ • Number of iterations: [100      ]     ││
│  │ • Tests random sample of space          ││
│  └─────────────────────────────────────────┘│
│                                              │
│  ┌─────────────────────────────────────────┐│
│  │ Bayesian Settings (when selected)       ││
│  │ • Iterations: [50        ]              ││
│  │ • Initial points: [10        ]          ││
│  │ • Acquisition: [EI           ▼]        ││
│  └─────────────────────────────────────────┘│
│                                              │
│  [Cancel]                    [Optimize]      │
└─────────────────────────────────────────────┘
```

---

## Code Reuse: What's Shared vs New

### ✅ **Reused from Phases 1-3** (No Duplication!)

All new optimizers inherit/reuse:

1. **Parallel Execution** (Phase 1)
   - `_test_single_params()` function
   - ProcessPoolExecutor infrastructure
   - Worker management

2. **Progress Tracking** (Phase 2)
   - ETA calculation
   - Progress percentage
   - Real-time logging

3. **Result Caching** (Phase 3)
   - Cache key generation
   - Memory + disk caching
   - TTL management
   - Cache statistics

4. **CSV Export** (Phase 2)
   - Results export
   - Sensitivity analysis
   - Timing statistics

5. **Core Infrastructure**
   - Engine configuration
   - Error handling
   - Validation
   - Metric extraction

### 🆕 **New Code** (Algorithm-Specific)

Each optimizer adds:

1. **Random Search**: Random sampling logic (~100 lines)
2. **Bayesian**: Gaussian process surrogate model (~150 lines)
3. **Genetic**: Crossover, mutation, selection (~200 lines)
4. **Base Class**: Common interface extraction (~100 lines)

**Total New Code**: ~550 lines
**Reused Code**: ~1,200 lines (from Phases 1-3)
**Reuse Ratio**: 70% reused!

---

## Performance Comparison

Example: Optimize 4 parameters with large search space

```
Parameter Space:
- fast_window: 5-30 (26 values)
- slow_window: 40-120 (81 values)
- threshold: 0.01-0.10 (100 values)
- lookback: 10-50 (41 values)

Total combinations: 26 * 81 * 100 * 41 = 8,614,600 combinations
```

| Method | Tests Needed | Time (est) | Quality | Use Case |
|--------|-------------|------------|---------|----------|
| **Grid Search** | 8,614,600 | 30 days | 100% | Impossible |
| **Random Search** | 1,000 | 2 hours | 80% | Large spaces |
| **Bayesian** | 200 | 25 mins | 90% | Expensive tests |
| **Genetic** | 500 | 1 hour | 85% | Multi-modal |

---

## Migration Path

### Step 1: Refactor GridSearchOptimizer
Extract common methods into BaseOptimizer without changing API.

### Step 2: Implement RandomSearchOptimizer
Simplest new optimizer - validates architecture.

### Step 3: Add BayesianOptimizer
Most powerful - requires scikit-optimize dependency.

### Step 4: Add GeneticOptimizer
Most complex - validates full flexibility.

### Step 5: GUI Integration
Add method selector to optimization dialog.

---

## Dependencies

**No New Dependencies** for Random Search & Genetic
- Uses existing libraries (numpy, pandas)

**Optional Dependency** for Bayesian:
- `scikit-optimize` (for Gaussian process)
- Falls back gracefully if not installed

---

## Testing Strategy

Each new optimizer needs:

1. **Unit Tests** (~5 tests each)
   - Basic optimization works
   - Finds correct optimum
   - Respects iteration limits
   - Handles invalid parameters
   - Caching integration works

2. **Integration Tests** (~3 tests)
   - Works with BacktestEngine
   - CSV export works
   - Progress tracking works

3. **Performance Tests** (~2 benchmarks)
   - Speedup vs grid search
   - Convergence quality

**Total New Tests**: ~40 tests (10 per optimizer)

---

## Estimated Effort

| Task | Lines of Code | Estimated Time |
|------|---------------|----------------|
| BaseOptimizer refactor | 100 | 2 hours |
| RandomSearchOptimizer | 150 | 3 hours |
| BayesianOptimizer | 200 | 4 hours |
| GeneticOptimizer | 250 | 5 hours |
| GUI integration | 100 | 2 hours |
| Tests | 600 | 8 hours |
| Documentation | - | 3 hours |
| **Total** | **~1,400 lines** | **~27 hours** |

---

## Decision: Grid Search vs Advanced Methods

### When to Use Grid Search (Phases 1-3)
✅ Small parameter spaces (< 100 combinations)
✅ Need guaranteed optimum
✅ Parameters are discrete/categorical
✅ Cheap objective function (fast backtests)

### When to Use Random Search
✅ Large continuous spaces
✅ Time-constrained optimization
✅ Quick exploration
✅ Good enough > perfect

### When to Use Bayesian Optimization
✅ Expensive objective function (slow backtests)
✅ Limited iteration budget
✅ Want near-optimal quickly
✅ Smooth parameter response

### When to Use Genetic Algorithm
✅ Multi-modal objective (multiple peaks)
✅ Complex parameter interactions
✅ Need diverse solution set
✅ Discrete + continuous mix

---

## Recommendation

**Start with Random Search** (Phase 4a):
- Simplest to implement (~3-4 hours)
- Immediately useful for large grids
- Validates architecture
- No new dependencies

**Then add Bayesian** (Phase 4b):
- Most powerful
- Dramatic speedup for expensive objectives
- Requires scikit-optimize

**Genetic is optional** (Phase 4c):
- Most complex
- Niche use cases
- Can defer

---

## Example: Before vs After Phase 4

### Before (Grid Search Only)

```python
# User has large parameter space
param_grid = {
    'fast_window': list(range(5, 31)),     # 26 values
    'slow_window': list(range(40, 121)),   # 81 values
    'threshold': [i/1000 for i in range(10, 101)]  # 91 values
}

# Total combinations: 26 * 81 * 91 = 191,386 tests
# Estimated time: 53 hours (at 1 second per test)
# Solution: Give up or reduce grid drastically
```

### After Phase 4

```python
# Option 1: Random Search (fast exploration)
random_optimizer.optimize(
    param_ranges={
        'fast_window': (5, 30),
        'slow_window': (40, 120),
        'threshold': (0.01, 0.10)
    },
    n_iterations=1000  # Test 1000 random combos
)
# Time: 17 minutes
# Quality: Likely 80-90% as good as exhaustive


# Option 2: Bayesian (smart exploration)
bayes_optimizer.optimize(
    param_space=[
        Integer(5, 30, name='fast_window'),
        Integer(40, 120, name='slow_window'),
        Real(0.01, 0.10, name='threshold')
    ],
    n_iterations=200  # Intelligently chosen
)
# Time: 3.3 minutes
# Quality: Likely 90-95% as good as exhaustive
```

---

## Conclusion

Phase 4 would:
- ✅ **Integrate seamlessly** - Reuses 70% of existing code
- ✅ **Maintain compatibility** - Old code still works
- ✅ **Add flexibility** - Multiple optimization strategies
- ✅ **Solve real problems** - Makes large spaces tractable
- ✅ **Clean architecture** - Strategy pattern with shared base

**Would you like to proceed with Phase 4?**
