# Phase 4c: Genetic Algorithm Optimization - Design Plan

**Status**: Design Document (Not Yet Implemented)
**Prerequisites**: Phase 4a (Random Search) complete
**Estimated Effort**: 8-10 hours
**Date**: November 8, 2025

---

## Overview

Genetic Algorithms (GA) use evolutionary principles (selection, crossover, mutation) to evolve a population of parameter sets toward optimal solutions. Particularly effective for complex,multi-modal objective functions.

**Key Advantage**: Finds diverse solutions, good for multi-modal landscapes
**Best For**: Complex optimization landscapes with multiple local optima

---

## How Genetic Algorithms Work

### Evolutionary Process

```
1. Initialize Population
   └─ Generate N random parameter sets (individuals)

2. For each generation:
   a. Evaluate Fitness
      └─ Run backtest for each individual
      └─ Fitness = performance metric

   b. Selection
      └─ Select parents based on fitness (tournament selection)
      └─ Better individuals more likely to reproduce

   c. Crossover (Reproduction)
      └─ Combine parameters from two parents
      └─ Create offspring with mixed traits

   d. Mutation
      └─ Randomly modify some parameters
      └─ Maintain diversity, explore new areas

   e. Elitism
      └─ Keep best N individuals unchanged
      └─ Prevent losing good solutions

3. Return best individual found
```

### Genetic Operators

**Selection** (Tournament):
```
def tournament_selection(population, fitness_scores, tournament_size=3):
    """Select parent via tournament."""
    tournament = random.sample(zip(population, fitness_scores), tournament_size)
    winner = max(tournament, key=lambda x: x[1])
    return winner[0]
```

**Crossover** (Single-point):
```
def crossover(parent1, parent2, crossover_rate=0.7):
    """Combine two parents to create offspring."""
    if random.random() < crossover_rate:
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    return parent1, parent2
```

**Mutation** (Gaussian):
```
def mutate(individual, mutation_rate=0.1, mutation_scale=0.1):
    """Randomly modify individual parameters."""
    mutated = []
    for gene in individual:
        if random.random() < mutation_rate:
            # Add gaussian noise
            gene = gene * (1 + random.gauss(0, mutation_scale))
        mutated.append(gene)
    return mutated
```

---

## Implementation Design

### Class Structure

```python
# src/backtesting/optimization/genetic_optimizer.py

from backtesting.optimization.base_optimizer import BaseOptimizer
import numpy as np

class GeneticOptimizer(BaseOptimizer):
    """
    Genetic algorithm optimization for strategy parameters.

    Evolves a population of parameter sets over multiple generations
    using selection, crossover, and mutation operators.

    Features:
    - Tournament selection
    - Single-point/uniform crossover
    - Gaussian mutation
    - Elitism (preserve best individuals)
    - Diversity tracking
    - Convergence detection

    Example:
        optimizer = GeneticOptimizer(engine)
        result = optimizer.optimize(
            strategy_class=MovingAverageCrossover,
            param_ranges={
                'fast_window': (5, 30),
                'slow_window': (40, 120),
                'threshold': (0.01, 0.10, 'log')
            },
            symbols='AAPL',
            start_date='2023-01-01',
            end_date='2024-01-01',
            population_size=50,
            n_generations=20,
            mutation_rate=0.1,
            crossover_rate=0.7,
            elitism=0.2  # Keep top 20%
        )
    """

    def optimize(
        self,
        strategy_class: type,
        param_ranges: Dict[str, Union[Tuple, List]],
        symbols: Union[str, List[str]],
        start_date: str,
        end_date: str,
        metric: str = 'sharpe_ratio',
        population_size: int = 50,
        n_generations: int = 20,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elitism: float = 0.2,
        tournament_size: int = 3,
        max_workers: Optional[int] = None,
        use_cache: bool = True,
        cache_config: Optional[Any] = None,
        export_results: bool = True,
        output_dir: Optional[Any] = None,
        random_seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Optimize using genetic algorithm.

        Args:
            param_ranges: Same format as RandomSearchOptimizer
            population_size: Number of individuals per generation
            n_generations: Number of evolutionary generations
            mutation_rate: Probability of mutation per gene
            crossover_rate: Probability of crossover
            elitism: Fraction of top performers to keep unchanged
            tournament_size: Size of tournament for selection
            Other args same as RandomSearchOptimizer

        Returns:
            Same as RandomSearchOptimizer, plus:
            {
                'evolution_history': List of best fitness per generation,
                'diversity_history': List of population diversity per generation,
                'final_population': Top N individuals from final generation
            }
        """
```

### Key Methods

```python
def _initialize_population(self, param_ranges, population_size):
    """Generate initial random population."""
    population = []
    for _ in range(population_size):
        individual = self._random_individual(param_ranges)
        population.append(individual)
    return population

def _evaluate_population(self, population, strategy_class, data,
                        symbols, metric, max_workers, use_cache):
    """Evaluate fitness for all individuals (parallel!)."""
    # Convert individuals to parameter dicts
    # Use parallel execution (reuses Phase 1 infrastructure)
    # Returns list of fitness scores

def _tournament_selection(self, population, fitness_scores, tournament_size):
    """Select parent using tournament selection."""
    # Randomly sample tournament_size individuals
    # Return the one with best fitness

def _crossover(self, parent1, parent2, crossover_rate):
    """Create offspring via crossover."""
    if random.random() < crossover_rate:
        # Single-point crossover
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    return parent1.copy(), parent2.copy()

def _mutate(self, individual, param_ranges, mutation_rate):
    """Mutate individual genes."""
    mutated = []
    for i, gene in enumerate(individual):
        if random.random() < mutation_rate:
            # Gaussian mutation within bounds
            param_name = list(param_ranges.keys())[i]
            param_spec = param_ranges[param_name]
            gene = self._mutate_gene(gene, param_spec)
        mutated.append(gene)
    return mutated

def _apply_elitism(self, population, offspring, fitness_scores, elitism_rate):
    """Keep top performers from previous generation."""
    n_elite = int(len(population) * elitism_rate)

    # Get indices of top performers
    elite_indices = np.argsort(fitness_scores)[-n_elite:]
    elite = [population[i] for i in elite_indices]

    # Replace worst offspring with elite
    new_population = offspring[:-n_elite] + elite
    return new_population

def _calculate_diversity(self, population):
    """Calculate population diversity (genetic variance)."""
    # Convert population to numpy array
    pop_array = np.array(population)
    # Calculate variance across population for each parameter
    diversity = np.mean(np.var(pop_array, axis=0))
    return diversity
```

---

## Genetic Operators in Detail

### 1. Selection

**Tournament Selection** (Default):
```
Pros:
- Simple to implement
- Preserves diversity
- Adjustable selection pressure (via tournament_size)

Algorithm:
1. Randomly select K individuals
2. Return the fittest one
3. Higher K = more selection pressure
```

**Alternative: Roulette Wheel**:
```
Pros:
- Fitness-proportional selection
- Natural for maximization

Cons:
- Can lose diversity quickly
- Sensitive to fitness scaling
```

### 2. Crossover

**Single-Point Crossover** (Default):
```python
Parent1: [10, 50, 0.02]
Parent2: [20, 80, 0.08]
         ↓  ↓  ↓
Point:       ↑
         ↓  ↓  ↓
Child1:  [10, 80, 0.08]
Child2:  [20, 50, 0.02]
```

**Alternative: Uniform Crossover**:
```python
# Each gene has 50% chance from each parent
Parent1: [10, 50, 0.02]
Parent2: [20, 80, 0.08]
         ↓  ↓  ↓
Child:   [10, 80, 0.02]  # Random mix
```

### 3. Mutation

**Gaussian Mutation** (Continuous Parameters):
```python
gene_new = gene_old + N(0, σ)

Where:
- N(0, σ) = Gaussian noise
- σ = mutation_scale * parameter_range
```

**Uniform Mutation** (Discrete Parameters):
```python
if random() < mutation_rate:
    gene = random_choice(valid_values)
```

---

## Performance Characteristics

### Computational Cost

```
Total tests = population_size × n_generations

Example:
- Population: 50
- Generations: 20
- Total tests: 1,000

Comparable to Random Search with n=1,000
But explores space differently (evolutionary vs random)
```

### Convergence Behavior

```
Generation | Best Fitness | Diversity | Status
------------------------------------------------
1          | 1.2          | High      | Initial exploration
5          | 1.8          | High      | Diverse search
10         | 2.1          | Medium    | Converging
15         | 2.3          | Low       | Fine-tuning
20         | 2.35         | Very Low  | Converged

Warning: Low diversity = risk of premature convergence
Solution: Higher mutation rate or restart with new population
```

---

## Multi-Modal Optimization

### Why Genetic Algorithms Excel

```
Fitness Landscape:

Sharpe
  ↑
  │    ╱╲         ╱╲
  │   ╱  ╲       ╱  ╲
  │  ╱    ╲  ╱╲ ╱    ╲
  │ ╱      ╲╱  ╲      ╲
  └──────────────────────→ Parameters

Local Optimum (λ₁)  Global Optimum (λ₂)

Genetic Algorithm:
- Maintains diverse population
- Explores multiple peaks simultaneously
- Less likely to get stuck in local optimum

vs

Bayesian/Random:
- Single search trajectory
- May get trapped in local optimum
- Harder to escape poor regions
```

---

## GUI Integration

### Dialog Changes

```
Optimization Method: [Genetic Algorithm ▼]

┌─────────────────────────────────────────────┐
│ Genetic Algorithm Settings                   │
├─────────────────────────────────────────────┤
│ Define parameter ranges (min/max):          │
│                                              │
│ Fast Window: [5] to [30]                    │
│ Slow Window: [40] to [120]                  │
│                                              │
│ Population size: [50      ]                 │
│ Number of generations: [20      ]           │
│                                              │
│ Mutation rate: [0.10    ] (10%)            │
│ Crossover rate: [0.70    ] (70%)           │
│ Elitism: [0.20    ] (keep top 20%)         │
│                                              │
│ Total tests: ~1,000 (50 × 20)               │
│                                              │
│ [Run Optimization]                           │
└─────────────────────────────────────────────┘
```

---

## Testing Strategy

### Unit Tests (10 tests)

```python
def test_initialize_population():
    """Test random population initialization."""

def test_tournament_selection():
    """Test tournament selection favors fitter individuals."""

def test_crossover():
    """Test crossover creates valid offspring."""

def test_mutation():
    """Test mutation maintains parameter bounds."""

def test_elitism():
    """Test elite individuals are preserved."""

def test_evolution():
    """Test population improves over generations."""

def test_convergence():
    """Test algorithm converges to good solution."""

def test_diversity_tracking():
    """Test diversity calculation."""

def test_caching_integration():
    """Test caching works with genetic algorithm."""

def test_export_evolution_history():
    """Test evolution history export."""
```

---

## Implementation Checklist

### Core Implementation (~5 hours)
- [ ] Create `genetic_optimizer.py`
- [ ] Implement `optimize()` method
- [ ] Implement population initialization
- [ ] Implement tournament selection
- [ ] Implement crossover (single-point + uniform)
- [ ] Implement mutation (Gaussian + uniform)
- [ ] Implement elitism
- [ ] Integrate with caching
- [ ] Track evolution history
- [ ] Track diversity history

### Testing (~2 hours)
- [ ] Create `test_genetic_optimizer.py`
- [ ] Write 10 unit tests
- [ ] Test with real strategies
- [ ] Verify convergence quality
- [ ] Test multi-modal landscapes

### Documentation (~1 hour)
- [ ] API documentation
- [ ] Usage examples
- [ ] Operator guide (selection, crossover, mutation)
- [ ] Parameter tuning guide
- [ ] Performance comparison

### GUI Integration (~2 hours)
- [ ] Add Genetic to method dropdown
- [ ] Create Genetic settings panel
- [ ] Add sliders for rates
- [ ] Update runner to handle Genetic

**Total Estimated Time**: 8-10 hours

---

## Example Usage

### Command Line

```python
from backtesting.optimization import GeneticOptimizer

optimizer = GeneticOptimizer(engine)

result = optimizer.optimize(
    strategy_class=MovingAverageCrossover,
    param_ranges={
        'fast_window': (5, 30),
        'slow_window': (40, 120),
        'threshold': (0.01, 0.10, 'log')
    },
    symbols='AAPL',
    start_date='2023-01-01',
    end_date='2024-01-01',
    metric='sharpe_ratio',
    population_size=50,
    n_generations=20,
    mutation_rate=0.1,
    crossover_rate=0.7,
    elitism=0.2
)

print(f"Best params: {result['best_params']}")
print(f"Best Sharpe: {result['best_value']:.2f}")
print(f"Evolution: {result['evolution_history']}")
print(f"Final diversity: {result['diversity_history'][-1]:.3f}")
```

### Expected Output

```
===============================================================================
Optimizing MovingAverageCrossover (GENETIC ALGORITHM)
 Parameter ranges: {'fast_window': (5,30), 'slow_window': (40,120), ...}
 Population size: 50
 Generations: 20
 Mutation rate: 10%
 Crossover rate: 70%
 Elitism: 20%
===============================================================================

Generation 1/20:
  Evaluating 50 individuals... Done
  Best fitness: 1.45 | Avg fitness: 0.82 | Diversity: 0.52

Generation 2/20:
  Selection, crossover, mutation... Done
  Evaluating 40 new individuals (10 elite cached)... Done
  Best fitness: 1.67 (↑) | Avg fitness: 1.12 (↑) | Diversity: 0.48

Generation 3/20:
  Selection, crossover, mutation... Done
  Evaluating 40 new individuals... Done
  Best fitness: 1.89 (↑) | Avg fitness: 1.34 (↑) | Diversity: 0.43

...

Generation 18/20:
  Selection, crossover, mutation... Done
  Evaluating 40 new individuals... Done
  Best fitness: 2.34 (=) | Avg fitness: 2.21 (↑) | Diversity: 0.08

Warning: Low diversity detected (0.08 < 0.10)
Consider increasing mutation rate or population size.

Generation 19/20:
  Selection, crossover, mutation... Done
  Evaluating 40 new individuals... Done
  Best fitness: 2.35 (↑) | Avg fitness: 2.24 (↑) | Diversity: 0.06

Generation 20/20:
  Selection, crossover, mutation... Done
  Evaluating 40 new individuals... Done
  Best fitness: 2.35 (=) | Avg fitness: 2.26 (↑) | Diversity: 0.05

===============================================================================
[+] Best parameters: {'fast_window': 17, 'slow_window': 61, 'threshold': 0.041}
[^] Best sharpe_ratio: 2.35
 Evaluated 1,000 individuals over 20 generations
 Time: 28 minutes
 Convergence: Generation 18
 Evolution history exported to: logs/.../evolution_plot.png
===============================================================================
```

---

## Parameter Tuning Guide

### Population Size

```
Small (20-30):
  ✅ Faster
  ❌ Less diversity
  ❌ Risk premature convergence

Medium (50-80):
  ✅ Good balance
  ✅ Recommended default

Large (100+):
  ✅ High diversity
  ✅ Better for complex landscapes
  ❌ Slower
```

### Mutation Rate

```
Low (0.01-0.05):
  ✅ Fine-tuning
  ❌ Slow exploration
  Use: Late generations, simple landscapes

Medium (0.1-0.2):
  ✅ Balanced
  ✅ Recommended default

High (0.3-0.5):
  ✅ High exploration
  ❌ Risk destroying good solutions
  Use: Complex landscapes, early generations
```

### Crossover Rate

```
Low (0.3-0.5):
  ✅ Preserve good individuals
  ❌ Slow mixing

Medium (0.6-0.8):
  ✅ Good mixing
  ✅ Recommended default

High (0.9-1.0):
  ✅ Aggressive mixing
  ❌ May lose good combinations
```

### Elitism

```
None (0):
  ❌ Can lose best solution
  Not recommended

Low (0.1-0.2):
  ✅ Preserve best
  ✅ Recommended default

High (0.3-0.5):
  ❌ Reduces exploration
  ❌ Risk stagnation
```

---

## Advantages vs Other Methods

### vs Grid Search
✅ Handles large spaces
✅ Finds good solutions quickly
❌ No guarantee of optimum

### vs Random Search
✅ More directed search
✅ Good for multi-modal
≈ Similar speed

### vs Bayesian
❌ Needs more iterations
✅ Better for multi-modal
✅ Returns diverse solutions

### When to Use Genetic Algorithm

✅ Multi-modal objective (multiple good solutions)
✅ Want diverse solution set
✅ Complex parameter interactions
✅ Mixed discrete/continuous parameters
❌ Simple convex landscape (use Bayesian)
❌ Very expensive objective (use Bayesian)
❌ Need guaranteed optimum (use Grid)

---

## Conclusion

Genetic Algorithms provide:
- **Robust** optimization for complex landscapes
- **Diverse** solution sets
- **Flexible** operator customization
- **Intuitive** evolutionary metaphor

**Recommendation**: Implement as optional third optimization method for users with complex, multi-modal optimization problems.

**Priority**: Low (after Grid Search, Random Search, and Bayesian)

---

**Design Document by**: Claude (Anthropic AI)
**Date**: November 8, 2025
**Status**: Design Phase - Ready for Implementation
