"""
Genetic Algorithm optimization for parameter tuning.

Uses evolutionary principles (selection, crossover, mutation) to evolve
populations of parameter sets toward optimal solutions. Particularly effective
for multi-modal optimization landscapes.
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, List, Any, Union, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

from src.utils import logger
from src.backtesting.optimization.base_optimizer import BaseOptimizer


@dataclass
class Individual:
    """Represents a single individual in the population."""
    params: Dict[str, Any]
    fitness: Optional[float] = None
    stats: Optional[Dict] = None
    error: Optional[str] = None


class GeneticOptimizer(BaseOptimizer):
    """
    Genetic Algorithm optimizer for strategy parameters.

    Uses evolutionary principles to evolve populations of parameter sets:
    - Selection: Tournament selection chooses best individuals
    - Crossover: Combines parent parameters to create offspring
    - Mutation: Random parameter changes for diversity
    - Elitism: Best individuals always survive

    Features:
    - Multi-modal optimization (finds multiple local optima)
    - Population diversity tracking
    - Convergence detection
    - Diversity plots and analysis
    - Full cache integration

    Example:
        optimizer = GeneticOptimizer(engine)
        result = optimizer.optimize(
            strategy_class=MovingAverageCrossover,
            param_ranges={
                'fast_window': (5, 30),
                'slow_window': (40, 120)
            },
            symbols='AAPL',
            start_date='2023-01-01',
            end_date='2024-01-01',
            population_size=50,
            n_generations=20,
            mutation_rate=0.1,
            crossover_rate=0.7,
            elitism_rate=0.2
        )

    Advantages:
    - Handles discrete and continuous parameters
    - Explores multiple regions simultaneously
    - Maintains population diversity
    - Less prone to local optima than gradient methods
    """

    def __init__(self, engine: 'BacktestEngine'):
        """Initialize Genetic optimizer."""
        super().__init__(engine)
        self._diversity_history = []
        self._best_fitness_history = []
        self._avg_fitness_history = []

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
        elitism_rate: float = 0.2,
        tournament_size: int = 3,
        max_workers: Optional[int] = None,
        price_type: str = 'close',
        use_cache: bool = True,
        cache_config: Optional[Any] = None,
        export_results: bool = True,
        output_dir: Optional[Any] = None,
        random_seed: Optional[int] = None,
        enable_plots: bool = True,
        convergence_patience: int = 5
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters using Genetic Algorithm.

        Args:
            strategy_class: Strategy class to optimize
            param_ranges: Parameter ranges in format:
                {'param_name': (min, max)} for numeric
                {'param_name': [val1, val2, ...]} for discrete
            symbols: Symbol or list of symbols
            start_date: Start date for backtest period
            end_date: End date for backtest period
            metric: Metric to optimize ('sharpe_ratio', 'total_return', 'max_drawdown')
            population_size: Number of individuals in population (typically 30-100)
            n_generations: Number of generations to evolve (typically 10-50)
            mutation_rate: Probability of mutation (0.01-0.3, default: 0.1)
            crossover_rate: Probability of crossover (0.5-1.0, default: 0.7)
            elitism_rate: Fraction of best to preserve (0.1-0.3, default: 0.2)
            tournament_size: Number of individuals in tournament (2-5, default: 3)
            max_workers: Maximum parallel workers
            price_type: Price column to use
            use_cache: If True, use result cache
            cache_config: Optional cache configuration
            export_results: If True, export results to CSV
            output_dir: Optional custom output directory
            random_seed: Random seed for reproducibility
            enable_plots: If True, generate diversity plots
            convergence_patience: Generations without improvement before stopping

        Returns:
            Dictionary with optimization results
        """
        import os

        if isinstance(symbols, str):
            symbols = [symbols]

        # Validate metric
        valid_metrics = ['sharpe_ratio', 'total_return', 'max_drawdown']
        if metric not in valid_metrics:
            raise ValueError(f"Unknown metric: {metric}")

        # Validate parameters
        if not (0 <= mutation_rate <= 1):
            raise ValueError("mutation_rate must be between 0 and 1")
        if not (0 <= crossover_rate <= 1):
            raise ValueError("crossover_rate must be between 0 and 1")
        if not (0 <= elitism_rate <= 0.5):
            raise ValueError("elitism_rate must be between 0 and 0.5")

        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)

        # Determine number of workers
        if max_workers is None:
            max_workers = min(4, os.cpu_count() or 1)

        # Load data
        logger.info("Loading market data for optimization period...")
        data = self.engine.data_loader.load_symbols(symbols, start_date, end_date)

        # Log optimization header
        logger.blank()
        logger.separator()
        logger.header(f"Optimizing {strategy_class.__name__} (GENETIC ALGORITHM)")
        logger.info(f"Parameter ranges: {param_ranges}")
        logger.info(f"Population size: {population_size}")
        logger.info(f"Generations: {n_generations}")
        logger.info(f"Mutation rate: {mutation_rate}")
        logger.info(f"Crossover rate: {crossover_rate}")
        logger.info(f"Elitism rate: {elitism_rate}")
        logger.separator()
        logger.blank()

        # Initialize cache
        cache = None
        cache_hits = 0
        cache_misses = 0
        if use_cache:
            from backtesting.optimization.result_cache import ResultCache
            cache = ResultCache(cache_config)
            logger.info("Result cache enabled")

        # Initialize tracking
        best_individual = None
        all_individuals = []
        self._diversity_history = []
        self._best_fitness_history = []
        self._avg_fitness_history = []

        # Progress tracking
        start_time = time.time()
        total_evaluations = 0
        no_improvement_count = 0

        # Initialize population
        logger.info(f"Initializing population of {population_size} individuals...")
        population = self._initialize_population(param_ranges, population_size)

        # Evaluate initial population
        population, evals = self._evaluate_population(
            population, strategy_class, data, symbols, price_type,
            metric, cache, start_date, end_date
        )
        total_evaluations += len(population)  # Count all evaluations (cache + new)
        gen_cache_hits = len(population) - evals
        cache_hits += gen_cache_hits
        cache_misses += evals

        # Track best
        best_individual = max(
            [ind for ind in population if ind.fitness is not None],
            key=lambda ind: -ind.fitness if metric == 'max_drawdown' else ind.fitness
        )
        all_individuals.extend(population)

        # Calculate initial diversity
        diversity = self._calculate_diversity(population, param_ranges)
        self._diversity_history.append(diversity)
        self._best_fitness_history.append(best_individual.fitness)
        self._avg_fitness_history.append(np.mean([ind.fitness for ind in population if ind.fitness is not None]))

        logger.success(f"Initial population evaluated. Best {metric}: {best_individual.fitness:.4f}")
        logger.info(f"Population diversity: {diversity:.4f}")
        logger.blank()

        # Evolution loop
        for generation in range(n_generations):
            gen_start = time.time()
            logger.info(f"Generation {generation + 1}/{n_generations}")

            previous_best = best_individual.fitness

            # Selection
            parents = self._tournament_selection(
                population, int(population_size * crossover_rate), tournament_size, metric
            )

            # Crossover
            offspring = []
            for i in range(0, len(parents) - 1, 2):
                if np.random.random() < crossover_rate:
                    child1, child2 = self._crossover(parents[i], parents[i + 1], param_ranges)
                    offspring.extend([child1, child2])
                else:
                    offspring.extend([parents[i], parents[i + 1]])

            # Mutation
            for individual in offspring:
                if np.random.random() < mutation_rate:
                    self._mutate(individual, param_ranges, mutation_rate)

            # Evaluate offspring
            offspring, evals = self._evaluate_population(
                offspring, strategy_class, data, symbols, price_type,
                metric, cache, start_date, end_date
            )
            total_evaluations += len(offspring)  # Count all evaluations (cache + new)
            gen_cache_hits = len(offspring) - evals
            cache_hits += gen_cache_hits
            cache_misses += evals

            # Elitism: keep best individuals
            n_elite = int(population_size * elitism_rate)
            elite = sorted(
                [ind for ind in population if ind.fitness is not None],
                key=lambda ind: -ind.fitness if metric == 'max_drawdown' else ind.fitness,
                reverse=False if metric == 'max_drawdown' else True
            )[:n_elite]

            # Combine elite + offspring, select best
            combined = elite + offspring
            combined = [ind for ind in combined if ind.fitness is not None]
            combined = sorted(
                combined,
                key=lambda ind: -ind.fitness if metric == 'max_drawdown' else ind.fitness,
                reverse=False if metric == 'max_drawdown' else True
            )[:population_size]

            population = combined
            all_individuals.extend(offspring)

            # Update best
            current_best = population[0]
            if self._is_better(current_best.fitness, best_individual.fitness, metric):
                best_individual = current_best
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Track diversity and fitness
            diversity = self._calculate_diversity(population, param_ranges)
            avg_fitness = np.mean([ind.fitness for ind in population if ind.fitness is not None])

            self._diversity_history.append(diversity)
            self._best_fitness_history.append(best_individual.fitness)
            self._avg_fitness_history.append(avg_fitness)

            # Log progress
            gen_time = time.time() - gen_start
            improvement = best_individual.fitness - previous_best if metric != 'max_drawdown' else previous_best - best_individual.fitness
            improvement_marker = f" (+{improvement:.4f})" if improvement > 0 else ""

            logger.metric(
                f"  Best: {best_individual.fitness:.4f}{improvement_marker} | "
                f"Avg: {avg_fitness:.4f} | Diversity: {diversity:.4f} | "
                f"Time: {gen_time:.1f}s"
            )

            # Check convergence
            if no_improvement_count >= convergence_patience:
                logger.warning(f"Early stopping: No improvement for {convergence_patience} generations")
                break

        # Calculate final statistics
        total_time = time.time() - start_time
        total_mins = total_time / 60
        actual_generations = generation + 1

        # Prepare convergence data
        convergence_data = {
            'generations': list(range(actual_generations + 1)),
            'best_fitness': self._best_fitness_history,
            'avg_fitness': self._avg_fitness_history,
            'diversity': self._diversity_history,
            'early_stopped': no_improvement_count >= convergence_patience
        }

        # Log results
        logger.blank()
        logger.separator()
        logger.success(f"Best parameters: {best_individual.params}")
        logger.profit(f"Best {metric}: {best_individual.fitness:.4f}")
        logger.info(f"Generations: {actual_generations}/{n_generations}")
        logger.info(f"Total evaluations: {total_evaluations}")
        if cache:
            logger.info(f"Cache hits: {cache_hits}")
            logger.info(f"Cache misses: {cache_misses}")
        logger.info(f"Total time: {total_mins:.2f} minutes ({total_time:.1f}s)")
        logger.separator()
        logger.blank()

        # Re-run best to get portfolio
        best_strategy = strategy_class(**best_individual.params)
        if len(symbols) == 1:
            best_portfolio = self.engine._run_single_symbol(
                best_strategy, data, symbols[0], price_type
            )
        else:
            best_portfolio = self.engine._run_multiple_symbols(
                best_strategy, data, symbols, price_type
            )

        # Export results
        csv_path = None
        if export_results:
            csv_path = self._export_results_to_csv(
                all_results=[
                    {'params': ind.params, 'value': ind.fitness, 'stats': ind.stats, 'error': ind.error}
                    for ind in all_individuals
                ],
                best_params=best_individual.params,
                best_value=best_individual.fitness,
                metric=metric,
                strategy_name=strategy_class.__name__,
                symbols=symbols,
                output_dir=output_dir,
                method_name="GeneticAlgorithm",
                convergence_data=convergence_data
            )

        # Export plots
        plots_path = None
        if enable_plots:
            plots_path = self._export_diversity_plots(
                convergence_data=convergence_data,
                metric=metric,
                strategy_name=strategy_class.__name__,
                symbols=symbols,
                output_dir=output_dir
            )

        return {
            'best_params': best_individual.params,
            'best_value': best_individual.fitness,
            'best_portfolio': best_portfolio,
            'metric': metric,
            'all_results': [
                {'params': ind.params, 'value': ind.fitness, 'stats': ind.stats, 'error': ind.error}
                for ind in all_individuals
            ],
            'convergence_data': convergence_data,
            'n_generations': actual_generations,
            'total_evaluations': total_evaluations,
            'early_stopped': convergence_data['early_stopped'],
            'total_time': total_time,
            'avg_time_per_evaluation': total_time / total_evaluations if total_evaluations > 0 else 0,
            'cache_hits': cache_hits if cache else 0,
            'cache_misses': cache_misses if cache else 0,
            'method': 'genetic_algorithm',
            'csv_path': csv_path,
            'plots_path': plots_path
        }

    def _initialize_population(
        self,
        param_ranges: Dict[str, Union[Tuple, List]],
        population_size: int
    ) -> List[Individual]:
        """Initialize random population."""
        population = []

        for _ in range(population_size):
            params = {}
            for param_name, param_spec in param_ranges.items():
                if isinstance(param_spec, list):
                    # Discrete choice - convert to native Python type
                    value = np.random.choice(param_spec)
                    params[param_name] = int(value) if isinstance(value, (np.integer, np.int64, np.int32)) else value
                elif isinstance(param_spec, tuple):
                    # Numeric range
                    min_val, max_val = param_spec[0], param_spec[1]
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        # Integer parameter - ensure native Python int
                        params[param_name] = int(np.random.randint(min_val, max_val + 1))
                    else:
                        # Float parameter - ensure native Python float
                        params[param_name] = float(np.random.uniform(min_val, max_val))

            population.append(Individual(params=params))

        return population

    def _evaluate_population(
        self,
        population: List[Individual],
        strategy_class: type,
        data: pd.DataFrame,
        symbols: List[str],
        price_type: str,
        metric: str,
        cache: Optional[Any],
        start_date: str,
        end_date: str
    ) -> Tuple[List[Individual], int]:
        """Evaluate fitness of all individuals in population."""
        evaluations = 0

        for individual in population:
            # Check cache
            if cache:
                cache_key = cache.generate_cache_key(
                    strategy_class=strategy_class,
                    params=individual.params,
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    price_type=price_type,
                    engine_config=self._engine_config,
                    metric=metric
                )
                cached_result = cache.get(cache_key)

                if cached_result:
                    individual.fitness = cached_result['value']
                    individual.stats = cached_result['stats']
                    individual.error = cached_result['error']
                    continue

            # Evaluate
            evaluations += 1
            try:
                strategy = strategy_class(**individual.params)

                if len(symbols) == 1:
                    portfolio = self.engine._run_single_symbol(
                        strategy, data, symbols[0], price_type
                    )
                else:
                    portfolio = self.engine._run_multiple_symbols(
                        strategy, data, symbols, price_type
                    )

                stats = portfolio.stats()
                individual.fitness = self._extract_metric_value(stats, metric)
                individual.stats = stats
                individual.error = None

                # Store in cache
                if cache:
                    cache.put(
                        cache_key=cache_key,
                        params=individual.params,
                        metric_value=individual.fitness,
                        stats=stats,
                        error=None
                    )

            except Exception as e:
                individual.fitness = float('-inf') if metric != 'max_drawdown' else float('inf')
                individual.stats = None
                individual.error = str(e)

        return population, evaluations

    def _tournament_selection(
        self,
        population: List[Individual],
        n_select: int,
        tournament_size: int,
        metric: str
    ) -> List[Individual]:
        """Select parents using tournament selection."""
        selected = []

        # Filter valid individuals once for efficiency
        valid_individuals = [ind for ind in population if ind.fitness is not None]

        for _ in range(n_select):
            # Random tournament
            tournament = np.random.choice(
                valid_individuals,
                size=min(tournament_size, len(valid_individuals)),
                replace=False
            ).tolist()

            # Select best from tournament
            best = max(
                tournament,
                key=lambda ind: -ind.fitness if metric == 'max_drawdown' else ind.fitness
            )

            selected.append(best)

        return selected

    def _crossover(
        self,
        parent1: Individual,
        parent2: Individual,
        param_ranges: Dict[str, Union[Tuple, List]]
    ) -> Tuple[Individual, Individual]:
        """Uniform crossover: randomly mix parent parameters."""
        child1_params = {}
        child2_params = {}

        for param_name in param_ranges.keys():
            if np.random.random() < 0.5:
                child1_params[param_name] = parent1.params[param_name]
                child2_params[param_name] = parent2.params[param_name]
            else:
                child1_params[param_name] = parent2.params[param_name]
                child2_params[param_name] = parent1.params[param_name]

        return Individual(params=child1_params), Individual(params=child2_params)

    def _mutate(
        self,
        individual: Individual,
        param_ranges: Dict[str, Union[Tuple, List]],
        mutation_rate: float
    ) -> None:
        """Mutate individual parameters with given probability."""
        for param_name, param_spec in param_ranges.items():
            if np.random.random() < mutation_rate:
                if isinstance(param_spec, list):
                    # Discrete: choose random value (convert NumPy types to native Python)
                    value = np.random.choice(param_spec)
                    individual.params[param_name] = int(value) if isinstance(value, (np.integer, np.int64, np.int32)) else value
                elif isinstance(param_spec, tuple):
                    # Numeric: Gaussian mutation with bounds
                    min_val, max_val = param_spec[0], param_spec[1]
                    current = individual.params[param_name]

                    if isinstance(min_val, int) and isinstance(max_val, int):
                        # Integer mutation
                        mutation = np.random.randint(-5, 6)
                        new_value = int(np.clip(current + mutation, min_val, max_val))
                    else:
                        # Float mutation (convert to native Python float)
                        range_size = max_val - min_val
                        mutation = np.random.normal(0, range_size * 0.1)
                        new_value = float(np.clip(current + mutation, min_val, max_val))

                    individual.params[param_name] = new_value

    def _calculate_diversity(
        self,
        population: List[Individual],
        param_ranges: Dict[str, Union[Tuple, List]]
    ) -> float:
        """Calculate population diversity (average pairwise distance)."""
        if len(population) < 2:
            return 0.0

        # Normalize parameters to [0, 1]
        normalized_pop = []
        for individual in population:
            normalized = []
            for param_name, param_spec in param_ranges.items():
                value = individual.params[param_name]

                if isinstance(param_spec, list):
                    # Discrete: binary distance
                    normalized.append(param_spec.index(value) / len(param_spec))
                elif isinstance(param_spec, tuple):
                    # Numeric: normalize to [0, 1]
                    min_val, max_val = param_spec[0], param_spec[1]
                    normalized.append((value - min_val) / (max_val - min_val) if max_val > min_val else 0)

            normalized_pop.append(normalized)

        # Calculate average pairwise Euclidean distance
        distances = []
        for i in range(len(normalized_pop)):
            for j in range(i + 1, len(normalized_pop)):
                dist = np.linalg.norm(np.array(normalized_pop[i]) - np.array(normalized_pop[j]))
                distances.append(dist)

        return np.mean(distances) if distances else 0.0

    def _export_diversity_plots(
        self,
        convergence_data: Dict,
        metric: str,
        strategy_name: str,
        symbols: List[str],
        output_dir: Optional[Any] = None
    ) -> Optional[Path]:
        """Export diversity and convergence plots."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available - skipping plot generation")
            return None

        from datetime import datetime
        from src.config import get_backtest_results_dir

        if output_dir is None:
            base_dir = get_backtest_results_dir()
        else:
            base_dir = Path(output_dir)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        symbols_str = '_'.join(symbols)
        dir_name = f"{timestamp}_{strategy_name}_{symbols_str}_GeneticAlgorithm"
        output_path = base_dir / dir_name

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        generations = convergence_data['generations']
        best_fitness = convergence_data['best_fitness']
        avg_fitness = convergence_data['avg_fitness']
        diversity = convergence_data['diversity']

        # Plot 1: Fitness evolution
        ax1.plot(generations, best_fitness, 'b-', linewidth=2, label='Best Fitness')
        ax1.plot(generations, avg_fitness, 'g--', linewidth=1.5, label='Avg Fitness')
        ax1.set_xlabel('Generation', fontsize=12)
        ax1.set_ylabel(f'{metric}', fontsize=12)
        ax1.set_title(f'Genetic Algorithm Convergence - {strategy_name}', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: Diversity evolution
        ax2.plot(generations, diversity, 'r-', linewidth=2)
        ax2.set_xlabel('Generation', fontsize=12)
        ax2.set_ylabel('Population Diversity', fontsize=12)
        ax2.set_title('Population Diversity Over Time', fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        plot_path = output_path / 'evolution_plots.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Exported evolution plots to: {plot_path}")

        return plot_path

    def _export_results_to_csv(
        self,
        all_results: List[Dict[str, Any]],
        best_params: Dict[str, Any],
        best_value: float,
        metric: str,
        strategy_name: str,
        symbols: List[str],
        output_dir: Optional[Any] = None,
        method_name: str = "GeneticAlgorithm",
        convergence_data: Optional[Dict] = None
    ) -> Path:
        """Export optimization results to CSV."""
        from datetime import datetime
        from src.config import get_backtest_results_dir

        if output_dir is None:
            base_dir = get_backtest_results_dir()
        else:
            base_dir = Path(output_dir)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        symbols_str = '_'.join(symbols)
        dir_name = f"{timestamp}_{strategy_name}_{symbols_str}_{method_name}"
        output_path = base_dir / dir_name
        output_path.mkdir(parents=True, exist_ok=True)

        # Prepare data
        rows = []
        for i, result in enumerate(all_results):
            row = {'generation': i // 50, 'method': method_name}  # Approximate generation
            for param_name, param_value in result['params'].items():
                row[f'param_{param_name}'] = param_value
            row[metric] = result['value']
            row['error'] = result['error'] if result['error'] else ''
            rows.append(row)

        df = pd.DataFrame(rows)
        ascending = (metric == 'max_drawdown')
        df = df.sort_values(by=metric, ascending=ascending)
        df['distance_from_best'] = abs(df[metric] - best_value)

        csv_path = output_path / 'optimization_results.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"Exported optimization results to: {csv_path}")

        # Export summary
        summary_path = output_path / 'optimization_summary.txt'
        with open(summary_path, 'w') as f:
            f.write(f"Genetic Algorithm Optimization Summary\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Method: {method_name}\n")
            f.write(f"Strategy: {strategy_name}\n")
            f.write(f"Symbols: {', '.join(symbols)}\n")
            f.write(f"Metric: {metric}\n")
            f.write(f"Total evaluations: {len(all_results)}\n")
            if convergence_data:
                f.write(f"Generations: {len(convergence_data['generations']) - 1}\n")
                if convergence_data.get('early_stopped'):
                    f.write(f"Early stopped: Yes\n")
            f.write(f"\nBest Parameters:\n")
            for param_name, param_value in best_params.items():
                f.write(f"  {param_name}: {param_value}\n")
            f.write(f"\nBest {metric}: {best_value:.4f}\n")

        logger.info(f"Exported optimization summary to: {summary_path}")

        return csv_path
