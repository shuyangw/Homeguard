"""Experiment registry per docs/methodology/backtesting.md Section 9.

Single DuckDB store at output/experiments.duckdb. Every backtest and
optimization run appends. The portfolio-integrator queries.
"""
from src.experiments.registry import (
    DEFAULT_DB_PATH,
    append_run,
    duplicate_spec_run_ids,
    incumbent_return_streams,
    init_db,
    make_trial_callback,
    n_trials_project_wide,
)

__all__ = [
    "DEFAULT_DB_PATH",
    "append_run",
    "duplicate_spec_run_ids",
    "incumbent_return_streams",
    "init_db",
    "make_trial_callback",
    "n_trials_project_wide",
]
