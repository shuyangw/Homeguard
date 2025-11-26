"""
Default configuration values for backtesting.

These defaults are used when config values are not specified.
They represent sensible, moderate-risk settings suitable for most backtests.
"""

from typing import Dict, Any


DEFAULT_CONFIG: Dict[str, Any] = {
    # Execution mode
    "mode": "single",

    # Backtest core settings
    "backtest": {
        "initial_capital": 100000.0,
        "fees": 0.001,
        "slippage": 0.0005,
        "benchmark": "SPY",
        "market_hours_only": True,
        "allow_shorts": False,
    },

    # Risk management (moderate profile)
    "risk": {
        "enabled": True,
        "position_sizing_method": "fixed_percent",
        "position_size_pct": 0.10,  # 10% per trade
        "max_positions": 5,
        "stop_loss_pct": None,  # Disabled by default
        "stop_loss_type": "fixed",
    },

    # Sweep mode settings
    "sweep": {
        "sort_by": "Sharpe Ratio",
        "top_n": None,
        "parallel": True,
        "max_workers": 4,
        "export_csv": True,
        "export_html": True,
    },

    # Optimization settings
    "optimization": {
        "metric": "sharpe_ratio",
        "param_grid": {},
        "n_jobs": 4,
        "max_iterations": None,
    },

    # Walk-forward validation settings
    "walk_forward": {
        "train_months": 12,
        "test_months": 6,
        "step_months": 6,
        "n_splits": None,
        "export_results": True,
    },

    # Output settings
    "output": {
        "directory": None,  # Auto-generate
        "save_trades": True,
        "save_reports": True,
        "quantstats": True,
        "visualize": True,
        "timestamp_files": True,
        "verbosity": 1,
    },
}


# Predefined date ranges (can be referenced via dates.preset)
DATE_PRESETS: Dict[str, Dict[str, str]] = {
    # Full periods
    "full_periods.max_history": {"start": "2015-01-01", "end": "2024-12-31"},
    "full_periods.five_years": {"start": "2020-01-01", "end": "2024-12-31"},
    "full_periods.three_years": {"start": "2022-01-01", "end": "2024-12-31"},
    "full_periods.one_year": {"start": "2024-01-01", "end": "2024-12-31"},

    # Market regimes
    "regimes.bull_2019_2021": {"start": "2019-01-01", "end": "2021-12-31"},
    "regimes.bear_2022": {"start": "2022-01-01", "end": "2022-12-31"},
    "regimes.covid_crash": {"start": "2020-02-01", "end": "2020-04-30"},
    "regimes.covid_recovery": {"start": "2020-04-01", "end": "2021-12-31"},

    # Walk-forward periods
    "walk_forward.in_sample": {"start": "2020-01-01", "end": "2022-12-31"},
    "walk_forward.out_of_sample": {"start": "2023-01-01", "end": "2024-12-31"},

    # Optimization
    "optimization.fast_test": {"start": "2023-01-01", "end": "2024-12-31"},
    "optimization.robust_test": {"start": "2020-01-01", "end": "2024-12-31"},
}


# Predefined symbol universes (can be referenced via symbols.universe)
SYMBOL_UNIVERSES: Dict[str, list] = {
    # Indices
    "indices.all": ["SPY", "QQQ", "DIA", "IWM"],
    "indices.major": ["SPY", "QQQ"],

    # Technology
    "technology.faang": ["AAPL", "AMZN", "GOOGL", "META", "NFLX"],
    "technology.semiconductors": ["NVDA", "AMD", "INTC", "MU", "AVGO"],
    "technology.mag7": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],

    # Leveraged ETFs
    "leveraged.triple_long": ["TQQQ", "UPRO", "SOXL", "TECL", "FAS"],
    "leveraged.triple_short": ["SQQQ", "SPXU", "SOXS"],
    "leveraged.pairs": ["TQQQ", "SQQQ", "UPRO", "SPXU"],

    # Sectors
    "sectors.all": ["XLK", "XLF", "XLV", "XLE", "XLY", "XLP", "XLI", "XLB", "XLRE", "XLU", "XLC"],
    "sectors.cyclical": ["XLY", "XLI", "XLB", "XLF"],
    "sectors.defensive": ["XLP", "XLV", "XLU"],

    # Production portfolios
    "production.conservative": ["SPY", "QQQ", "IWM"],
    "production.moderate": ["SPY", "QQQ", "IWM", "XLK", "XLF"],
    "production.aggressive": ["TQQQ", "SOXL", "TECL"],
    "production.omr_top_5": ["QQQ", "TQQQ", "SOXL", "TECL", "SPY"],

    # Testing
    "testing.single": ["SPY"],
    "testing.pair": ["SPY", "QQQ"],
    "testing.small": ["AAPL", "MSFT", "GOOGL"],
}


def get_date_preset(preset_name: str) -> Dict[str, str]:
    """
    Get dates from a preset name.

    Args:
        preset_name: Preset name (e.g., 'full_periods.five_years')

    Returns:
        Dict with 'start' and 'end' keys

    Raises:
        KeyError: If preset not found
    """
    if preset_name not in DATE_PRESETS:
        available = ', '.join(sorted(DATE_PRESETS.keys()))
        raise KeyError(f"Unknown date preset '{preset_name}'. Available: {available}")
    return DATE_PRESETS[preset_name]


def get_symbol_universe(universe_name: str) -> list:
    """
    Get symbols from a universe name.

    Args:
        universe_name: Universe name (e.g., 'production.conservative')

    Returns:
        List of symbols

    Raises:
        KeyError: If universe not found
    """
    if universe_name not in SYMBOL_UNIVERSES:
        available = ', '.join(sorted(SYMBOL_UNIVERSES.keys()))
        raise KeyError(f"Unknown symbol universe '{universe_name}'. Available: {available}")
    return SYMBOL_UNIVERSES[universe_name]
