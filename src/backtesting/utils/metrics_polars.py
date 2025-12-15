"""
Polars-native metrics calculations for backtesting.

This module provides vectorized performance metric calculations using Polars
for use with LazyFrames and streaming data. All functions are designed to work
with Polars expressions for maximum query optimization.

Usage:
    import polars as pl
    from src.backtesting.utils.metrics_polars import (
        calculate_returns,
        calculate_sharpe_ratio,
        calculate_max_drawdown,
        calculate_all_metrics
    )

    # With DataFrame
    df = pl.DataFrame({"equity": [100, 105, 103, 110]})
    metrics = calculate_all_metrics(df["equity"])

    # With LazyFrame (deferred execution)
    lf = pl.LazyFrame({"equity": [100, 105, 103, 110]})
    result = lf.select(
        calculate_returns("equity").alias("returns"),
        calculate_sharpe_ratio("equity", periods_per_year=252).alias("sharpe")
    ).collect()
"""

import polars as pl
from typing import Union, Optional


def calculate_returns(
    equity_col: Union[str, pl.Expr],
    fill_null: float = 0.0
) -> pl.Expr:
    """
    Calculate percentage returns from equity series.

    Args:
        equity_col: Column name or expression containing equity values
        fill_null: Value to fill for null returns (default: 0.0)

    Returns:
        Polars expression for returns calculation
    """
    col = pl.col(equity_col) if isinstance(equity_col, str) else equity_col
    return col.pct_change().fill_null(fill_null)


def calculate_cumulative_returns(
    equity_col: Union[str, pl.Expr],
    init_value: float = 1.0
) -> pl.Expr:
    """
    Calculate cumulative returns from equity series.

    Args:
        equity_col: Column name or expression containing equity values
        init_value: Initial value (default: 1.0 for percentage, or initial cash)

    Returns:
        Polars expression for cumulative returns
    """
    col = pl.col(equity_col) if isinstance(equity_col, str) else equity_col
    return col / col.first() * init_value


def calculate_total_return(equity: pl.Series) -> float:
    """
    Calculate total return percentage.

    Args:
        equity: Polars Series of equity values

    Returns:
        Total return as percentage
    """
    if equity.is_empty() or equity.len() < 2:
        return 0.0

    first_val = equity[0]
    last_val = equity[-1]

    if first_val == 0:
        return 0.0

    return ((last_val - first_val) / first_val) * 100


def calculate_sharpe_ratio(
    equity: pl.Series,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0
) -> float:
    """
    Calculate Sharpe ratio from equity series.

    Args:
        equity: Polars Series of equity values
        periods_per_year: Number of periods per year for annualization
        risk_free_rate: Annual risk-free rate (default: 0)

    Returns:
        Annualized Sharpe ratio
    """
    if equity.is_empty() or equity.len() < 2:
        return 0.0

    # Calculate returns
    returns = equity.pct_change().drop_nulls()

    if returns.is_empty():
        return 0.0

    mean_return = returns.mean()
    std_return = returns.std()

    if std_return is None or std_return == 0:
        return 0.0

    # Annualize
    excess_return = mean_return - (risk_free_rate / periods_per_year)
    sharpe = (excess_return / std_return) * (periods_per_year ** 0.5)

    return float(sharpe) if sharpe is not None else 0.0


def calculate_max_drawdown(equity: pl.Series) -> float:
    """
    Calculate maximum drawdown percentage.

    Args:
        equity: Polars Series of equity values

    Returns:
        Maximum drawdown as negative percentage
    """
    if equity.is_empty() or equity.len() < 2:
        return 0.0

    # Calculate running maximum
    cummax = equity.cum_max()

    # Calculate drawdown at each point
    drawdown = (equity - cummax) / cummax * 100

    # Return minimum (most negative) drawdown
    min_dd = drawdown.min()

    return float(min_dd) if min_dd is not None else 0.0


def calculate_sortino_ratio(
    equity: pl.Series,
    periods_per_year: int = 252,
    target_return: float = 0.0
) -> float:
    """
    Calculate Sortino ratio (uses downside deviation instead of std).

    Args:
        equity: Polars Series of equity values
        periods_per_year: Number of periods per year
        target_return: Target return for downside calculation

    Returns:
        Annualized Sortino ratio
    """
    if equity.is_empty() or equity.len() < 2:
        return 0.0

    returns = equity.pct_change().drop_nulls()

    if returns.is_empty():
        return 0.0

    mean_return = returns.mean()

    # Calculate downside deviation (only negative returns)
    downside_returns = returns.filter(returns < target_return)

    if downside_returns.is_empty():
        return float('inf') if mean_return > 0 else 0.0

    downside_std = downside_returns.std()

    if downside_std is None or downside_std == 0:
        return float('inf') if mean_return > 0 else 0.0

    # Annualize
    excess_return = mean_return - (target_return / periods_per_year)
    sortino = (excess_return / downside_std) * (periods_per_year ** 0.5)

    return float(sortino) if sortino is not None else 0.0


def calculate_calmar_ratio(
    equity: pl.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Calmar ratio (annual return / max drawdown).

    Args:
        equity: Polars Series of equity values
        periods_per_year: Number of periods per year

    Returns:
        Calmar ratio
    """
    total_return = calculate_total_return(equity)
    max_dd = calculate_max_drawdown(equity)

    if max_dd == 0:
        return float('inf') if total_return > 0 else 0.0

    # Annualize return
    n_periods = equity.len()
    years = n_periods / periods_per_year if periods_per_year > 0 else 1

    if years <= 0:
        return 0.0

    # CAGR approximation
    annual_return = total_return / years

    return float(annual_return / abs(max_dd))


def calculate_win_rate(pnl_series: pl.Series) -> float:
    """
    Calculate win rate from P&L series.

    Args:
        pnl_series: Polars Series of P&L values

    Returns:
        Win rate as percentage (0-100)
    """
    if pnl_series.is_empty():
        return 0.0

    total = pnl_series.len()
    wins = pnl_series.filter(pnl_series > 0).len()

    return (wins / total) * 100 if total > 0 else 0.0


def calculate_profit_factor(pnl_series: pl.Series) -> float:
    """
    Calculate profit factor (gross profit / gross loss).

    Args:
        pnl_series: Polars Series of P&L values

    Returns:
        Profit factor
    """
    if pnl_series.is_empty():
        return 0.0

    gross_profit = pnl_series.filter(pnl_series > 0).sum()
    gross_loss = abs(pnl_series.filter(pnl_series < 0).sum())

    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0

    return float(gross_profit / gross_loss)


def calculate_all_metrics(
    equity: pl.Series,
    pnl_series: Optional[pl.Series] = None,
    periods_per_year: int = 252
) -> dict:
    """
    Calculate all performance metrics.

    Args:
        equity: Polars Series of equity values
        pnl_series: Optional Polars Series of trade P&L values
        periods_per_year: Number of periods per year

    Returns:
        Dictionary of all metrics
    """
    metrics = {
        'total_return_pct': calculate_total_return(equity),
        'sharpe_ratio': calculate_sharpe_ratio(equity, periods_per_year),
        'max_drawdown_pct': calculate_max_drawdown(equity),
        'sortino_ratio': calculate_sortino_ratio(equity, periods_per_year),
        'calmar_ratio': calculate_calmar_ratio(equity, periods_per_year),
    }

    if pnl_series is not None and not pnl_series.is_empty():
        metrics.update({
            'win_rate_pct': calculate_win_rate(pnl_series),
            'profit_factor': calculate_profit_factor(pnl_series),
            'total_trades': pnl_series.len(),
            'avg_win': float(pnl_series.filter(pnl_series > 0).mean() or 0),
            'avg_loss': float(pnl_series.filter(pnl_series < 0).mean() or 0),
        })

    return metrics
