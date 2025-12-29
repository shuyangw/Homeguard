"""
Enhanced trade logger v2 with custom column support.

Exports all trade columns including strategy-specific enrichments.
"""

from pathlib import Path
from typing import Optional, Union, TYPE_CHECKING

import pandas as pd

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.backtesting_v2.engine.portfolio_simulator import PortfolioV2

logger = get_logger(__name__)


class TradeLoggerV2:
    """
    Enhanced trade logger that exports all columns including custom enrichment.

    Key improvements over original TradeLogger:
    - Exports ALL columns (including strategy-specific enrichment)
    - Exports bar-by-bar portfolio state
    - Supports full analysis export (trades, state, summary)
    """

    @staticmethod
    def export_trades_csv(
        portfolio: "PortfolioV2",
        output_path: Union[str, Path],
        symbol: str = "",
    ) -> Path:
        """
        Export trades with ALL columns (including custom enrichment).

        Args:
            portfolio: PortfolioV2 instance
            output_path: Path for CSV output
            symbol: Optional symbol for logging

        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Prefer enriched trades_df over raw trades list
        if hasattr(portfolio, "trades_df") and portfolio.trades_df is not None:
            trades = portfolio.trades_df
        elif hasattr(portfolio, "trades"):
            trades = pd.DataFrame(portfolio.trades) if isinstance(portfolio.trades, list) else portfolio.trades
        else:
            trades = pd.DataFrame()

        if trades.empty:
            logger.warning(f"No trades to export for {symbol or 'portfolio'}")
            return output_path

        # Export ALL columns (not just predefined)
        trades.to_csv(output_path, index=False)
        logger.success(f"Trades exported: {output_path} ({len(trades)} trades, {len(trades.columns)} columns)")

        return output_path

    @staticmethod
    def export_portfolio_state_csv(
        portfolio: "PortfolioV2",
        output_path: Union[str, Path],
        symbol: str = "",
    ) -> Path:
        """
        Export bar-by-bar portfolio state.

        Args:
            portfolio: PortfolioV2 instance
            output_path: Path for CSV output
            symbol: Optional symbol for logging

        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if hasattr(portfolio, "portfolio_state"):
            state = portfolio.portfolio_state
        else:
            state = pd.DataFrame()

        if state.empty:
            logger.warning(f"No portfolio state to export for {symbol or 'portfolio'}")
            return output_path

        state.to_csv(output_path, index=False)
        logger.success(f"Portfolio state exported: {output_path} ({len(state)} bars)")

        return output_path

    @staticmethod
    def export_summary_csv(
        portfolio: "PortfolioV2",
        output_path: Union[str, Path],
        symbol: str = "",
    ) -> Path:
        """
        Export portfolio statistics summary.

        Args:
            portfolio: PortfolioV2 instance
            output_path: Path for CSV output
            symbol: Optional symbol for logging

        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        stats = portfolio.stats() if hasattr(portfolio, "stats") else {}

        if not stats:
            logger.warning(f"No stats to export for {symbol or 'portfolio'}")
            return output_path

        # Convert to DataFrame with symbol column
        summary = pd.DataFrame([{
            "symbol": symbol or "PORTFOLIO",
            **stats,
        }])

        summary.to_csv(output_path, index=False)
        logger.success(f"Summary exported: {output_path}")

        return output_path

    @staticmethod
    def export_full_analysis(
        portfolio: "PortfolioV2",
        output_dir: Union[str, Path],
        symbol: str = "",
        prefix: str = "",
    ) -> dict:
        """
        Export all analysis files: trades, state, and summary.

        Args:
            portfolio: PortfolioV2 instance
            output_dir: Output directory
            symbol: Symbol for filenames
            prefix: Optional prefix for filenames (e.g., timestamp)

        Returns:
            Dict with paths to all exported files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        name = f"{prefix}_{symbol}" if prefix else symbol
        name = name.strip("_") or "portfolio"

        paths = {}

        # Export trades
        trades_path = output_dir / f"{name}_trades.csv"
        paths["trades"] = TradeLoggerV2.export_trades_csv(portfolio, trades_path, symbol)

        # Export state
        state_path = output_dir / f"{name}_state.csv"
        paths["state"] = TradeLoggerV2.export_portfolio_state_csv(portfolio, state_path, symbol)

        # Export summary
        summary_path = output_dir / f"{name}_summary.csv"
        paths["summary"] = TradeLoggerV2.export_summary_csv(portfolio, summary_path, symbol)

        return paths

    @staticmethod
    def export_equity_curve_csv(
        portfolio: "PortfolioV2",
        output_path: Union[str, Path],
        symbol: str = "",
    ) -> Path:
        """
        Export equity curve as CSV.

        Args:
            portfolio: PortfolioV2 instance
            output_path: Path for CSV output
            symbol: Optional symbol for logging

        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if hasattr(portfolio, "equity_curve") and portfolio.equity_curve is not None:
            equity = portfolio.equity_curve
            df = pd.DataFrame({
                "timestamp": equity.index,
                "equity": equity.values,
            })
            df.to_csv(output_path, index=False)
            logger.success(f"Equity curve exported: {output_path} ({len(df)} bars)")
        else:
            logger.warning(f"No equity curve to export for {symbol or 'portfolio'}")

        return output_path


def export_trades_v2(
    portfolio: "PortfolioV2",
    output_path: Union[str, Path],
    symbol: str = "",
) -> Path:
    """Convenience function to export trades."""
    return TradeLoggerV2.export_trades_csv(portfolio, output_path, symbol)


def export_analysis_v2(
    portfolio: "PortfolioV2",
    output_dir: Union[str, Path],
    symbol: str = "",
    prefix: str = "",
) -> dict:
    """Convenience function to export full analysis."""
    return TradeLoggerV2.export_full_analysis(portfolio, output_dir, symbol, prefix)
