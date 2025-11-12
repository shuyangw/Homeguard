"""
Trade logging and portfolio tracking for backtesting results.

Exports detailed trade-by-trade logs and bar-by-bar portfolio state.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from src.utils import logger


class TradeLogger:
    """
    Logs detailed trade information and portfolio state progression.
    """

    @staticmethod
    def export_trades_csv(portfolio, output_path: Path | str, symbol: str = "") -> None:
        """
        Export all trades to CSV with entry/exit details and P&L.

        Args:
            portfolio: VectorBT Portfolio object
            output_path: Path to save CSV file
            symbol: Symbol name for labeling
        """
        try:
            # Access trades - handle MultiAssetPortfolio, custom Portfolio (list), and VectorBT
            trades_obj = portfolio.trades  # type: ignore[attr-defined]

            # Custom Portfolio: trades is a list of dicts
            if isinstance(trades_obj, list):
                if len(trades_obj) == 0:
                    logger.warning(f"No trades found for {symbol or 'portfolio'}")
                    return

                # Convert custom Portfolio trades list to DataFrame
                trades_df = pd.DataFrame(trades_obj)

                # Split into buy and sell rows (matching MultiAssetPortfolio format)
                buy_rows = []
                sell_rows = []

                for _, trade in trades_df.iterrows():
                    trade_type = trade.get('type', '')

                    if trade_type == 'entry':
                        # Buy row
                        buy_rows.append({
                            'Symbol': symbol,
                            'Date': trade.get('timestamp'),
                            'Price': trade.get('price'),
                            'Size': trade.get('shares'),
                            'Direction': 'Buy',
                            'PnL': np.nan,
                            'Return': np.nan,
                            'Status': ''
                        })
                    elif trade_type == 'exit':
                        # Sell row
                        sell_rows.append({
                            'Symbol': symbol,
                            'Date': trade.get('timestamp'),
                            'Price': trade.get('price'),
                            'Size': trade.get('shares'),
                            'Direction': 'Sell',
                            'PnL': trade.get('pnl'),
                            'Return': trade.get('pnl_pct'),
                            'Status': trade.get('exit_reason', '')
                        })

                # Combine and sort by date
                all_rows = buy_rows + sell_rows
                trades = pd.DataFrame(all_rows)
                if len(trades) > 0 and 'Date' in trades.columns:
                    trades = trades.sort_values('Date').reset_index(drop=True)

            # VectorBT or MultiAssetPortfolio: trades is already a DataFrame
            elif isinstance(trades_obj, pd.DataFrame):
                trades = trades_obj

                # Check if this is MultiAssetPortfolio format (lowercase columns)
                # and split into separate buy/sell rows
                if 'entry_timestamp' in trades.columns:
                    # Create buy rows (entry transactions)
                    buy_rows = []
                    for _, trade in trades.iterrows():
                        buy_rows.append({
                            'Symbol': trade.get('symbol', ''),
                            'Date': trade['entry_timestamp'],
                            'Price': trade['entry_price'],
                            'Size': trade['shares'],
                            'Direction': 'Buy',
                            'Status': '',
                            'PnL': np.nan,
                            'Return': np.nan
                        })

                    # Create sell rows (exit transactions)
                    sell_rows = []
                    for _, trade in trades.iterrows():
                        sell_rows.append({
                            'Symbol': trade.get('symbol', ''),
                            'Date': trade['exit_timestamp'],
                            'Price': trade['exit_price'],
                            'Size': trade['shares'],
                            'Direction': 'Sell',
                            'Status': trade.get('exit_reason', ''),
                            'PnL': trade['pnl'],
                            'Return': trade['pnl_pct']
                        })

                    # Combine buy and sell rows, sorted by date
                    all_rows = buy_rows + sell_rows
                    trades = pd.DataFrame(all_rows).sort_values('Date').reset_index(drop=True)

            elif hasattr(trades_obj, 'records_readable'):
                trades = trades_obj.records_readable  # type: ignore[attr-defined]
            elif hasattr(trades_obj, 'records'):
                trades = trades_obj.records  # type: ignore[attr-defined]
            else:
                trades = pd.DataFrame(trades_obj) if trades_obj else pd.DataFrame()

            if len(trades) == 0:
                logger.warning(f"No trades found for {symbol or 'portfolio'}")
                return

            # Handle different trade formats
            # New format (buy/sell rows): 'Symbol', 'Date', 'Price', 'Size', 'Direction', 'Status', 'PnL', 'Return'
            # Old format (round-trip): 'Symbol', 'Entry Date', 'Entry Price', 'Exit Date', 'Exit Price', 'Size', 'PnL', 'Return', 'Direction', 'Status'

            if 'Date' in trades.columns:
                # New format with buy/sell rows
                trade_cols = ['Symbol', 'Date', 'Price', 'Size', 'Direction', 'PnL', 'Return', 'Status']
                available_cols = [col for col in trade_cols if col in trades.columns]
                trades_export = trades[available_cols].copy()

                # Add symbol column if provided and not already present
                if symbol and 'Symbol' not in trades_export.columns:
                    trades_export.insert(0, 'Symbol', symbol)

                # Format numeric columns
                if 'Price' in trades_export.columns:
                    trades_export['Price'] = trades_export['Price'].round(2)
                if 'PnL' in trades_export.columns:
                    trades_export['PnL'] = trades_export['PnL'].round(2)
                if 'Return' in trades_export.columns:
                    trades_export['Return'] = (trades_export['Return'] * 100).round(2)
                    trades_export.rename(columns={'Return': 'Return %'}, inplace=True)
            else:
                # Old format with round-trip trades
                trade_cols = [
                    'Symbol', 'Entry Date', 'Entry Price', 'Exit Date', 'Exit Price',
                    'Size', 'PnL', 'Return', 'Direction', 'Status'
                ]
                available_cols = [col for col in trade_cols if col in trades.columns]
                trades_export = trades[available_cols].copy()

                # Add symbol column if provided and not already present
                if symbol and 'Symbol' not in trades_export.columns:
                    trades_export.insert(0, 'Symbol', symbol)

                # Format numeric columns
                if 'Entry Price' in trades_export.columns:
                    trades_export['Entry Price'] = trades_export['Entry Price'].round(2)
                if 'Exit Price' in trades_export.columns:
                    trades_export['Exit Price'] = trades_export['Exit Price'].round(2)
                if 'PnL' in trades_export.columns:
                    trades_export['PnL'] = trades_export['PnL'].round(2)
                if 'Return' in trades_export.columns:
                    trades_export['Return'] = (trades_export['Return'] * 100).round(2)
                    trades_export.rename(columns={'Return': 'Return %'}, inplace=True)

            # Save to CSV
            output_path = Path(output_path)
            trades_export.to_csv(output_path, index=False)

            logger.success(f"Trades log exported: {output_path} ({len(trades_export)} trades)")

        except Exception as e:
            logger.error(f"Failed to export trades CSV for {symbol}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Create error file
            output_path = Path(output_path)
            pd.DataFrame({'Error': [str(e)]}).to_csv(output_path, index=False)

    @staticmethod
    def export_equity_curve_csv(portfolio, output_path: Path | str, symbol: str = "") -> None:
        """
        Export bar-by-bar equity curve (portfolio value at each timestamp).

        Args:
            portfolio: VectorBT Portfolio object
            output_path: Path to save CSV file
            symbol: Symbol name for labeling
        """
        try:
            # Get equity curve using VectorBT-compatible methods
            equity_curve = None

            # Method 1: Try VectorBT-compatible .value() method (works for both VectorBT and MultiAssetPortfolio)
            if hasattr(portfolio, 'value'):
                try:
                    value_attr = portfolio.value  # type: ignore[attr-defined]
                    if callable(value_attr):
                        equity_curve = value_attr()
                    else:
                        equity_curve = value_attr
                except Exception:
                    pass

            # Method 2: Try .total_value
            if equity_curve is None and hasattr(portfolio, 'total_value'):
                try:
                    total_value = portfolio.total_value  # type: ignore[attr-defined]
                    if callable(total_value):
                        equity_curve = total_value()
                    else:
                        equity_curve = total_value
                except Exception:
                    pass

            # Method 3: Try raw .equity_curve attribute (convert to Series if needed)
            if equity_curve is None and hasattr(portfolio, 'equity_curve'):
                raw_curve = portfolio.equity_curve  # type: ignore[attr-defined]

                # If it's a list (MultiAssetPortfolio raw attribute), convert to Series
                if isinstance(raw_curve, list):
                    if hasattr(portfolio, 'equity_timestamps'):
                        timestamps = portfolio.equity_timestamps  # type: ignore[attr-defined]
                        equity_curve = pd.Series(raw_curve, index=timestamps)
                    else:
                        # No timestamps available, use numeric index
                        equity_curve = pd.Series(raw_curve)
                else:
                    # Already a Series
                    equity_curve = raw_curve

            if equity_curve is None or len(equity_curve) == 0:
                logger.warning(f"No equity curve data for {symbol or 'portfolio'}")
                return

            # Create DataFrame with timestamp and portfolio value
            equity_df = pd.DataFrame({
                'Timestamp': equity_curve.index,
                'Portfolio Value': equity_curve.values.round(2)
            })

            # Add symbol column if provided
            if symbol:
                equity_df.insert(0, 'Symbol', symbol)

            # Calculate daily returns (use .values to ensure positional alignment)
            daily_returns = equity_curve.pct_change().fillna(0) * 100
            equity_df['Daily Return %'] = daily_returns.values.round(4)

            # Calculate cumulative return (use .values to ensure positional alignment)
            initial_value = equity_curve.iloc[0]
            cumulative_returns = ((equity_curve / initial_value - 1) * 100).round(2)
            equity_df['Cumulative Return %'] = cumulative_returns.values

            # Save to CSV
            output_path = Path(output_path)
            equity_df.to_csv(output_path, index=False)

            logger.success(f"Equity curve exported: {output_path} ({len(equity_df)} bars)")

        except Exception as e:
            logger.error(f"Failed to export equity curve CSV for {symbol}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Create error file
            output_path = Path(output_path)
            pd.DataFrame({'Error': [str(e)]}).to_csv(output_path, index=False)

    @staticmethod
    def export_portfolio_state_csv(portfolio, output_path: Path | str, symbol: str = "") -> None:
        """
        Export comprehensive bar-by-bar portfolio state (value, cash, position, P&L).

        Args:
            portfolio: VectorBT Portfolio object
            output_path: Path to save CSV file
            symbol: Symbol name for labeling
        """
        try:
            # Get equity curve (portfolio value over time) using VectorBT-compatible methods
            value = None

            # Method 1: Try VectorBT-compatible .value() method (works for both VectorBT and MultiAssetPortfolio)
            if hasattr(portfolio, 'value'):
                try:
                    value_attr = portfolio.value  # type: ignore[attr-defined]
                    if callable(value_attr):
                        value = value_attr()
                    else:
                        value = value_attr
                except Exception:
                    pass

            # Method 2: Try .total_value
            if value is None and hasattr(portfolio, 'total_value'):
                try:
                    total_value = portfolio.total_value  # type: ignore[attr-defined]
                    if callable(total_value):
                        value = total_value()
                    else:
                        value = total_value
                except Exception:
                    pass

            # Method 3: Try raw .equity_curve attribute (convert to Series if needed)
            if value is None and hasattr(portfolio, 'equity_curve'):
                raw_curve = portfolio.equity_curve  # type: ignore[attr-defined]

                # If it's a list (MultiAssetPortfolio raw attribute), convert to Series
                if isinstance(raw_curve, list):
                    if hasattr(portfolio, 'equity_timestamps'):
                        timestamps = portfolio.equity_timestamps  # type: ignore[attr-defined]
                        value = pd.Series(raw_curve, index=timestamps)
                    else:
                        # No timestamps available, use numeric index
                        value = pd.Series(raw_curve)
                else:
                    # Already a Series
                    value = raw_curve

            if value is None or len(value) == 0:
                logger.warning(f"No portfolio state data for {symbol or 'portfolio'}")
                return

            # Try to get cash (custom Portfolio doesn't expose this separately)
            cash = None
            for attr in ['cash', 'cash_balance']:
                if hasattr(portfolio, attr):
                    try:
                        cash_val = getattr(portfolio, attr)
                        if callable(cash_val):
                            cash = cash_val()
                        else:
                            cash = cash_val
                        if cash is not None:
                            break
                    except:
                        continue

            # Create comprehensive DataFrame
            data_dict = {
                'Timestamp': value.index,
                'Portfolio Value': value.values.round(2),
            }

            # Add cash if available
            if cash is not None:
                if isinstance(cash, pd.Series):
                    data_dict['Cash'] = cash.values.round(2)
                elif hasattr(cash, 'values'):
                    # NumPy array or similar
                    data_dict['Cash'] = cash.values.round(2)
                else:
                    # Scalar value - MultiAssetPortfolio returns current cash as float
                    data_dict['Cash'] = cash

            state_df = pd.DataFrame(data_dict)

            # Add symbol column if provided
            if symbol:
                state_df.insert(0, 'Symbol', symbol)

            # Add position size if available
            position = None

            # For multi-symbol portfolios, use position count history instead of individual positions
            if hasattr(portfolio, 'position_count_history'):
                try:
                    import json
                    count_history = portfolio.position_count_history  # type: ignore[attr-defined]
                    if count_history and len(count_history) > 0:
                        # Create a Series from position count history
                        timestamps = [ts for ts, _ in count_history]
                        counts = [count for _, count in count_history]
                        count_series = pd.Series(counts, index=timestamps)

                        # Reindex to match state_df timestamps
                        state_df['Position Count'] = count_series.reindex(state_df['Timestamp'], method='ffill').fillna(0).astype(int).values
                        position = 'handled'  # Mark as handled
                except Exception:
                    pass

            # If not handled yet, try standard position attributes
            if position is None:
                for attr in ['assets', 'positions', 'holdings']:
                    if hasattr(portfolio, attr):
                        try:
                            position = getattr(portfolio, attr)
                            if callable(position):
                                position = position()
                            if position is not None:
                                # Check type and handle accordingly
                                if isinstance(position, pd.Series):
                                    # Pandas Series - use .values to get numpy array
                                    state_df['Position Size'] = position.values
                                elif isinstance(position, dict):
                                    # Multi-symbol portfolio: positions is Dict[str, Position] (current state only)
                                    # This is already handled via position_count_history above
                                    # Don't try to convert dict to column (causes the <built-in method> error)
                                    pass
                                elif hasattr(position, 'values') and not callable(getattr(position, 'values', None)):
                                    # Has a values attribute that's not a method (like numpy array)
                                    state_df['Position Size'] = position.values
                                else:
                                    # Scalar value - use directly
                                    state_df['Position Size'] = position
                                break
                        except:
                            continue

            # Calculate metrics (use .values to ensure positional alignment)
            initial_value = value.iloc[0]
            cumulative_returns = ((value / initial_value - 1) * 100).round(2)
            state_df['Cumulative Return %'] = cumulative_returns.values

            daily_returns = (value.pct_change().fillna(0) * 100).round(4)
            state_df['Daily Return %'] = daily_returns.values

            # Add drawdown if available
            drawdown = None
            for attr in ['drawdown', 'drawdowns', 'dd']:
                if hasattr(portfolio, attr):
                    try:
                        drawdown = getattr(portfolio, attr)
                        if callable(drawdown):
                            drawdown = drawdown()
                        if drawdown is not None:
                            state_df['Drawdown %'] = (drawdown * 100).round(2)
                            break
                    except:
                        continue

            # Save to CSV
            output_path = Path(output_path)
            state_df.to_csv(output_path, index=False)

            logger.success(f"Portfolio state exported: {output_path} ({len(state_df)} bars)")

        except Exception as e:
            logger.error(f"Failed to export portfolio state CSV for {symbol}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Create error file
            output_path = Path(output_path)
            pd.DataFrame({'Error': [str(e)]}).to_csv(output_path, index=False)

    @staticmethod
    def get_trades_summary(portfolio) -> Dict[str, Any]:
        """
        Get summary statistics about trades.

        Returns:
            Dictionary with trade summary metrics
        """
        try:
            # Access trades - handle custom Portfolio (list) vs VectorBT (object)
            trades_obj = portfolio.trades  # type: ignore[attr-defined]

            # Custom Portfolio: trades is a list of dicts
            if isinstance(trades_obj, list):
                if len(trades_obj) == 0:
                    return {
                        'total_trades': 0,
                        'winning_trades': 0,
                        'losing_trades': 0,
                        'avg_win': 0,
                        'avg_loss': 0,
                        'largest_win': 0,
                        'largest_loss': 0,
                        'total_profit': 0,
                        'total_loss': 0
                    }

                trades = pd.DataFrame(trades_obj)
            # VectorBT: trades has .records_readable
            elif isinstance(trades_obj, pd.DataFrame):
                trades = trades_obj
            elif hasattr(trades_obj, 'records_readable'):
                trades = trades_obj.records_readable  # type: ignore[attr-defined]
            elif hasattr(trades_obj, 'records'):
                trades = trades_obj.records  # type: ignore[attr-defined]
            else:
                trades = pd.DataFrame(trades_obj) if trades_obj else pd.DataFrame()

            if trades is None or len(trades) == 0:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'avg_win': 0,
                    'avg_loss': 0,
                    'largest_win': 0,
                    'largest_loss': 0,
                    'total_profit': 0,
                    'total_loss': 0
                }

            # Handle both 'PnL' (VectorBT) and 'pnl' (custom Portfolio) column names
            pnl_col = 'PnL' if 'PnL' in trades.columns else 'pnl'

            # Filter exit trades only (custom Portfolio has both entry and exit in same list)
            if 'type' in trades.columns:
                exit_trades = trades[trades['type'] == 'exit']
            else:
                exit_trades = trades

            if pnl_col not in exit_trades.columns:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'avg_win': 0,
                    'avg_loss': 0,
                    'largest_win': 0,
                    'largest_loss': 0,
                    'total_profit': 0,
                    'total_loss': 0
                }

            winning = exit_trades[exit_trades[pnl_col] > 0]
            losing = exit_trades[exit_trades[pnl_col] < 0]

            return {
                'total_trades': len(exit_trades),
                'winning_trades': len(winning),
                'losing_trades': len(losing),
                'avg_win': float(winning[pnl_col].mean()) if len(winning) > 0 else 0,
                'avg_loss': float(losing[pnl_col].mean()) if len(losing) > 0 else 0,
                'largest_win': float(winning[pnl_col].max()) if len(winning) > 0 else 0,
                'largest_loss': float(losing[pnl_col].min()) if len(losing) > 0 else 0,
                'total_profit': float(winning[pnl_col].sum()) if len(winning) > 0 else 0,
                'total_loss': float(losing[pnl_col].sum()) if len(losing) > 0 else 0
            }

        except Exception as e:
            logger.warning(f"Could not get trades summary: {e}")
            return {}
