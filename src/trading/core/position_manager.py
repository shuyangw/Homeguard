"""
Position Manager - Broker-Agnostic Position Tracking and Risk Management

Tracks all positions, calculates P&L, enforces risk limits, and maintains
position state. This component is completely broker-agnostic.

Design Principles:
- Single Responsibility: Only handles position tracking and risk management
- Broker-Agnostic: No dependencies on specific broker implementations
- State Persistence: Save/load positions for recovery
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
import uuid
import json
from pathlib import Path
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger()  # Use global logger (no file creation)


class PositionManager:
    """
    Broker-agnostic position manager.

    Tracks positions, calculates P&L, enforces risk limits.
    Works with any broker implementation.
    """

    def __init__(self, config: Dict):
        """
        Initialize position manager.

        Args:
            config: Configuration dict with risk limits:
                - max_position_size_pct: Max position size (e.g., 0.15 = 15%)
                - max_concurrent_positions: Max concurrent positions
                - max_total_exposure_pct: Max total exposure
                - stop_loss_pct: Stop-loss threshold
        """
        self.config = config

        # Extract risk limits from config
        self.max_position_size_pct = config.get('max_position_size_pct', 0.15)
        self.max_concurrent_positions = config.get('max_concurrent_positions', 3)
        self.max_total_exposure_pct = config.get('max_total_exposure_pct', 0.45)
        self.stop_loss_pct = config.get('stop_loss_pct', -0.02)

        # Position tracking
        self.positions: Dict[str, Dict] = {}  # position_id -> position data
        self.closed_positions: List[Dict] = []

        # Performance metrics
        self.total_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0

        logger.info(
            f"Initialized PositionManager (max_positions={self.max_concurrent_positions}, "
            f"max_size={self.max_position_size_pct:.1%}, stop_loss={self.stop_loss_pct:.1%})"
        )

    # ==================== Position Management ====================

    def add_position(
        self,
        symbol: str,
        entry_price: float,
        qty: int,
        timestamp: datetime,
        order_id: str,
        signal: Optional[Dict] = None
    ) -> str:
        """
        Record new position.

        Args:
            symbol: Stock symbol
            entry_price: Entry price
            qty: Number of shares
            timestamp: Entry timestamp
            order_id: Broker order ID
            signal: Optional signal details

        Returns:
            position_id: Unique position identifier
        """
        position_id = str(uuid.uuid4())

        position = {
            'position_id': position_id,
            'symbol': symbol,
            'entry_price': entry_price,
            'current_price': entry_price,
            'quantity': qty,
            'entry_time': timestamp,
            'exit_time': None,
            'exit_price': None,
            'pnl': 0.0,
            'pnl_pct': 0.0,
            'status': 'open',
            'order_ids': [order_id],
            'signal': signal or {},
            'value': entry_price * abs(qty),
        }

        self.positions[position_id] = position

        logger.info(
            f"Position opened: {symbol} {qty} shares @ ${entry_price:.2f} "
            f"(value: ${position['value']:.2f})"
        )

        return position_id

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        timestamp: datetime,
        reason: str = "scheduled_exit"
    ) -> Dict:
        """
        Close position and calculate P&L.

        Args:
            position_id: Position ID to close
            exit_price: Exit price
            timestamp: Exit timestamp
            reason: Reason for closing (scheduled_exit, stop_loss, etc.)

        Returns:
            Dict with position details and P&L

        Raises:
            ValueError: If position doesn't exist
        """
        if position_id not in self.positions:
            raise ValueError(f"Position {position_id} not found")

        position = self.positions[position_id]

        # Calculate P&L
        entry_price = position['entry_price']
        qty = position['quantity']
        pnl = (exit_price - entry_price) * qty
        pnl_pct = (exit_price - entry_price) / entry_price

        # Update position
        position['exit_price'] = exit_price
        position['exit_time'] = timestamp
        position['current_price'] = exit_price
        position['pnl'] = pnl
        position['pnl_pct'] = pnl_pct
        position['status'] = 'closed'
        position['close_reason'] = reason

        # Update metrics
        self.total_pnl += pnl
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        # Move to closed positions
        self.closed_positions.append(position)
        del self.positions[position_id]

        logger.info(
            f"Position closed: {position['symbol']} @ ${exit_price:.2f} | "
            f"P&L: ${pnl:+.2f} ({pnl_pct:+.2%}) | Reason: {reason}"
        )

        return position

    def update_position_price(self, position_id: str, current_price: float) -> Dict:
        """
        Update position with current market price.

        Args:
            position_id: Position ID
            current_price: Current market price

        Returns:
            Updated position dict
        """
        if position_id not in self.positions:
            raise ValueError(f"Position {position_id} not found")

        position = self.positions[position_id]
        position['current_price'] = current_price

        # Calculate unrealized P&L
        entry_price = position['entry_price']
        qty = position['quantity']
        position['pnl'] = (current_price - entry_price) * qty
        position['pnl_pct'] = (current_price - entry_price) / entry_price

        return position

    def get_position(self, position_id: str) -> Optional[Dict]:
        """Get position by ID."""
        return self.positions.get(position_id)

    def get_position_by_symbol(self, symbol: str) -> Optional[Dict]:
        """Get open position by symbol."""
        for position in self.positions.values():
            if position['symbol'] == symbol:
                return position
        return None

    def get_open_positions(self) -> List[Dict]:
        """Get all open positions."""
        return list(self.positions.values())

    def get_closed_positions(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get closed positions.

        Args:
            limit: Optional limit on number of positions to return

        Returns:
            List of closed positions (most recent first)
        """
        positions = sorted(
            self.closed_positions,
            key=lambda p: p['exit_time'],
            reverse=True
        )
        return positions[:limit] if limit else positions

    # ==================== P&L Calculations ====================

    def calculate_pnl(self, position: Dict, current_price: float) -> float:
        """
        Calculate unrealized P&L for position.

        Args:
            position: Position dict
            current_price: Current market price

        Returns:
            Unrealized P&L
        """
        entry_price = position['entry_price']
        qty = position['quantity']
        return (current_price - entry_price) * qty

    def calculate_pnl_pct(self, position: Dict, current_price: float) -> float:
        """
        Calculate unrealized P&L percentage for position.

        Args:
            position: Position dict
            current_price: Current market price

        Returns:
            Unrealized P&L percentage
        """
        entry_price = position['entry_price']
        return (current_price - entry_price) / entry_price

    def calculate_portfolio_pnl(self, current_prices: Dict[str, float]) -> Dict:
        """
        Calculate total portfolio P&L.

        Args:
            current_prices: Dict of symbol -> current price

        Returns:
            Dict with portfolio P&L metrics
        """
        total_unrealized_pnl = 0.0
        total_position_value = 0.0

        for position in self.positions.values():
            symbol = position['symbol']
            if symbol in current_prices:
                current_price = current_prices[symbol]
                pnl = self.calculate_pnl(position, current_price)
                total_unrealized_pnl += pnl
                total_position_value += current_price * abs(position['quantity'])

        return {
            'total_realized_pnl': self.total_pnl,
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_pnl': self.total_pnl + total_unrealized_pnl,
            'total_position_value': total_position_value,
            'open_positions': len(self.positions),
        }

    # ==================== Risk Management ====================

    def check_risk_limits(
        self,
        new_position_value: Optional[float] = None,
        portfolio_value: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Check if new position violates risk limits.

        Args:
            new_position_value: Value of new position to add
            portfolio_value: Current portfolio value

        Returns:
            Tuple of (is_valid, reason)
        """
        # Check max concurrent positions
        if len(self.positions) >= self.max_concurrent_positions:
            return False, f"Max positions reached ({self.max_concurrent_positions})"

        # If checking new position
        if new_position_value and portfolio_value:
            # Check position size
            position_size_pct = new_position_value / portfolio_value
            if position_size_pct > self.max_position_size_pct:
                return False, f"Position size {position_size_pct:.1%} exceeds max {self.max_position_size_pct:.1%}"

            # Check total exposure
            current_exposure = sum(p['value'] for p in self.positions.values())
            total_exposure = (current_exposure + new_position_value) / portfolio_value
            if total_exposure > self.max_total_exposure_pct:
                return False, f"Total exposure {total_exposure:.1%} exceeds max {self.max_total_exposure_pct:.1%}"

        return True, "Validation passed"

    def check_stop_losses(self, current_prices: Dict[str, float]) -> List[Dict]:
        """
        Check if any positions hit stop-loss.

        Args:
            current_prices: Dict of symbol -> current price

        Returns:
            List of positions that hit stop-loss
        """
        positions_to_stop = []

        for position in self.positions.values():
            symbol = position['symbol']
            if symbol in current_prices:
                current_price = current_prices[symbol]
                pnl_pct = self.calculate_pnl_pct(position, current_price)

                if pnl_pct <= self.stop_loss_pct:
                    logger.warning(
                        f"Stop-loss triggered: {symbol} {pnl_pct:.2%} <= {self.stop_loss_pct:.2%}"
                    )
                    positions_to_stop.append(position)

        return positions_to_stop

    # ==================== Metrics & Analytics ====================

    def get_portfolio_metrics(self) -> Dict:
        """
        Calculate portfolio performance metrics.

        Returns:
            Dict with performance metrics
        """
        if self.total_trades == 0:
            return {
                'total_pnl': 0.0,
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
            }

        # Calculate win rate
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0

        # Calculate average win/loss
        winning_pnls = [p['pnl'] for p in self.closed_positions if p['pnl'] > 0]
        losing_pnls = [p['pnl'] for p in self.closed_positions if p['pnl'] < 0]

        avg_win = sum(winning_pnls) / len(winning_pnls) if winning_pnls else 0.0
        avg_loss = sum(losing_pnls) / len(losing_pnls) if losing_pnls else 0.0

        # Calculate profit factor
        total_wins = sum(winning_pnls) if winning_pnls else 0.0
        total_losses = abs(sum(losing_pnls)) if losing_pnls else 0.0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Calculate Sharpe ratio (simplified)
        returns = [p['pnl_pct'] for p in self.closed_positions]
        if len(returns) > 1:
            mean_return = sum(returns) / len(returns)
            std_return = (sum((r - mean_return) ** 2 for r in returns) / len(returns)) ** 0.5
            sharpe_ratio = mean_return / std_return if std_return > 0 else 0.0
        else:
            sharpe_ratio = 0.0

        return {
            'total_pnl': self.total_pnl,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
        }

    def get_trade_history_df(self) -> pd.DataFrame:
        """
        Get trade history as DataFrame.

        Returns:
            DataFrame with trade history
        """
        if not self.closed_positions:
            return pd.DataFrame()

        df = pd.DataFrame(self.closed_positions)
        return df[[
            'position_id', 'symbol', 'entry_time', 'entry_price',
            'exit_time', 'exit_price', 'quantity', 'pnl', 'pnl_pct',
            'status', 'close_reason'
        ]]

    # ==================== State Persistence ====================

    def save_state(self, filepath: str) -> None:
        """
        Save position state to file.

        Args:
            filepath: Path to save state file
        """
        state = {
            'positions': self.positions,
            'closed_positions': self.closed_positions,
            'total_pnl': self.total_pnl,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
        }

        # Convert datetime objects to ISO format
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(state, f, default=convert_datetime, indent=2)

        logger.info(f"Position state saved to: {filepath}")

    def load_state(self, filepath: str) -> None:
        """
        Load position state from file.

        Args:
            filepath: Path to state file
        """
        if not Path(filepath).exists():
            logger.warning(f"State file not found: {filepath}")
            return

        with open(filepath, 'r') as f:
            state = json.load(f)

        # Convert ISO format back to datetime
        def convert_datetime(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, str) and 'T' in value:
                        try:
                            obj[key] = datetime.fromisoformat(value)
                        except:
                            pass
                    elif isinstance(value, dict):
                        convert_datetime(value)
            return obj

        self.positions = convert_datetime(state.get('positions', {}))
        self.closed_positions = convert_datetime(state.get('closed_positions', []))
        self.total_pnl = state.get('total_pnl', 0.0)
        self.total_trades = state.get('total_trades', 0)
        self.winning_trades = state.get('winning_trades', 0)
        self.losing_trades = state.get('losing_trades', 0)

        logger.info(f"Position state loaded from: {filepath}")
        logger.info(
            f"Loaded {len(self.positions)} open positions, "
            f"{len(self.closed_positions)} closed positions"
        )
