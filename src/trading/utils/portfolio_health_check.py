"""
Portfolio Health Check Module

Performs comprehensive portfolio state validation before trading decisions.
Critical for risk management and preventing trading errors.

Author: Homeguard Risk Management
Date: 2025-11-13
"""

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from datetime import datetime, timedelta
from dataclasses import dataclass
from src.utils.logger import logger

if TYPE_CHECKING:
    from src.trading.state.strategy_state_manager import StrategyStateManager


@dataclass
class HealthCheckResult:
    """Result of portfolio health check."""
    passed: bool
    warnings: List[str]
    errors: List[str]
    info: Dict[str, any]


class PortfolioHealthChecker:
    """
    Validates portfolio state before trading decisions.

    Checks:
    - Account balance and buying power
    - Existing positions
    - Pending orders
    - Risk limits
    - Position age and staleness
    """

    def __init__(
        self,
        broker,
        min_buying_power: float = 1000.0,
        min_portfolio_value: float = 5000.0,
        max_positions: int = 10,
        max_position_age_hours: int = 48,
        state_manager: Optional["StrategyStateManager"] = None
    ):
        """
        Initialize health checker.

        Args:
            broker: Broker interface
            min_buying_power: Minimum buying power required
            min_portfolio_value: Minimum portfolio value
            max_positions: Maximum concurrent positions
            max_position_age_hours: Max hours a position should be held
            state_manager: Optional strategy state manager for multi-strategy support
        """
        self.broker = broker
        self.min_buying_power = min_buying_power
        self.min_portfolio_value = min_portfolio_value
        self.max_positions = max_positions
        self.max_position_age_hours = max_position_age_hours
        self.state_manager = state_manager

    def check_before_entry(
        self,
        required_capital: Optional[float] = None,
        allow_existing_positions: bool = False,
        strategy_name: Optional[str] = None
    ) -> HealthCheckResult:
        """
        Comprehensive check before entering new positions.

        Args:
            required_capital: Capital required for planned trades
            allow_existing_positions: If False, error if positions exist
            strategy_name: If provided, only count positions owned by this strategy
                          for max_positions check (requires state_manager)

        Returns:
            HealthCheckResult with validation status
        """
        warnings = []
        errors = []
        info = {}

        strategy_label = f"[{strategy_name.upper()}] " if strategy_name else ""
        logger.info("=" * 60)
        logger.info(f"{strategy_label}PRE-ENTRY PORTFOLIO HEALTH CHECK")
        logger.info("=" * 60)

        # 1. Check account status
        try:
            account = self.broker.get_account()

            buying_power = float(account['buying_power'])
            portfolio_value = float(account['portfolio_value'])
            cash = float(account['cash'])

            info['buying_power'] = buying_power
            info['portfolio_value'] = portfolio_value
            info['cash'] = cash

            logger.info(f"Account Status:")
            logger.info(f"  Portfolio Value: ${portfolio_value:,.2f}")
            logger.info(f"  Buying Power:    ${buying_power:,.2f}")
            logger.info(f"  Cash:            ${cash:,.2f}")

            # Check minimum requirements
            if buying_power < self.min_buying_power:
                errors.append(
                    f"Insufficient buying power: ${buying_power:,.2f} "
                    f"< ${self.min_buying_power:,.2f}"
                )

            if portfolio_value < self.min_portfolio_value:
                errors.append(
                    f"Portfolio value too low: ${portfolio_value:,.2f} "
                    f"< ${self.min_portfolio_value:,.2f}"
                )

            # Check if capital is sufficient for planned trades
            if required_capital and buying_power < required_capital:
                errors.append(
                    f"Insufficient capital for trades: ${buying_power:,.2f} "
                    f"< ${required_capital:,.2f} required"
                )

        except Exception as e:
            errors.append(f"Failed to fetch account info: {e}")

        # 2. Check existing positions
        try:
            positions = self.broker.get_positions()
            info['broker_position_count'] = len(positions)

            # For multi-strategy: get strategy-specific position count
            strategy_position_count = len(positions)  # Default to all positions
            if strategy_name and self.state_manager:
                strategy_positions = self.state_manager.get_positions(strategy_name)
                strategy_position_count = len(strategy_positions)
                info['strategy_position_count'] = strategy_position_count
                logger.info(f"\n{strategy_label}Strategy-owned positions: {strategy_position_count}")
                if strategy_positions:
                    for sym in strategy_positions.keys():
                        logger.info(f"  - {sym}")

            info['position_count'] = strategy_position_count

            if positions:
                logger.info(f"\nTotal Broker Positions: {len(positions)}")

                total_position_value = 0.0

                for pos in positions:
                    # Calculate position details
                    qty = float(pos['quantity'])
                    current_price = float(pos['current_price'])
                    avg_entry = float(pos['avg_entry_price'])

                    position_value = qty * current_price
                    unrealized_pl = qty * (current_price - avg_entry)
                    unrealized_pl_pct = ((current_price - avg_entry) / avg_entry) * 100

                    total_position_value += position_value

                    # Check position age
                    # Note: created_at might not be available for positions
                    position_age = "Unknown"
                    if 'created_at' in pos and pos['created_at']:
                        age_delta = datetime.now() - pos['created_at']
                        position_age = f"{age_delta.total_seconds() / 3600:.1f} hours"

                        # Warn if position is too old
                        if age_delta.total_seconds() / 3600 > self.max_position_age_hours:
                            warnings.append(
                                f"Position {pos['symbol']} is stale ({position_age})"
                            )

                    logger.info(f"  {pos['symbol']}:")
                    logger.info(f"    Qty: {qty}, Price: ${current_price:.2f}")
                    logger.info(f"    Value: ${position_value:,.2f}")
                    logger.info(f"    P&L: ${unrealized_pl:,.2f} ({unrealized_pl_pct:+.2f}%)")
                    logger.info(f"    Age: {position_age}")

                info['total_position_value'] = total_position_value
                logger.info(f"  Total Position Value: ${total_position_value:,.2f}")

                # Decide if existing positions are a problem
                # Only warn if strategy has its own positions (or no strategy specified)
                if not allow_existing_positions and strategy_position_count > 0:
                    warnings.append(
                        f"Existing {strategy_label}positions detected ({strategy_position_count}). "
                        f"New entry may violate position limits."
                    )

                # Check position count limit - use strategy-specific count
                if strategy_position_count >= self.max_positions:
                    errors.append(
                        f"Max positions reached ({strategy_position_count}/{self.max_positions})"
                    )
            else:
                logger.info("\nNo existing positions")

        except Exception as e:
            errors.append(f"Failed to fetch positions: {e}")

        # 3. Check pending orders
        try:
            orders = self.broker.get_open_orders()
            info['pending_orders'] = len(orders)

            if orders:
                logger.info(f"\nPending Orders: {len(orders)}")

                for order in orders:
                    logger.info(f"  {order['symbol']}: {order['side']} {order['quantity']} @ {order['order_type']}")

                    # Check order age
                    if 'created_at' in order and order['created_at']:
                        order_age = datetime.now() - order['created_at']
                        if order_age > timedelta(hours=1):
                            warnings.append(
                                f"Stale order: {order['symbol']} ({order_age.total_seconds()/3600:.1f}h old)"
                            )

                warnings.append(
                    f"Pending orders detected ({len(orders)}). "
                    f"Consider canceling before new entry."
                )
            else:
                logger.info("\nNo pending orders")

        except Exception as e:
            errors.append(f"Failed to fetch orders: {e}")

        # 4. Summary
        logger.info("\n" + "=" * 60)
        logger.info("HEALTH CHECK SUMMARY")
        logger.info("=" * 60)

        if errors:
            logger.error(f"FAILED: {len(errors)} error(s)")
            for err in errors:
                logger.error(f"  X {err}")

        if warnings:
            logger.warning(f"WARNINGS: {len(warnings)} warning(s)")
            for warn in warnings:
                logger.warning(f"  ! {warn}")

        if not errors and not warnings:
            logger.info("All checks passed - safe to proceed")

        logger.info("=" * 60)

        passed = len(errors) == 0

        return HealthCheckResult(
            passed=passed,
            warnings=warnings,
            errors=errors,
            info=info
        )

    def check_before_exit(self) -> HealthCheckResult:
        """
        Comprehensive check before exiting positions.

        Returns:
            HealthCheckResult with validation status
        """
        warnings = []
        errors = []
        info = {}

        logger.info("=" * 60)
        logger.info("PRE-EXIT PORTFOLIO HEALTH CHECK")
        logger.info("=" * 60)

        # 1. Check if positions exist
        try:
            positions = self.broker.get_positions()
            info['position_count'] = len(positions)

            if not positions:
                logger.info("No positions to close")
                logger.info("=" * 60)
                return HealthCheckResult(
                    passed=True,
                    warnings=[],
                    errors=[],
                    info=info
                )

            logger.info(f"Positions to Close: {len(positions)}")

            # 2. Verify each position
            total_unrealized_pl = 0.0

            for pos in positions:
                qty = float(pos['quantity'])
                current_price = float(pos['current_price'])
                avg_entry = float(pos['avg_entry_price'])

                unrealized_pl = qty * (current_price - avg_entry)
                unrealized_pl_pct = ((current_price - avg_entry) / avg_entry) * 100

                total_unrealized_pl += unrealized_pl

                # Check position age
                position_age = "Unknown"
                age_hours = None

                if 'created_at' in pos and pos['created_at']:
                    age_delta = datetime.now() - pos['created_at']
                    age_hours = age_delta.total_seconds() / 3600
                    position_age = f"{age_hours:.1f} hours"

                    # For overnight strategy, expect 12-20 hours
                    if age_hours < 12:
                        warnings.append(
                            f"Position {pos['symbol']} held too short ({position_age}). "
                            f"Expected overnight hold (12-20h)."
                        )

                    if age_hours > 48:
                        errors.append(
                            f"Position {pos['symbol']} held too long ({position_age}). "
                            f"Position is stale!"
                        )

                logger.info(f"\n  {pos['symbol']}:")
                logger.info(f"    Qty: {qty} @ ${avg_entry:.2f} entry")
                logger.info(f"    Current: ${current_price:.2f}")
                logger.info(f"    P&L: ${unrealized_pl:,.2f} ({unrealized_pl_pct:+.2f}%)")
                logger.info(f"    Age: {position_age}")

            info['total_unrealized_pl'] = total_unrealized_pl

            logger.info(f"\n  Total Unrealized P&L: ${total_unrealized_pl:,.2f}")

        except Exception as e:
            errors.append(f"Failed to fetch positions: {e}")

        # 3. Summary
        logger.info("\n" + "=" * 60)
        logger.info("EXIT CHECK SUMMARY")
        logger.info("=" * 60)

        if errors:
            logger.error(f"ERRORS: {len(errors)} error(s)")
            for err in errors:
                logger.error(f"  X {err}")

        if warnings:
            logger.warning(f"WARNINGS: {len(warnings)} warning(s)")
            for warn in warnings:
                logger.warning(f"  ! {warn}")

        if not errors and not warnings:
            logger.info("All positions valid - safe to close")

        logger.info("=" * 60)

        # For exit, we proceed even with warnings (still close positions)
        # Only fail if there are critical errors
        passed = len(errors) == 0

        return HealthCheckResult(
            passed=passed,
            warnings=warnings,
            errors=errors,
            info=info
        )

    def quick_status_check(self) -> Dict:
        """
        Quick portfolio status check (non-blocking).

        Returns:
            Dictionary with current portfolio state
        """
        try:
            account = self.broker.get_account()
            positions = self.broker.get_positions()
            orders = self.broker.get_open_orders()

            return {
                'account_value': float(account['portfolio_value']),
                'buying_power': float(account['buying_power']),
                'cash': float(account['cash']),
                'position_count': len(positions),
                'pending_orders': len(orders),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Quick status check failed: {e}")
            return {}


if __name__ == "__main__":
    # Example usage
    from src.trading.brokers.broker_factory import BrokerFactory

    # Initialize broker
    broker = BrokerFactory.create_broker('alpaca', paper=True)

    # Create health checker
    checker = PortfolioHealthChecker(
        broker=broker,
        min_buying_power=1000.0,
        max_positions=5
    )

    # Check before entry
    result = checker.check_before_entry(
        required_capital=5000.0,
        allow_existing_positions=False
    )

    if result.passed:
        print("\n✓ Safe to enter new positions")
    else:
        print("\n✗ Entry blocked due to errors")
