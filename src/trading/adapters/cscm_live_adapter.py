"""
CSCM (Cross-Sectional Crypto Momentum) Live Trading Adapter.

DEPRECATED for paper trading: Uses Alpaca which has significant limitations
for crypto paper trading (integer-only quantities, limited universe).

For paper trading, use CSCMDemoAdapter with DemoBroker instead.
This module is retained for potential future live trading with real money.

Key Features:
- Weekly rebalance (Sunday 0:00 UTC)
- 24/7 crypto trading via Coinbase/Alpaca
- BTC regime filter
- Top N coins by 28-day momentum
- Equal weight allocation

Deployment (live trading only):
    systemctl start homeguard-cscm
"""

from decimal import Decimal
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
import pickle

import pandas as pd

from src.data.providers.binance import BinanceDataProvider
from src.strategies.advanced.cscm_signals import CSCMSignals, CSCMRiskSignals
from src.trading.brokers.crypto_router import CryptoBrokerRouter
from src.trading.brokers.interfaces import OrderSide, OrderType
from src.trading.state.strategy_state_manager import StrategyStateManager
from src.utils.logger import get_logger
from src.utils.timezone import tz
from src.utils.trading_logger import get_trade_log_writer

logger = get_logger(__name__)

# Strategy identifier
STRATEGY_NAME = 'cscm'

# Cache configuration
CACHE_DIR = Path.home() / '.homeguard' / 'cache'
CACHE_FILE = CACHE_DIR / 'cscm_state.pkl'

# Per-fill BUY headroom factor. Each BUY's target notional is multiplied by this
# before computing target_qty so the actual fill (with adverse slippage + fees
# applied by the broker) still fits within available cash. 0.998 = 20 bps buffer;
# DemoBroker drag empirically ~12.5 bps (5 bps slippage + 10 bps fees), so this
# leaves ~7 bps margin for randomization. Tunable via cscm_live.yaml.
DEFAULT_BUY_SAFETY_FACTOR = Decimal('0.998')


def _resolve_broker_name(broker) -> str:
    """Return a human-readable broker name from a broker or CryptoBrokerRouter."""
    if hasattr(broker, '_primary') or hasattr(broker, '_secondary'):
        active_slot = getattr(broker, '_active_broker_name', 'primary')
        primary_broker = getattr(broker, '_primary', None)
        secondary_broker = getattr(broker, '_secondary', None)
        if active_slot == 'primary' and primary_broker is not None:
            active_broker = primary_broker
        elif secondary_broker is not None:
            active_broker = secondary_broker
        elif primary_broker is not None:
            active_broker = primary_broker
        else:
            active_broker = None
    else:
        active_broker = broker
    if active_broker is None:
        return 'crypto'
    cls_name = active_broker.__class__.__name__.lower()
    if 'coinbase' in cls_name:
        return 'coinbase'
    if 'alpaca' in cls_name:
        return 'alpaca'
    if 'demo' in cls_name:
        return 'demo'
    return cls_name


def _maybe_emit_decision(
    *,
    risk_signals,
    target_positions: dict,
    executed: list,
    account: dict,
    broker_name: str,
    initial_capital: float,
    positions_after: Optional[List[dict]] = None,
) -> None:
    """Emit one DecisionRecord per CSCM rebalance attempt.

    `executed` is a list of tuples:
        (symbol, action, status, fill_price_or_None, order_id_or_None, reason_or_None)

    `positions_after` is an optional list of broker position dicts
    (the same shape `broker.get_crypto_positions()` returns) used to
    populate `PostState.positions_after`. CSCM doesn't write to
    StrategyStateManager (it goes through DemoBroker directly), so
    the post-state snapshot must come from the broker, not from the
    state manager.

    Decision-log failures are never fatal -- all exceptions are swallowed.
    """
    try:
        from src.trading.decision_log import (
            begin_decision, write_decision, stage,
        )
        from src.trading.decision_log.record import (
            GateResult, StrategyInputs, LogicDecisions, Execution, PostState,
            PositionSnapshot,
        )

        rec = begin_decision(
            strategy="cscm",
            trigger_kind="scheduled_rebalance",
            schedule_time="Sunday 00:00 UTC",
            broker_name=broker_name,
            data_provider="binance-streaming",
            initial_capital_usd=initial_capital,
            strategy_version=1,
        )
        try:
            with stage(rec, "preconditions"):
                ok = GateResult(passed=True, details={})
                rec.preconditions.strategy_enabled = ok
                rec.preconditions.shutdown_requested = ok
                rec.preconditions.execution_lock_acquired = ok
                rec.preconditions.health_check = ok
                rec.preconditions.data_freshness = ok
                rec.preconditions.all_passed = True

            with stage(rec, "inputs"):
                rec.inputs = StrategyInputs(
                    regime=risk_signals.regime,
                    regime_confidence=None,
                    regime_params=None,
                    vix=None,
                    spy_drawdown_pct=None,
                    universe_size=len(risk_signals.momentum_scores or {}),
                    data_completeness_pct=100.0,
                    cache_source="binance-streaming",
                    momentum_scores=dict(risk_signals.momentum_scores or {}),
                    extra={
                        "btc_price": risk_signals.btc_price,
                        "btc_sma": risk_signals.btc_sma,
                        "is_rebalance_day": risk_signals.is_rebalance_day,
                        "reduce_exposure": risk_signals.reduce_exposure,
                        "exposure_pct": risk_signals.exposure_pct,
                        "top_symbols": list(risk_signals.top_symbols or []),
                    },
                )

            with stage(rec, "logic"):
                target_symbols = [s for s, w in target_positions.items() if w > 0]
                target_value_usd = {
                    s: w * initial_capital for s, w in target_positions.items() if w > 0
                }
                rec.logic_decisions = LogicDecisions(
                    top_n=len(target_symbols),
                    target_symbols=target_symbols,
                    target_weights={s: w for s, w in target_positions.items() if w > 0},
                    target_value_usd=target_value_usd,
                    reduce_exposure=bool(risk_signals.reduce_exposure),
                    exposure_pct=float(risk_signals.exposure_pct),
                    exit_signals=[],
                    hold_signals=[],
                    skip_reasons={},
                )

            with stage(rec, "execution"):
                for sym, action, status, fill_price, order_id, reason in executed:
                    rec.executions.append(Execution(
                        symbol=sym,
                        action=action,
                        target_qty=0,
                        target_value_usd=target_value_usd.get(sym, 0.0),
                        status=status,
                        fill_price=fill_price,
                        filled_qty=None,
                        order_id=order_id,
                        reason=reason,
                    ))

            with stage(rec, "post_state"):
                positions_snapshot: Dict[str, PositionSnapshot] = {}
                for pos in (positions_after or []):
                    sym = pos.get("symbol")
                    if not sym:
                        continue
                    try:
                        positions_snapshot[sym] = PositionSnapshot(
                            qty=float(pos.get("quantity", 0) or 0),
                            avg_price=float(pos.get("avg_entry_price") or pos.get("entry_price") or 0),
                            unrealized_pnl=float(pos.get("unrealized_pnl", 0) or 0),
                        )
                    except (TypeError, ValueError):
                        continue
                rec.post_state = PostState(
                    positions_after=positions_snapshot,
                    cash_after=float(account.get("cash", 0) or 0),
                    strategy_equity_after=float(account.get("portfolio_value", 0) or 0),
                    state_writes=[],
                )
        except Exception as e:
            import traceback
            from src.trading.decision_log.record import ErrorInfo
            rec.error = ErrorInfo(
                type=type(e).__name__,
                message=str(e),
                traceback=traceback.format_exc(),
                stage=getattr(rec, "_current_stage", "unknown"),
            )
        finally:
            write_decision(rec)
    except Exception as outer_e:
        try:
            logger.error(f"[decision_log] _maybe_emit_decision failed: {outer_e}")
        except Exception:
            pass


class CSCMLiveAdapter:
    """
    Live trading adapter for CSCM strategy.

    Manages weekly rebalancing of crypto positions based on
    cross-sectional momentum with BTC regime filter.

    Usage:
        adapter = CSCMLiveAdapter(
            universe=['BTC/USD', 'ETH/USD', 'SOL/USD', ...],
            top_n=5,
        )

        # Run main loop
        adapter.run()
    """

    # Default parameters (optimized from backtest)
    # Universe: All Alpaca-tradeable alts (BTC kept for regime filter)
    DEFAULT_UNIVERSE = [
        'BTC/USD', 'ETH/USD', 'SOL/USD', 'AVAX/USD', 'LINK/USD',
        'DOGE/USD', 'DOT/USD', 'LTC/USD', 'BCH/USD', 'UNI/USD',
        'AAVE/USD', 'SUSHI/USD', 'XRP/USD', 'CRV/USD', 'GRT/USD'
    ]
    DEFAULT_TOP_N = 7
    DEFAULT_MOMENTUM_PERIOD = 28
    DEFAULT_BTC_SMA_PERIOD = 40  # Optimized: 40-day SMA
    DEFAULT_TRAILING_STOP = 0.25  # 25% trailing stop

    def __init__(
        self,
        universe: Optional[List[str]] = None,
        top_n: int = DEFAULT_TOP_N,
        momentum_period: int = DEFAULT_MOMENTUM_PERIOD,
        btc_sma_period: int = DEFAULT_BTC_SMA_PERIOD,
        trailing_stop_pct: float = DEFAULT_TRAILING_STOP,
        rebalance_day: str = 'sunday',
        go_to_cash_in_bear: bool = True,
        broker: Optional[CryptoBrokerRouter] = None,
        paper: bool = True,
        signal_logger=None,
        max_capital_usd: Optional[float] = None,
        buy_safety_factor: Optional[float] = None,
    ):
        """
        Initialize CSCM live adapter.

        Args:
            universe: Crypto symbols to trade
            top_n: Number of top coins to hold
            momentum_period: Days for momentum calculation
            btc_sma_period: BTC SMA period for regime
            trailing_stop_pct: Trailing stop percentage (0.25 = 25%)
            rebalance_day: Day of week to rebalance
            go_to_cash_in_bear: Exit all in bearish regime
            broker: Crypto broker (auto-created if None)
            paper: Use paper trading (only affects Alpaca)
        """
        self.universe = universe or self.DEFAULT_UNIVERSE
        self.top_n = top_n
        self.momentum_period = momentum_period
        self.btc_sma_period = btc_sma_period
        self.trailing_stop_pct = trailing_stop_pct
        self.rebalance_day = rebalance_day
        self.go_to_cash_in_bear = go_to_cash_in_bear
        self.paper = paper
        self.max_capital_usd = max_capital_usd
        if max_capital_usd is not None:
            logger.info(f"[CSCM] Capital cap: ${max_capital_usd:,.0f} per strategy")

        self.buy_safety_factor = (
            Decimal(str(buy_safety_factor)) if buy_safety_factor is not None
            else DEFAULT_BUY_SAFETY_FACTOR
        )

        # Initialize signal generator
        self.signals = CSCMSignals(
            symbols=self.universe,
            top_n=top_n,
            momentum_period=momentum_period,
            btc_sma_period=btc_sma_period,
            rebalance_day=rebalance_day,
            go_to_cash_in_bear=go_to_cash_in_bear,
            trailing_stop_pct=trailing_stop_pct
        )

        # Initialize broker (lazy)
        self._broker = broker
        self._broker_initialized = False

        # Data provider (Binance REST API - no credentials needed)
        self._data_provider = BinanceDataProvider()

        # Signal logger (for hypothetical performance tracking)
        self._signal_logger = signal_logger

        # State tracking
        self._last_rebalance: Optional[datetime] = None
        self._current_positions: Dict[str, Decimal] = {}
        self._trade_log_writer = None

        # Shared state manager (enables per-strategy attribution + metrics)
        self.state_manager = StrategyStateManager()

        # Load saved state
        self._load_state()

        logger.info(f"[CSCM] Initialized live adapter")
        logger.info(f"[CSCM]   Universe: {len(self.universe)} symbols")
        logger.info(f"[CSCM]   Top N: {top_n}")
        logger.info(f"[CSCM]   BTC SMA Period: {btc_sma_period}")
        logger.info(f"[CSCM]   Trailing Stop: {trailing_stop_pct:.0%}")
        logger.info(f"[CSCM]   Rebalance Day: {rebalance_day}")
        logger.info(f"[CSCM]   Paper Mode: {paper}")

    @property
    def broker(self) -> CryptoBrokerRouter:
        """Get broker with lazy initialization."""
        if not self._broker_initialized:
            if self._broker is None:
                self._broker = CryptoBrokerRouter()
            self._broker_initialized = True
        return self._broker

    def _load_state(self) -> None:
        """Load saved state from disk."""
        try:
            if CACHE_FILE.exists():
                with open(CACHE_FILE, 'rb') as f:
                    state = pickle.load(f)
                    self._last_rebalance = state.get('last_rebalance')
                    self._current_positions = state.get('positions', {})
                    # Load trailing stop state
                    peak_value = state.get('peak_value', 0.0)
                    if peak_value > 0:
                        self.signals._peak_portfolio_value = peak_value
                    self.signals._trailing_stop_triggered = state.get('trailing_stop_triggered', False)
                    logger.info(f"[CSCM] Loaded state: last rebalance {self._last_rebalance}")
                    if self.signals._trailing_stop_triggered:
                        logger.warning("[CSCM] Trailing stop was triggered in previous session")
        except Exception as e:
            logger.warning(f"[CSCM] Failed to load state: {e}")

    def _save_state(self) -> None:
        """Save state to disk."""
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(CACHE_FILE, 'wb') as f:
                pickle.dump({
                    'last_rebalance': self._last_rebalance,
                    'positions': self._current_positions,
                    'saved_at': tz.now(),
                    # Save trailing stop state
                    'peak_value': self.signals._peak_portfolio_value,
                    'trailing_stop_triggered': self.signals._trailing_stop_triggered,
                }, f)
            logger.debug("[CSCM] Saved state to disk")
        except Exception as e:
            logger.warning(f"[CSCM] Failed to save state: {e}")

    def _fetch_historical_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for all symbols."""
        end = datetime.now()
        start = end - timedelta(days=60)  # Need enough for momentum + SMA

        data_dict = self._data_provider.get_historical_bars_batch(
            self.universe,
            start,
            end,
            '1D'
        )

        logger.info(f"[CSCM] Fetched data for {len(data_dict)} symbols")
        return data_dict

    def _get_current_positions(self) -> Dict[str, Decimal]:
        """Get current positions from broker."""
        try:
            positions = self.broker.get_crypto_positions()
            pos_dict = {}
            for pos in positions:
                symbol = pos['symbol']
                if symbol in self.universe:
                    pos_dict[symbol] = pos['quantity']
            return pos_dict
        except Exception as e:
            logger.error(f"[CSCM] Failed to get positions: {e}")
            return {}

    def _get_account_balance(self) -> float:
        """Get available account balance."""
        try:
            balance = self.broker.get_crypto_balance('USD')
            return balance.get('available', 0.0)
        except Exception as e:
            logger.error(f"[CSCM] Failed to get balance: {e}")
            return 0.0

    def _get_total_portfolio_value(self) -> float:
        """Get total portfolio value (cash + positions)."""
        try:
            total = self._get_account_balance()
            positions = self._get_current_positions()

            for symbol, qty in positions.items():
                try:
                    quote = self.broker.get_crypto_quote(symbol)
                    total += float(qty) * quote['last']
                except (KeyError, ValueError, ConnectionError):
                    pass

            return total
        except Exception as e:
            logger.error(f"[CSCM] Failed to get portfolio value: {e}")
            return 0.0

    def _check_trailing_stop(self) -> bool:
        """
        Check and update trailing stop.

        Returns:
            True if stop was triggered and positions should be closed
        """
        portfolio_value = self._get_total_portfolio_value()
        if portfolio_value <= 0:
            return False

        # Update portfolio value in signals (this checks trailing stop)
        self.signals.update_portfolio_value(portfolio_value)

        # If stop just triggered, close all positions
        if self.signals.is_trailing_stop_active():
            current_positions = self._get_current_positions()
            if current_positions:
                logger.warning(f"[CSCM] Trailing stop triggered - closing all positions")
                for symbol in list(current_positions.keys()):
                    try:
                        self.broker.close_crypto_position(symbol)
                        logger.info(f"[CSCM] Closed {symbol} due to trailing stop")
                    except Exception as e:
                        logger.error(f"[CSCM] Failed to close {symbol}: {e}")
                self._save_state()
                return True

        return False

    def _should_rebalance(self) -> bool:
        """Check if we should rebalance now.

        Uses UTC for the day-of-week / day-boundary check to align with the
        EventBridge weekend EC2 schedule (also UTC). Previously used
        `tz.now()` (ET), which silently missed CSCM's weekly rebalance every
        weekend because the EventBridge wake window (Sat 23:00 UTC -> Sun
        00:10 UTC) is entirely on Saturday in ET, so the day check returned
        False and the bot never fired before EC2 was stopped again.
        """
        now = datetime.now(timezone.utc)

        # Check if it's rebalance day (in UTC)
        if not self.signals.should_rebalance(now):
            return False

        # Check if we already rebalanced today (UTC). Normalize the persisted
        # _last_rebalance to UTC if it carries a different tzinfo (older state
        # files were written with ET-aware datetimes).
        if self._last_rebalance:
            last = self._last_rebalance
            if hasattr(last, 'tzinfo') and last.tzinfo is not None:
                last_utc = last.astimezone(timezone.utc)
            else:
                last_utc = last
            last_date = last_utc.date() if hasattr(last_utc, 'date') else last_utc
            today = now.date()
            if last_date == today:
                logger.debug("[CSCM] Already rebalanced today")
                return False

        return True

    def rebalance(self) -> List[Dict]:
        """
        Execute rebalancing.

        Returns:
            List of executed orders
        """
        logger.info("[CSCM] Starting rebalance...")

        # Fetch latest data
        data_dict = self._fetch_historical_data()
        if not data_dict or 'BTC/USD' not in data_dict:
            logger.error("[CSCM] Failed to fetch required data, skipping rebalance")
            return []

        # Update signals with latest data
        self.signals.update_historical_data(data_dict)

        # Get risk signals and target positions
        risk_signals = self.signals.get_risk_signals()
        target_positions = self.signals.get_target_positions()

        logger.info(f"[CSCM] Regime: {risk_signals.regime}")
        logger.info(f"[CSCM] BTC: ${risk_signals.btc_price:,.2f} (SMA: ${risk_signals.btc_sma:,.2f})")
        logger.info(f"[CSCM] Top symbols: {risk_signals.top_symbols}")

        # Get current positions and balance
        current_positions = self._get_current_positions()
        available_balance = self._get_account_balance()

        logger.info(f"[CSCM] Available balance: ${available_balance:,.2f}")
        logger.info(f"[CSCM] Current positions: {list(current_positions.keys())}")

        # Calculate trades needed
        orders = []
        executed = []  # decision-log bookkeeping: (symbol, action, status, fill_price, order_id, reason)
        total_value = available_balance

        # Add current position values
        for symbol, qty in current_positions.items():
            try:
                quote = self.broker.get_crypto_quote(symbol)
                total_value += float(qty) * quote['last']
            except (KeyError, ValueError, ConnectionError):
                pass

        logger.info(f"[CSCM] Total portfolio value: ${total_value:,.2f}")

        sizing_base = min(total_value, self.max_capital_usd) if self.max_capital_usd else total_value
        if self.max_capital_usd and sizing_base < total_value:
            logger.info(f"[CSCM] Capped sizing base at ${sizing_base:,.2f} (max_capital_usd)")

        # Close positions not in target
        for symbol in list(current_positions.keys()):
            if target_positions.get(symbol, 0) == 0:
                try:
                    logger.info(f"[CSCM] Closing position: {symbol}")
                    closed_qty = float(current_positions.get(symbol, 0))
                    order = self.broker.close_crypto_position(symbol)
                    orders.append(order)
                    self._log_close(symbol, closed_qty, order)
                    fill_price = float(order.get('filled_avg_price') or 0) if isinstance(order, dict) else None
                    order_id = order.get('order_id') if isinstance(order, dict) else None
                    executed.append((symbol, "sell", "filled", fill_price, order_id, None))
                except Exception as e:
                    logger.error(f"[CSCM] Failed to close {symbol}: {e}")
                    executed.append((symbol, "sell", "error", None, None, str(e)))

        # ---- Classify each target into hold / sell-down / buy ----
        # Initial classification uses the snapshot price + sizing_base. Final BUY
        # sizing happens in Phase D against fresh cash so that cumulative slippage
        # and fees from earlier fills don't starve the last symbol.
        holds: List[tuple] = []          # (symbol, current_price)
        sell_downs: List[tuple] = []     # (symbol, sell_qty, snapshot_price)
        pending_buys: List[tuple] = []   # (symbol, current_qty, snapshot_price)

        for symbol, weight in target_positions.items():
            if weight == 0:
                continue

            target_value = sizing_base * weight
            current_qty = current_positions.get(symbol, Decimal('0'))

            try:
                quote = self.broker.get_crypto_quote(symbol)
                current_price = quote['last']
                target_qty = Decimal(str(target_value / current_price))
                qty_diff = target_qty - current_qty

                # Skip small adjustments (< 5% of target)
                if abs(qty_diff) < target_qty * Decimal('0.05'):
                    holds.append((symbol, current_price))
                    continue

                if qty_diff < 0:
                    sell_downs.append((symbol, abs(qty_diff), current_price))
                else:
                    pending_buys.append((symbol, current_qty, current_price))

            except Exception as e:
                logger.error(f"[CSCM] Failed to classify {symbol}: {e}")
                executed.append((symbol, "buy", "error", None, None, str(e)))

        # ---- Phase: holds (just record skip) ----
        for symbol, current_price in holds:
            executed.append((symbol, "hold", "skipped", float(current_price), None, "adjustment < 5% threshold"))

        # ---- Phase: sell-downs first (they free cash for subsequent buys) ----
        for symbol, sell_qty, snapshot_price in sell_downs:
            try:
                logger.info(f"[CSCM] Selling {sell_qty:.6f} {symbol}")
                order = self.broker.place_crypto_order(
                    symbol=symbol,
                    quantity=sell_qty,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET
                )
                orders.append(order)
                self._log_partial_sell(symbol, float(sell_qty), float(snapshot_price), order)
                fill_price = float(order.get('filled_avg_price') or snapshot_price) if isinstance(order, dict) else float(snapshot_price)
                order_id = order.get('order_id') if isinstance(order, dict) else None
                executed.append((symbol, "sell", "filled", fill_price, order_id, None))
            except Exception as e:
                logger.error(f"[CSCM] Failed to sell-down {symbol}: {e}")
                executed.append((symbol, "sell", "error", None, None, str(e)))

        # ---- Phase: buys with sequential cash tracking + safety factor ----
        # Re-read cash before each fill and divide by remaining slots so the
        # actual broker drag (slippage + fees) is absorbed by smaller subsequent
        # targets rather than starving the final symbol. The safety factor adds
        # per-fill headroom so even the last fill (where target = remaining_cash
        # / 1) still fits after the broker applies adverse slippage and fees.
        remaining_buys = len(pending_buys)
        for symbol, current_qty, snapshot_price in pending_buys:
            try:
                # Read fresh available cash from the broker
                try:
                    account = self.broker.get_account() or {}
                    available_cash = float(account.get('cash', 0) or 0)
                except Exception:
                    available_cash = float(self._get_account_balance())

                if remaining_buys <= 0 or available_cash <= 0:
                    executed.append((symbol, "buy", "skipped", float(snapshot_price), None, "insufficient cash for sequential allocation"))
                    continue  # `finally` decrements remaining_buys

                target_value = (
                    Decimal(str(available_cash)) * self.buy_safety_factor
                    / Decimal(remaining_buys)
                )

                # Re-fetch quote (price may have moved during sell-downs)
                quote = self.broker.get_crypto_quote(symbol)
                current_price = quote['last']
                target_qty = Decimal(str(float(target_value) / current_price))
                qty_diff = target_qty - current_qty

                if qty_diff <= 0:
                    # Sell-downs freed enough that no buy is needed for this symbol
                    executed.append((symbol, "hold", "skipped", float(current_price), None, "no additional buy needed after sell-downs"))
                    continue  # `finally` decrements remaining_buys

                logger.info(f"[CSCM] Buying {qty_diff:.6f} {symbol}")
                order = self.broker.place_crypto_order(
                    symbol=symbol,
                    quantity=qty_diff,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET
                )
                orders.append(order)
                self._log_buy(symbol, float(qty_diff), float(current_price), order)
                fill_price = float(order.get('filled_avg_price') or current_price) if isinstance(order, dict) else float(current_price)
                order_id = order.get('order_id') if isinstance(order, dict) else None
                executed.append((symbol, "buy", "filled", fill_price, order_id, None))
            except Exception as e:
                logger.error(f"[CSCM] Failed to buy {symbol}: {e}")
                executed.append((symbol, "buy", "error", None, None, str(e)))
            finally:
                remaining_buys = max(remaining_buys - 1, 0)

        # Mark rebalance complete. Stored in UTC to match the day-boundary
        # semantics in _should_rebalance (which uses UTC to align with the
        # EventBridge weekend EC2 schedule).
        self._last_rebalance = datetime.now(timezone.utc)
        self.signals.mark_rebalanced(self._last_rebalance)
        self._current_positions = self._get_current_positions()
        self._save_state()

        # Emit decision record (replaces legacy signal_logger.log_rebalance)
        try:
            account = self.broker.get_account() or {}
        except Exception:
            account = {"portfolio_value": total_value, "cash": available_balance}
        try:
            positions_after = self.broker.get_crypto_positions() or []
        except Exception as e:
            logger.debug(f"[CSCM] Failed to snapshot positions for decision log: {e}")
            positions_after = []
        broker_name = _resolve_broker_name(self.broker)
        initial_capital = float(self.max_capital_usd or total_value)
        _maybe_emit_decision(
            risk_signals=risk_signals,
            target_positions=target_positions,
            executed=executed,
            account=account,
            broker_name=broker_name,
            initial_capital=initial_capital,
            positions_after=positions_after,
        )

        logger.info(f"[CSCM] Rebalance complete: {len(orders)} orders executed")
        return orders

    def _log_buy(self, symbol: str, qty: float, price: float, order: Dict) -> None:
        """Log a buy fill, persist to trade log, and update state manager."""
        order_id = order.get('order_id') if isinstance(order, dict) else None
        fill_price = float(order.get('filled_avg_price') or price) if isinstance(order, dict) else price
        try:
            get_trade_log_writer().log_entry(
                strategy=STRATEGY_NAME,
                symbol=symbol,
                qty=qty,
                price=fill_price,
                order_id=order_id,
                metadata={'broker': 'crypto'},
            )
        except Exception as e:
            logger.warning(f"[CSCM] Trade-log entry failed (non-blocking): {e}")
        try:
            self.state_manager.add_or_update_position(
                STRATEGY_NAME, symbol, qty, fill_price, order_id, broker='crypto'
            )
        except Exception as e:
            logger.warning(f"[CSCM] State-manager update failed (non-blocking): {e}")

    def _log_partial_sell(self, symbol: str, qty: float, price: float, order: Dict) -> None:
        """Log a partial sell, record realized PnL, and decrement state manager qty."""
        order_id = order.get('order_id') if isinstance(order, dict) else None
        fill_price = float(order.get('filled_avg_price') or price) if isinstance(order, dict) else price
        position_info = {}
        try:
            position_info = self.state_manager.get_positions(STRATEGY_NAME).get(symbol, {}) or {}
        except Exception as e:
            logger.warning(f"[CSCM] State lookup failed for {symbol} (non-blocking): {e}")
        try:
            get_trade_log_writer().log_exit(
                strategy=STRATEGY_NAME,
                symbol=symbol,
                qty=qty,
                exit_price=fill_price,
                order_id=order_id,
                entry_price=position_info.get('entry_price'),
                entry_time=position_info.get('entry_time'),
                metadata={'broker': 'crypto', 'partial': True},
            )
        except Exception as e:
            logger.warning(f"[CSCM] Trade-log exit failed (non-blocking): {e}")
        try:
            # Decrement state qty (negative delta) while preserving entry price/time.
            self.state_manager.add_or_update_position(
                STRATEGY_NAME, symbol, -qty, fill_price, order_id
            )
        except Exception as e:
            logger.warning(f"[CSCM] State-manager decrement failed (non-blocking): {e}")

    def _log_close(self, symbol: str, qty: float, order: Dict) -> None:
        """Log a full close, record realized PnL, and remove from state manager."""
        order_id = order.get('order_id') if isinstance(order, dict) else None
        fill_price = float(order.get('filled_avg_price') or 0) if isinstance(order, dict) else 0.0
        position_info = {}
        try:
            position_info = self.state_manager.get_positions(STRATEGY_NAME).get(symbol, {}) or {}
        except Exception as e:
            logger.warning(f"[CSCM] State lookup failed for {symbol} (non-blocking): {e}")
        try:
            get_trade_log_writer().log_exit(
                strategy=STRATEGY_NAME,
                symbol=symbol,
                qty=qty,
                exit_price=fill_price,
                order_id=order_id,
                entry_price=position_info.get('entry_price'),
                entry_time=position_info.get('entry_time'),
                metadata={'broker': 'crypto'},
            )
        except Exception as e:
            logger.warning(f"[CSCM] Trade-log exit failed (non-blocking): {e}")
        try:
            self.state_manager.remove_position(STRATEGY_NAME, symbol)
        except Exception as e:
            logger.warning(f"[CSCM] State-manager remove failed (non-blocking): {e}")

    def get_status(self) -> Dict:
        """Get current strategy status."""
        try:
            data_dict = self._fetch_historical_data()
            if data_dict:
                self.signals.update_historical_data(data_dict)

            risk_signals = self.signals.get_risk_signals()
            target_positions = self.signals.get_target_positions()
            current_positions = self._get_current_positions()
            balance = self._get_account_balance()

            return {
                'strategy': STRATEGY_NAME,
                'regime': risk_signals.regime,
                'btc_price': risk_signals.btc_price,
                'btc_sma': risk_signals.btc_sma,
                'is_rebalance_day': risk_signals.is_rebalance_day,
                'top_symbols': risk_signals.top_symbols,
                'target_positions': target_positions,
                'current_positions': {k: float(v) for k, v in current_positions.items()},
                'available_balance': balance,
                'last_rebalance': str(self._last_rebalance) if self._last_rebalance else None,
                # Trailing stop info
                'trailing_stop_triggered': risk_signals.trailing_stop_triggered,
                'current_drawdown': risk_signals.current_drawdown,
                'peak_value': risk_signals.peak_value,
                'trailing_stop_pct': self.trailing_stop_pct,
            }
        except Exception as e:
            logger.error(f"[CSCM] Failed to get status: {e}")
            return {'error': str(e)}

    def run_once(self) -> None:
        """Run single iteration of the strategy."""
        now = tz.now()
        logger.info(f"[CSCM] Run at {now}")

        try:
            # Fetch data and compute signals every cycle (for logging)
            data_dict = self._fetch_historical_data()
            if data_dict and 'BTC/USD' in data_dict:
                self.signals.update_historical_data(data_dict)
                risk_signals = self.signals.get_risk_signals()

                logger.info(f"[CSCM] Regime: {risk_signals.regime}")
                logger.info(f"[CSCM] BTC: ${risk_signals.btc_price:,.2f} (SMA: ${risk_signals.btc_sma:,.2f})")

                # Log signal snapshot
                if self._signal_logger:
                    self._signal_logger.log_signal(
                        regime=risk_signals.regime,
                        btc_price=risk_signals.btc_price,
                        btc_sma=risk_signals.btc_sma,
                        is_rebalance_day=risk_signals.is_rebalance_day,
                        reduce_exposure=risk_signals.reduce_exposure,
                        exposure_pct=risk_signals.exposure_pct,
                        top_symbols=risk_signals.top_symbols,
                        momentum_scores=risk_signals.momentum_scores,
                    )

            # Check trailing stop first (may close positions)
            stop_triggered = self._check_trailing_stop()
            if stop_triggered:
                logger.info("[CSCM] Trailing stop active - waiting for next rebalance")
                return

            # Check if we should rebalance
            if self._should_rebalance():
                # Reset trailing stop on rebalance day (allows re-entry)
                self.signals.reset_trailing_stop()
                self.rebalance()
            else:
                logger.info("[CSCM] No rebalance needed")

        except Exception as e:
            logger.error(f"[CSCM] Error in run_once: {e}")

    def run(self, check_interval_hours: int = 1) -> None:
        """
        Run main trading loop.

        Args:
            check_interval_hours: How often to check for rebalance
        """
        import time as time_module

        logger.info("[CSCM] Starting main trading loop...")
        logger.info(f"[CSCM] Check interval: {check_interval_hours} hours")

        while True:
            try:
                self.run_once()
            except KeyboardInterrupt:
                logger.info("[CSCM] Received shutdown signal")
                break
            except Exception as e:
                logger.error(f"[CSCM] Error in main loop: {e}")

            # Sleep until next check
            time_module.sleep(check_interval_hours * 3600)
