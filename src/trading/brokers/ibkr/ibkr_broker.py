"""
IBKR Broker - Full implementation of BrokerInterface + OptionsTradingInterface.

Follows the same Adapter Pattern as AlpacaBroker: translates between
broker-agnostic interfaces and IBKR-specific API.  All ib_async types
are converted at the boundary.

Implements 22 abstract methods across 6 interfaces:
    AccountInterface (2), MarketHoursInterface (2), MarketDataInterface (3),
    OrderManagementInterface (3), StockTradingInterface (4+),
    OptionsTradingInterface (7+)
"""

import asyncio
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pytz

from ib_async import (
    IB, Contract, LimitOrder, MarketOrder, Option, Stock, StopOrder,
    StopLimitOrder, Trade, ComboLeg,
)

from src.trading.brokers.interfaces import (
    AccountInterface,
    MarketHoursInterface,
    MarketDataInterface,
    StockTradingInterface,
    OptionsTradingInterface,
    OptionLeg,
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce,
    OptionType,
    BrokerConnectionError,
    InvalidOrderError,
    InsufficientFundsError,
    OrderNotFoundError,
    NoPositionError,
    SymbolNotFoundError,
)
from src.trading.brokers.ibkr.config import IBKRConfig
from src.trading.brokers.ibkr.connection import IBKRConnectionManager
from src.trading.brokers.ibkr.contracts import ContractResolver
from src.trading.brokers.ibkr.data_download import IBKRDataProvider
from src.trading.brokers.ibkr.pacing import PacingManager
from src.trading.brokers.ibkr.symbols import from_ibkr_symbol
from src.utils.logger import get_logger

logger = get_logger(__name__)

ET = pytz.timezone('America/New_York')


class IBKRBroker(
    AccountInterface,
    MarketHoursInterface,
    MarketDataInterface,
    StockTradingInterface,
    OptionsTradingInterface,
):
    """
    Interactive Brokers implementation.

    First Homeguard broker to implement OptionsTradingInterface.
    """

    def __init__(self, config: Optional[IBKRConfig] = None):
        self._config = config or IBKRConfig()
        self._conn: Optional[IBKRConnectionManager] = None
        self._resolver: Optional[ContractResolver] = None
        self._data: Optional[IBKRDataProvider] = None

    def start(self) -> None:
        """Initialize connection and sub-components."""
        self._conn = IBKRConnectionManager(self._config)
        self._conn.start()
        self._resolver = ContractResolver(self._conn)
        self._data = IBKRDataProvider(self._conn, self._resolver)
        logger.info(f"[IBKR] Broker started ({self._config.gateway_type})")

    def stop(self) -> None:
        """Graceful shutdown."""
        if self._conn:
            self._conn.stop()
        logger.info("[IBKR] Broker stopped")

    @property
    def data_provider(self) -> IBKRDataProvider:
        assert self._data is not None, "Call start() first"
        return self._data

    # ==================== AccountInterface ====================

    def get_account(self) -> Dict:
        """Get account information in standardized format."""
        try:
            ib = self._conn.ib
            account_id = ib.managedAccounts()[0] if ib.managedAccounts() else ""

            summary = self._conn.run_sync(self._fetch_account(account_id))
            return summary
        except Exception as e:
            logger.error(f"[IBKR] Failed to get account: {e}")
            raise BrokerConnectionError(f"IBKR API error: {e}")

    async def _fetch_account(self, account_id: str) -> Dict:
        # ib.accountSummaryAsync() is idempotent: it subscribes via
        # reqAccountSummary only if the internal cache is empty, then returns
        # cached values on every subsequent call. Calling reqAccountSummaryAsync
        # explicitly here (and every get_account tick) opened a fresh server-side
        # subscription per call, tripping IBKR error 322 "Maximum number of
        # account summary requests exceeded" after the first one.
        ib = self._conn.ib
        values = await ib.accountSummaryAsync(account_id)
        result = {
            'account_id': account_id,
            'buying_power': 0.0,
            'cash': 0.0,
            'portfolio_value': 0.0,
            'equity': 0.0,
            'currency': 'USD',
        }

        tag_map = {
            'BuyingPower': 'buying_power',
            'TotalCashValue': 'cash',
            'NetLiquidation': 'portfolio_value',
            'EquityWithLoanValue': 'equity',
            'Currency': 'currency',
        }

        for item in values:
            field = tag_map.get(item.tag)
            if field:
                try:
                    result[field] = item.value if field == 'currency' else float(item.value)
                except (ValueError, TypeError):
                    result[field] = item.value

        return result

    def test_connection(self) -> bool:
        """Test broker connection."""
        try:
            if self._conn and self._conn.is_connected:
                logger.success("[IBKR] Connection successful")
                return True
            return False
        except Exception:
            return False

    # ==================== MarketHoursInterface ====================

    def is_market_open(self) -> bool:
        """Check if US equity market is currently open."""
        try:
            from src.utils.timezone import tz
            now = tz.now()
            weekday = now.weekday()
            if weekday >= 5:
                return False
            from datetime import time
            market_open = time(9, 30)
            market_close = time(16, 0)
            return market_open <= now.time() <= market_close
        except Exception as e:
            logger.error(f"[IBKR] Failed to check market hours: {e}")
            raise BrokerConnectionError(f"IBKR API error: {e}")

    def get_market_hours(self, date: datetime) -> Tuple[datetime, datetime]:
        """Get market hours for a specific date. Returns (open, close) tuple."""
        try:
            target = date.date() if hasattr(date, 'date') else date
            # Standard US equity hours
            open_dt = datetime.combine(target, datetime.min.time().replace(hour=9, minute=30))
            close_dt = datetime.combine(target, datetime.min.time().replace(hour=16))

            open_dt = ET.localize(open_dt)
            close_dt = ET.localize(close_dt)

            return (open_dt, close_dt)
        except Exception as e:
            logger.error(f"[IBKR] Failed to get market hours: {e}")
            raise BrokerConnectionError(f"IBKR API error: {e}")

    # ==================== MarketDataInterface ====================

    def get_latest_quote(self, symbol: str) -> Dict:
        """Get latest bid/ask quote in standardized format."""
        try:
            contract = self._resolver.resolve_stock(symbol)
            ticker = self._conn.run_sync(self._snapshot(contract), timeout=10)

            def safe(val):
                return float(val) if val == val else 0.0

            return {
                'symbol': symbol,
                'bid': safe(ticker.bid),
                'ask': safe(ticker.ask),
                'bid_size': int(safe(ticker.bidSize)),
                'ask_size': int(safe(ticker.askSize)),
                'timestamp': datetime.now(ET),
            }
        except SymbolNotFoundError:
            raise
        except Exception as e:
            logger.error(f"[IBKR] Failed to get quote for {symbol}: {e}")
            raise BrokerConnectionError(f"IBKR API error: {e}")

    def get_latest_trade(self, symbol: str) -> Dict:
        """Get latest trade in standardized format."""
        try:
            contract = self._resolver.resolve_stock(symbol)
            ticker = self._conn.run_sync(self._snapshot(contract), timeout=10)

            last = ticker.last if ticker.last == ticker.last else 0.0

            return {
                'symbol': symbol,
                'price': float(last),
                'size': int(ticker.lastSize) if ticker.lastSize == ticker.lastSize else 0,
                'timestamp': datetime.now(ET),
            }
        except SymbolNotFoundError:
            raise
        except Exception as e:
            logger.error(f"[IBKR] Failed to get trade for {symbol}: {e}")
            raise BrokerConnectionError(f"IBKR API error: {e}")

    def get_bars(
        self,
        symbols: List[str],
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        Get historical bars with MultiIndex (symbol, timestamp).

        Matches AlpacaBroker.get_bars() contract exactly.
        """
        try:
            frames = []
            for symbol in symbols:
                df = self._data.get_historical_bars(symbol, start, end, timeframe)
                if df is not None and not df.empty:
                    df['symbol'] = symbol
                    frames.append(df)

            if not frames:
                return pd.DataFrame()

            combined = pd.concat(frames)
            combined = combined.reset_index()
            combined = combined.set_index(['symbol', combined.columns[0]])
            combined.index.names = ['symbol', 'timestamp']

            return self._ensure_et_timezone(combined)
        except Exception as e:
            logger.error(f"[IBKR] Failed to get bars: {e}")
            raise BrokerConnectionError(f"IBKR API error: {e}")

    def get_historical_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = '1D',
    ) -> pd.DataFrame:
        """Single-symbol convenience method (used by strategy adapters)."""
        df = self.get_bars([symbol], timeframe, start, end)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level='symbol')
        return df

    # ==================== StockTradingInterface ====================

    def get_stock_positions(self) -> List[Dict]:
        """Get all stock positions in standardized format.

        Uses ib.portfolio() rather than ib.positions() so each position
        carries marketPrice / marketValue / unrealizedPNL populated by IBKR's
        internal market-data feed. ib.positions() returns bare contract+qty
        only, which previously surfaced as None unrealized_pnl/market_value
        on every position -- breaking Grafana panels and metric attribution.
        """
        try:
            positions = self._conn.ib.portfolio()
            result = []
            for pos in positions:
                if pos.contract.secType != 'STK':
                    continue
                result.append(self._translate_stock_position(pos))
            return result
        except Exception as e:
            logger.error(f"[IBKR] Failed to get positions: {e}")
            raise BrokerConnectionError(f"IBKR API error: {e}")

    def get_stock_position(self, symbol: str) -> Optional[Dict]:
        """Get specific stock position by symbol."""
        positions = self.get_stock_positions()
        return next((p for p in positions if p['symbol'] == symbol), None)

    def place_stock_order(
        self,
        symbol: str,
        quantity: int,
        side: OrderSide,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
        **kwargs,
    ) -> Dict:
        """Place a stock order. Returns standardized order dict."""
        try:
            contract = self._resolver.resolve_stock(symbol)
            ibkr_action = self._side_to_ibkr(side)
            order = self._build_order(
                ibkr_action, quantity, order_type, limit_price, stop_price, time_in_force,
            )

            trade = self._conn.run_sync(self._place(contract, order))

            logger.info(
                f"[IBKR] Order placed: {symbol} {side.value} {quantity} @ {order_type.value}"
            )
            return self._translate_order(trade)
        except InvalidOrderError:
            raise
        except Exception as e:
            logger.error(f"[IBKR] Failed to place order: {e}")
            if "insufficient" in str(e).lower():
                raise InsufficientFundsError(f"Insufficient funds: {e}")
            raise BrokerConnectionError(f"IBKR API error: {e}")

    def close_stock_position(self, symbol: str, quantity: Optional[int] = None) -> Dict:
        """Close a stock position."""
        position = self.get_stock_position(symbol)
        if position is None:
            raise NoPositionError(f"No position for {symbol}")

        qty_to_close = quantity if quantity is not None else abs(position['quantity'])
        close_side = OrderSide.SELL if position['quantity'] > 0 else OrderSide.BUY

        return self.place_stock_order(symbol, qty_to_close, close_side, OrderType.MARKET)

    def close_all_stock_positions(self) -> List[Dict]:
        """Close all open stock positions."""
        positions = self.get_stock_positions()
        results = []
        for pos in positions:
            try:
                result = self.close_stock_position(pos['symbol'])
                results.append(result)
            except Exception as e:
                logger.error(f"[IBKR] Failed to close {pos['symbol']}: {e}")
        return results

    # ==================== OrderManagementInterface ====================

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        try:
            trades = self._conn.ib.openTrades()
            for trade in trades:
                if str(trade.order.orderId) == order_id:
                    self._conn.ib.cancelOrder(trade.order)
                    logger.info(f"[IBKR] Cancelled order {order_id}")
                    return True
            raise OrderNotFoundError(f"Order {order_id} not found")
        except OrderNotFoundError:
            raise
        except Exception as e:
            logger.error(f"[IBKR] Failed to cancel order {order_id}: {e}")
            return False

    def get_order(self, order_id: str) -> Dict:
        """Get order details by ID."""
        all_trades = self._conn.ib.trades()
        for trade in all_trades:
            if str(trade.order.orderId) == order_id:
                return self._translate_order(trade)
        raise OrderNotFoundError(f"Order {order_id} not found")

    def get_orders(
        self,
        status: Optional[OrderStatus] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict]:
        """Get orders with optional filters.

        Combines current session trades with completed orders from IBKR
        so the full order history is visible regardless of connection lifecycle.
        """
        try:
            # Current session orders (open + cancelled this session)
            trades = self._conn.ib.trades()
            seen_ids = set()
            result = []
            for t in trades:
                order_dict = self._translate_order(t)
                result.append(order_dict)
                seen_ids.add(order_dict['order_id'])

            # Completed orders from previous sessions
            async def _fetch_completed():
                return await self._conn.ib.reqCompletedOrdersAsync(apiOnly=False)

            try:
                completed = self._conn.run_sync(_fetch_completed())
                for t in completed:
                    order_dict = self._translate_order(t)
                    if order_dict['order_id'] not in seen_ids:
                        result.append(order_dict)
                        seen_ids.add(order_dict['order_id'])
            except Exception as e:
                logger.warning(f"[IBKR] Could not fetch completed orders: {e}")

            if status is not None:
                result = [o for o in result if o['status'] == status.value]

            return result
        except Exception as e:
            logger.error(f"[IBKR] Failed to get orders: {e}")
            raise BrokerConnectionError(f"IBKR API error: {e}")

    # ==================== OptionsTradingInterface ====================

    def get_options_chain(
        self,
        underlying: str,
        expiration: Optional[date] = None,
    ) -> List[Dict]:
        """Get options chain for an underlying symbol."""
        exp_str = expiration.strftime("%Y%m%d") if expiration else None
        return self._data.get_options_chain_data(underlying, exp_str)

    def get_options_positions(self) -> List[Dict]:
        """Get all current options positions.

        Uses ib.portfolio() (PortfolioItem) for live market_value /
        unrealized_pnl. See get_stock_positions for details.
        """
        try:
            positions = self._conn.ib.portfolio()
            result = []
            for pos in positions:
                if pos.contract.secType != 'OPT':
                    continue
                result.append(self._translate_option_position(pos))
            return result
        except Exception as e:
            logger.error(f"[IBKR] Failed to get options positions: {e}")
            raise BrokerConnectionError(f"IBKR API error: {e}")

    def get_options_position(self, contract_id: str) -> Optional[Dict]:
        """Get specific options position by contract ID."""
        positions = self.get_options_positions()
        return next((p for p in positions if p['contract_id'] == contract_id), None)

    def place_options_order(
        self,
        underlying: str,
        expiration: date,
        strike: float,
        option_type: OptionType,
        quantity: int,
        side: OrderSide,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
        **kwargs,
    ) -> Dict:
        """Place a single-leg options order."""
        try:
            right = 'C' if option_type == OptionType.CALL else 'P'
            contract = self._resolver.resolve_option(
                underlying, expiration.strftime("%Y%m%d"), strike, right,
            )
            ibkr_action = self._side_to_ibkr(side)
            order = self._build_order(
                ibkr_action, quantity, order_type, limit_price, None, time_in_force,
            )

            trade = self._conn.run_sync(self._place(contract, order))

            logger.info(
                f"[IBKR] Options order: {side.value} {quantity} "
                f"{underlying} {expiration} {strike}{right}"
            )
            return self._translate_order(trade)
        except Exception as e:
            logger.error(f"[IBKR] Failed to place options order: {e}")
            raise BrokerConnectionError(f"IBKR API error: {e}")

    def place_multi_leg_order(
        self,
        legs: List[OptionLeg],
        order_type: OrderType = OrderType.LIMIT,
        limit_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
        **kwargs,
    ) -> Dict:
        """Place a multi-leg options order (spreads, collars, etc.)."""
        try:
            combo_legs = []
            for leg in legs:
                right = 'C' if leg.option_type == OptionType.CALL else 'P'
                opt_contract = self._resolver.resolve_option(
                    leg.underlying,
                    leg.expiration.strftime("%Y%m%d"),
                    leg.strike,
                    right,
                )
                ibkr_leg = ComboLeg(
                    conId=opt_contract.conId,
                    ratio=leg.quantity,
                    action=self._side_to_ibkr(leg.side),
                    exchange=opt_contract.exchange or "SMART",
                )
                combo_legs.append(ibkr_leg)

            bag = Contract()
            bag.symbol = legs[0].underlying
            bag.secType = "BAG"
            bag.exchange = "SMART"
            bag.currency = "USD"
            bag.comboLegs = combo_legs

            order = self._build_order(
                'BUY', 1, order_type, limit_price, None, time_in_force,
            )

            trade = self._conn.run_sync(self._place(bag, order))

            logger.info(f"[IBKR] Multi-leg order: {len(legs)} legs")
            return self._translate_order(trade)
        except Exception as e:
            logger.error(f"[IBKR] Failed to place multi-leg order: {e}")
            raise BrokerConnectionError(f"IBKR API error: {e}")

    def get_greeks(self, contract_id: str) -> Dict:
        """Get option Greeks for a specific contract."""
        try:
            # Parse contract_id: "AAPL_20260417_190.0_C"
            parts = contract_id.split('_')
            if len(parts) != 4:
                raise InvalidOrderError(f"Invalid contract_id format: {contract_id}")

            underlying, exp, strike, right = parts
            contract = self._resolver.resolve_option(underlying, exp, float(strike), right)

            ticker = self._conn.run_sync(self._snapshot(contract), timeout=10)

            greeks = ticker.modelGreeks if hasattr(ticker, 'modelGreeks') and ticker.modelGreeks else None

            return {
                'delta': greeks.delta if greeks else None,
                'gamma': greeks.gamma if greeks else None,
                'theta': greeks.theta if greeks else None,
                'vega': greeks.vega if greeks else None,
                'rho': None,  # Not always available from IBKR
                'implied_volatility': greeks.impliedVol if greeks else None,
            }
        except Exception as e:
            logger.error(f"[IBKR] Failed to get greeks for {contract_id}: {e}")
            raise BrokerConnectionError(f"IBKR API error: {e}")

    def close_options_position(
        self, contract_id: str, quantity: Optional[int] = None,
    ) -> Dict:
        """Close an options position."""
        position = self.get_options_position(contract_id)
        if position is None:
            raise NoPositionError(f"No options position for {contract_id}")

        qty = quantity if quantity is not None else abs(position['quantity'])
        close_side = OrderSide.SELL if position['quantity'] > 0 else OrderSide.BUY

        parts = contract_id.split('_')
        opt_type = OptionType.CALL if parts[3] == 'C' else OptionType.PUT
        exp = datetime.strptime(parts[1], "%Y%m%d").date()

        return self.place_options_order(
            underlying=parts[0],
            expiration=exp,
            strike=float(parts[2]),
            option_type=opt_type,
            quantity=qty,
            side=close_side,
            order_type=OrderType.MARKET,
        )

    def close_all_options_positions(self) -> List[Dict]:
        """Close all open options positions."""
        positions = self.get_options_positions()
        results = []
        for pos in positions:
            try:
                result = self.close_options_position(pos['contract_id'])
                results.append(result)
            except Exception as e:
                logger.error(f"[IBKR] Failed to close {pos['contract_id']}: {e}")
        return results

    # ==================== Internal Helpers ====================

    @staticmethod
    def _side_to_ibkr(side: OrderSide) -> str:
        return 'BUY' if side == OrderSide.BUY else 'SELL'

    @staticmethod
    def _tif_to_ibkr(tif: TimeInForce) -> str:
        return {
            TimeInForce.DAY: 'DAY', TimeInForce.GTC: 'GTC',
            TimeInForce.IOC: 'IOC', TimeInForce.FOK: 'FOK',
        }.get(tif, 'DAY')

    def _build_order(self, action, qty, order_type, limit_price, stop_price, tif):
        if order_type == OrderType.MARKET:
            order = MarketOrder(action, qty)
        elif order_type == OrderType.LIMIT:
            if limit_price is None:
                raise InvalidOrderError("Limit price required for LIMIT orders")
            order = LimitOrder(action, qty, limit_price)
        elif order_type == OrderType.STOP:
            if stop_price is None:
                raise InvalidOrderError("Stop price required for STOP orders")
            order = StopOrder(action, qty, stop_price)
        elif order_type == OrderType.STOP_LIMIT:
            if stop_price is None or limit_price is None:
                raise InvalidOrderError("Both prices required for STOP_LIMIT orders")
            order = StopLimitOrder(action, qty, stop_price, limit_price)
        else:
            raise InvalidOrderError(f"Unsupported order type: {order_type}")

        order.tif = self._tif_to_ibkr(tif)
        return order

    async def _place(self, contract, order):
        trade = self._conn.ib.placeOrder(contract, order)
        await asyncio.sleep(0.5)
        return trade

    async def _snapshot(self, contract):
        ticker = self._conn.ib.reqMktData(contract, '', True, False)
        await asyncio.sleep(2)
        return ticker

    def _translate_order(self, trade) -> Dict:
        """Translate ib_async Trade to standardized order dict."""
        order = trade.order
        status = trade.orderStatus

        status_map = {
            'PendingSubmit': 'pending', 'PreSubmitted': 'pending',
            'Submitted': 'pending', 'Filled': 'filled',
            'Cancelled': 'cancelled', 'ApiCancelled': 'cancelled',
            'Inactive': 'rejected',
        }

        return {
            'order_id': str(order.orderId),
            'symbol': from_ibkr_symbol(trade.contract.symbol),
            'quantity': int(order.totalQuantity),
            'side': 'buy' if order.action == 'BUY' else 'sell',
            'order_type': order.orderType.lower() if order.orderType else 'market',
            'status': status_map.get(status.status, status.status.lower()),
            'limit_price': float(order.lmtPrice) if order.lmtPrice else None,
            'stop_price': float(order.auxPrice) if order.auxPrice else None,
            'created_at': None,
            'filled_qty': int(status.filled) if status.filled else 0,
            'filled_avg_price': float(status.avgFillPrice) if status.avgFillPrice else None,
        }

    def _translate_stock_position(self, pos) -> Dict:
        """Translate IBKR position to standardized stock position dict.

        Accepts either a `Position` (from ib.positions(): contract + qty +
        avgCost only) or a `PortfolioItem` (from ib.portfolio(): adds
        marketPrice, marketValue, unrealizedPNL, averageCost). Reads the
        market fields via getattr so legacy callers/test fixtures still work.
        """
        qty = int(pos.position)
        # PortfolioItem uses `averageCost`; Position uses `avgCost`.
        avg_cost = float(getattr(pos, 'averageCost', None) or getattr(pos, 'avgCost', 0))

        market_price_raw = getattr(pos, 'marketPrice', None)
        market_value_raw = getattr(pos, 'marketValue', None)
        unrealized_raw = getattr(pos, 'unrealizedPNL', None)

        # IBKR sometimes reports 0.0 placeholders before the first market
        # data tick; treat 0 as "not yet known" and fall back to None so
        # downstream callers can distinguish a real $0 from missing data.
        current_price = float(market_price_raw) if market_price_raw else None
        market_value = float(market_value_raw) if market_value_raw else None
        unrealized = float(unrealized_raw) if unrealized_raw is not None else None
        unrealized_pct = (
            (unrealized / (avg_cost * abs(qty)) * 100.0)
            if unrealized is not None and avg_cost and qty
            else None
        )

        return {
            'symbol': from_ibkr_symbol(pos.contract.symbol),
            'quantity': qty,
            'avg_entry_price': avg_cost,
            'current_price': current_price,
            'market_value': market_value,
            'unrealized_pnl': unrealized,
            'unrealized_pnl_pct': unrealized_pct,
            'side': 'long' if qty > 0 else 'short',
        }

    def _translate_option_position(self, pos) -> Dict:
        """Translate IBKR position to standardized options position dict.

        Accepts either Position or PortfolioItem; PortfolioItem populates
        marketPrice/marketValue/unrealizedPNL via IBKR's market data feed.
        """
        c = pos.contract
        underlying = from_ibkr_symbol(c.symbol)
        contract_id = f"{underlying}_{c.lastTradeDateOrContractMonth}_{c.strike}_{c.right}"
        opt_type = 'call' if c.right == 'C' else 'put'

        avg_cost = float(getattr(pos, 'averageCost', None) or getattr(pos, 'avgCost', 0))
        market_price_raw = getattr(pos, 'marketPrice', None)
        market_value_raw = getattr(pos, 'marketValue', None)
        unrealized_raw = getattr(pos, 'unrealizedPNL', None)

        return {
            'contract_id': contract_id,
            'underlying': underlying,
            'expiration': c.lastTradeDateOrContractMonth,
            'strike': float(c.strike),
            'option_type': opt_type,
            'quantity': int(pos.position),
            'avg_entry_price': avg_cost,
            'current_price': float(market_price_raw) if market_price_raw else None,
            'market_value': float(market_value_raw) if market_value_raw else None,
            'unrealized_pnl': float(unrealized_raw) if unrealized_raw is not None else None,
            'delta': None,
            'gamma': None,
            'theta': None,
            'vega': None,
        }

    def _ensure_et_timezone(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert DataFrame index to Eastern Time (matching AlpacaBroker)."""
        if df.empty:
            return df

        if isinstance(df.index, pd.MultiIndex):
            ts_level = df.index.get_level_values(1)
            if hasattr(ts_level, 'tz') and ts_level.tz is not None:
                new_ts = ts_level.tz_convert(ET)
                df.index = pd.MultiIndex.from_arrays(
                    [df.index.get_level_values(0), new_ts],
                    names=df.index.names,
                )
            elif hasattr(ts_level, 'tz_localize'):
                new_ts = ts_level.tz_localize('UTC').tz_convert(ET)
                df.index = pd.MultiIndex.from_arrays(
                    [df.index.get_level_values(0), new_ts],
                    names=df.index.names,
                )
        else:
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_convert(ET)
            elif hasattr(df.index, 'tz_localize'):
                df.index = df.index.tz_localize('UTC').tz_convert(ET)

        return df
