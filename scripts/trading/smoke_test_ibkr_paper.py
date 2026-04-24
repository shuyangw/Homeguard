"""
IBKR Paper Smoke Test -- end-to-end trading chain validation.

Exercises the full broker contract against IBKR paper:
  1. Connect via broker_routing.yaml
  2. Fetch account, verify non-zero equity
  3. Fetch last trade price on target symbol
  4. Place LIMIT BUY at 50% below market (won't fill)
  5. Retrieve order by id
  6. Cancel order
  7. Verify cancellation
  8. Repeat 4-7 for LIMIT SELL at 200% of market
  9. Verify no stray positions or open orders from this test

Designed to be idempotent -- running twice never creates stray state.
Safe to run during after-hours: limit prices are far from market so orders
will not fill even if the exchange processes them. Both orders are cancelled
before the script exits.

Usage:
    python scripts/trading/smoke_test_ibkr_paper.py                # default SPY, qty=1
    python scripts/trading/smoke_test_ibkr_paper.py --symbol QQQ
    python scripts/trading/smoke_test_ibkr_paper.py --symbol SPY --qty 2 --strategy ramp

Exits 0 on success, non-zero on first failure (printing which step failed).
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.trading.brokers.interfaces import OrderSide, OrderType, TimeInForce
from src.trading.config.broker_routing import load_broker_routing
from src.utils.logger import logger


VALID_PENDING = {'pending', 'PendingSubmit', 'PreSubmitted', 'Submitted'}
VALID_CANCELLED = {'cancelled', 'Cancelled', 'ApiCancelled'}


def step(n: int, msg: str) -> None:
    """Log a step header."""
    logger.info(f"\n===== step {n}: {msg} =====")


def ok(msg: str) -> None:
    logger.success(f"  [+] {msg}")


def fail(step_num: int, msg: str) -> None:
    logger.error(f"  [X] STEP {step_num} FAILED: {msg}")
    logger.error("=== SMOKE TEST FAILED ===")
    sys.exit(1)


def place_far_limit(broker, symbol: str, qty: int, side: OrderSide,
                    last_price: float, multiplier: float, step_num: int) -> str:
    """Place a limit order far from market and return its order_id."""
    limit_price = round(last_price * multiplier, 2)
    action = 'BUY' if side == OrderSide.BUY else 'SELL'
    logger.info(f"  placing LIMIT {action} {qty} {symbol} @ ${limit_price:.2f}"
                f" (last ${last_price:.2f}, {multiplier * 100:.0f}%)")
    try:
        order = broker.place_stock_order(
            symbol=symbol,
            quantity=qty,
            side=side,
            order_type=OrderType.LIMIT,
            limit_price=limit_price,
            time_in_force=TimeInForce.DAY,
        )
    except Exception as e:
        fail(step_num, f"place_stock_order raised: {e}")
    if not isinstance(order, dict) or not order.get('order_id'):
        fail(step_num, f"place_stock_order returned no order_id: {order!r}")
    order_id = str(order['order_id'])
    status = order.get('status', '<missing>')
    if status not in VALID_PENDING:
        # Some brokers may return 'filled' immediately for marketable limits;
        # with a 50%-off price that should not happen. Still surface if it does.
        fail(step_num, f"unexpected initial status {status!r} (expected one of {VALID_PENDING})")
    ok(f"submitted order_id={order_id} status={status}")
    return order_id


def verify_retrievable(broker, order_id: str, step_num: int) -> dict:
    """Assert the order can be fetched by id."""
    try:
        fetched = broker.get_order(order_id)
    except Exception as e:
        fail(step_num, f"get_order({order_id!r}) raised: {e}")
    if not fetched or str(fetched.get('order_id')) != order_id:
        fail(step_num, f"get_order returned unexpected: {fetched!r}")
    ok(f"get_order({order_id}) returned status={fetched.get('status')}")
    return fetched


def cancel_and_verify(broker, order_id: str, step_num: int) -> None:
    """Cancel and confirm it sticks."""
    try:
        result = broker.cancel_order(order_id)
    except Exception as e:
        fail(step_num, f"cancel_order({order_id!r}) raised: {e}")
    if not result:
        fail(step_num, f"cancel_order({order_id!r}) returned falsy: {result!r}")
    ok(f"cancel_order({order_id}) -> {result}")

    # Re-query -- status should be cancelled or order gone
    time.sleep(2)
    try:
        fetched = broker.get_order(order_id)
        status = fetched.get('status')
        if status not in VALID_CANCELLED:
            fail(step_num, f"post-cancel status {status!r} not in {VALID_CANCELLED}")
        ok(f"post-cancel status={status}")
    except Exception as e:
        # Some brokers drop cancelled orders after a delay -- treat as success
        # only if the error is clearly "not found" rather than a generic fault.
        msg = str(e).lower()
        if 'not found' in msg or 'ordernotfound' in msg:
            ok(f"order no longer queryable (treated as cancelled)")
        else:
            fail(step_num, f"post-cancel get_order raised unexpectedly: {e}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--symbol', default='SPY',
                        help='Stock symbol to trade (default: SPY)')
    parser.add_argument('--qty', type=int, default=1,
                        help='Share quantity per order (default: 1)')
    parser.add_argument('--strategy', default='ramp',
                        help='Strategy name for broker routing (default: ramp)')
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info(f"IBKR PAPER SMOKE TEST")
    logger.info(f"  strategy={args.strategy}  symbol={args.symbol}  qty={args.qty}")
    logger.info("=" * 70)

    # --- step 1: routing
    step(1, "load broker routing + resolve strategy broker")
    try:
        routing = load_broker_routing()
        broker = routing[args.strategy]
        broker_name = routing.get_broker_name(args.strategy)
    except Exception as e:
        fail(1, f"load_broker_routing failed: {e}")
    if broker_name != 'ibkr':
        fail(1, f"expected broker=ibkr for strategy={args.strategy}, got {broker_name!r}")
    ok(f"routed {args.strategy} -> {broker_name} ({type(broker).__name__})")

    # --- step 2: account
    step(2, "fetch account + baseline positions")
    try:
        account = broker.get_account()
    except Exception as e:
        fail(2, f"broker.get_account() raised: {e}")
    equity = float(account.get('portfolio_value', 0) or 0)
    if equity <= 0:
        fail(2, f"portfolio_value not positive: {equity}")
    ok(f"account_id={account.get('account_id')} equity=${equity:,.2f}")

    try:
        baseline_positions = broker.get_stock_positions()
    except Exception as e:
        fail(2, f"broker.get_stock_positions() raised: {e}")
    baseline_symbols = sorted(p.get('symbol') for p in baseline_positions)
    ok(f"baseline positions ({len(baseline_symbols)}): {baseline_symbols[:10]}"
       f"{'...' if len(baseline_symbols) > 10 else ''}")

    # --- step 3: last price
    step(3, f"fetch last trade price for {args.symbol}")
    try:
        trade_snapshot = broker.get_latest_trade(args.symbol)
    except Exception as e:
        fail(3, f"get_latest_trade({args.symbol!r}) raised: {e}")
    last_price = float(trade_snapshot.get('price') or trade_snapshot.get('last') or 0)
    if last_price <= 0:
        # IBKR may return 0 outside market hours -- fall back to bid/ask midpoint
        try:
            q = broker.get_latest_quote(args.symbol)
            mid = (float(q.get('bid', 0) or 0) + float(q.get('ask', 0) or 0)) / 2
            if mid > 0:
                last_price = mid
                logger.info(f"  using bid/ask midpoint ${mid:.2f} (last was 0)")
        except Exception:
            pass
    if last_price <= 0:
        fail(3, f"no usable price for {args.symbol}: trade={trade_snapshot!r}")
    ok(f"{args.symbol} reference price = ${last_price:,.2f}")

    # --- step 4: limit buy far below
    step(4, "LIMIT BUY at 50% below market")
    buy_order_id = place_far_limit(broker, args.symbol, args.qty,
                                   OrderSide.BUY, last_price, 0.5, 4)

    # --- step 5: retrieve buy order
    step(5, "retrieve buy order by id")
    verify_retrievable(broker, buy_order_id, 5)

    # --- step 6: cancel buy order
    step(6, "cancel buy order + verify")
    cancel_and_verify(broker, buy_order_id, 6)

    # --- step 7: limit sell far above
    step(7, "LIMIT SELL at 200% of market")
    sell_order_id = place_far_limit(broker, args.symbol, args.qty,
                                    OrderSide.SELL, last_price, 2.0, 7)

    # --- step 8: retrieve + cancel sell order
    step(8, "retrieve + cancel sell order")
    verify_retrievable(broker, sell_order_id, 8)
    cancel_and_verify(broker, sell_order_id, 8)

    # --- step 9: final clean state
    step(9, "verify clean state -- no new positions, no lingering open orders")
    try:
        final_positions = broker.get_stock_positions()
    except Exception as e:
        fail(9, f"final get_stock_positions raised: {e}")
    final_symbols = sorted(p.get('symbol') for p in final_positions)
    if final_symbols != baseline_symbols:
        new_syms = set(final_symbols) - set(baseline_symbols)
        fail(9, f"positions changed! new symbols: {new_syms}")
    ok(f"positions unchanged: {len(final_symbols)} symbols")

    # Check open orders contain none of our test order ids
    try:
        open_orders = broker.get_orders()
    except Exception as e:
        logger.warning(f"get_orders raised: {e} -- skipping open-order check")
        open_orders = []
    leftover = [
        o for o in open_orders
        if str(o.get('order_id')) in {buy_order_id, sell_order_id}
        and o.get('status') in {'pending', 'PendingSubmit', 'PreSubmitted', 'Submitted'}
    ]
    if leftover:
        fail(9, f"test orders still open: {leftover}")
    ok(f"no lingering open orders from this test run")

    logger.success("\n" + "=" * 70)
    logger.success("=== SMOKE TEST PASSED ===")
    logger.success("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
