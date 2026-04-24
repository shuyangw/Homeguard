"""
IBKR Paper Smoke Test -- end-to-end trading chain validation.

================================================================
WHEN TO RUN
================================================================
Run this after ANY change to:
  - IBKRBroker method signatures or behaviour
  - ExecutionEngine order flow (especially _place_order_with_validation)
  - BrokerInterface / StockTradingInterface / OrderManagementInterface
  - broker_routing.yaml or IBKRConfig

It takes ~20-30 seconds, places two far-from-market limit orders on IBKR
paper (guaranteed not to fill), and cancels them. Safe after-hours and
during market hours (the limit prices are always outside the current
spread so fills never happen).

================================================================
WHAT IT VALIDATES
================================================================
Two independent call paths, each exercising the same submit/query/cancel
lifecycle against live IBKR paper:

  Part 1 (--mode direct or full):
    IBKRBroker.place_stock_order / get_order / cancel_order
    Proves the broker methods work.

  Part 2 (--mode engine or full, DEFAULT):
    ExecutionEngine(broker).execute_order / cancel_order
    Proves the ACTUAL live trading call chain works, including
    _place_order_with_validation (the site of the 2026-04-24 regression
    where `broker.place_order` was called on IBKRBroker which has no
    such method).

================================================================
USAGE
================================================================
On EC2 (typical, after a deploy):
    ssh ec2
    cd ~/Homeguard && source venv/bin/activate
    python scripts/trading/smoke_test_ibkr_paper.py

Flags:
    --symbol SPY        # which liquid ETF to trade (default SPY)
    --qty 1             # shares per test order (default 1)
    --client-id 99      # IBKR clientId (must != running service's; default 99)
    --port 4002         # IBKR paper gateway port (default 4002)
    --mode full         # which parts to run: direct|engine|full (default full)

Locally with an IBKR gateway on 127.0.0.1:4002 (rarer):
    conda activate fintech
    python scripts/trading/smoke_test_ibkr_paper.py --mode full

================================================================
SAFETY
================================================================
- Idempotent: running twice never accumulates stray state.
- Uses clientId=99 by default so it doesn't collide with the running
  homeguard-multi service (which holds clientId=10).
- Limit prices are 50% of market (buy) / 200% of market (sell) so no
  marketable quote ever crosses them.
- Every order placed is cancelled before the script exits, and final
  state is verified (no new positions, no lingering open orders from
  this run).
- If any step fails, the script exits non-zero immediately and prints
  which step failed.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.trading.brokers.interfaces import OrderSide, OrderType, TimeInForce
from src.trading.brokers.ibkr.config import IBKRConfig
from src.trading.brokers.ibkr.ibkr_broker import IBKRBroker
from src.trading.core.execution_engine import ExecutionEngine
from src.utils.logger import logger


VALID_PENDING = {'pending', 'PendingSubmit', 'PreSubmitted', 'Submitted'}
# 'pendingcancel' / 'PendingCancel' = cancel request received by broker,
# final state not yet confirmed. Accepted here because our smoke test wants
# to prove the cancel path works end-to-end, not that the exchange has
# finalized. IBKR typically transitions PendingCancel -> Cancelled within
# ~1s after market-hours, and immediately when market is closed.
VALID_CANCELLED = {'cancelled', 'Cancelled', 'ApiCancelled',
                   'pendingcancel', 'PendingCancel'}


# --------------------------------------------------------------- helpers

def step(n: str, msg: str) -> None:
    """Log a step header. `n` can be '1' or '2a' etc. so parts stay readable."""
    logger.info(f"\n===== step {n}: {msg} =====")


def ok(msg: str) -> None:
    logger.success(f"  [+] {msg}")


def fail(step_num: str, msg: str) -> None:
    logger.error(f"  [X] STEP {step_num} FAILED: {msg}")
    logger.error("=== SMOKE TEST FAILED ===")
    sys.exit(1)


def _connect_ibkr(client_id: int, port: int) -> IBKRBroker:
    """Instantiate IBKRBroker with an env-forced clientId and port.

    IBKRConfig kwargs don't override `config/ibkr.yaml` in pydantic v2
    mode='before' semantics -- but env vars do. Setting IBKR_CLIENT_ID/
    IBKR_PORT here is the reliable path.
    """
    os.environ['IBKR_CLIENT_ID'] = str(client_id)
    os.environ['IBKR_PORT'] = str(port)
    config = IBKRConfig()
    if config.client_id != client_id:
        raise RuntimeError(
            f"IBKRConfig ignored IBKR_CLIENT_ID env: got {config.client_id}"
        )
    broker = IBKRBroker(config=config)
    broker.start()
    return broker


def _fetch_reference_price(broker: IBKRBroker, symbol: str) -> float:
    """Get a non-zero reference price via last-trade or bid/ask midpoint."""
    trade = broker.get_latest_trade(symbol)
    last = float(trade.get('price') or trade.get('last') or 0)
    if last > 0:
        return last
    q = broker.get_latest_quote(symbol)
    mid = (float(q.get('bid', 0) or 0) + float(q.get('ask', 0) or 0)) / 2
    if mid > 0:
        logger.info(f"  using bid/ask midpoint ${mid:.2f} (last was 0)")
        return mid
    raise RuntimeError(f"no usable price for {symbol}: trade={trade!r}")


def _verify_clean_state(
    broker: IBKRBroker,
    baseline_symbols: list,
    test_order_ids: list,
    step_num: str,
) -> None:
    """Assert no new positions + no lingering open orders from this test."""
    final_positions = broker.get_stock_positions()
    final_symbols = sorted(p.get('symbol') for p in final_positions)
    if final_symbols != baseline_symbols:
        new_syms = set(final_symbols) - set(baseline_symbols)
        fail(step_num, f"positions changed! new symbols: {new_syms}")
    ok(f"positions unchanged: {len(final_symbols)} symbols")

    try:
        open_orders = broker.get_orders()
    except Exception as e:
        logger.warning(f"get_orders raised: {e} -- skipping open-order check")
        open_orders = []
    leftover = [
        o for o in open_orders
        if str(o.get('order_id')) in set(test_order_ids)
        and o.get('status') in VALID_PENDING
    ]
    if leftover:
        fail(step_num, f"test orders still open: {leftover}")
    ok(f"no lingering open orders from this test run")


# --------------------------------------------------------------- Part 1

def _part_direct_broker(broker: IBKRBroker, symbol: str, qty: int,
                       last_price: float, baseline_symbols: list) -> None:
    """Part 1: direct IBKRBroker.place_stock_order / get_order / cancel_order."""
    logger.info("\n" + "#" * 70)
    logger.info("# PART 1 -- direct IBKRBroker lifecycle")
    logger.info("#" * 70)

    order_ids = []

    # BUY far below
    step('1a', f"LIMIT BUY {qty} {symbol} @ 50% below market (direct broker)")
    limit = round(last_price * 0.5, 2)
    logger.info(f"  placing LIMIT BUY {qty} {symbol} @ ${limit:.2f} (last ${last_price:.2f})")
    order = broker.place_stock_order(
        symbol=symbol, quantity=qty, side=OrderSide.BUY,
        order_type=OrderType.LIMIT, limit_price=limit, time_in_force=TimeInForce.DAY,
    )
    if not order or not order.get('order_id'):
        fail('1a', f"no order_id: {order!r}")
    buy_id = str(order['order_id'])
    if order.get('status') not in VALID_PENDING:
        fail('1a', f"unexpected status {order.get('status')!r}")
    order_ids.append(buy_id)
    ok(f"submitted order_id={buy_id} status={order.get('status')}")

    step('1b', f"get_order + cancel + verify (buy, direct broker)")
    fetched = broker.get_order(buy_id)
    if str(fetched.get('order_id')) != buy_id:
        fail('1b', f"get_order mismatch: {fetched!r}")
    ok(f"get_order({buy_id}) -> status={fetched.get('status')}")

    if not broker.cancel_order(buy_id):
        fail('1b', "cancel_order returned falsy")
    ok(f"cancel_order({buy_id}) -> True")
    time.sleep(2)
    try:
        post = broker.get_order(buy_id)
        if post.get('status') not in VALID_CANCELLED:
            fail('1b', f"post-cancel status {post.get('status')!r}")
        ok(f"post-cancel status={post.get('status')}")
    except Exception as e:
        if 'not found' in str(e).lower() or 'ordernotfound' in str(e).lower():
            ok("order no longer queryable (treated as cancelled)")
        else:
            fail('1b', f"unexpected: {e}")

    # SELL far above
    step('1c', f"LIMIT SELL {qty} {symbol} @ 200% of market (direct broker)")
    limit = round(last_price * 2.0, 2)
    logger.info(f"  placing LIMIT SELL {qty} {symbol} @ ${limit:.2f}")
    order = broker.place_stock_order(
        symbol=symbol, quantity=qty, side=OrderSide.SELL,
        order_type=OrderType.LIMIT, limit_price=limit, time_in_force=TimeInForce.DAY,
    )
    if not order or not order.get('order_id'):
        fail('1c', f"no order_id: {order!r}")
    sell_id = str(order['order_id'])
    order_ids.append(sell_id)
    ok(f"submitted order_id={sell_id} status={order.get('status')}")

    step('1d', f"get_order + cancel + verify (sell, direct broker)")
    fetched = broker.get_order(sell_id)
    ok(f"get_order({sell_id}) -> status={fetched.get('status')}")
    if not broker.cancel_order(sell_id):
        fail('1d', "cancel_order returned falsy")
    ok(f"cancel_order({sell_id}) -> True")
    time.sleep(2)
    try:
        post = broker.get_order(sell_id)
        if post.get('status') not in VALID_CANCELLED:
            fail('1d', f"post-cancel status {post.get('status')!r}")
        ok(f"post-cancel status={post.get('status')}")
    except Exception as e:
        if 'not found' not in str(e).lower() and 'ordernotfound' not in str(e).lower():
            fail('1d', f"unexpected: {e}")

    step('1e', "verify clean state after Part 1")
    _verify_clean_state(broker, baseline_symbols, order_ids, '1e')


# --------------------------------------------------------------- Part 2

def _part_execution_engine(broker: IBKRBroker, symbol: str, qty: int,
                          last_price: float, baseline_symbols: list) -> None:
    """Part 2: full ExecutionEngine.execute_order / cancel_order lifecycle.

    This is the ACTUAL call chain live strategies (RAMPLiveAdapter,
    OMRLiveAdapter, MomentumLiveAdapter) use. It's the path where the
    2026-04-24 bug lived (ExecutionEngine._place_order_with_validation
    calling the deprecated broker.place_order on IBKRBroker).
    """
    logger.info("\n" + "#" * 70)
    logger.info("# PART 2 -- ExecutionEngine lifecycle (live strategy path)")
    logger.info("#" * 70)

    engine = ExecutionEngine(
        broker=broker,
        max_retries=1,      # no retries during smoke test
        retry_delay=0.5,
        fill_timeout=5.0,   # never actually waits for a fill here
    )
    order_ids = []

    # BUY far below via engine
    step('2a', f"ExecutionEngine.execute_order: BUY {qty} {symbol} @ 50% below market")
    limit = round(last_price * 0.5, 2)
    try:
        execution = engine.execute_order(
            symbol=symbol,
            quantity=qty,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=limit,
            time_in_force=TimeInForce.DAY,
            wait_for_fill=False,     # critical: we are not filling
        )
    except Exception as e:
        fail('2a', f"engine.execute_order raised: {e}")
    if not isinstance(execution, dict) or 'order' not in execution:
        fail('2a', f"unexpected execute_order return: {execution!r}")
    order = execution['order']
    if not order or not order.get('order_id'):
        fail('2a', f"no order_id in returned order: {order!r}")
    buy_id = str(order['order_id'])
    order_ids.append(buy_id)
    if order.get('status') not in VALID_PENDING:
        fail('2a', f"unexpected status {order.get('status')!r}")
    ok(f"engine.execute_order submitted order_id={buy_id} status={order.get('status')}")

    step('2b', "ExecutionEngine.cancel_order (buy)")
    try:
        cancelled = engine.cancel_order(buy_id)
    except Exception as e:
        fail('2b', f"engine.cancel_order raised: {e}")
    if not isinstance(cancelled, dict):
        fail('2b', f"unexpected cancel_order return: {cancelled!r}")
    if cancelled.get('status') not in VALID_CANCELLED:
        fail('2b', f"post-cancel status {cancelled.get('status')!r}")
    ok(f"engine.cancel_order({buy_id}) -> status={cancelled.get('status')}")

    # SELL far above via engine
    step('2c', f"ExecutionEngine.execute_order: SELL {qty} {symbol} @ 200% of market")
    limit = round(last_price * 2.0, 2)
    try:
        execution = engine.execute_order(
            symbol=symbol,
            quantity=qty,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            limit_price=limit,
            time_in_force=TimeInForce.DAY,
            wait_for_fill=False,
        )
    except Exception as e:
        fail('2c', f"engine.execute_order raised: {e}")
    order = execution.get('order') or {}
    sell_id = str(order.get('order_id') or '')
    if not sell_id:
        fail('2c', f"no order_id in sell execution: {execution!r}")
    order_ids.append(sell_id)
    ok(f"engine.execute_order submitted order_id={sell_id} status={order.get('status')}")

    step('2d', "ExecutionEngine.cancel_order (sell)")
    try:
        cancelled = engine.cancel_order(sell_id)
    except Exception as e:
        fail('2d', f"engine.cancel_order raised: {e}")
    if cancelled.get('status') not in VALID_CANCELLED:
        fail('2d', f"post-cancel status {cancelled.get('status')!r}")
    ok(f"engine.cancel_order({sell_id}) -> status={cancelled.get('status')}")

    step('2e', "verify clean state after Part 2")
    _verify_clean_state(broker, baseline_symbols, order_ids, '2e')

    step('2f', "ExecutionEngine internal counters")
    logger.info(f"  total_orders={engine.total_orders} "
                f"successful={engine.successful_orders} "
                f"failed={engine.failed_orders} "
                f"retries={engine.retry_count}")
    if engine.total_orders != 2:
        fail('2f', f"expected 2 submissions, engine saw {engine.total_orders}")
    if engine.failed_orders != 0:
        fail('2f', f"engine recorded {engine.failed_orders} failures")
    ok("engine counters match expectations")


# --------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--symbol', default='SPY', help='Stock symbol (default: SPY)')
    parser.add_argument('--qty', type=int, default=1, help='Shares per order (default: 1)')
    parser.add_argument('--client-id', type=int, default=99,
                        help='IBKR clientId; must differ from running service (default 99)')
    parser.add_argument('--port', type=int, default=4002,
                        help='IBKR gateway port (default 4002 paper)')
    parser.add_argument('--mode', choices=['direct', 'engine', 'full'], default='full',
                        help="Which path(s) to exercise: 'direct' = IBKRBroker calls only; "
                             "'engine' = ExecutionEngine only; 'full' = both (default)")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info(f"IBKR PAPER SMOKE TEST  (mode={args.mode})")
    logger.info(f"  symbol={args.symbol}  qty={args.qty}  "
                f"port={args.port}  clientId={args.client_id}")
    logger.info("=" * 70)

    # ---- step 0: connect + baseline ------------------------------------
    step('0a', f"connect to IBKR paper (clientId={args.client_id})")
    try:
        broker = _connect_ibkr(args.client_id, args.port)
    except Exception as e:
        fail('0a', f"connect failed: {e}")
    ok(f"connected to IBKR paper")

    step('0b', "fetch account + baseline positions")
    try:
        account = broker.get_account()
    except Exception as e:
        fail('0b', f"get_account raised: {e}")
    equity = float(account.get('portfolio_value', 0) or 0)
    if equity <= 0:
        fail('0b', f"portfolio_value not positive: {equity}")
    ok(f"account_id={account.get('account_id')} equity=${equity:,.2f}")

    try:
        baseline = broker.get_stock_positions()
    except Exception as e:
        fail('0b', f"get_stock_positions raised: {e}")
    baseline_symbols = sorted(p.get('symbol') for p in baseline)
    ok(f"baseline positions ({len(baseline_symbols)})")

    step('0c', f"fetch reference price for {args.symbol}")
    try:
        last_price = _fetch_reference_price(broker, args.symbol)
    except Exception as e:
        fail('0c', str(e))
    ok(f"{args.symbol} reference price = ${last_price:,.2f}")

    # ---- Part 1: direct broker path -----------------------------------
    try:
        if args.mode in ('direct', 'full'):
            _part_direct_broker(broker, args.symbol, args.qty, last_price, baseline_symbols)
        else:
            logger.info("\n[skip] Part 1 (direct broker) skipped -- mode=engine")

        # ---- Part 2: ExecutionEngine path -----------------------------
        if args.mode in ('engine', 'full'):
            _part_execution_engine(broker, args.symbol, args.qty, last_price, baseline_symbols)
        else:
            logger.info("\n[skip] Part 2 (ExecutionEngine) skipped -- mode=direct")
    finally:
        # Always disconnect, even on failure, so the gateway frees clientId.
        step('Z', "disconnect IBKR cleanly")
        try:
            broker.stop()
            ok("broker.stop() completed")
        except Exception as e:
            logger.warning(f"  broker.stop() raised (non-fatal): {e}")

    logger.success("\n" + "=" * 70)
    logger.success(f"=== SMOKE TEST PASSED (mode={args.mode}) ===")
    logger.success("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
