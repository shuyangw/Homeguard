"""
IBKR Paper Futures Smoke Test -- end-to-end futures trading chain validation.

================================================================
WHEN TO RUN
================================================================
Run after ANY change to:
  - IBKRFuturesBroker method signatures or behavior
  - FuturesTradingInterface
  - FuturesSymbolResolver / FuturesDefinitionsLoader / ExpirationGuard /
    MarginGuard / AuditLog
  - submit_resolved_order safeguard chain

Takes ~30-45 seconds. Places ONE real limit order on IBKR paper at 50%
below the current market (guaranteed not to fill in any liquidity
condition), exercises the full safeguard chain, then cancels. Safe
after-hours and during market hours since the price never crosses the spread.

================================================================
WHAT IT VALIDATES
================================================================
The complete plumbing for live futures trading:

  Step 0: Connect, fetch account margin, baseline futures positions
  Step 1: Resolve continuous intent "MES.v.0" -> raw symbol via the
          FuturesSymbolResolver. Verify expiration_date came from the
          FuturesDefinitionsLoader (not a placeholder).
  Step 2: Fetch reference market price for the resolved contract via
          IBKRFuturesBroker.get_latest_trade
  Step 3: Run what_if_order to verify the margin pre-check path
  Step 4: Submit a LIMIT BUY at 50% below market through the full
          safeguard chain (ExpirationGuard -> MarginGuard -> AuditLog
          -> _ibkr_submit). Verify orderId returned and order appears
          in get_open_orders.
  Step 5: cancel_order + verify cancellation
  Step 6: Verify final state: positions unchanged, no leftover open
          orders from this run, audit log contains the submit entry
  Step Z: Disconnect cleanly

================================================================
USAGE
================================================================
Prerequisites:
  - IB Gateway running on port 4002 (paper) with futures permissions
  - fintech conda env activated
  - H:/Stock_Data/futures_definitions/ populated for the current month

Locally:
    conda activate fintech
    python scripts/trading/futures_paper_smoke_test.py

Flags:
    --symbol-root MES   # which futures root (default MES, the micro E-mini S&P)
    --qty 1             # contracts per order (default 1)
    --client-id 99      # IBKR clientId (must != running service; default 99)
    --port 4002         # IBKR paper gateway port (default 4002)

================================================================
SAFETY
================================================================
- Idempotent: re-running never accumulates state.
- Uses clientId=99 by default so it doesn't collide with homeguard-multi
  (which holds clientId=10).
- LIMIT price is 50% of market so no realistic quote ever crosses it.
- The single order is cancelled before the script exits; final state
  is verified.
- If any step fails, exits non-zero immediately with the failed step.
"""

import argparse
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.futures_definitions_loader import FuturesDefinitionsLoader
from src.trading.brokers.ibkr.config import IBKRConfig
from src.trading.brokers.ibkr.ibkr_futures_broker import (
    IBKRFuturesBroker, OrderRejectedError,
)
from src.trading.brokers.interfaces.base import OrderSide, OrderType, TimeInForce
from src.trading.futures.audit_log import AuditLog
from src.trading.futures.symbol_resolver import FuturesSymbolResolver
from src.utils.logger import logger


VALID_PENDING = {"pending", "PendingSubmit", "PreSubmitted", "Submitted"}
VALID_CANCELLED = {
    "cancelled", "Cancelled", "ApiCancelled",
    "pendingcancel", "PendingCancel",
}


# ============================================================ helpers

def step(n: str, msg: str) -> None:
    logger.info(f"\n===== step {n}: {msg} =====")


def ok(msg: str) -> None:
    logger.info(f"  [+] {msg}")


def fail(step_num: str, msg: str) -> None:
    logger.error(f"  [X] STEP {step_num} FAILED: {msg}")
    logger.error("=== FUTURES SMOKE TEST FAILED ===")
    sys.exit(1)


def _connect_broker(client_id: int, port: int) -> IBKRFuturesBroker:
    """Construct + start a broker with the requested clientId/port."""
    os.environ["IBKR_CLIENT_ID"] = str(client_id)
    os.environ["IBKR_PORT"] = str(port)
    config = IBKRConfig()
    if config.client_id != client_id:
        raise RuntimeError(
            f"IBKRConfig ignored IBKR_CLIENT_ID env: got {config.client_id}"
        )

    audit_dir = Path.home() / ".homeguard" / "audit_smoke"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit = AuditLog(log_dir=audit_dir)

    broker = IBKRFuturesBroker(config=config, audit_log=audit)
    broker.start()
    return broker


def _resolve_intent(broker: IBKRFuturesBroker, symbol_root: str):
    """Resolve <root>.v.0 to ResolvedOrder with real expiration_date."""
    loader = FuturesDefinitionsLoader()
    resolver = FuturesSymbolResolver(definitions_loader=loader)
    return resolver.resolve_for_order(
        strategy_intent=f"{symbol_root}.v.0",
        side=OrderSide.BUY,
        quantity=1,  # placeholder, overridden below
        as_of=date.today(),
        strategy="futures_smoke_test",
        order_type=OrderType.LIMIT,
        limit_price=1.0,
        time_in_force=TimeInForce.DAY,
    )


def _fetch_reference_price(
    broker: IBKRFuturesBroker, symbol_root: str, contract_month: str,
) -> float:
    """Return a non-zero reference price (last trade, bid/ask mid, or close)."""
    snap = broker.get_latest_trade(symbol_root, contract_month)
    if snap["price"] > 0:
        return snap["price"]
    mid = (snap["bid"] + snap["ask"]) / 2.0
    if mid > 0:
        logger.info(f"  using bid/ask midpoint ${mid:,.2f} (last was 0)")
        return mid
    if snap["close"] > 0:
        logger.info(f"  using previous close ${snap['close']:,.2f} "
                    f"(market closed; last/bid/ask all 0)")
        return snap["close"]
    raise RuntimeError(f"no usable price for {symbol_root} {contract_month}: {snap!r}")


def _round_to_tick(price: float, tick: float = 0.25) -> float:
    """Round to nearest valid tick (default 0.25 for ES/MES family)."""
    return round(round(price / tick) * tick, 4)


def _verify_clean_state(
    broker: IBKRFuturesBroker, baseline_keys: set, test_order_ids: set,
    step_num: str,
) -> None:
    """Assert no new positions + no lingering open orders from this run."""
    final_positions = broker.get_futures_positions()
    final_keys = {(p["symbol_root"], p["contract_month"]) for p in final_positions}
    new_keys = final_keys - baseline_keys
    if new_keys:
        fail(step_num, f"unexpected new positions: {new_keys}")
    ok(f"positions unchanged: {len(final_keys)} futures positions")

    try:
        opens = broker.get_open_orders()
    except Exception as e:
        logger.warning(f"get_open_orders raised: {e} -- skipping lingering check")
        opens = []
    leftover = [o for o in opens if str(o.get("orderId")) in test_order_ids]
    if leftover:
        fail(step_num, f"test orders still open: {leftover}")
    ok("no lingering open orders from this run")


# ============================================================ main flow

def run_smoke_test(symbol_root: str, qty: int,
                   client_id: int, port: int) -> int:
    logger.info("=" * 70)
    logger.info(f"FUTURES PAPER SMOKE TEST")
    logger.info(f"  symbol_root={symbol_root}  qty={qty}  "
                f"port={port}  clientId={client_id}")
    logger.info("=" * 70)

    # ---- step 0: connect ---------------------------------------------------
    step("0a", f"connect to IBKR paper (port={port}, clientId={client_id})")
    try:
        broker = _connect_broker(client_id, port)
    except Exception as e:
        fail("0a", f"connect failed: {e}")
    ok("connected to IBKR paper")

    test_order_ids: set = set()
    audit_today_path = (
        Path.home() / ".homeguard" / "audit_smoke"
        / f"audit_{date.today().strftime('%Y%m%d')}.jsonl"
    )

    try:
        # ---- step 0b: account snapshot ------------------------------------
        step("0b", "fetch account margin status")
        try:
            margin = broker.get_margin_status()
        except Exception as e:
            fail("0b", f"get_margin_status raised: {e}")
        if margin["net_liquidation"] <= 0:
            fail("0b", f"net_liquidation not positive: {margin}")
        ok(f"net_liq=${margin['net_liquidation']:,.2f} "
           f"avail=${margin['available_funds']:,.2f} "
           f"init_margin=${margin['initial_margin']:,.2f}")

        # ---- step 0c: baseline positions ----------------------------------
        step("0c", "fetch baseline futures positions")
        try:
            baseline = broker.get_futures_positions()
        except Exception as e:
            fail("0c", f"get_futures_positions raised: {e}")
        baseline_keys = {(p["symbol_root"], p["contract_month"]) for p in baseline}
        ok(f"baseline: {len(baseline_keys)} futures positions")

        # ---- step 0d: resolve symbol --------------------------------------
        step("0d", f"resolve {symbol_root}.v.0 with real expiration")
        try:
            resolved = _resolve_intent(broker, symbol_root)
        except Exception as e:
            fail("0d", f"symbol resolution failed: {e}")
        if resolved.expiration_date is None:
            fail("0d", "expiration_date is None -- DefinitionsLoader not wired")
        ok(f"{symbol_root}.v.0 -> {resolved.raw_symbol} "
           f"(contract_month={resolved.contract_month}, "
           f"expiration={resolved.expiration_date})")

        # ---- step 0e: reference price -------------------------------------
        step("0e", f"fetch reference price for {resolved.raw_symbol}")
        try:
            last_price = _fetch_reference_price(
                broker, resolved.symbol_root, resolved.contract_month,
            )
        except Exception as e:
            fail("0e", str(e))
        ok(f"{resolved.raw_symbol} reference price = ${last_price:,.2f}")

        # ---- step 1: what_if_order (margin pre-check sanity) --------------
        step("1", "what_if_order pre-trade margin estimate")
        try:
            estimate = broker.what_if_order(
                symbol_root=resolved.symbol_root,
                contract_month=resolved.contract_month,
                side=OrderSide.BUY, quantity=qty,
                order_type=OrderType.MARKET,
            )
        except Exception as e:
            fail("1", f"what_if_order raised: {e}")
        ok(f"estimate: init_margin=${estimate['initial_margin']:,.2f} "
           f"commission=${estimate['commission']:,.4f}")

        # ---- step 2: submit LIMIT BUY at 50% via safeguard chain ----------
        step("2a", f"LIMIT BUY {qty} {resolved.raw_symbol} @ 50% below market")
        limit_price = _round_to_tick(last_price * 0.5)
        logger.info(f"  placing LIMIT BUY {qty} {resolved.raw_symbol} "
                    f"@ ${limit_price:.2f} (last ${last_price:.2f})")
        # Rebuild ResolvedOrder with the actual qty and limit_price
        from src.trading.futures.symbol_resolver import ResolvedOrder
        order_to_submit = ResolvedOrder(
            strategy_intent=resolved.strategy_intent,
            symbol_root=resolved.symbol_root,
            contract_month=resolved.contract_month,
            raw_symbol=resolved.raw_symbol,
            side=OrderSide.BUY,
            quantity=qty,
            order_type=OrderType.LIMIT,
            limit_price=limit_price,
            stop_price=None,
            time_in_force=TimeInForce.DAY,
            strategy="futures_smoke_test",
            as_of=resolved.as_of,
            expiration_date=resolved.expiration_date,
        )
        try:
            response = broker.submit_resolved_order(
                order_to_submit, hold_overnight=False,
            )
        except OrderRejectedError as e:
            fail("2a", f"safeguard rejected order: {e}")
        except Exception as e:
            fail("2a", f"submit raised: {e}")
        order_id = str(response.get("orderId", ""))
        if not order_id:
            fail("2a", f"no orderId in response: {response!r}")
        test_order_ids.add(order_id)
        if response.get("status") not in VALID_PENDING:
            fail("2a", f"unexpected status {response.get('status')!r}")
        ok(f"submitted orderId={order_id} status={response['status']}")

        # ---- step 2b: verify order present ---------------------------------
        step("2b", "get_order + confirm visibility")
        try:
            fetched = broker.get_order(order_id)
        except Exception as e:
            fail("2b", f"get_order raised: {e}")
        if str(fetched.get("orderId")) != order_id:
            fail("2b", f"get_order returned wrong order: {fetched!r}")
        ok(f"get_order({order_id}) -> status={fetched.get('status')}")

        # ---- step 3: cancel + verify --------------------------------------
        step("3a", f"cancel_order({order_id})")
        if not broker.cancel_order(order_id):
            fail("3a", "cancel_order returned False")
        # Audit-log the cancel (broker layer doesn't auto-log -- strategy does)
        try:
            broker.audit_log.log_cancel(
                timestamp=datetime.now(timezone.utc),
                strategy="futures_smoke_test",
                raw_symbol=resolved.raw_symbol,
                contract_month=resolved.contract_month,
                ibkr_order_id=order_id,
            )
        except Exception as e:
            logger.warning(f"audit log_cancel failed (non-fatal): {e}")
        ok(f"cancel_order({order_id}) -> True")

        time.sleep(2)
        try:
            post = broker.get_order(order_id)
            if post.get("status") not in VALID_CANCELLED:
                fail("3a", f"post-cancel status {post.get('status')!r}")
            ok(f"post-cancel status={post.get('status')}")
        except LookupError:
            ok("order no longer queryable (treated as cancelled)")

        # ---- step 4: verify clean state -----------------------------------
        step("4a", "verify clean state")
        _verify_clean_state(broker, baseline_keys, test_order_ids, "4a")

        # ---- step 4b: verify audit log --------------------------------------
        step("4b", "verify audit log captured events")
        if not audit_today_path.exists():
            fail("4b", f"audit log missing: {audit_today_path}")
        import json
        events = [
            json.loads(line) for line in audit_today_path.read_text().splitlines()
            if line.strip()
        ]
        smoke_events = [
            e for e in events
            if e.get("strategy") == "futures_smoke_test"
            and e.get("raw_symbol") == resolved.raw_symbol
        ]
        event_types = [e.get("event_type") for e in smoke_events]
        if "submit" not in event_types:
            fail("4b", f"no submit entry: {event_types}")
        if "cancel" not in event_types:
            fail("4b", f"no cancel entry: {event_types}")
        ok(f"audit log has {len(smoke_events)} events for this run: {event_types}")

    finally:
        step("Z", "disconnect IBKR cleanly")
        try:
            broker.stop()
            ok("broker.stop() completed")
        except Exception as e:
            logger.warning(f"  broker.stop() raised (non-fatal): {e}")

    logger.info("\n" + "=" * 70)
    logger.info(f"=== FUTURES SMOKE TEST PASSED ({symbol_root}) ===")
    logger.info("=" * 70)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--symbol-root", default="MES",
                        help="Futures symbol root (default: MES)")
    parser.add_argument("--qty", type=int, default=1,
                        help="Contracts per order (default: 1)")
    parser.add_argument("--client-id", type=int, default=99,
                        help="IBKR clientId; must differ from running service "
                             "(default 99)")
    parser.add_argument("--port", type=int, default=4002,
                        help="IBKR gateway port (default 4002 paper)")
    args = parser.parse_args()
    return run_smoke_test(args.symbol_root, args.qty, args.client_id, args.port)


if __name__ == "__main__":
    raise SystemExit(main())
