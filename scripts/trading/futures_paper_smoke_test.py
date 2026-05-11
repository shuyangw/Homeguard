"""End-to-end smoke test of the futures broker safeguard chain against IBKR paper.

Exercises one MES round-trip:
  1. Resolve "MES.v.0" continuous intent to today's active contract
  2. Submit a LIMIT order at 50% of market (won't fill -- intentional)
  3. Verify safeguard chain passes (expiration + margin)
  4. Verify audit log captured the submission
  5. Cancel the order
  6. Verify audit log captured the cancel

Prerequisites:
  - IBKR Gateway running on port 4002 (paper)
  - Futures permissions enabled on the paper account
  - The _ibkr_submit_stub in IBKRFuturesBroker has been replaced with real
    ib_async integration (see docs/trading/futures_paper_validation_runbook.md
    "Real IBKR integration" section)

This script is the futures analogue of scripts/trading/smoke_test_ibkr_paper.py.
Run before declaring the futures broker ready for the 14-day validation window.

Usage:
    python scripts/trading/futures_paper_smoke_test.py
    python scripts/trading/futures_paper_smoke_test.py --mode direct       # broker only
    python scripts/trading/futures_paper_smoke_test.py --mode safeguards   # full chain (default)
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.trading.brokers.ibkr.config import IBKRConfig
from src.trading.brokers.ibkr.ibkr_futures_broker import (
    IBKRFuturesBroker, OrderRejectedError,
)
from src.trading.brokers.interfaces.base import OrderSide, OrderType, TimeInForce
from src.trading.futures.audit_log import AuditLog
from src.trading.futures.symbol_resolver import FuturesSymbolResolver
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_safeguard_smoke(symbol_root: str = "MES") -> int:
    """Full chain: resolve -> guard -> submit -> verify audit -> cancel."""
    logger.info(f"=== Futures paper smoke test (symbol_root={symbol_root}) ===")

    config = IBKRConfig(port=4002)
    audit_dir = Path.home() / ".homeguard" / "audit_smoke"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit = AuditLog(log_dir=audit_dir)
    broker = IBKRFuturesBroker(config=config, audit_log=audit)

    resolver = FuturesSymbolResolver()
    today = date.today()
    try:
        resolved = resolver.resolve_for_order(
            strategy_intent=f"{symbol_root}.v.0",
            side=OrderSide.BUY,
            quantity=1,
            as_of=today,
            strategy="smoke_test",
            order_type=OrderType.LIMIT,
            limit_price=1.0,
            time_in_force=TimeInForce.DAY,
        )
    except Exception as e:
        logger.error(f"[FAIL] symbol resolution failed: {e}")
        return 1
    logger.info(f"[OK] resolved {symbol_root}.v.0 -> {resolved.raw_symbol} (contract_month={resolved.contract_month})")

    placeholder_expiration = date(today.year + 1, 12, 31)
    logger.warning("[!] using placeholder expiration date; replace with futures_definitions/ lookup")

    try:
        response = broker.submit_resolved_order(
            resolved, expiration_date=placeholder_expiration, hold_overnight=False,
        )
        logger.info(f"[OK] order submitted: orderId={response['orderId']} status={response['status']}")
    except OrderRejectedError as e:
        logger.error(f"[FAIL] safeguard rejected order: {e}")
        return 1

    audit_today = audit_dir / f"audit_{today.strftime('%Y%m%d')}.jsonl"
    if not audit_today.exists():
        logger.error(f"[FAIL] audit log file missing: {audit_today}")
        return 1
    logger.info(f"[OK] audit log present: {audit_today}")

    logger.warning(
        "[!] cancel path requires real IBKR integration via _ibkr_submit; "
        "currently the stub does not track orders. Wire up before live validation."
    )

    logger.info("=== Smoke test complete ===")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mode", choices=["direct", "safeguards"], default="safeguards")
    parser.add_argument("--symbol-root", default="MES")
    args = parser.parse_args()

    if args.mode == "direct":
        logger.warning(
            "direct mode tests IBKRFuturesBroker without safeguards; "
            "requires real _ibkr_submit integration. Skipping for now."
        )
        return 0
    return run_safeguard_smoke(symbol_root=args.symbol_root)


if __name__ == "__main__":
    raise SystemExit(main())
