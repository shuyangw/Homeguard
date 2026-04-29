"""
PriceOracle: unified live-price abstraction.

Codifies the price-resolution policy in one place so callers don't reach into
the broker, the streaming buffer, or a position dict's stale field directly.

Architecture:
    PriceOracle holds an ordered list of PriceProviders. Each call walks the
    list and returns the first PricePoint a provider can produce. Default
    chain for stock strategies on IBKR + Alpaca streaming:

        StreamingPriceProvider(alpaca_streaming)   # Alpaca live IEX
            -> BrokerQuotePriceProvider(broker)    # REST quote
            -> BrokerPortfolioPriceProvider(broker) # ib.portfolio() snapshot

    Source-of-truth boundaries unchanged:
      - Broker:   cost basis, fills, qty held
      - DataLayer: historical bars, live streaming
      - Oracle:   live mark for held positions (composes the above)

The oracle does not modify broker behavior. Callers ask the oracle when they
need a live price; raw `broker.get_stock_positions()` still returns whatever
IBKR thinks (which may be delayed).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# PricePoint -- result type with provenance
# ============================================================================

@dataclass(frozen=True)
class PricePoint:
    """A live price with provenance.

    Attributes:
        price: The resolved price (always > 0 if returned by a provider).
        source: Identifier for which provider supplied it (e.g., "streaming",
            "broker_quote", "broker_portfolio"). For diagnostic logging.
        age_seconds: How old the underlying data is, if known. None means
            unknown (provider didn't supply a timestamp).
    """
    price: float
    source: str
    age_seconds: Optional[float] = None


# ============================================================================
# PriceProvider protocol -- pluggable source
# ============================================================================

@runtime_checkable
class PriceProvider(Protocol):
    """Pluggable price source. Returns None if it can't price the symbol.

    Implementations should NOT raise for "no data available" -- they should
    return None so the oracle can fall through. Raise only for programming
    errors (bad symbol type, etc.).
    """
    def get_live_price(self, symbol: str) -> Optional[PricePoint]: ...


# ============================================================================
# StreamingPriceProvider -- preferred source when available
# ============================================================================

class StreamingPriceProvider:
    """Reads the latest bar from a `StreamingProviderInterface`.

    Returns None when:
        - streaming is None or not connected
        - symbol has no bar in the buffer
        - bar.close is None or <= 0
        - bar is older than `max_age_seconds` (default 300s)
    """

    SOURCE = "streaming"

    def __init__(self, streaming: Any, max_age_seconds: float = 300.0) -> None:
        # Duck-typed: needs `get_bar(symbol) -> Optional[Bar]` and optionally
        # `is_connected() -> bool`. We don't import StreamingProviderInterface
        # to keep this module dependency-light.
        self._streaming = streaming
        self._max_age = float(max_age_seconds)

    def get_live_price(self, symbol: str) -> Optional[PricePoint]:
        if self._streaming is None:
            return None

        # Optional connectivity check -- if exposed and false, skip.
        is_connected = getattr(self._streaming, "is_connected", None)
        try:
            if callable(is_connected) and is_connected() is False:
                return None
        except Exception:
            pass  # connectivity check is best-effort

        try:
            bar = self._streaming.get_bar(symbol)
        except Exception as e:
            logger.debug(f"[oracle:streaming] get_bar({symbol}) raised: {e}")
            return None

        if bar is None:
            return None

        close = getattr(bar, "close", None)
        if close is None or close <= 0:
            return None

        age = self._compute_age(getattr(bar, "timestamp", None))
        if age is not None and age > self._max_age:
            return None  # too stale; fall through

        return PricePoint(price=float(close), source=self.SOURCE, age_seconds=age)

    @staticmethod
    def _compute_age(ts: Optional[datetime]) -> Optional[float]:
        if ts is None:
            return None
        try:
            now = datetime.now(timezone.utc)
            ts_aware = ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)
            return max(0.0, (now - ts_aware).total_seconds())
        except Exception:
            return None


# ============================================================================
# BrokerQuotePriceProvider -- second-tier fallback (REST)
# ============================================================================

class BrokerQuotePriceProvider:
    """Reads a stock quote via `broker.get_latest_quote(symbol)`.

    Prefers `last`, falls back to mid of `bid`/`ask`, then `ask`. Returns None
    if all are missing or zero. One REST round-trip per call -- intended as
    second-tier fallback only (after streaming).
    """

    SOURCE = "broker_quote"

    def __init__(self, broker: Any) -> None:
        self._broker = broker

    def get_live_price(self, symbol: str) -> Optional[PricePoint]:
        try:
            quote = self._broker.get_latest_quote(symbol)
        except Exception as e:
            logger.debug(f"[oracle:broker_quote] get_latest_quote({symbol}) raised: {e}")
            return None

        if not quote or not isinstance(quote, dict):
            return None

        price = self._resolve_price_from_quote(quote)
        if price is None:
            return None
        return PricePoint(price=price, source=self.SOURCE, age_seconds=None)

    @staticmethod
    def _resolve_price_from_quote(quote: Dict[str, Any]) -> Optional[float]:
        last = quote.get("last")
        if last and last > 0:
            return float(last)
        bid = quote.get("bid") or 0
        ask = quote.get("ask") or 0
        if bid > 0 and ask > 0:
            return float((bid + ask) / 2)
        if ask > 0:
            return float(ask)
        if bid > 0:
            return float(bid)
        return None


# ============================================================================
# BrokerCryptoQuotePriceProvider -- crypto equivalent
# ============================================================================

class BrokerCryptoQuotePriceProvider:
    """Same shape as BrokerQuotePriceProvider but uses `get_crypto_quote`.

    For CSCM's DemoBroker / Coinbase / AlpacaCrypto path.
    """

    SOURCE = "broker_crypto_quote"

    def __init__(self, broker: Any) -> None:
        self._broker = broker

    def get_live_price(self, symbol: str) -> Optional[PricePoint]:
        try:
            quote = self._broker.get_crypto_quote(symbol)
        except Exception as e:
            logger.debug(f"[oracle:broker_crypto_quote] get_crypto_quote({symbol}) raised: {e}")
            return None

        if not quote or not isinstance(quote, dict):
            return None

        price = BrokerQuotePriceProvider._resolve_price_from_quote(quote)
        if price is None:
            return None
        return PricePoint(price=price, source=self.SOURCE, age_seconds=None)


# ============================================================================
# BrokerPortfolioPriceProvider -- last-resort, cached
# ============================================================================

class BrokerPortfolioPriceProvider:
    """Reads `current_price` from `broker.get_stock_positions()` snapshot.

    For IBKR this comes from `ib.portfolio()` (PortfolioItem.marketPrice),
    which is delayed when market-data subscriptions are limited. Used as
    last resort after streaming + REST quote both miss.

    Caches the entire positions list with TTL to amortize across N calls
    in a single metric tick (5 enrich_position calls = 1 broker hit).
    """

    SOURCE = "broker_portfolio"

    def __init__(self, broker: Any, cache_ttl_seconds: float = 30.0) -> None:
        self._broker = broker
        self._ttl = float(cache_ttl_seconds)
        self._cache: Dict[str, float] = {}
        self._cache_at: float = 0.0
        self._lock = threading.Lock()

    def get_live_price(self, symbol: str) -> Optional[PricePoint]:
        with self._lock:
            now = time.time()
            age = now - self._cache_at
            if age > self._ttl or not self._cache:
                self._refresh(now)
                age = 0.0

            price = self._cache.get(symbol)
            if price is None or price <= 0:
                return None
            return PricePoint(price=price, source=self.SOURCE, age_seconds=age)

    def _refresh(self, now: float) -> None:
        try:
            positions = self._broker.get_stock_positions() or []
        except Exception as e:
            logger.debug(f"[oracle:broker_portfolio] get_stock_positions raised: {e}")
            return  # leave cache as-is

        new_cache: Dict[str, float] = {}
        for pos in positions:
            sym = pos.get("symbol")
            cp = pos.get("current_price")
            if sym and cp is not None and cp > 0:
                new_cache[sym] = float(cp)
        self._cache = new_cache
        self._cache_at = now


# ============================================================================
# PriceOracle -- composes providers in priority order
# ============================================================================

@dataclass
class PriceOracle:
    """Ordered chain of `PriceProvider`s. First non-None wins.

    Usage::

        oracle = PriceOracle(providers=[
            StreamingPriceProvider(streaming=alpaca_market_data),
            BrokerQuotePriceProvider(broker=ibkr_broker),
            BrokerPortfolioPriceProvider(broker=ibkr_broker),
        ])
        pp = oracle.get_live_price("AAPL")
        if pp is not None:
            print(f"AAPL @ ${pp.price} (source={pp.source})")

        # Most consumers want this:
        enriched = oracle.enrich_position(broker_position_dict)
        # enriched['unrealized_pnl'] now reflects live price, not broker's view
    """

    providers: List[PriceProvider] = field(default_factory=list)

    def get_live_price(self, symbol: str) -> Optional[PricePoint]:
        if not symbol:
            return None
        for p in self.providers:
            if p is None:
                continue
            try:
                result = p.get_live_price(symbol)
            except Exception as e:
                logger.warning(f"[oracle] provider {type(p).__name__} raised on {symbol}: {e}")
                continue
            if result is not None:
                return result
        return None

    def enrich_position(self, pos: Dict[str, Any]) -> Dict[str, Any]:
        """Return a NEW dict with `current_price`/`market_value`/`unrealized_pnl`/
        `unrealized_pnl_pct` recomputed from the oracle's resolved live price.

        Preserves `avg_entry_price` and `quantity` from the original (broker
        is the source of truth for cost basis). If the oracle can't resolve
        the symbol, returns a copy of the original unchanged.

        Adds `_price_source` field (diagnostic) when the oracle resolves.
        """
        out = dict(pos)
        sym = pos.get("symbol")
        qty_raw = pos.get("quantity")
        avg_raw = pos.get("avg_entry_price")

        try:
            qty = float(qty_raw) if qty_raw is not None else 0.0
            avg = float(avg_raw) if avg_raw is not None else 0.0
        except (TypeError, ValueError):
            return out

        if not sym or qty == 0 or avg <= 0:
            return out

        pp = self.get_live_price(sym)
        if pp is None:
            return out

        live = pp.price
        out["current_price"] = live
        out["market_value"] = live * qty
        out["unrealized_pnl"] = (live - avg) * qty
        out["unrealized_pnl_pct"] = (live - avg) / avg * 100.0
        out["_price_source"] = pp.source
        return out

    def enrich_positions(self, positions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Vectorized helper. Returns a list of enriched dicts in input order."""
        return [self.enrich_position(p) for p in (positions or [])]
