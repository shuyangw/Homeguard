# IBKR Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Interactive Brokers as a mix-and-match broker backend (execution + optional data) alongside Alpaca, with config-driven routing and EC2 Gateway deployment.

**Architecture:** IBKR module lives at `src/trading/brokers/ibkr/` implementing existing Homeguard interfaces (no new abstractions except `StreamingProviderInterface` extracted from `LiveDataProvider`). Strategies remain broker-agnostic. A new `broker_routing.yaml` config maps strategies to brokers at startup.

**Tech Stack:** Python 3.11, ib_async, Pydantic, pytest. IB Gateway + IBC + Xvfb on EC2 ARM64.

**Spec:** `docs/architecture/2026-04-07-ibkr-integration-design.md`

**V2 reference code:** Extracted from zip at `/tmp/ibkr_files/module_v2/ibkr_v2/` -- use as starting point, apply fixes noted in spec Section 10.

**Environment:** `conda activate fintech` for all Python commands.

**Logging:** Always use `from src.utils.logger import get_logger; logger = get_logger(__name__)`. Never `import logging` or `print()`.

**ASCII only:** No emojis in code or docs. Use `[+]`, `[-]`, `[!]` for status.

---

## File Structure

### New files

| File | Responsibility |
|---|---|
| `src/streaming/interface.py` | StreamingProviderInterface ABC |
| `src/trading/brokers/ibkr/__init__.py` | Package exports |
| `src/trading/brokers/ibkr/config.py` | IBKRConfig Pydantic model |
| `src/trading/brokers/ibkr/errors.py` | IBKR error code -> Homeguard exception mapper |
| `src/trading/brokers/ibkr/pacing.py` | Historical data rate limiter |
| `src/trading/brokers/ibkr/connection.py` | Async event loop bridge + reconnection |
| `src/trading/brokers/ibkr/contracts.py` | Symbol -> IBKR Contract resolver + OCC bridge |
| `src/trading/brokers/ibkr/data_download.py` | IBKRDataProvider (DataProviderInterface) |
| `src/trading/brokers/ibkr/streaming.py` | IBKRStreamingProvider (StreamingProviderInterface) |
| `src/trading/brokers/ibkr/ibkr_broker.py` | IBKRBroker (6 interfaces, 22+ methods) |
| `src/trading/config/broker_routing.py` | Strategy -> broker routing loader |
| `config/ibkr.yaml` | IBKR connection config |
| `config/trading/broker_routing.yaml` | Strategy -> broker assignments |
| `tests/trading/brokers/ibkr/__init__.py` | Package marker |
| `tests/trading/brokers/ibkr/conftest.py` | Test fixtures |
| `tests/trading/brokers/ibkr/test_config_and_errors.py` | Config + error mapping tests |
| `tests/trading/brokers/ibkr/test_contracts.py` | OCC parsing + contract tests |
| `tests/trading/brokers/ibkr/test_pacing.py` | Pacing manager tests |
| `tests/trading/brokers/ibkr/test_broker_routing.py` | Routing loader tests |
| `tests/streaming/test_interface.py` | StreamingProviderInterface conformance tests |
| `infra/ec2/services/homeguard-gateway.service` | Systemd service for IB Gateway |
| `infra/ec2/install_ibkr_gateway.sh` | EC2 Gateway installer |
| `config/ibkr/ibc-config.ini.template` | IBC config template |

### Modified files

| File | Change |
|---|---|
| `src/streaming/live_data_provider.py` | Add `(StreamingProviderInterface)`, `name` -> `@property` |
| `src/streaming/__init__.py` | Add `StreamingProviderInterface` to exports |
| `src/trading/adapters/omr_live_adapter.py` | `hasattr` -> `isinstance` check |
| `src/trading/adapters/ramp_live_adapter.py` | `hasattr` -> `isinstance` check |
| `src/trading/brokers/broker_factory.py` | Replace IBKR `NotImplementedError` with real creation |
| `src/data/providers/factory.py` | Add `ibkr` case |
| `.env.example` | Add IBKR credential placeholders |

---

## Task 1: Install ib_async dependency

**Files:**
- Modify: `requirements.txt` (or conda env)

- [ ] **Step 1: Install ib_async**

```bash
conda activate fintech && pip install ib_async
```

- [ ] **Step 2: Verify import**

```bash
conda activate fintech && python -c "import ib_async; print(ib_async.__version__)"
```

Expected: Version number printed, no errors.

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "chore: add ib_async dependency for IBKR integration"
```

---

## Task 2: StreamingProviderInterface ABC

**Files:**
- Create: `src/streaming/interface.py`
- Create: `tests/streaming/test_interface.py`
- Modify: `src/streaming/__init__.py`
- Modify: `src/streaming/live_data_provider.py:67,116`

- [ ] **Step 1: Write the interface conformance test**

Create `tests/streaming/test_interface.py`:

```python
"""Tests that LiveDataProvider conforms to StreamingProviderInterface."""

import pytest
from unittest.mock import MagicMock, patch

from src.streaming.interface import StreamingProviderInterface
from src.streaming.live_data_provider import LiveDataProvider


class TestStreamingProviderConformance:
    """Verify LiveDataProvider implements StreamingProviderInterface."""

    def test_is_subclass(self):
        assert issubclass(LiveDataProvider, StreamingProviderInterface)

    def test_has_name_property(self):
        assert isinstance(
            LiveDataProvider.name, property
        ), "name must be a @property, not a plain attribute"

    def test_all_abstract_methods_implemented(self):
        """Check all abstract methods are present on LiveDataProvider."""
        required = [
            'start', 'stop', 'is_connected',
            'get_price', 'get_quote', 'get_trade', 'get_bar', 'get_bars',
            'get_vwap', 'get_spread',
            'on_bar', 'on_quote', 'on_trade', 'unsubscribe',
            'get_subscribed_symbols',
        ]
        for method_name in required:
            assert hasattr(LiveDataProvider, method_name), (
                f"LiveDataProvider missing {method_name}"
            )


class TestStreamingProviderIsinstance:
    """Verify isinstance checks work for duck-typing replacement."""

    @patch('src.streaming.live_data_provider._get_alpaca_credentials',
           return_value=('fake_key', 'fake_secret'))
    @patch('src.streaming.live_data_provider.MarketDataHub')
    def test_isinstance_check(self, mock_hub, mock_creds):
        provider = LiveDataProvider()
        assert isinstance(provider, StreamingProviderInterface)

    def test_non_provider_fails_isinstance(self):
        assert not isinstance("not_a_provider", StreamingProviderInterface)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda activate fintech && python -m pytest tests/streaming/test_interface.py -v
```

Expected: ImportError for `src.streaming.interface`.

- [ ] **Step 3: Create `src/streaming/interface.py`**

```python
"""
Streaming Data Provider Interface.

Extracted from LiveDataProvider's public API so that multiple streaming
backends (Alpaca WebSocket, IBKR socket) can be used interchangeably.

Strategy adapters should type-hint against this interface:

    def __init__(self, broker: BrokerInterface,
                 data_provider: Optional[StreamingProviderInterface] = None):

Implementors:
    - LiveDataProvider        (Alpaca WebSocket)
    - IBKRStreamingProvider   (IBKR socket, future)
"""

from abc import ABC, abstractmethod
from typing import Callable, List, Optional

import pandas as pd

from src.streaming.types import Bar, Quote, Trade


class StreamingProviderInterface(ABC):
    """
    Contract for real-time market data providers.

    Methods and return types match the actual LiveDataProvider implementation.
    """

    # ---- Identity ----

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging."""
        ...

    # ---- Lifecycle ----

    @abstractmethod
    def start(self, symbols: Optional[List[str]] = None) -> None:
        """Start streaming connection and optionally subscribe to symbols."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop streaming and cleanup."""
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if streaming connection is active."""
        ...

    # ---- On-Demand Data ----

    @abstractmethod
    def get_price(self, symbol: str) -> Optional[float]:
        """Latest trade price, or None if no data."""
        ...

    @abstractmethod
    def get_quote(self, symbol: str) -> Optional[Quote]:
        """Latest bid/ask as a Quote dataclass, or None."""
        ...

    @abstractmethod
    def get_trade(self, symbol: str) -> Optional[Trade]:
        """Latest trade as a Trade dataclass, or None."""
        ...

    @abstractmethod
    def get_bar(self, symbol: str) -> Optional[Bar]:
        """Latest bar as a Bar dataclass, or None."""
        ...

    @abstractmethod
    def get_bars(self, symbol: str, n: Optional[int] = None) -> pd.DataFrame:
        """
        Recent bars from in-memory buffer.

        Returns DataFrame with OHLCV columns indexed by timestamp.
        Returns empty DataFrame if no data (NOT None).
        """
        ...

    @abstractmethod
    def get_vwap(self, symbol: str) -> Optional[float]:
        """Current VWAP, or None."""
        ...

    @abstractmethod
    def get_spread(self, symbol: str) -> Optional[float]:
        """Current bid-ask spread in dollars, or None."""
        ...

    # ---- Real-Time Callbacks ----

    @abstractmethod
    def on_bar(self, symbols: List[str], handler: Callable[[Bar], None]) -> str:
        """Register callback for new bars. Returns subscription ID."""
        ...

    @abstractmethod
    def on_quote(self, symbols: List[str], handler: Callable[[Quote], None]) -> str:
        """Register callback for new quotes. Returns subscription ID."""
        ...

    @abstractmethod
    def on_trade(self, symbols: List[str], handler: Callable[[Trade], None]) -> str:
        """Register callback for new trades. Returns subscription ID."""
        ...

    @abstractmethod
    def unsubscribe(self, subscription_id: str) -> None:
        """Remove a callback subscription."""
        ...

    # ---- Utility ----

    @abstractmethod
    def get_subscribed_symbols(self) -> set:
        """Get set of currently subscribed symbols."""
        ...
```

- [ ] **Step 4: Modify `src/streaming/live_data_provider.py`**

Add the import and interface to class declaration. At top of file after existing imports (line ~34), add:

```python
from src.streaming.interface import StreamingProviderInterface
```

Change class declaration (line 67) from:

```python
class LiveDataProvider:
```

to:

```python
class LiveDataProvider(StreamingProviderInterface):
```

Change `self.name` attribute (line 116) from:

```python
        self.name = f"streaming-{feed}"
```

to:

```python
        self._name = f"streaming-{feed}"
```

Add a `name` property after `self._feed = feed` (after line 113):

```python
    @property
    def name(self) -> str:
        """Provider name for logging."""
        return self._name
```

- [ ] **Step 5: Update `src/streaming/__init__.py`**

Add to imports (after line 50):

```python
from src.streaming.interface import StreamingProviderInterface
```

Add to `__all__` list:

```python
    "StreamingProviderInterface",
```

- [ ] **Step 6: Run tests**

```bash
conda activate fintech && python -m pytest tests/streaming/test_interface.py -v
```

Expected: All tests PASS.

- [ ] **Step 7: Run existing streaming tests to verify no regressions**

```bash
conda activate fintech && python -m pytest tests/streaming/ -v
```

Expected: All existing tests still pass.

- [ ] **Step 8: Commit**

```bash
git add src/streaming/interface.py src/streaming/__init__.py src/streaming/live_data_provider.py tests/streaming/test_interface.py
git commit -m "feat: extract StreamingProviderInterface ABC from LiveDataProvider"
```

---

## Task 3: Update adapter isinstance checks

**Files:**
- Modify: `src/trading/adapters/omr_live_adapter.py:357`
- Modify: `src/trading/adapters/ramp_live_adapter.py:734`

- [ ] **Step 1: Update OMR adapter**

In `src/trading/adapters/omr_live_adapter.py`, add import near top (with other TYPE_CHECKING imports):

```python
from src.streaming.interface import StreamingProviderInterface
```

Change line 357 from:

```python
        if self._data_provider is None or not hasattr(self._data_provider, 'get_bars'):
```

to:

```python
        if self._data_provider is None or not isinstance(self._data_provider, StreamingProviderInterface):
```

- [ ] **Step 2: Update RAMP adapter**

In `src/trading/adapters/ramp_live_adapter.py`, add import near top:

```python
from src.streaming.interface import StreamingProviderInterface
```

Change line 734 from:

```python
            if self._data_provider is not None and hasattr(self._data_provider, 'get_bars'):
```

to:

```python
            if self._data_provider is not None and isinstance(self._data_provider, StreamingProviderInterface):
```

- [ ] **Step 3: Run adapter tests**

```bash
conda activate fintech && python -m pytest tests/trading/adapters/ -v --timeout=60
```

Expected: All pass (or skip if they require live connection).

- [ ] **Step 4: Commit**

```bash
git add src/trading/adapters/omr_live_adapter.py src/trading/adapters/ramp_live_adapter.py
git commit -m "refactor: replace hasattr duck-typing with isinstance(StreamingProviderInterface)"
```

---

## Task 4: IBKR config and errors

**Files:**
- Create: `src/trading/brokers/ibkr/__init__.py`
- Create: `src/trading/brokers/ibkr/config.py`
- Create: `src/trading/brokers/ibkr/errors.py`
- Create: `config/ibkr.yaml`
- Create: `tests/trading/brokers/ibkr/__init__.py`
- Create: `tests/trading/brokers/ibkr/test_config_and_errors.py`

- [ ] **Step 1: Write tests**

Create `tests/trading/brokers/ibkr/__init__.py` (empty file).

Create `tests/trading/brokers/ibkr/test_config_and_errors.py` -- copy from V2 reference at `/tmp/ibkr_files/module_v2/ibkr_v2/tests/trading/brokers/ibkr/test_config_and_errors.py`. The content is already validated and correct (see V2 reference code exploration above). No changes needed.

- [ ] **Step 2: Run test to verify it fails**

```bash
conda activate fintech && python -m pytest tests/trading/brokers/ibkr/test_config_and_errors.py -v
```

Expected: ImportError.

- [ ] **Step 3: Create package directory and `__init__.py`**

Create `src/trading/brokers/ibkr/__init__.py` -- use V2 reference. For now, only export config and errors (broker/data/streaming added in later tasks):

```python
"""
IBKR Integration Module for Homeguard.

Provides data download, live streaming, and order execution via Interactive
Brokers through the ib_async library.

Public API:
    IBKRConfig             - Configuration model

Usage:
    from src.trading.brokers.ibkr import IBKRConfig

    config = IBKRConfig(port=4002)  # Paper trading gateway

Dependencies:
    pip install ib_async  (no ibapi needed)
"""

from src.trading.brokers.ibkr.config import IBKRConfig

__all__ = [
    "IBKRConfig",
]
```

- [ ] **Step 4: Create `config.py`**

Copy from V2 reference at `/tmp/ibkr_files/module_v2/ibkr_v2/src/trading/brokers/ibkr/config.py`. No fixes needed -- V2 code is correct.

- [ ] **Step 5: Create `errors.py`**

Copy from V2 reference at `/tmp/ibkr_files/module_v2/ibkr_v2/src/trading/brokers/ibkr/errors.py`. No fixes needed -- V2 code is correct (already uses Homeguard exceptions).

- [ ] **Step 6: Create `config/ibkr.yaml`**

Copy from V2 reference at `/tmp/ibkr_files/module_v2/ibkr_v2/config/ibkr.yaml`. No fixes needed.

- [ ] **Step 7: Run tests**

```bash
conda activate fintech && python -m pytest tests/trading/brokers/ibkr/test_config_and_errors.py -v
```

Expected: All tests PASS.

- [ ] **Step 8: Commit**

```bash
git add src/trading/brokers/ibkr/__init__.py src/trading/brokers/ibkr/config.py src/trading/brokers/ibkr/errors.py config/ibkr.yaml tests/trading/brokers/ibkr/
git commit -m "feat(ibkr): add config model and error code mapper"
```

---

## Task 5: Pacing manager

**Files:**
- Create: `src/trading/brokers/ibkr/pacing.py`
- Create: `tests/trading/brokers/ibkr/test_pacing.py`
- Create: `tests/trading/brokers/ibkr/conftest.py`

- [ ] **Step 1: Write test fixtures and tests**

Create `tests/trading/brokers/ibkr/conftest.py`:

```python
"""Shared fixtures for IBKR tests."""

import pytest

from src.trading.brokers.ibkr.pacing import PacingManager


@pytest.fixture
def pacer():
    """Fresh PacingManager with default settings for unit tests."""
    p = PacingManager(max_per_10min=58, identical_cooldown=0.0)
    yield p
    p.clear()
```

Copy `tests/trading/brokers/ibkr/test_pacing.py` from V2 reference at `/tmp/ibkr_files/module_v2/ibkr_v2/tests/trading/brokers/ibkr/test_pacing.py`. No fixes needed.

- [ ] **Step 2: Run test to verify it fails**

```bash
conda activate fintech && python -m pytest tests/trading/brokers/ibkr/test_pacing.py -v
```

Expected: ImportError.

- [ ] **Step 3: Create `pacing.py`**

Copy from V2 reference at `/tmp/ibkr_files/module_v2/ibkr_v2/src/trading/brokers/ibkr/pacing.py`.

**Apply fix from spec Section 10:** Change line 22 from:

```python
import logging

logger = logging.getLogger(__name__)
```

to:

```python
from src.utils.logger import get_logger

logger = get_logger(__name__)
```

- [ ] **Step 4: Run tests**

```bash
conda activate fintech && python -m pytest tests/trading/brokers/ibkr/test_pacing.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trading/brokers/ibkr/pacing.py tests/trading/brokers/ibkr/conftest.py tests/trading/brokers/ibkr/test_pacing.py
git commit -m "feat(ibkr): add pacing manager for historical data rate limiting"
```

---

## Task 6: Connection manager

**Files:**
- Create: `src/trading/brokers/ibkr/connection.py`

- [ ] **Step 1: Create `connection.py`**

Copy from V2 reference at `/tmp/ibkr_files/module_v2/ibkr_v2/src/trading/brokers/ibkr/connection.py`. No fixes needed -- V2 code already uses `from src.utils.logger import get_logger` and Homeguard exceptions.

- [ ] **Step 2: Verify import works**

```bash
conda activate fintech && python -c "from src.trading.brokers.ibkr.connection import IBKRConnectionManager; print('OK')"
```

Expected: `OK` printed (no connection attempt, just import).

- [ ] **Step 3: Commit**

```bash
git add src/trading/brokers/ibkr/connection.py
git commit -m "feat(ibkr): add connection manager with async-sync bridge"
```

---

## Task 7: Contract resolver

**Files:**
- Create: `src/trading/brokers/ibkr/contracts.py`
- Create: `tests/trading/brokers/ibkr/test_contracts.py`

- [ ] **Step 1: Write tests**

Copy `tests/trading/brokers/ibkr/test_contracts.py` from V2 reference at `/tmp/ibkr_files/module_v2/ibkr_v2/tests/trading/brokers/ibkr/test_contracts.py`.

**Fix one issue in V2 tests:** The integration test `test_resolve_nonexistent_raises` references `ContractNotFoundError` which doesn't exist -- it should be `SymbolNotFoundError`. Change:

```python
    def test_resolve_nonexistent_raises(self, ibkr_connection):
        from src.trading.brokers.ibkr.errors import ContractNotFoundError
        resolver = ContractResolver(ibkr_connection)
        with pytest.raises(ContractNotFoundError):
            resolver.resolve_stock("ZZZZZNOTREAL")
```

to:

```python
    def test_resolve_nonexistent_raises(self, ibkr_connection):
        from src.trading.brokers.interfaces.base import SymbolNotFoundError
        resolver = ContractResolver(ibkr_connection)
        with pytest.raises(SymbolNotFoundError):
            resolver.resolve_stock("ZZZZZNOTREAL")
```

- [ ] **Step 2: Run OCC parser tests only (no connection needed)**

```bash
conda activate fintech && python -m pytest tests/trading/brokers/ibkr/test_contracts.py -v -k "not ibkr"
```

Expected: ImportError (contracts.py doesn't exist yet).

- [ ] **Step 3: Create `contracts.py`**

Copy from V2 reference at `/tmp/ibkr_files/module_v2/ibkr_v2/src/trading/brokers/ibkr/contracts.py`. No fixes needed.

- [ ] **Step 4: Run OCC parser tests**

```bash
conda activate fintech && python -m pytest tests/trading/brokers/ibkr/test_contracts.py -v -k "not ibkr"
```

Expected: All OCC parsing tests PASS. Integration tests skipped (marked `@pytest.mark.ibkr`).

- [ ] **Step 5: Commit**

```bash
git add src/trading/brokers/ibkr/contracts.py tests/trading/brokers/ibkr/test_contracts.py
git commit -m "feat(ibkr): add contract resolver with OCC symbology bridge"
```

---

## Task 8: IBKRDataProvider

**Files:**
- Create: `src/trading/brokers/ibkr/data_download.py`
- Modify: `src/data/providers/factory.py:67-80`

- [ ] **Step 1: Create `data_download.py`**

Copy from V2 reference at `/tmp/ibkr_files/module_v2/ibkr_v2/src/trading/brokers/ibkr/data_download.py`. No fixes needed.

- [ ] **Step 2: Add IBKR to data provider factory**

In `src/data/providers/factory.py`, add after the `elif name_lower == 'yfinance':` block (after line 77):

```python
        elif name_lower == 'ibkr':
            try:
                from src.trading.brokers.ibkr.data_download import IBKRDataProvider
                from src.trading.brokers.ibkr.connection import IBKRConnectionManager
                conn = IBKRConnectionManager.get_instance()
                if conn.is_connected:
                    providers.append(IBKRDataProvider(conn))
                else:
                    logger.warning("IBKR provider requested but not connected")
            except ImportError:
                logger.warning("IBKR module not available")
```

- [ ] **Step 3: Verify import**

```bash
conda activate fintech && python -c "from src.trading.brokers.ibkr.data_download import IBKRDataProvider; print('OK')"
```

Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add src/trading/brokers/ibkr/data_download.py src/data/providers/factory.py
git commit -m "feat(ibkr): add IBKRDataProvider and wire into data provider factory"
```

---

## Task 9: IBKRStreamingProvider

**Files:**
- Create: `src/trading/brokers/ibkr/streaming.py`

- [ ] **Step 1: Create `streaming.py`**

This file implements `StreamingProviderInterface`. It uses `ib_async.reqMktData()` and `reqRealTimeBars()` under the hood, converting results to Homeguard `Bar`/`Quote`/`Trade` dataclasses.

```python
"""
IBKR Streaming Provider - Implements StreamingProviderInterface.

Uses ib_async's reqMktData() for quotes/trades and reqRealTimeBars()
for 5-second bars (aggregated to 1-min in buffer). All data converted
to Homeguard Bar/Quote/Trade dataclasses at the boundary.
"""

from __future__ import annotations

import uuid
from collections import defaultdict, deque
from datetime import datetime
from typing import Callable, Dict, List, Optional, Set

import pandas as pd
import pytz

from src.streaming.interface import StreamingProviderInterface
from src.streaming.types import Bar, Quote, Trade
from src.trading.brokers.ibkr.connection import IBKRConnectionManager
from src.trading.brokers.ibkr.contracts import ContractResolver
from src.utils.logger import get_logger

logger = get_logger(__name__)

ET = pytz.timezone('America/New_York')


class IBKRStreamingProvider(StreamingProviderInterface):
    """
    Real-time market data from IBKR via ib_async.

    Implements StreamingProviderInterface so strategy adapters can use
    IBKR streaming interchangeably with Alpaca's LiveDataProvider.
    """

    def __init__(
        self,
        connection: IBKRConnectionManager,
        resolver: Optional[ContractResolver] = None,
        max_bars_per_symbol: int = 500,
    ):
        self._conn = connection
        self._resolver = resolver or ContractResolver(connection)
        self._max_bars = max_bars_per_symbol

        self._subscribed: Set[str] = set()
        self._tickers: Dict[str, object] = {}
        self._bar_buffers: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self._max_bars)
        )
        self._callbacks: Dict[str, dict] = {}
        self._started = False

    @property
    def name(self) -> str:
        return "IBKR-streaming"

    # ---- Lifecycle ----

    def start(self, symbols: Optional[List[str]] = None) -> None:
        if self._started:
            logger.warning("[IBKR Stream] Already started")
            return

        self._started = True
        logger.info("[IBKR Stream] Started")

        if symbols:
            self._subscribe_symbols(symbols)

    def stop(self) -> None:
        for symbol in list(self._subscribed):
            self._unsubscribe_symbol(symbol)
        self._started = False
        logger.info("[IBKR Stream] Stopped")

    def is_connected(self) -> bool:
        return self._conn.is_connected and self._started

    # ---- On-Demand Data ----

    def get_price(self, symbol: str) -> Optional[float]:
        ticker = self._tickers.get(symbol)
        if ticker is None:
            return None
        last = ticker.last
        if last != last:  # NaN check
            mid = self._mid_price(ticker)
            return mid if mid is not None else None
        return float(last)

    def get_quote(self, symbol: str) -> Optional[Quote]:
        ticker = self._tickers.get(symbol)
        if ticker is None:
            return None

        def safe(val):
            return float(val) if val == val else 0.0

        return Quote(
            symbol=symbol,
            timestamp=datetime.now(ET),
            bid_price=safe(ticker.bid),
            bid_size=safe(ticker.bidSize),
            ask_price=safe(ticker.ask),
            ask_size=safe(ticker.askSize),
        )

    def get_trade(self, symbol: str) -> Optional[Trade]:
        ticker = self._tickers.get(symbol)
        if ticker is None:
            return None
        last = ticker.last
        if last != last:
            return None
        return Trade(
            symbol=symbol,
            timestamp=datetime.now(ET),
            price=float(last),
            size=float(ticker.lastSize) if ticker.lastSize == ticker.lastSize else 0.0,
        )

    def get_bar(self, symbol: str) -> Optional[Bar]:
        buf = self._bar_buffers.get(symbol)
        if not buf:
            return None
        return buf[-1]

    def get_bars(self, symbol: str, n: Optional[int] = None) -> pd.DataFrame:
        buf = self._bar_buffers.get(symbol)
        if not buf:
            return pd.DataFrame()

        bars = list(buf) if n is None else list(buf)[-n:]
        if not bars:
            return pd.DataFrame()

        records = [{
            'timestamp': b.timestamp,
            'open': b.open,
            'high': b.high,
            'low': b.low,
            'close': b.close,
            'volume': b.volume,
        } for b in bars]

        df = pd.DataFrame(records)
        df = df.set_index('timestamp')
        return df

    def get_vwap(self, symbol: str) -> Optional[float]:
        buf = self._bar_buffers.get(symbol)
        if not buf:
            return None
        total_vol = sum(b.volume for b in buf if b.volume)
        if total_vol == 0:
            return None
        weighted = sum(
            ((b.high + b.low + b.close) / 3) * b.volume
            for b in buf if b.volume
        )
        return weighted / total_vol

    def get_spread(self, symbol: str) -> Optional[float]:
        ticker = self._tickers.get(symbol)
        if ticker is None:
            return None
        bid = ticker.bid
        ask = ticker.ask
        if bid != bid or ask != ask:
            return None
        return float(ask) - float(bid)

    # ---- Callbacks ----

    def on_bar(self, symbols: List[str], handler: Callable[[Bar], None]) -> str:
        sub_id = str(uuid.uuid4())
        self._callbacks[sub_id] = {'type': 'bar', 'symbols': set(symbols), 'handler': handler}
        self._subscribe_symbols(symbols)
        return sub_id

    def on_quote(self, symbols: List[str], handler: Callable[[Quote], None]) -> str:
        sub_id = str(uuid.uuid4())
        self._callbacks[sub_id] = {'type': 'quote', 'symbols': set(symbols), 'handler': handler}
        self._subscribe_symbols(symbols)
        return sub_id

    def on_trade(self, symbols: List[str], handler: Callable[[Trade], None]) -> str:
        sub_id = str(uuid.uuid4())
        self._callbacks[sub_id] = {'type': 'trade', 'symbols': set(symbols), 'handler': handler}
        self._subscribe_symbols(symbols)
        return sub_id

    def unsubscribe(self, subscription_id: str) -> None:
        self._callbacks.pop(subscription_id, None)

    def get_subscribed_symbols(self) -> set:
        return set(self._subscribed)

    # ---- Internal ----

    def _subscribe_symbols(self, symbols: List[str]) -> None:
        for symbol in symbols:
            if symbol in self._subscribed:
                continue
            try:
                contract = self._resolver.resolve_stock(symbol)
                ticker = self._conn.run_sync(
                    self._req_market_data(contract)
                )
                self._tickers[symbol] = ticker
                self._subscribed.add(symbol)
                logger.debug(f"[IBKR Stream] Subscribed to {symbol}")
            except Exception as e:
                logger.warning(f"[IBKR Stream] Failed to subscribe {symbol}: {e}")

    def _unsubscribe_symbol(self, symbol: str) -> None:
        ticker = self._tickers.pop(symbol, None)
        if ticker:
            try:
                self._conn.ib.cancelMktData(ticker.contract)
            except Exception:
                pass
        self._subscribed.discard(symbol)

    async def _req_market_data(self, contract):
        ticker = self._conn.ib.reqMktData(contract, '', False, False)
        await self._conn.ib.sleepAsync(0.5)
        return ticker

    @staticmethod
    def _mid_price(ticker) -> Optional[float]:
        bid = ticker.bid
        ask = ticker.ask
        if bid != bid or ask != ask:
            return None
        return (float(bid) + float(ask)) / 2
```

- [ ] **Step 2: Update `__init__.py`**

Add to `src/trading/brokers/ibkr/__init__.py`:

```python
from src.trading.brokers.ibkr.streaming import IBKRStreamingProvider
```

And add `"IBKRStreamingProvider"` to `__all__`.

- [ ] **Step 3: Verify import**

```bash
conda activate fintech && python -c "from src.trading.brokers.ibkr.streaming import IBKRStreamingProvider; print('OK')"
```

Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add src/trading/brokers/ibkr/streaming.py src/trading/brokers/ibkr/__init__.py
git commit -m "feat(ibkr): add IBKRStreamingProvider implementing StreamingProviderInterface"
```

---

## Task 10: IBKRBroker

**Files:**
- Create: `src/trading/brokers/ibkr/ibkr_broker.py`

- [ ] **Step 1: Create `ibkr_broker.py`**

Copy from V2 reference at `/tmp/ibkr_files/module_v2/ibkr_v2/src/trading/brokers/ibkr/ibkr_broker.py`.

**Apply fix from spec Section 10:** The `close_stock_position` signature is already correct in the V2 code (`quantity: Optional[int] = None`). Verify this after copying.

- [ ] **Step 2: Update `__init__.py`**

Update `src/trading/brokers/ibkr/__init__.py` to its final form:

```python
"""
IBKR Integration Module for Homeguard.

Provides data download, live streaming, and order execution via Interactive
Brokers through the ib_async library.

Public API:
    IBKRBroker             - BrokerInterface + OptionsTradingInterface
    IBKRDataProvider       - DataProviderInterface for historical data
    IBKRStreamingProvider  - StreamingProviderInterface for real-time data
    IBKRConnectionManager  - Managed connection lifecycle
    IBKRConfig             - Configuration model

Usage:
    from src.trading.brokers.ibkr import IBKRBroker, IBKRConfig

    config = IBKRConfig(port=4002)  # Paper trading gateway
    broker = IBKRBroker(config)
    broker.start()

    account = broker.get_account()
    positions = broker.get_stock_positions()
    chain = broker.get_options_chain('AAPL')

    broker.stop()

Dependencies:
    pip install ib_async  (no ibapi needed)
"""

from src.trading.brokers.ibkr.config import IBKRConfig
from src.trading.brokers.ibkr.connection import IBKRConnectionManager
from src.trading.brokers.ibkr.ibkr_broker import IBKRBroker
from src.trading.brokers.ibkr.data_download import IBKRDataProvider
from src.trading.brokers.ibkr.streaming import IBKRStreamingProvider

__all__ = [
    "IBKRConfig",
    "IBKRConnectionManager",
    "IBKRBroker",
    "IBKRDataProvider",
    "IBKRStreamingProvider",
]
```

- [ ] **Step 3: Verify import**

```bash
conda activate fintech && python -c "from src.trading.brokers.ibkr import IBKRBroker, IBKRConfig; print('OK')"
```

Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add src/trading/brokers/ibkr/ibkr_broker.py src/trading/brokers/ibkr/__init__.py
git commit -m "feat(ibkr): add IBKRBroker implementing stock + options trading interfaces"
```

---

## Task 11: Broker factory update

**Files:**
- Modify: `src/trading/brokers/broker_factory.py:69-75`

- [ ] **Step 1: Replace NotImplementedError with IBKR creation**

In `src/trading/brokers/broker_factory.py`, replace lines 69-75:

```python
        elif broker_type in ['ib', 'interactive_brokers', 'interactivebrokers']:
            # Future implementation
            logger.error("Interactive Brokers not implemented yet")
            raise NotImplementedError(
                "Interactive Brokers support not implemented yet. "
                "To add IB support, implement IBBroker class in ib_broker.py"
            )
```

with:

```python
        elif broker_type in ['ib', 'interactive_brokers', 'interactivebrokers', 'ibkr']:
            from .ibkr import IBKRBroker, IBKRConfig
            logger.info("Creating IBKRBroker instance")
            ibkr_config = IBKRConfig(
                host=config.get('host', '127.0.0.1'),
                port=int(config.get('port', 4002)),
                client_id=int(config.get('client_id', 1)),
                readonly=config.get('readonly', False),
                account=config.get('account', ''),
            )
            broker = IBKRBroker(ibkr_config)
            broker.start()
            return broker
```

- [ ] **Step 2: Run existing broker factory tests**

```bash
conda activate fintech && python -m pytest tests/trading/brokers/ -v -k "factory" --timeout=30
```

Expected: Pass (or no existing factory tests, which is fine).

- [ ] **Step 3: Commit**

```bash
git add src/trading/brokers/broker_factory.py
git commit -m "feat: wire IBKRBroker into BrokerFactory"
```

---

## Task 12: Broker routing

**Files:**
- Create: `src/trading/config/broker_routing.py`
- Create: `config/trading/broker_routing.yaml`
- Create: `tests/trading/brokers/ibkr/test_broker_routing.py`

- [ ] **Step 1: Write routing tests**

Create `tests/trading/brokers/ibkr/test_broker_routing.py`:

```python
"""Tests for broker routing configuration loader."""

import pytest
from unittest.mock import patch, MagicMock

from src.trading.config.broker_routing import load_broker_routing


class TestBrokerRouting:

    def test_load_returns_dict(self, tmp_path):
        config_file = tmp_path / "routing.yaml"
        config_file.write_text("""
brokers:
  alpaca:
    paper: true

strategies:
  omr:
    broker: alpaca

default_broker: alpaca
""")
        with patch('src.trading.config.broker_routing.BrokerFactory') as mock_factory:
            mock_broker = MagicMock()
            mock_factory.create_broker.return_value = mock_broker

            result = load_broker_routing(str(config_file))

        assert 'omr' in result
        assert result['omr'] is mock_broker

    def test_shared_broker_instances(self, tmp_path):
        config_file = tmp_path / "routing.yaml"
        config_file.write_text("""
brokers:
  alpaca:
    paper: true

strategies:
  omr:
    broker: alpaca
  ramp:
    broker: alpaca

default_broker: alpaca
""")
        with patch('src.trading.config.broker_routing.BrokerFactory') as mock_factory:
            mock_broker = MagicMock()
            mock_factory.create_broker.return_value = mock_broker

            result = load_broker_routing(str(config_file))

        assert result['omr'] is result['ramp']

    def test_default_broker_for_unlisted_strategy(self, tmp_path):
        config_file = tmp_path / "routing.yaml"
        config_file.write_text("""
brokers:
  alpaca:
    paper: true

strategies: {}

default_broker: alpaca
""")
        with patch('src.trading.config.broker_routing.BrokerFactory') as mock_factory:
            mock_broker = MagicMock()
            mock_factory.create_broker.return_value = mock_broker

            result = load_broker_routing(str(config_file))

        assert result.get_default() is mock_broker
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda activate fintech && python -m pytest tests/trading/brokers/ibkr/test_broker_routing.py -v
```

Expected: ImportError.

- [ ] **Step 3: Create `src/trading/config/broker_routing.py`**

```python
"""
Broker Routing - Config-driven strategy-to-broker assignment.

Reads broker_routing.yaml and creates shared broker instances.
Strategies get their assigned broker; unlisted strategies get the default.
"""

from typing import Dict, Optional

import yaml

from src.trading.brokers.broker_factory import BrokerFactory
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BrokerRoutingMap:
    """Maps strategy names to broker instances."""

    def __init__(self, strategy_map: Dict, default_broker):
        self._map = strategy_map
        self._default = default_broker

    def __getitem__(self, strategy_name: str):
        return self._map.get(strategy_name, self._default)

    def __contains__(self, strategy_name: str) -> bool:
        return strategy_name in self._map

    def get(self, strategy_name: str, fallback=None):
        return self._map.get(strategy_name, fallback or self._default)

    def get_default(self):
        return self._default


def load_broker_routing(config_path: str = "config/trading/broker_routing.yaml") -> BrokerRoutingMap:
    """
    Load broker routing config and create broker instances.

    Brokers are shared: two strategies assigned to 'alpaca' get the same instance.

    Args:
        config_path: Path to broker_routing.yaml

    Returns:
        BrokerRoutingMap mapping strategy names to broker instances
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    brokers_config = config.get('brokers', {})
    strategies_config = config.get('strategies', {})
    default_broker_name = config.get('default_broker', 'alpaca')

    # Create broker instances (shared)
    broker_instances: Dict[str, object] = {}
    for broker_name, broker_cfg in brokers_config.items():
        try:
            broker_type = broker_cfg.pop('type', broker_name)
            broker_instances[broker_name] = BrokerFactory.create_broker(broker_type, broker_cfg)
            logger.info(f"[Routing] Created broker: {broker_name}")
        except Exception as e:
            logger.error(f"[Routing] Failed to create broker '{broker_name}': {e}")

    # Map strategies to broker instances
    strategy_map = {}
    for strategy_name, strategy_cfg in strategies_config.items():
        broker_name = strategy_cfg.get('broker', default_broker_name)
        if broker_name in broker_instances:
            strategy_map[strategy_name] = broker_instances[broker_name]
        else:
            logger.warning(
                f"[Routing] Strategy '{strategy_name}' references unknown broker "
                f"'{broker_name}', using default"
            )

    default = broker_instances.get(default_broker_name)

    logger.info(
        f"[Routing] Loaded {len(strategy_map)} strategy assignments, "
        f"default broker: {default_broker_name}"
    )

    return BrokerRoutingMap(strategy_map, default)
```

- [ ] **Step 4: Create `config/trading/broker_routing.yaml`**

```yaml
# Strategy-to-broker routing configuration.
#
# Brokers are created once and shared across strategies that use the same broker.
# Strategies not listed here use the default_broker.
#
# Paper vs live is controlled by broker config:
#   Alpaca: paper: true/false
#   IBKR: port: 4002 (paper) / 4001 (live)

brokers:
  alpaca:
    paper: true
    # Credentials from env: ALPACA_PAPER_KEY_ID, ALPACA_PAPER_SECRET_KEY

  # Uncomment when IBKR Gateway is running:
  # ibkr:
  #   port: 4002
  #   client_id: 10
  #   readonly: false

strategies:
  omr:
    broker: alpaca
  ramp:
    broker: alpaca
  cscm:
    broker: coinbase

default_broker: alpaca
```

- [ ] **Step 5: Run tests**

```bash
conda activate fintech && python -m pytest tests/trading/brokers/ibkr/test_broker_routing.py -v
```

Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add src/trading/config/broker_routing.py config/trading/broker_routing.yaml tests/trading/brokers/ibkr/test_broker_routing.py
git commit -m "feat: add config-driven broker routing for strategy-to-broker assignment"
```

---

## Task 13: .env.example update

**Files:**
- Modify: `.env.example`

- [ ] **Step 1: Add IBKR section to `.env.example`**

Append after the DATABENTO section (end of file):

```
# ============================================================================
# IBKR GATEWAY (Interactive Brokers via IB Gateway + IBC)
# ============================================================================
# IB Gateway connection settings (used by IBKRBroker)
# Download IB Gateway from: https://www.interactivebrokers.com/en/trading/ibgateway-stable.php
IBKR_HOST="127.0.0.1"
IBKR_PORT="4002"
IBKR_CLIENT_ID="10"

# IBC automated login credentials (used by homeguard-gateway.service on EC2)
# These are your Interactive Brokers account credentials
IBKR_USERNAME="<YOUR_IBKR_USERNAME>"
IBKR_PASSWORD="<YOUR_IBKR_PASSWORD>"
IBKR_TRADING_MODE="paper"
IBKR_GATEWAY_PORT="4002"
```

- [ ] **Step 2: Commit**

```bash
git add .env.example
git commit -m "docs: add IBKR credential placeholders to .env.example"
```

---

## Task 14: EC2 Gateway deployment files

**Files:**
- Create: `infra/ec2/services/homeguard-gateway.service`
- Create: `infra/ec2/install_ibkr_gateway.sh`
- Create: `config/ibkr/ibc-config.ini.template`

- [ ] **Step 1: Create systemd service**

Create `infra/ec2/services/homeguard-gateway.service`:

```ini
[Unit]
Description=IB Gateway via IBC (automated login)
After=network.target
Before=homeguard-trading.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user

# Source credentials from .env
EnvironmentFile=/home/ec2-user/Homeguard/.env

# Virtual display for Gateway's Swing UI
Environment="DISPLAY=:1"

# Start Xvfb then launch Gateway via IBC
ExecStartPre=/usr/bin/Xvfb :1 -screen 0 1024x768x24 &
ExecStart=/opt/ibc/scripts/ibcstart.sh -g \
    --ibc-ini=/home/ec2-user/Homeguard/config/ibkr/ibc-config.ini \
    --java-path=/usr/lib/jvm/bellsoft-java17-full-aarch64/bin

Restart=on-failure
RestartSec=30
StandardOutput=journal
StandardError=journal
SyslogIdentifier=homeguard-gateway

# Memory limit (Gateway + Xvfb)
MemoryMax=768M

[Install]
WantedBy=homeguard-trading.target
```

- [ ] **Step 2: Create IBC config template**

Create `config/ibkr/ibc-config.ini.template`:

```ini
# IBC configuration for automated IB Gateway login.
# Values with ${...} are resolved from .env at service start.
#
# Docs: https://github.com/IbcAlpha/IBC/blob/master/userguide.md

LogToConsole=yes
FIX=no

IbLoginId=${IBKR_USERNAME}
IbPassword=${IBKR_PASSWORD}
TradingMode=${IBKR_TRADING_MODE}

# Accept incoming API connections (localhost only)
AcceptIncomingConnectionAction=accept

# Don't auto-close; systemd manages lifecycle
ClosedownAt=

# Accept non-brokerage account warning
AcceptNonBrokerageAccountWarning=yes

# If another session exists, take over as primary
ExistingSessionDetectedAction=primary

# JVM memory (match systemd MemoryMax budget)
JavaHeapSize=512

# Override API port from env
OverrideTwsApiPort=${IBKR_GATEWAY_PORT}
```

- [ ] **Step 3: Create installer script**

Create `infra/ec2/install_ibkr_gateway.sh`:

```bash
#!/bin/bash
# Install IB Gateway stack on EC2 (ARM64).
# Idempotent: safe to run multiple times.
# Usage: bash infra/ec2/install_ibkr_gateway.sh

set -e

echo "[1/5] Installing Xvfb..."
sudo yum install -y xorg-x11-server-Xvfb

echo "[2/5] Installing Bellsoft Liberica JDK 17 (aarch64 Full)..."
cd /tmp
if ! java -version 2>&1 | grep -q "17"; then
    wget -q https://download.bell-sw.com/java/17.0.14+10/bellsoft-jdk17.0.14+10-linux-aarch64-full.rpm
    sudo rpm -ivh bellsoft-jdk17.0.14+10-linux-aarch64-full.rpm || true
fi

echo "[3/5] Installing IB Gateway (stable)..."
cd /tmp
if [ ! -d /home/ec2-user/ibgateway ]; then
    wget -q https://download2.interactivebrokers.com/installers/ibgateway/stable-standalone/ibgateway-stable-standalone-linux-x64.sh
    chmod +x ibgateway-stable-standalone-linux-x64.sh
    sudo -u ec2-user bash ibgateway-stable-standalone-linux-x64.sh -q -dir /home/ec2-user/ibgateway
fi

echo "[4/5] Installing IBC..."
IBC_VERSION="3.19.0"
cd /tmp
if [ ! -d /opt/ibc ]; then
    wget -q "https://github.com/IbcAlpha/IBC/releases/download/${IBC_VERSION}/IBCLinux-${IBC_VERSION}.zip"
    sudo mkdir -p /opt/ibc
    sudo unzip -o "IBCLinux-${IBC_VERSION}.zip" -d /opt/ibc
    sudo chmod +x /opt/ibc/scripts/*.sh
fi

echo "[5/5] Installing systemd service..."
sudo cp /home/ec2-user/Homeguard/infra/ec2/services/homeguard-gateway.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable homeguard-gateway

echo ""
echo "[+] IB Gateway stack installed."
echo "    Configure credentials in .env:"
echo "      IBKR_USERNAME, IBKR_PASSWORD, IBKR_TRADING_MODE"
echo "    Then: sudo systemctl start homeguard-gateway"
echo "    Verify: journalctl -u homeguard-gateway -f"
```

- [ ] **Step 4: Commit**

```bash
git add infra/ec2/services/homeguard-gateway.service infra/ec2/install_ibkr_gateway.sh config/ibkr/ibc-config.ini.template
git commit -m "infra: add IB Gateway deployment files for EC2 ARM64"
```

---

## Task 15: Run full test suite and final validation

- [ ] **Step 1: Run all IBKR tests**

```bash
conda activate fintech && python -m pytest tests/trading/brokers/ibkr/ tests/streaming/test_interface.py -v
```

Expected: All unit tests PASS. Integration tests marked `@pytest.mark.ibkr` are skipped.

- [ ] **Step 2: Run full project test suite**

```bash
conda activate fintech && python -m pytest tests/ -v --timeout=120 -x
```

Expected: No regressions in existing tests.

- [ ] **Step 3: Verify all imports work**

```bash
conda activate fintech && python -c "
from src.streaming.interface import StreamingProviderInterface
from src.streaming import LiveDataProvider, StreamingProviderInterface
from src.trading.brokers.ibkr import IBKRBroker, IBKRConfig, IBKRDataProvider, IBKRStreamingProvider
from src.trading.brokers.ibkr.connection import IBKRConnectionManager
from src.trading.brokers.ibkr.contracts import ContractResolver
from src.trading.brokers.ibkr.pacing import PacingManager
from src.trading.brokers.ibkr.errors import classify_error, ibkr_code_to_exception
from src.trading.config.broker_routing import load_broker_routing
print('[+] All imports successful')
"
```

Expected: `[+] All imports successful`

- [ ] **Step 4: Final commit if any cleanup needed**

```bash
git status
```

If clean, done. If any uncommitted changes, commit with appropriate message.
