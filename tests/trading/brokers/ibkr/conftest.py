"""Shared fixtures for IBKR tests."""

import os

import pytest

from src.trading.brokers.ibkr.pacing import PacingManager


@pytest.fixture
def pacer():
    """Fresh PacingManager with default settings for unit tests."""
    p = PacingManager(max_per_10min=58, identical_cooldown=0.0)
    yield p
    p.clear()


@pytest.fixture
def ibkr_connection():
    """Live IBKRConnectionManager for @pytest.mark.ibkr integration tests.

    Auto-skips unless HOMEGUARD_RUN_IBKR_TESTS=1 is set AND the gateway is
    reachable. Keeps integration tests off the default CI path.
    """
    if os.environ.get("HOMEGUARD_RUN_IBKR_TESTS") != "1":
        pytest.skip("Set HOMEGUARD_RUN_IBKR_TESTS=1 to run IBKR integration tests")

    from src.trading.brokers.ibkr.config import IBKRConfig
    from src.trading.brokers.ibkr.connection import IBKRConnectionManager

    mgr = IBKRConnectionManager(IBKRConfig())
    try:
        mgr.start()
    except Exception as e:
        pytest.skip(f"IBKR gateway unreachable: {e}")
    yield mgr
    mgr.stop()
