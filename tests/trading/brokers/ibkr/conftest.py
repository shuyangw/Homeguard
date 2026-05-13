"""Shared fixtures for IBKR tests."""

import os
import socket

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
    """Real IBKRConnectionManager bound to a paper-trading Gateway.

    Used by `@pytest.mark.ibkr` integration tests. Skips when no Gateway
    is reachable on 127.0.0.1:4002 (the paper port) -- this is the case
    on CI and during routine unit-test runs.
    """
    host = os.environ.get("IBKR_TEST_HOST", "127.0.0.1")
    port = int(os.environ.get("IBKR_TEST_PORT", "4002"))

    # Cheap reachability probe; skip integration tests if Gateway is down.
    try:
        with socket.create_connection((host, port), timeout=1.0):
            pass
    except OSError:
        pytest.skip(f"IBKR Gateway not reachable at {host}:{port} -- skipping integration test")

    from src.trading.brokers.ibkr.config import IBKRConfig
    from src.trading.brokers.ibkr.connection import IBKRConnectionManager

    cfg = IBKRConfig(host=host, port=port, client_id=99, readonly=True)
    conn = IBKRConnectionManager(cfg)
    conn.start()
    try:
        yield conn
    finally:
        conn.stop()
