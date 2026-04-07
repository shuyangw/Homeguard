"""Shared fixtures for IBKR tests."""

import pytest

from src.trading.brokers.ibkr.pacing import PacingManager


@pytest.fixture
def pacer():
    """Fresh PacingManager with default settings for unit tests."""
    p = PacingManager(max_per_10min=58, identical_cooldown=0.0)
    yield p
    p.clear()
