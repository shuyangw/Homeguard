"""
Tests for IBKR PacingManager.

These are pure unit tests -- no IBKR connection required.
"""

import time
import threading
import pytest

from src.trading.brokers.ibkr.pacing import PacingManager


class TestPacingManagerBasic:
    """Basic rate limiting behaviour."""

    def test_first_request_passes_immediately(self, pacer):
        start = time.time()
        pacer.acquire("req1")
        elapsed = time.time() - start
        assert elapsed < 0.1

    def test_headroom_decreases_after_request(self, pacer):
        initial = pacer.headroom
        pacer.acquire("req1")
        assert pacer.headroom == initial - 1

    def test_requests_in_window_tracks_count(self, pacer):
        assert pacer.requests_in_window == 0
        pacer.acquire("a")
        pacer.acquire("b")
        pacer.acquire("c")
        assert pacer.requests_in_window == 3

    def test_clear_resets_state(self, pacer):
        pacer.acquire("x")
        pacer.acquire("y")
        pacer.clear()
        assert pacer.requests_in_window == 0
        assert pacer.headroom == pacer._max_10min


class TestIdenticalRequestCooldown:
    """Identical request deduplication / cooldown."""

    def test_identical_key_blocks(self):
        pacer = PacingManager(max_per_10min=100, identical_cooldown=1.0)
        pacer.acquire("same_key")

        start = time.time()
        pacer.acquire("same_key")
        elapsed = time.time() - start

        # Should have waited ~1 second for cooldown
        assert elapsed >= 0.9
        pacer.clear()

    def test_different_keys_dont_block(self):
        pacer = PacingManager(max_per_10min=100, identical_cooldown=2.0)
        pacer.acquire("key_a")

        start = time.time()
        pacer.acquire("key_b")
        elapsed = time.time() - start

        assert elapsed < 0.5
        pacer.clear()

    def test_none_key_skips_dedup(self, pacer):
        """If no key provided, skip identical-request check."""
        start = time.time()
        pacer.acquire(None)
        pacer.acquire(None)
        elapsed = time.time() - start
        assert elapsed < 0.5


class TestWindowLimit:
    """10-minute window enforcement."""

    def test_window_blocks_at_limit(self):
        pacer = PacingManager(max_per_10min=3, identical_cooldown=0.0)

        # Fill the window
        pacer.acquire("a")
        pacer.acquire("b")
        pacer.acquire("c")

        assert pacer.headroom == 0
        assert pacer.requests_in_window == 3
        pacer.clear()


class TestViolationBackoff:
    """Post-violation exponential backoff."""

    def test_violation_increases_count(self, pacer):
        assert pacer._violation_count == 0
        pacer.on_violation()
        assert pacer._violation_count == 1
        pacer.on_violation()
        assert pacer._violation_count == 2


class TestRequestKey:
    """Request key hashing."""

    def test_same_params_same_key(self):
        k1 = PacingManager.request_key("SPY", "1 D", "1 min", "TRADES", "20260101 16:00:00")
        k2 = PacingManager.request_key("SPY", "1 D", "1 min", "TRADES", "20260101 16:00:00")
        assert k1 == k2

    def test_different_params_different_key(self):
        k1 = PacingManager.request_key("SPY", "1 D", "1 min", "TRADES", "20260101")
        k2 = PacingManager.request_key("QQQ", "1 D", "1 min", "TRADES", "20260101")
        assert k1 != k2

    def test_key_is_deterministic(self):
        k1 = PacingManager.request_key("AAPL", "1 Y", "1 day", "TRADES", "")
        k2 = PacingManager.request_key("AAPL", "1 Y", "1 day", "TRADES", "")
        assert k1 == k2


class TestThreadSafety:
    """Verify pacing works under concurrent access."""

    def test_concurrent_acquires(self):
        pacer = PacingManager(max_per_10min=50, identical_cooldown=0.0)
        errors = []

        def worker(n):
            try:
                for i in range(10):
                    pacer.acquire(f"thread_{n}_req_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Errors during concurrent access: {errors}"
        assert pacer.requests_in_window == 50
        pacer.clear()
