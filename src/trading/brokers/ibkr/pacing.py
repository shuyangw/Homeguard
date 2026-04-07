"""
IBKR Historical Data Pacing Manager.

IBKR enforces strict rate limits on historical data requests:
    - Max 60 requests per 10-minute window (we use 58 for headroom)
    - Max 6 requests per 2 seconds
    - Identical requests must be 15 seconds apart
    - Max ~3 concurrent requests (queued server-side)

Violating these results in error 162 and a temporary ban (~10 minutes).
This module makes pacing invisible to callers: acquire() blocks until
it's safe to proceed.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import deque
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PacingManager:
    """
    Token-bucket rate limiter tuned for IBKR historical data pacing rules.

    Usage:
        pacer = PacingManager()
        key = pacer.request_key('SPY', '1 D', '1 min', 'TRADES', '')
        pacer.acquire(key)       # blocks if necessary
        bars = ib.reqHistoricalData(...)

    Thread-safe: Multiple threads can call acquire() concurrently.
    """

    def __init__(
        self,
        max_per_10min: int = 58,
        max_per_2sec: int = 5,
        identical_cooldown: float = 15.0,
    ):
        self._max_10min = max_per_10min
        self._max_2sec = max_per_2sec
        self._identical_cooldown = identical_cooldown

        # Sliding window of request timestamps
        self._timestamps: deque = deque()
        self._recent_requests: dict[str, float] = {}  # key -> last request time
        self._lock = threading.Lock()

        # Violation tracking (for exponential backoff after 162)
        self._violation_count = 0
        self._last_violation_time = 0.0

    @staticmethod
    def request_key(
        symbol: str,
        duration: str,
        bar_size: str,
        what_to_show: str,
        end_datetime: str,
    ) -> str:
        """
        Compute a hash key for a historical data request.
        Used to detect and throttle identical requests.
        """
        raw = f"{symbol}|{duration}|{bar_size}|{what_to_show}|{end_datetime}"
        return hashlib.md5(raw.encode()).hexdigest()

    def acquire(self, request_key: Optional[str] = None) -> None:
        """
        Block until it's safe to make a historical data request.

        Enforces:
            1. Identical-request cooldown (15s)
            2. 10-minute rolling window (58 requests max)
            3. 2-second burst limit (5 requests max)
            4. Post-violation backoff (if we recently hit a 162 error)
        """
        with self._lock:
            now = time.time()

            # 0. Post-violation backoff
            if self._violation_count > 0:
                backoff = min(10 * (2 ** (self._violation_count - 1)), 600)
                elapsed_since_violation = now - self._last_violation_time
                if elapsed_since_violation < backoff:
                    wait = backoff - elapsed_since_violation
                    logger.warning(
                        f"[IBKR Pacing] Post-violation backoff: {wait:.1f}s "
                        f"(violation #{self._violation_count})"
                    )
                    self._lock.release()
                    time.sleep(wait)
                    self._lock.acquire()
                    now = time.time()

            # 1. Identical-request cooldown
            if request_key and request_key in self._recent_requests:
                elapsed = now - self._recent_requests[request_key]
                if elapsed < self._identical_cooldown:
                    sleep_time = self._identical_cooldown - elapsed + 0.1
                    logger.debug(
                        f"[IBKR Pacing] Identical request cooldown: {sleep_time:.1f}s"
                    )
                    self._lock.release()
                    time.sleep(sleep_time)
                    self._lock.acquire()
                    now = time.time()

            # 2. 10-minute rolling window
            cutoff_10min = now - 600  # 10 minutes
            while self._timestamps and self._timestamps[0] < cutoff_10min:
                self._timestamps.popleft()

            if len(self._timestamps) >= self._max_10min:
                oldest = self._timestamps[0]
                wait = 600 - (now - oldest) + 0.5
                if wait > 0:
                    logger.info(
                        f"[IBKR Pacing] 10-min window full ({len(self._timestamps)} requests). "
                        f"Waiting {wait:.1f}s"
                    )
                    self._lock.release()
                    time.sleep(wait)
                    self._lock.acquire()
                    now = time.time()
                    # Re-prune after sleep
                    cutoff_10min = now - 600
                    while self._timestamps and self._timestamps[0] < cutoff_10min:
                        self._timestamps.popleft()

            # 3. 2-second burst limit
            cutoff_2sec = now - 2.0
            recent_burst = sum(1 for t in self._timestamps if t > cutoff_2sec)
            if recent_burst >= self._max_2sec:
                wait = 2.0 - (now - self._timestamps[-self._max_2sec]) + 0.1
                if wait > 0:
                    logger.debug(f"[IBKR Pacing] Burst limit: waiting {wait:.1f}s")
                    self._lock.release()
                    time.sleep(wait)
                    self._lock.acquire()
                    now = time.time()

            # Record this request
            self._timestamps.append(now)
            if request_key:
                self._recent_requests[request_key] = now

            # Decay violation count over time (reset after 10 clean minutes)
            if self._violation_count > 0 and (now - self._last_violation_time) > 600:
                self._violation_count = 0
                logger.info("[IBKR Pacing] Violation count reset after clean 10-min window")

    def on_violation(self) -> None:
        """
        Called when we receive error 162 (pacing violation).
        Increases backoff for subsequent requests.
        """
        with self._lock:
            self._violation_count += 1
            self._last_violation_time = time.time()
            logger.warning(
                f"[IBKR Pacing] Pacing violation #{self._violation_count}. "
                f"Increasing backoff."
            )

    @property
    def requests_in_window(self) -> int:
        """Current count of requests in the 10-minute window."""
        with self._lock:
            cutoff = time.time() - 600
            return sum(1 for t in self._timestamps if t > cutoff)

    @property
    def headroom(self) -> int:
        """How many more requests can be made before hitting the limit."""
        return max(0, self._max_10min - self.requests_in_window)

    def clear(self) -> None:
        """Reset all state (for tests)."""
        with self._lock:
            self._timestamps.clear()
            self._recent_requests.clear()
            self._violation_count = 0
            self._last_violation_time = 0.0
