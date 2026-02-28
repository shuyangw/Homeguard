"""Background worker that periodically takes portfolio snapshots.

Runs as a daemon thread, calling the broker at a configurable interval
during market hours and writing snapshots via PortfolioLogger.
"""

import threading
import time as _time
from datetime import time as dt_time

from src.trading.logging.portfolio_logger import PortfolioLogger
from src.utils.logger import get_logger
from src.utils.timezone import tz

logger = get_logger()

MARKET_OPEN = dt_time(9, 30)
MARKET_CLOSE = dt_time(16, 0)


class PortfolioSnapshotWorker:
    """Daemon thread that takes periodic portfolio snapshots.

    Snapshots are taken at a configurable interval during US equity
    market hours (Mon-Fri, 9:30 AM - 4:00 PM ET). An initial snapshot
    is taken when the worker starts, and a final snapshot is taken
    when it stops.

    Args:
        broker: Broker instance with get_account() and get_positions().
        portfolio_logger: PortfolioLogger instance for writing snapshots.
        interval_minutes: Minutes between snapshots (default 15).
    """

    def __init__(
        self,
        broker,
        portfolio_logger: PortfolioLogger,
        interval_minutes: int = 15,
    ) -> None:
        self._broker = broker
        self._logger = portfolio_logger
        self._interval_seconds = interval_minutes * 60
        self._running = False
        self._thread = None

    def _is_market_hours(self, current_time: dt_time, weekday: int) -> bool:
        """Check whether the given time falls within US equity market hours.

        Args:
            current_time: Time of day to check.
            weekday: Day of week (0=Monday .. 6=Sunday).

        Returns:
            True if Mon-Fri and 9:30 AM <= current_time < 4:00 PM.
        """
        if weekday > 4:
            return False
        return MARKET_OPEN <= current_time < MARKET_CLOSE

    def _take_snapshot(self) -> None:
        """Fetch account/positions from broker and log a snapshot.

        Never raises; all exceptions are caught and logged.
        """
        try:
            account = self._broker.get_account()
            positions = self._broker.get_positions()
            self._logger.log_snapshot(account, positions)
        except Exception as exc:
            logger.error(f"PortfolioSnapshotWorker._take_snapshot failed: {exc}")

    def _run(self) -> None:
        """Thread target: take snapshots at the configured interval."""
        logger.info("PortfolioSnapshotWorker started")

        # Initial snapshot on start
        self._take_snapshot()

        while self._running:
            # Sleep in 1-second increments for responsive stop
            elapsed = 0
            while self._running and elapsed < self._interval_seconds:
                _time.sleep(1)
                elapsed += 1

            if not self._running:
                break

            now = tz.now()
            if self._is_market_hours(now.time(), now.weekday()):
                self._take_snapshot()

        logger.info("PortfolioSnapshotWorker stopped")

    def start(self) -> None:
        """Start the background snapshot worker."""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the worker, take a final snapshot, and join the thread."""
        self._running = False

        # Final snapshot before shutdown
        self._take_snapshot()

        if self._thread is not None:
            self._thread.join(timeout=5)
