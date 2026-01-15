"""
Portfolio Tracker - Monitor portfolio value with persistence.

Tracks portfolio value over time with:
- Scheduled updates during equity hours
- On-demand refresh capability
- Value history persistence (parquet)
- Equity hours detection (9:30 AM - 4:00 PM EST)

While crypto trades 24/7, we track portfolio value during equity hours
to align with typical monitoring schedules. On-demand refresh is always
available via EC2 commands or Discord.

Usage:
    from src.trading.adapters import CSCMPaperAdapter
    from src.trading.portfolio import PortfolioTracker

    adapter = CSCMPaperAdapter()
    tracker = PortfolioTracker(adapter)

    # Get status
    status = tracker.get_status()

    # Update value (respects equity hours unless forced)
    value = tracker.update_value(force=True)

    # Check value history
    history = tracker.get_value_history()
"""

from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, TYPE_CHECKING
import pickle

import pandas as pd

from src.utils.logger import get_logger
from src.utils.timezone import tz

if TYPE_CHECKING:
    from src.trading.adapters.cscm_paper_adapter import CSCMPaperAdapter

logger = get_logger(__name__)


class PortfolioTracker:
    """
    Tracks portfolio value with scheduled updates and persistence.

    Features:
    - Equity hours detection (9:30 AM - 4:00 PM EST, Mon-Fri)
    - Scheduled updates during equity hours
    - On-demand refresh (ignores equity hours)
    - Value history persistence to parquet
    - Integration with CSCM paper trading adapter

    The tracker stores:
    - Current portfolio value
    - Historical values (parquet file)
    - Last update timestamp
    - Tracker state (pickle file)

    Storage locations:
    - State: ~/.homeguard/cache/cscm_portfolio/tracker_state.pkl
    - History: ~/.homeguard/cache/cscm_portfolio/portfolio_history.parquet

    Usage:
        tracker = PortfolioTracker(adapter)

        # Auto-update during equity hours
        tracker.update_value()

        # Force update anytime
        tracker.update_value(force=True)

        # Get detailed status
        status = tracker.get_status()
    """

    # Equity hours (EST)
    EQUITY_HOURS_START = time(9, 30)  # 9:30 AM EST
    EQUITY_HOURS_END = time(16, 0)    # 4:00 PM EST

    # Update interval during equity hours
    UPDATE_INTERVAL_MINUTES = 15

    # Storage paths
    CACHE_DIR = Path.home() / '.homeguard' / 'cache' / 'cscm_portfolio'
    VALUE_HISTORY_FILE = CACHE_DIR / 'portfolio_history.parquet'
    STATE_FILE = CACHE_DIR / 'tracker_state.pkl'

    # History retention
    MAX_HISTORY_DAYS = 90  # Keep 90 days of history
    SAVE_HISTORY_INTERVAL = 10  # Save every N updates

    def __init__(self, adapter: 'CSCMPaperAdapter'):
        """
        Initialize portfolio tracker.

        Args:
            adapter: CSCM paper trading adapter to track
        """
        self._adapter = adapter
        self._last_update: Optional[datetime] = None
        self._current_value: float = 0.0
        self._value_history: List[Dict] = []
        self._update_count: int = 0

        # Load persisted state
        self._load_state()

        logger.info("[PortfolioTracker] Initialized")
        if self._last_update:
            logger.info(f"[PortfolioTracker] Last update: {self._last_update}")

    def is_equity_hours(self) -> bool:
        """
        Check if current time is within equity trading hours.

        Returns:
            True if within 9:30 AM - 4:00 PM EST, Monday-Friday
        """
        now = tz.now()

        # Only Monday-Friday (0=Monday, 6=Sunday)
        if now.weekday() >= 5:  # Saturday or Sunday
            return False

        current_time = now.time()
        return self.EQUITY_HOURS_START <= current_time <= self.EQUITY_HOURS_END

    def is_market_day(self) -> bool:
        """
        Check if today is a market day (weekday).

        Returns:
            True if Monday-Friday
        """
        return tz.now().weekday() < 5

    def should_update(self) -> bool:
        """
        Check if we should update based on interval.

        Returns:
            True if enough time has passed since last update
        """
        if self._last_update is None:
            return True

        elapsed = tz.now() - self._last_update
        return elapsed >= timedelta(minutes=self.UPDATE_INTERVAL_MINUTES)

    def update_value(self, force: bool = False) -> float:
        """
        Update portfolio value.

        Args:
            force: If True, update regardless of equity hours or interval

        Returns:
            Current portfolio value in USD
        """
        # Check if we should update
        if not force:
            if not self.is_equity_hours():
                logger.debug("[PortfolioTracker] Outside equity hours, skipping update")
                return self._current_value

            if not self.should_update():
                logger.debug("[PortfolioTracker] Update interval not reached")
                return self._current_value

        try:
            # Get current portfolio value from adapter
            total_value = self._adapter._get_total_portfolio_value()

            if total_value <= 0:
                logger.warning("[PortfolioTracker] Got zero portfolio value")
                return self._current_value

            # Record update
            self._current_value = total_value
            self._last_update = tz.now()
            self._update_count += 1

            # Get additional details
            positions = self._adapter._get_current_positions()
            cash = self._adapter._get_account_balance()

            # Add to history
            self._value_history.append({
                'timestamp': self._last_update.isoformat(),
                'value': total_value,
                'cash': cash,
                'positions_count': len(positions),
                'is_equity_hours': self.is_equity_hours(),
            })

            # Trim history if too long
            self._trim_history()

            # Persist state periodically
            if self._update_count % self.SAVE_HISTORY_INTERVAL == 0:
                self._save_history()

            # Always save state
            self._save_state()

            logger.info(f"[PortfolioTracker] Updated value: ${total_value:,.2f}")
            return total_value

        except Exception as e:
            logger.error(f"[PortfolioTracker] Failed to update value: {e}")
            return self._current_value

    def get_status(self) -> Dict[str, Any]:
        """
        Get current portfolio status.

        Returns:
            Dict with current value, positions, regime, etc.
        """
        try:
            positions = self._adapter._get_current_positions()
            cash = self._adapter._get_account_balance()

            # Get regime from adapter if available
            regime = 'unknown'
            try:
                status = self._adapter.get_status()
                regime = status.get('regime', 'unknown')
            except (AttributeError, KeyError, ValueError):
                pass

            return {
                'current_value': self._current_value,
                'last_update': str(self._last_update) if self._last_update else None,
                'is_equity_hours': self.is_equity_hours(),
                'is_market_day': self.is_market_day(),
                'positions': {k: float(v) for k, v in positions.items()},
                'positions_count': len(positions),
                'cash': cash,
                'regime': regime,
                'update_count': self._update_count,
                'history_length': len(self._value_history),
            }

        except Exception as e:
            logger.error(f"[PortfolioTracker] Failed to get status: {e}")
            return {
                'current_value': self._current_value,
                'error': str(e),
            }

    def get_value_history(
        self,
        days: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get portfolio value history as DataFrame.

        Args:
            days: Number of days to return (None for all)

        Returns:
            DataFrame with timestamp, value, cash, positions_count
        """
        if not self._value_history:
            return pd.DataFrame()

        df = pd.DataFrame(self._value_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()

        if days:
            # Use timezone-naive cutoff for comparison
            cutoff = datetime.now() - timedelta(days=days)
            # Make index timezone-naive for comparison if needed
            if df.index.tz is not None:
                cutoff = pd.Timestamp(cutoff).tz_localize(df.index.tz)
            df = df[df.index >= cutoff]

        return df

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary from value history.

        Returns:
            Dict with returns, drawdown, etc.
        """
        df = self.get_value_history()

        if df.empty or len(df) < 2:
            return {'error': 'Insufficient history'}

        try:
            first_value = df['value'].iloc[0]
            last_value = df['value'].iloc[-1]
            peak_value = df['value'].max()

            return {
                'first_value': first_value,
                'current_value': last_value,
                'peak_value': peak_value,
                'total_return_pct': (last_value / first_value - 1) * 100,
                'max_drawdown_pct': ((peak_value - df['value'].min()) / peak_value) * 100,
                'current_drawdown_pct': ((peak_value - last_value) / peak_value) * 100,
                'history_start': df.index.min().isoformat(),
                'history_end': df.index.max().isoformat(),
                'data_points': len(df),
            }
        except Exception as e:
            return {'error': str(e)}

    def _trim_history(self) -> None:
        """Trim history to max retention period."""
        if not self._value_history:
            return

        cutoff = tz.now() - timedelta(days=self.MAX_HISTORY_DAYS)
        cutoff_str = cutoff.isoformat()

        self._value_history = [
            h for h in self._value_history
            if h['timestamp'] >= cutoff_str
        ]

    def _save_state(self) -> None:
        """Save tracker state to disk."""
        try:
            self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

            with open(self.STATE_FILE, 'wb') as f:
                pickle.dump({
                    'last_update': self._last_update,
                    'current_value': self._current_value,
                    'update_count': self._update_count,
                }, f)

            logger.debug("[PortfolioTracker] Saved state")

        except Exception as e:
            logger.warning(f"[PortfolioTracker] Failed to save state: {e}")

    def _load_state(self) -> None:
        """Load tracker state from disk."""
        try:
            if self.STATE_FILE.exists():
                with open(self.STATE_FILE, 'rb') as f:
                    state = pickle.load(f)
                    self._last_update = state.get('last_update')
                    self._current_value = state.get('current_value', 0.0)
                    self._update_count = state.get('update_count', 0)

            # Load history if exists
            if self.VALUE_HISTORY_FILE.exists():
                df = pd.read_parquet(self.VALUE_HISTORY_FILE)
                self._value_history = df.to_dict('records')

        except Exception as e:
            logger.warning(f"[PortfolioTracker] Failed to load state: {e}")

    def _save_history(self) -> None:
        """Save value history to parquet file."""
        try:
            if not self._value_history:
                return

            self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

            df = pd.DataFrame(self._value_history)
            df.to_parquet(self.VALUE_HISTORY_FILE, index=False)

            logger.debug(f"[PortfolioTracker] Saved {len(df)} history records")

        except Exception as e:
            logger.warning(f"[PortfolioTracker] Failed to save history: {e}")

    def reset(self) -> None:
        """Reset tracker state and history."""
        self._last_update = None
        self._current_value = 0.0
        self._value_history = []
        self._update_count = 0

        # Remove persisted files
        try:
            if self.STATE_FILE.exists():
                self.STATE_FILE.unlink()
            if self.VALUE_HISTORY_FILE.exists():
                self.VALUE_HISTORY_FILE.unlink()
            logger.info("[PortfolioTracker] Reset complete")
        except Exception as e:
            logger.warning(f"[PortfolioTracker] Reset failed: {e}")
