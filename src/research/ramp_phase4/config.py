"""HarnessConfig dataclass for Phase B research harness.

Pure data; no logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class HarnessConfig:
    """Configuration for a single run of run_variant.

    Run the harness once per (variant, cost-tier, timing-mode) combination.
    """
    start_date: datetime
    end_date: datetime
    universe_csv: Path
    initial_capital: float
    cost_bps_per_side: float
    timing_mode: Literal['near_close', 'one_day_lag'] = 'near_close'
    rebalance_frequency: Literal['daily', 'weekly_friday', 'weekly_wednesday'] = 'daily'
    rounding_mode: Literal['whole_share', 'dollar_weight'] = 'whole_share'
    min_trade_value_usd: float = 100.0
    # Phase C Wave 1: V06 delta-rebalance percent-of-portfolio threshold.
    # Floor per trade = max(min_trade_value_usd, total_value * delta_rebalance_pct).
    # Default 0.0 preserves V01/V03 baseline behavior.
    delta_rebalance_pct: float = 0.0
