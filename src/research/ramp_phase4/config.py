"""HarnessConfig dataclass for Phase B research harness.

Pure data; no logic except validation in __post_init__.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Literal


_DEFAULT_REGIME_POSITIONS = {
    'STRONG_BULL':   'normal',
    'WEAK_BULL':     'normal',
    'SIDEWAYS':      'normal',
    'UNPREDICTABLE': 'normal',
    'BEAR':          'cash',
    'SAFE_MODE':     'hold',
}

_ALLOWED_POSITION_VALUES = frozenset({'normal', 'cash', 'hold'})


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
    delta_rebalance_pct: float = 0.0
    # V12 additions:
    regime_positions: Dict[str, str] = field(
        default_factory=lambda: dict(_DEFAULT_REGIME_POSITIONS)
    )
    min_regime_days: int = 0

    def __post_init__(self) -> None:
        for regime, mode in self.regime_positions.items():
            if mode not in _ALLOWED_POSITION_VALUES:
                raise ValueError(
                    f"regime_positions[{regime!r}] = {mode!r} is not in "
                    f"{sorted(_ALLOWED_POSITION_VALUES)}. Ticker/strategy "
                    f"values are reserved for V13+."
                )
        if self.min_regime_days < 0:
            raise ValueError(
                f"min_regime_days must be >= 0, got {self.min_regime_days}"
            )
