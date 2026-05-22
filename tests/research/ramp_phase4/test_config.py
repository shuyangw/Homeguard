"""Tests for HarnessConfig dataclass."""
from datetime import datetime
from pathlib import Path

from src.research.ramp_phase4.config import HarnessConfig


def test_harness_config_construction_defaults():
    cfg = HarnessConfig(
        start_date=datetime(2017, 1, 1),
        end_date=datetime(2026, 4, 30),
        universe_csv=Path('config/universes/sp500-2025.csv'),
        initial_capital=100000.0,
        cost_bps_per_side=5.0,
    )
    assert cfg.timing_mode == 'near_close'
    assert cfg.rebalance_frequency == 'daily'
    assert cfg.rounding_mode == 'whole_share'
    assert cfg.min_trade_value_usd == 100.0


def test_harness_config_is_frozen():
    cfg = HarnessConfig(
        start_date=datetime(2017, 1, 1),
        end_date=datetime(2026, 4, 30),
        universe_csv=Path('config/universes/sp500-2025.csv'),
        initial_capital=100000.0,
        cost_bps_per_side=5.0,
    )
    try:
        cfg.cost_bps_per_side = 10.0
    except Exception as e:
        msg = str(e).lower()
        assert 'frozen' in msg or 'attribute' in msg or 'field' in msg
    else:
        raise AssertionError("HarnessConfig should be frozen")


def test_harness_config_has_delta_rebalance_pct_default_zero():
    from datetime import datetime
    from pathlib import Path
    from src.research.ramp_phase4.config import HarnessConfig
    cfg = HarnessConfig(
        start_date=datetime(2017, 1, 1),
        end_date=datetime(2026, 4, 30),
        universe_csv=Path('config/universes/sp500-2025.csv'),
        initial_capital=100000.0,
        cost_bps_per_side=5.0,
    )
    assert cfg.delta_rebalance_pct == 0.0
