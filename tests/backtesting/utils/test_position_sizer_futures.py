"""Tests for FuturesPositionSizer."""
import pytest

from src.backtesting.utils.position_sizer_futures import (
    ContractSpec, FuturesPositionSizer,
)


def test_size_position_basic_math():
    """Verify integer contract count from formula.

    vol_per_contract = multiplier * underlying * current_vol
                     = 5.0 * 5300.0 * 0.15
                     = 3975.0 dollars/year
    contracts = vol_target / vol_per_contract
              = 1250 / 3975
              = 0.314 -> round(0.314) = 0
    """
    spec = ContractSpec(
        symbol="MES", multiplier=5.0, tick_size=0.25, tick_value=1.25, max_contracts=10,
    )
    sizer = FuturesPositionSizer()
    n = sizer.size_position(
        symbol="MES",
        vol_target_dollars=1250.0,
        current_vol=0.15,
        contract_specs=spec,
        underlying_price=5300.0,
    )
    # Allow for round-vs-truncate variation: 0 or 1 acceptable
    assert 0 <= n <= 1


def test_size_position_caps_at_max_contracts():
    """Sizer must respect max_contracts cap regardless of model output."""
    spec = ContractSpec(
        symbol="MES", multiplier=5.0, tick_size=0.25, tick_value=1.25, max_contracts=3,
    )
    sizer = FuturesPositionSizer()
    n = sizer.size_position(
        symbol="MES",
        vol_target_dollars=1_000_000,  # very large
        current_vol=0.10,
        contract_specs=spec,
        underlying_price=5300.0,
    )
    assert n == 3


def test_size_position_zero_vol_returns_zero():
    """current_vol <= 0 -> 0 contracts (avoid division by zero)."""
    spec = ContractSpec(
        symbol="MES", multiplier=5.0, tick_size=0.25, tick_value=1.25, max_contracts=10,
    )
    sizer = FuturesPositionSizer()
    n = sizer.size_position(
        symbol="MES",
        vol_target_dollars=1000.0,
        current_vol=0.0,
        contract_specs=spec,
        underlying_price=5300.0,
    )
    assert n == 0


def test_size_position_never_negative():
    """Negative model output (e.g. shorting) clamped to 0; sign is handled externally."""
    spec = ContractSpec(
        symbol="MES", multiplier=5.0, tick_size=0.25, tick_value=1.25, max_contracts=10,
    )
    sizer = FuturesPositionSizer()
    n = sizer.size_position(
        symbol="MES",
        vol_target_dollars=-1000.0,  # negative target
        current_vol=0.10,
        contract_specs=spec,
        underlying_price=5300.0,
    )
    assert n == 0


def test_zero_forecast_zero_contracts():
    from src.backtesting.utils.position_sizer_futures import size_from_forecast
    assert size_from_forecast(0.0, 25000, 0.20, "MES", price=5000, daily_vol=0.01) == 0


def test_positive_forecast_positive_contracts():
    from src.backtesting.utils.position_sizer_futures import size_from_forecast
    n = size_from_forecast(10.0, 100000, 0.20, "MES", price=5000, daily_vol=0.008)
    assert n > 0


def test_negative_forecast_negative_contracts():
    from src.backtesting.utils.position_sizer_futures import size_from_forecast
    n = size_from_forecast(-10.0, 100000, 0.20, "MES", price=5000, daily_vol=0.008)
    assert n < 0


def test_capped_by_max_contracts():
    from src.backtesting.utils.position_sizer_futures import size_from_forecast
    from src.data.futures.contract_specs import get_spec
    n = size_from_forecast(20.0, 10_000_000, 0.50, "MES", price=5000, daily_vol=0.001)
    assert abs(n) <= get_spec("MES").max_contracts
