import pytest
from src.strategies.registry import get_strategy_class

_SP_A = [
    "FuturesXSMomentum", "FuturesReversal", "FuturesTurnOfMonth",
    "FuturesSameMonthSeasonality", "FuturesCarryTrend",
]

@pytest.mark.parametrize("name", _SP_A)
def test_strategy_resolves_and_has_forecast_panel(name):
    cls = get_strategy_class(name)
    strat = cls(["ES", "NQ"])
    assert hasattr(strat, "forecast_panel")
