import pytest
from src.backtesting.utils.pre_registration import validate_pre_registration
from src.backtesting.engine.futures_backtest import run_futures_backtest


def test_validate_prereg_flag_gates_the_check():
    # Regression: the walk-forward runner builds per-window configs WITHOUT a
    # pre_registration block and calls run_futures_backtest(validate_prereg=False).
    # With the flag True the gate fires; with False it is skipped (the run then
    # fails LATER on missing data, not on the pre-registration gate).
    cfg = {"strategy": {"name": "FuturesCarry", "universe": ["NOTAREALROOT"]},
           "dates": {"start": "2020-01-01", "end": "2020-02-01"},
           "backtest": {}}
    with pytest.raises(ValueError, match="pre_registration"):
        run_futures_backtest(cfg, register=False, validate_prereg=True)
    with pytest.raises(Exception) as ei:
        run_futures_backtest(cfg, register=False, validate_prereg=False)
    assert "pre_registration" not in str(ei.value)

_GOOD = {
    "pre_registration": {
        "construction": "rank 12-1 return across commodity block",
        "expected_sign": "long_short",
        "hypothesis": "cross-sectional commodity momentum is positive OOS",
    }
}

def test_valid_block_passes():
    validate_pre_registration(_GOOD)  # must not raise

def test_missing_block_raises():
    with pytest.raises(ValueError, match="pre_registration"):
        validate_pre_registration({"strategy": {"name": "X"}})

def test_empty_field_raises():
    bad = {"pre_registration": {"construction": "", "expected_sign": "long", "hypothesis": "h"}}
    with pytest.raises(ValueError, match="construction"):
        validate_pre_registration(bad)

def test_bad_sign_raises():
    bad = {"pre_registration": {"construction": "c", "expected_sign": "up", "hypothesis": "h"}}
    with pytest.raises(ValueError, match="expected_sign"):
        validate_pre_registration(bad)
