import pytest
from src.backtesting.utils.pre_registration import validate_pre_registration

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
