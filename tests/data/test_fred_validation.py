import pandas as pd
import pytest
from src.data.acquisition.plugins.fred_rates import (
    validate_fred_series, FredValidationError,
)


def test_rejects_empty():
    with pytest.raises(FredValidationError):
        validate_fred_series(pd.Series(dtype=float), "TEST")


def test_rejects_implausible_rate():
    s = pd.Series([250.0, 300.0], index=pd.to_datetime(["2020-01-01", "2020-02-01"]))
    with pytest.raises(FredValidationError):
        validate_fred_series(s, "TEST")


def test_accepts_plausible():
    s = pd.Series([3.5, 3.75], index=pd.to_datetime(["2020-01-01", "2020-02-01"]))
    validate_fred_series(s, "TEST")  # no raise
