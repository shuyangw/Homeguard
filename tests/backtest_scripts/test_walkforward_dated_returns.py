import numpy as np
import pandas as pd
from datetime import date

from scripts.backtest_scripts.run_carver_walkforward import _oos_returns, _oos_returns_dated


def test_dated_matches_bare():
    eq = [100.0, 101.0, 102.0, 101.0, 103.0]
    dts = [date(2020, 1, d) for d in (2, 3, 6, 7, 8)]
    test_start = date(2020, 1, 3)

    bare = _oos_returns(eq, dts, test_start)
    dated = _oos_returns_dated(eq, dts, test_start)

    assert isinstance(dated, pd.Series)
    assert np.allclose(dated.to_numpy(), bare, rtol=0, atol=1e-12)
    assert len(dated.index) == len(bare)
    assert len(dated) == len(bare)


def test_dated_empty_matches_bare_empty():
    eq = [100.0, 101.0]
    dts = [date(2020, 1, 2), date(2020, 1, 3)]
    test_start = date(2021, 1, 1)

    bare = _oos_returns(eq, dts, test_start)
    dated = _oos_returns_dated(eq, dts, test_start)

    assert isinstance(dated, pd.Series)
    assert bare.size == 0
    assert len(dated) == 0
