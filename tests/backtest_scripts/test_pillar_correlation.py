from datetime import date

from scripts.backtest_scripts.pillar_correlation import daily_return_correlation


def test_perfectly_correlated():
    dts = [date(2020, 1, d) for d in range(1, 8)]
    eq = [100, 101, 102, 101, 103, 104, 103]
    assert daily_return_correlation(eq, eq, dts, dts) > 0.999


def test_common_dates_only():
    a_d = [date(2020, 1, d) for d in range(1, 6)]
    b_d = [date(2020, 1, d) for d in range(3, 8)]
    a_e = [100, 110, 100, 110, 100]
    b_e = [50, 55, 50, 55, 50]
    r = daily_return_correlation(a_e, b_e, a_d, b_d)
    assert -1.0 <= r <= 1.0
