"""RBA/RBNZ historical backfill (defect 3 of the #35 Kalman diagnostic).

Before the backfill the calendar held ONE RBA and ONE RBNZ date, both 2025, so
the AUD/NZD '+-7d central-bank entry blackout' was a no-op in 12 of 13
walk-forward windows: the spec that was gated was not the spec that was
described.
"""
import datetime as dt

from src.data.macro_calendar import load_cb_decisions


def test_rba_rbnz_cover_the_backtest_era():
    cb = load_cb_decisions()
    for bank, per_year_min in (("RBA", 8), ("RBNZ", 7)):
        dates = cb[bank]
        assert len(dates) > 100, f"{bank} still effectively empty: {len(dates)}"
        years = {d.year for d in dates}
        assert years >= set(range(2011, 2027)), f"{bank} missing years: {sorted(years)}"
        for y in (2012, 2018, 2023):
            n = sum(1 for d in dates if d.year == y)
            assert n >= per_year_min, f"{bank} only {n} dates in {y}"


def test_dates_are_the_expected_weekday():
    """RBA decisions are Tuesdays, RBNZ reviews are Wednesdays."""
    cb = load_cb_decisions()
    assert all(d.weekday() == 1 for d in cb["RBA"] if d.year >= 2011)
    assert all(d.weekday() == 2 for d in cb["RBNZ"] if d.year >= 2011)


def test_blackout_is_now_actually_active_historically():
    from src.strategies.advanced.fx_audnzd_pairs import AudNzdPairs
    s = AudNzdPairs()
    bd = s._blackout_dates()
    # a mid-sample year that previously had zero blackout coverage
    hits = sum(1 for day in (dt.date(2015, 1, 5) + dt.timedelta(weeks=w) for w in range(52))
               if s._in_blackout(day, bd))
    assert hits > 0, "blackout must no longer be a no-op in the historical sample"


# --------------------------------------------- spike cleaner causal mode (defect 6)

def test_causal_mode_uses_no_future_information():
    """The default cleaner reads r[t+1]; the causal mode must not. Perturbing
    bars AFTER t must not change whether t is flagged."""
    import numpy as np, pandas as pd
    from src.data.artifacts.spike_clean import scrub_spike_reverts
    rng = np.random.default_rng(0)
    s = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.004, 300))))
    s.iloc[150] *= 1.30                       # a bad print
    _, causal_a = scrub_spike_reverts(s, causal=True)
    s2 = s.copy()
    s2.iloc[200:] *= 2.0                      # mutate the FUTURE only
    _, causal_b = scrub_spike_reverts(s2, causal=True)
    assert [d for d in causal_a if d <= 150] == [d for d in causal_b if d <= 150]


def test_default_mode_preserves_a_real_non_reverting_jump():
    """Why the look-ahead default is kept: a real devaluation does NOT revert and
    must survive, or backtests lose genuine tail risk."""
    import numpy as np, pandas as pd
    from src.data.artifacts.spike_clean import scrub_spike_reverts
    s = pd.Series([1.0] * 50 + [1.25] * 50)   # permanent 25% step, never reverts
    _, flagged = scrub_spike_reverts(s)
    assert flagged == [], "a real regime break must not be nulled"


def test_default_mode_still_catches_a_reverting_spike():
    import pandas as pd
    from src.data.artifacts.spike_clean import scrub_spike_reverts
    s = pd.Series([1.0] * 50 + [1.30] + [1.0] * 49)   # spike and full revert
    _, flagged = scrub_spike_reverts(s)
    assert flagged == [50]
