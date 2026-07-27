"""Forward-paper log: evidence accumulated FORWARD, never backfilled.

A spec routed to forward paper is being tested on a sample that does not exist
yet. The single property that makes it forward validation rather than a backtest
is that an observation cannot be recorded for a date at or before the spec's
lock. Without that guard the log is just a backtest with extra ceremony, and it
would silently consume trials it never declared.
"""
import datetime as dt
import json

import pytest

from src.backtesting.validation.forward_paper import (ForwardPaperError,
                                                      ForwardSpec,
                                                      load_observations,
                                                      month_end_events,
                                                      quarter_end_events,
                                                      record_observation,
                                                      summarise)

_LOCK = dt.date(2026, 7, 26)


@pytest.fixture
def spec():
    return ForwardSpec(name="test-fix", locked_on=_LOCK,
                       params={"pairs": ["EURUSD"], "edge_bps": 6.0})


@pytest.fixture
def log(tmp_path):
    return tmp_path / "observations.jsonl"


def test_cannot_record_before_the_lock_date(spec, log):
    """THE guard. Backfilling would turn this into an undeclared backtest."""
    with pytest.raises(ForwardPaperError, match="on or before"):
        record_observation(spec, dt.date(2026, 6, 30), signal=1, return_bps=4.0,
                           path=log)


def test_cannot_record_on_the_lock_date_itself(spec, log):
    with pytest.raises(ForwardPaperError):
        record_observation(spec, _LOCK, signal=1, return_bps=4.0, path=log)


def test_records_an_observation_after_the_lock(spec, log):
    record_observation(spec, dt.date(2026, 7, 31), signal=1, return_bps=4.0, path=log)
    rows = load_observations(log)
    assert len(rows) == 1
    assert rows[0]["spec"] == "test-fix" and rows[0]["return_bps"] == 4.0


def test_the_log_is_append_only(spec, log):
    record_observation(spec, dt.date(2026, 7, 31), signal=1, return_bps=4.0, path=log)
    record_observation(spec, dt.date(2026, 8, 31), signal=-1, return_bps=-2.0, path=log)
    assert len(load_observations(log)) == 2


def test_the_same_event_cannot_be_recorded_twice(spec, log):
    """Re-recording an event is how a bad observation quietly disappears."""
    record_observation(spec, dt.date(2026, 7, 31), signal=1, return_bps=4.0, path=log)
    with pytest.raises(ForwardPaperError, match="already recorded"):
        record_observation(spec, dt.date(2026, 7, 31), signal=1, return_bps=9.9,
                           path=log)


def test_observation_carries_the_spec_fingerprint(spec, log):
    """If the params change, old observations must not silently pass as the same
    spec."""
    record_observation(spec, dt.date(2026, 7, 31), signal=1, return_bps=4.0, path=log)
    stored = json.loads(log.read_text().splitlines()[0])
    assert stored["spec_hash"] == spec.fingerprint()

    altered = ForwardSpec(name="test-fix", locked_on=_LOCK,
                          params={"pairs": ["EURUSD"], "edge_bps": 9.0})
    assert altered.fingerprint() != spec.fingerprint()


def test_summary_reports_count_and_mean_without_claiming_significance(spec, log):
    for i, r in enumerate([4.0, -2.0, 6.0]):
        record_observation(spec, dt.date(2026, 8 + i, 28), signal=1, return_bps=r,
                           path=log)
    s = summarise(log)["test-fix"]
    assert s["n"] == 3
    assert s["mean_bps"] == pytest.approx(8.0 / 3)
    assert s["conclusive"] is False, "3 observations can never be conclusive"


def test_month_end_events_are_business_days():
    ev = month_end_events(dt.date(2026, 1, 1), dt.date(2026, 12, 31))
    assert len(ev) == 12
    assert all(d.weekday() < 5 for d in ev)
    assert ev[0].month == 1 and ev[-1].month == 12


def test_quarter_end_events_are_four_per_year():
    ev = quarter_end_events(dt.date(2026, 1, 1), dt.date(2026, 12, 31))
    assert len(ev) == 4
    assert [d.month for d in ev] == [3, 6, 9, 12]
    assert all(d.weekday() < 5 for d in ev)


def test_quarter_ends_are_a_subset_of_month_ends():
    lo, hi = dt.date(2026, 1, 1), dt.date(2027, 12, 31)
    assert set(quarter_end_events(lo, hi)) <= set(month_end_events(lo, hi))
