"""Each scheduled action must fire once per day, not on every check inside its window.

should_run_now matches a +/-60 s window around each execution time and only de-duplicated
'entry' and 'exit', so RAMP's 'rebalance' re-fired on every 15 s check: 7-8 full rebalances
inside the 15:55 window on 2026-09-08 and 2026-09-09.
"""

from datetime import datetime

import pytz

from scripts.trading.run_live_paper_trading import LiveTradingRunner
from src.utils.timezone import tz

EASTERN = pytz.timezone("US/Eastern")


class StubAdapter:
    def __init__(self, execution_times):
        self._schedule = {"execution_times": execution_times, "market_hours_only": True}

    def get_schedule(self):
        return self._schedule


def make_runner(tmp_path, execution_times):
    return LiveTradingRunner(StubAdapter(execution_times), log_dir=tmp_path, assume_market_open=True)


def actions_at(runner, monkeypatch, moments):
    fired = []
    for year, month, day, hour, minute, second in moments:
        moment = EASTERN.localize(datetime(year, month, day, hour, minute, second))
        monkeypatch.setattr(tz, "now", lambda moment=moment: moment)
        fired.append(runner.should_run_now())
    return fired


def test_rebalance_fires_once_across_every_check_in_its_window(tmp_path, monkeypatch):
    runner = make_runner(tmp_path, [{"time": "15:55", "action": "rebalance"}])
    checks = [(2026, 10, 7, 15, 54, s) for s in (5, 20, 35, 50)] + [(2026, 10, 7, 15, 55, s) for s in (5, 20, 35, 50)]

    fired = actions_at(runner, monkeypatch, checks)

    assert fired == ["rebalance"] + [None] * 7


def test_rebalance_fires_again_the_next_day(tmp_path, monkeypatch):
    runner = make_runner(tmp_path, [{"time": "15:55", "action": "rebalance"}])

    fired = actions_at(runner, monkeypatch, [(2026, 10, 7, 15, 55, 0), (2026, 10, 7, 15, 55, 30), (2026, 10, 8, 15, 55, 0)])

    assert fired == ["rebalance", None, "rebalance"]


def test_outside_the_window_nothing_fires(tmp_path, monkeypatch):
    runner = make_runner(tmp_path, [{"time": "15:55", "action": "rebalance"}])

    fired = actions_at(runner, monkeypatch, [(2026, 10, 7, 15, 53, 59), (2026, 10, 7, 15, 56, 1)])

    assert fired == [None, None]


def test_entry_and_exit_each_fire_once_per_day(tmp_path, monkeypatch):
    runner = make_runner(tmp_path, [{"time": "09:31", "action": "exit"}, {"time": "15:50", "action": "entry"}])
    checks = [(2026, 10, 7, 9, 31, 0), (2026, 10, 7, 9, 31, 15), (2026, 10, 7, 15, 50, 0), (2026, 10, 7, 15, 50, 30)]

    fired = actions_at(runner, monkeypatch, checks)

    assert fired == ["exit", None, "entry", None]


def test_a_restart_inside_the_window_does_not_fire_again(tmp_path, monkeypatch):
    schedule = [{"time": "15:55", "action": "rebalance"}]
    first = make_runner(tmp_path, schedule)
    assert actions_at(first, monkeypatch, [(2026, 10, 7, 15, 55, 0)]) == ["rebalance"]

    restarted = make_runner(tmp_path, schedule)

    assert actions_at(restarted, monkeypatch, [(2026, 10, 7, 15, 55, 30)]) == [None]


def test_a_restart_on_a_later_day_fires_normally(tmp_path, monkeypatch):
    schedule = [{"time": "15:55", "action": "rebalance"}]
    actions_at(make_runner(tmp_path, schedule), monkeypatch, [(2026, 10, 7, 15, 55, 0)])

    restarted = make_runner(tmp_path, schedule)

    assert actions_at(restarted, monkeypatch, [(2026, 10, 8, 15, 55, 0)]) == ["rebalance"]


def test_an_unreadable_fired_actions_file_does_not_stop_the_runner(tmp_path, monkeypatch):
    (tmp_path / "fired_actions.json").write_text("{not json")
    runner = make_runner(tmp_path, [{"time": "15:55", "action": "rebalance"}])

    assert actions_at(runner, monkeypatch, [(2026, 10, 7, 15, 55, 0)]) == ["rebalance"]
