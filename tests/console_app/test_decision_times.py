"""The console's decision times must match the times the adapters hardcode."""

import re
from pathlib import Path

import pytest

from tools.console.decision_times import DECISION_TIMES

REPO_ROOT = Path(__file__).resolve().parents[2]
ADAPTERS = {
    "ramp": REPO_ROOT / "src" / "trading" / "adapters" / "ramp_live_adapter.py",
    "omr": REPO_ROOT / "src" / "trading" / "adapters" / "omr_live_adapter.py",
}
_BEGIN_DECISION = re.compile(r'_begin_decision\("scheduled_\w+", schedule_time="(\d\d:\d\d)"\)')


@pytest.mark.parametrize("strategy", sorted(ADAPTERS))
def test_decision_times_match_the_adapter(strategy):
    adapter_times = _BEGIN_DECISION.findall(ADAPTERS[strategy].read_text(encoding="utf-8"))

    assert adapter_times, f"no scheduled decisions found in {ADAPTERS[strategy].name}"
    assert sorted(adapter_times) == sorted(DECISION_TIMES[strategy])


def test_every_strategy_with_times_has_an_adapter_check():
    assert set(DECISION_TIMES) == set(ADAPTERS)
