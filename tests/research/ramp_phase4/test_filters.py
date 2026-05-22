"""Tests for src/research/ramp_phase4/filters.py."""
from datetime import datetime
import pandas as pd

from src.research.ramp_phase4.engine import HarnessState
from src.research.ramp_phase4.filters import rank_buffer


def test_rank_buffer_keeps_held_name_within_buffer_zone():
    # Held position AAA. Today's full ranking puts AAA at rank 12 (top_n=10, buffer=5
    # -> buffer zone is ranks 1..15). AAA should be retained.
    state = HarnessState(cash_usd=0.0, positions={'AAA': 100.0})
    proposed = {f'X{i}': 0.1 for i in range(10)}  # 10 proposed names
    ranking = pd.Series({**{f'X{i}': i + 1 for i in range(10)}, 'AAA': 12})
    result = rank_buffer(
        proposed_targets=proposed,
        state=state,
        buffer_size=5,
        universe_ranking=ranking,
        top_n=10,
    )
    assert 'AAA' in result


def test_rank_buffer_drops_held_name_past_buffer():
    # AAA held but ranks 20 with top_n=10, buffer=5 -> buffer zone is 1..15. Drop.
    state = HarnessState(cash_usd=0.0, positions={'AAA': 100.0})
    proposed = {f'X{i}': 0.1 for i in range(10)}
    ranking = pd.Series({**{f'X{i}': i + 1 for i in range(10)}, 'AAA': 20})
    result = rank_buffer(
        proposed_targets=proposed,
        state=state,
        buffer_size=5,
        universe_ranking=ranking,
        top_n=10,
    )
    assert 'AAA' not in result


def test_rank_buffer_renormalizes_to_sum_one():
    # 10 proposed + 1 retained = 11 names total; each should be 1/11.
    state = HarnessState(cash_usd=0.0, positions={'AAA': 100.0})
    proposed = {f'X{i}': 0.1 for i in range(10)}
    ranking = pd.Series({**{f'X{i}': i + 1 for i in range(10)}, 'AAA': 12})
    result = rank_buffer(
        proposed_targets=proposed,
        state=state,
        buffer_size=5,
        universe_ranking=ranking,
        top_n=10,
    )
    assert abs(sum(result.values()) - 1.0) < 1e-9
    expected_weight = 1.0 / 11
    for w in result.values():
        assert abs(w - expected_weight) < 1e-9


def test_rank_buffer_no_retained_when_no_held_position():
    # No held positions -> output is just the proposed names.
    state = HarnessState(cash_usd=100000.0, positions={})
    proposed = {f'X{i}': 0.1 for i in range(10)}
    ranking = pd.Series({f'X{i}': i + 1 for i in range(10)})
    result = rank_buffer(
        proposed_targets=proposed,
        state=state,
        buffer_size=5,
        universe_ranking=ranking,
        top_n=10,
    )
    assert set(result.keys()) == set(proposed.keys())
    assert abs(sum(result.values()) - 1.0) < 1e-9
