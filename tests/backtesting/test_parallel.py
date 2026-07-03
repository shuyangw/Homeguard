import pytest
from src.backtesting.parallel import parallel_map


def _double(x):
    return x * 2


def _boom(x):
    if x == 3:
        raise ValueError("boom at 3")
    return x


def test_returns_input_order():
    items = list(range(10))
    assert parallel_map(_double, items, max_workers=4) == [x * 2 for x in items]


def test_serial_path_max_workers_1():
    items = list(range(5))
    assert parallel_map(_double, items, max_workers=1) == [0, 2, 4, 6, 8]


def test_empty_items():
    assert parallel_map(_double, [], max_workers=4) == []


def test_worker_exception_propagates():
    with pytest.raises(ValueError):
        parallel_map(_boom, [1, 2, 3, 4], max_workers=2)
