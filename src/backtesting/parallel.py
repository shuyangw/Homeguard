"""Deterministic process-parallel map for CPU-bound backtest work.

Uses ProcessPoolExecutor (not threads -- the backtest loop is GIL-bound).
Results are returned in INPUT order, so callers whose aggregation is
order-sensitive (e.g. the walk-forward stitching OOS segments by window)
get byte-identical results to a serial run. Worker count is capped by
get_default_workers(). The first worker exception propagates (fail-fast).
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, List, Optional, Sequence, TypeVar

from src.backtesting.optimization.data_loader import get_default_workers

T = TypeVar("T")
R = TypeVar("R")


def parallel_map(fn: Callable[[T], R], items: Sequence[T],
                 max_workers: Optional[int] = None) -> List[R]:
    items = list(items)
    if not items:
        return []
    if max_workers is None:
        max_workers = min(get_default_workers(), len(items))
    if max_workers <= 1:
        return [fn(x) for x in items]

    results: List[Any] = [None] * len(items)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fn, item): i for i, item in enumerate(items)}
        for fut in as_completed(futures):
            idx = futures[fut]
            results[idx] = fut.result()  # re-raises worker exception in the parent
    return results
