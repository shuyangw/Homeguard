"""Combinatorial Purged Cross-Validation (CPCV).

Splits the backtest timeline into N blocks. For each C(N, n_test) combination,
the selected blocks are test sets and the rest are train. A purge gap prevents
data leakage at train/test boundaries.

Generic framework: accepts a callable that runs a backtest over a date range
and returns a Sharpe ratio (or other scalar metric).
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from itertools import combinations
from typing import Callable, List, Tuple

import numpy as np

from src.utils.logger import get_logger

logger = get_logger()


@dataclass
class CPCVSplit:
    split_index: int
    test_blocks: List[int]
    test_ranges: List[Tuple[date, date]]
    oos_sharpe: float = 0.0


@dataclass
class CPCVResult:
    n_blocks: int
    n_test_blocks: int
    n_splits: int
    purge_days: int
    splits: List[CPCVSplit] = field(default_factory=list)
    oos_sharpes: List[float] = field(default_factory=list)

    @property
    def pct_positive(self) -> float:
        if not self.oos_sharpes:
            return 0.0
        return sum(1 for s in self.oos_sharpes if s > 0) / len(self.oos_sharpes)

    @property
    def mean_sharpe(self) -> float:
        if not self.oos_sharpes:
            return 0.0
        return float(np.mean(self.oos_sharpes))

    @property
    def std_sharpe(self) -> float:
        if not self.oos_sharpes:
            return 0.0
        return float(np.std(self.oos_sharpes))

    @property
    def passed(self) -> bool:
        return self.pct_positive > 0.70


def generate_cpcv_splits(
    start_date: date,
    end_date: date,
    n_blocks: int = 13,
    n_test: int = 2,
    purge_days: int = 30,
) -> List[CPCVSplit]:
    """Generate all C(n_blocks, n_test) CPCV splits.

    Args:
        start_date: First date of the backtest window.
        end_date: Last date of the backtest window.
        n_blocks: Number of equal-length temporal blocks.
        n_test: Number of blocks used as test in each split.
        purge_days: Gap in calendar days between train and test boundaries.

    Returns:
        List of CPCVSplit objects with test block indices and date ranges.
    """
    total_days = (end_date - start_date).days
    block_length = total_days // n_blocks

    blocks = []
    for i in range(n_blocks):
        b_start = start_date + timedelta(days=i * block_length)
        if i == n_blocks - 1:
            b_end = end_date
        else:
            b_end = start_date + timedelta(days=(i + 1) * block_length - 1)
        blocks.append((b_start, b_end))

    splits = []
    for idx, combo in enumerate(combinations(range(n_blocks), n_test)):
        test_ranges = [blocks[b] for b in combo]
        splits.append(CPCVSplit(
            split_index=idx,
            test_blocks=list(combo),
            test_ranges=test_ranges,
        ))

    logger.info(
        f"CPCV: {len(splits)} splits from {n_blocks} blocks, "
        f"n_test={n_test}, purge={purge_days}d"
    )
    return splits


def run_cpcv(
    run_backtest_fn: Callable[[date, date], float],
    start_date: date,
    end_date: date,
    n_blocks: int = 13,
    n_test: int = 2,
    purge_days: int = 30,
    progress_callback: Callable[[int, int, float], None] = None,
) -> CPCVResult:
    """Run CPCV by executing backtest on each test block of each split.

    Args:
        run_backtest_fn: Callable(start_date, end_date) -> Sharpe ratio.
            Must use frozen parameters (no re-optimization).
        start_date: Start of full backtest window.
        end_date: End of full backtest window.
        n_blocks: Number of temporal blocks.
        n_test: Number of test blocks per split.
        purge_days: Purge gap in calendar days.
        progress_callback: Optional fn(split_idx, total_splits, sharpe) called
            after each split completes.

    Returns:
        CPCVResult with per-split OOS Sharpe ratios.
    """
    splits = generate_cpcv_splits(start_date, end_date, n_blocks, n_test, purge_days)

    result = CPCVResult(
        n_blocks=n_blocks,
        n_test_blocks=n_test,
        n_splits=len(splits),
        purge_days=purge_days,
    )

    for split in splits:
        # Run backtest on each test block, average the Sharpes
        block_sharpes = []
        for b_start, b_end in split.test_ranges:
            # Apply purge: shrink test block by purge_days on each side
            purged_start = b_start + timedelta(days=purge_days)
            purged_end = b_end - timedelta(days=purge_days)

            if purged_start >= purged_end:
                logger.debug(
                    f"Split {split.split_index}: block {b_start}-{b_end} "
                    f"too short after purge, skipping"
                )
                continue

            try:
                sharpe = run_backtest_fn(purged_start, purged_end)
                block_sharpes.append(sharpe)
            except Exception as e:
                logger.error(
                    f"Split {split.split_index}: backtest failed for "
                    f"{purged_start}-{purged_end}: {e}"
                )
                block_sharpes.append(0.0)

        if block_sharpes:
            split.oos_sharpe = float(np.mean(block_sharpes))
        else:
            split.oos_sharpe = 0.0

        result.splits.append(split)
        result.oos_sharpes.append(split.oos_sharpe)

        if progress_callback:
            progress_callback(
                split.split_index, len(splits), split.oos_sharpe
            )

        if (split.split_index + 1) % 10 == 0:
            logger.info(
                f"CPCV progress: {split.split_index + 1}/{len(splits)} splits, "
                f"pct_positive={result.pct_positive:.1%}"
            )

    logger.info(
        f"CPCV complete: {len(splits)} splits, "
        f"mean OOS Sharpe={result.mean_sharpe:.3f}, "
        f"pct_positive={result.pct_positive:.1%}"
    )
    return result
