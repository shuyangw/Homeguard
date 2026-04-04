"""Tests for Combinatorial Purged Cross-Validation module."""

from datetime import date
from math import comb

import numpy as np
import pytest

from src.backtesting.validation.cpcv import (
    CPCVResult,
    CPCVSplit,
    generate_cpcv_splits,
    run_cpcv,
)


class TestGenerateCPCVSplits:

    def test_correct_number_of_splits_6_blocks(self):
        splits = generate_cpcv_splits(
            date(2022, 1, 1), date(2024, 12, 31),
            n_blocks=6, n_test=2,
        )
        assert len(splits) == comb(6, 2)  # 15

    def test_correct_number_of_splits_13_blocks(self):
        splits = generate_cpcv_splits(
            date(2018, 7, 1), date(2024, 12, 31),
            n_blocks=13, n_test=2,
        )
        assert len(splits) == comb(13, 2)  # 78

    def test_split_indices_are_sequential(self):
        splits = generate_cpcv_splits(
            date(2020, 1, 1), date(2022, 12, 31),
            n_blocks=6, n_test=2,
        )
        for i, split in enumerate(splits):
            assert split.split_index == i

    def test_each_split_has_correct_test_blocks(self):
        splits = generate_cpcv_splits(
            date(2020, 1, 1), date(2022, 12, 31),
            n_blocks=4, n_test=2,
        )
        for split in splits:
            assert len(split.test_blocks) == 2
            assert len(split.test_ranges) == 2

    def test_test_ranges_are_within_bounds(self):
        start = date(2020, 1, 1)
        end = date(2022, 12, 31)
        splits = generate_cpcv_splits(start, end, n_blocks=6, n_test=2)
        for split in splits:
            for b_start, b_end in split.test_ranges:
                assert b_start >= start
                assert b_end <= end
                assert b_start < b_end

    def test_all_block_combinations_covered(self):
        splits = generate_cpcv_splits(
            date(2020, 1, 1), date(2022, 12, 31),
            n_blocks=4, n_test=2,
        )
        combos = {tuple(s.test_blocks) for s in splits}
        expected = {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}
        assert combos == expected


class TestRunCPCV:

    def test_positive_signal_passes(self):
        """A consistently positive strategy should pass CPCV."""
        def always_positive(start, end):
            return 0.5  # always positive Sharpe

        result = run_cpcv(
            always_positive,
            date(2020, 1, 1), date(2022, 12, 31),
            n_blocks=4, n_test=2, purge_days=5,
        )
        assert isinstance(result, CPCVResult)
        assert result.pct_positive == 1.0
        assert result.passed is True

    def test_negative_signal_fails(self):
        """A consistently negative strategy should fail CPCV."""
        def always_negative(start, end):
            return -0.3

        result = run_cpcv(
            always_negative,
            date(2020, 1, 1), date(2022, 12, 31),
            n_blocks=4, n_test=2, purge_days=5,
        )
        assert result.pct_positive == 0.0
        assert result.passed is False

    def test_mixed_signal_counts_correctly(self):
        """Mixed positive/negative results are counted correctly."""
        call_count = [0]

        def alternating(start, end):
            """Return positive for early dates, negative for late dates."""
            call_count[0] += 1
            # Use the start year to determine sign
            return 0.5 if start.year <= 2020 else -0.3

        result = run_cpcv(
            alternating,
            date(2020, 1, 1), date(2022, 12, 31),
            n_blocks=4, n_test=2, purge_days=5,
        )
        assert len(result.oos_sharpes) == comb(4, 2)
        # Some splits have early blocks (positive), some have late (negative)
        # The average per split determines sign -- at least some should differ
        positive_count = sum(1 for s in result.oos_sharpes if s > 0)
        negative_count = sum(1 for s in result.oos_sharpes if s <= 0)
        assert positive_count > 0
        assert negative_count > 0

    def test_progress_callback_called(self):
        calls = []

        def track_progress(idx, total, sharpe):
            calls.append((idx, total, sharpe))

        def constant(start, end):
            return 0.3

        run_cpcv(
            constant,
            date(2020, 1, 1), date(2022, 12, 31),
            n_blocks=4, n_test=2, purge_days=5,
            progress_callback=track_progress,
        )
        assert len(calls) == comb(4, 2)

    def test_exception_in_backtest_handled(self):
        """If backtest raises, it should be caught and logged as 0.0."""
        def failing(start, end):
            raise RuntimeError("test error")

        result = run_cpcv(
            failing,
            date(2020, 1, 1), date(2022, 12, 31),
            n_blocks=4, n_test=2, purge_days=5,
        )
        assert all(s == 0.0 for s in result.oos_sharpes)
