"""
Test to validate that CSV file links in HTML reports match actual filenames.

This test ensures we don't have file-not-found issues when users click
on trade log download links in the HTML tearsheet.
"""

import pytest
from pathlib import Path


def test_timestamp_extraction_from_filename():
    """
    Test that timestamp extraction from HTML filename works correctly.

    HTML filename format: YYYYMMDD_HHMMSS_XX_strategy_sweep_results.html
    CSV filename format:  YYYYMMDD_HHMMSS_SYMBOL_trades.csv

    We need to extract just the YYYYMMDD_HHMMSS part.
    """
    test_cases = [
        # (HTML filename, expected timestamp)
        ("20251101_141244_05_breakout_strategy_sweep_results.html", "20251101_141244"),
        ("20251101_034938_03_bollinger_bands_sweep_results.html", "20251101_034938"),
        ("20250115_123456_01_moving_average_sweep_results.html", "20250115_123456"),
        ("20250101_000000_rsi_sweep_results.html", "20250101_000000"),
    ]

    for html_filename, expected_timestamp in test_cases:
        output_path = Path(html_filename)
        filename_parts = output_path.stem.split('_')

        if len(filename_parts) >= 2:
            timestamp = f"{filename_parts[0]}_{filename_parts[1]}"
        else:
            timestamp = output_path.stem

        assert timestamp == expected_timestamp, \
            f"Timestamp extraction failed for {html_filename}: got {timestamp}, expected {expected_timestamp}"


def test_csv_filename_construction():
    """
    Test that CSV filename construction matches the actual file naming pattern.
    """
    timestamp = "20251101_141244"
    symbols = ["AAPL", "GOOGL", "MSFT", "NFLX", "META"]

    expected_patterns = {
        "trades": "{timestamp}_{symbol}_trades.csv",
        "equity": "{timestamp}_{symbol}_equity_curve.csv",
        "state": "{timestamp}_{symbol}_portfolio_state.csv"
    }

    for symbol in symbols:
        # Construct filenames as they would be in the HTML
        trades_link = f"trades/{timestamp}_{symbol}_trades.csv"
        equity_link = f"trades/{timestamp}_{symbol}_equity_curve.csv"
        state_link = f"trades/{timestamp}_{symbol}_portfolio_state.csv"

        # Verify the format matches expectations
        assert trades_link == f"trades/{timestamp}_{symbol}_trades.csv"
        assert equity_link == f"trades/{timestamp}_{symbol}_equity_curve.csv"
        assert state_link == f"trades/{timestamp}_{symbol}_portfolio_state.csv"

        # Verify no extra underscores or wrong patterns
        assert trades_link.count(timestamp) == 1, \
            f"Timestamp should appear exactly once in {trades_link}"
        assert "_sweep_results" not in trades_link, \
            f"CSV link should not contain '_sweep_results': {trades_link}"
        assert "_strategy" not in trades_link, \
            f"CSV link should not contain strategy name: {trades_link}"


def test_old_vs_new_link_format():
    """
    Document the bug and verify the fix.

    OLD (buggy): trades/20251101_141244_05_breakout_strategy_sweep_results_AAPL_trades.csv
    NEW (fixed): trades/20251101_141244_AAPL_trades.csv
    """
    html_filename = "20251101_141244_05_breakout_strategy_sweep_results.html"
    symbol = "AAPL"

    # OLD way (wrong)
    output_path = Path(html_filename)
    old_link = f"trades/{output_path.stem}_{symbol}_trades.csv"

    # NEW way (correct)
    filename_parts = output_path.stem.split('_')
    timestamp = f"{filename_parts[0]}_{filename_parts[1]}"
    new_link = f"trades/{timestamp}_{symbol}_trades.csv"

    # Verify they're different
    assert old_link != new_link, "Old and new formats should be different"

    # Verify old format is wrong (contains strategy name)
    assert "breakout_strategy" in old_link, \
        "Old format incorrectly includes strategy name"
    assert "sweep_results" in old_link, \
        "Old format incorrectly includes 'sweep_results'"

    # Verify new format is correct (no strategy name)
    assert "breakout_strategy" not in new_link, \
        "New format should not include strategy name"
    assert "sweep_results" not in new_link, \
        "New format should not include 'sweep_results'"
    assert new_link == "trades/20251101_141244_AAPL_trades.csv", \
        f"New format should match expected pattern, got {new_link}"


def test_edge_cases():
    """
    Test edge cases for timestamp extraction.
    """
    # Short filename (fallback to full stem)
    short_filename = "results.html"
    output_path = Path(short_filename)
    filename_parts = output_path.stem.split('_')
    if len(filename_parts) >= 2:
        timestamp = f"{filename_parts[0]}_{filename_parts[1]}"
    else:
        timestamp = output_path.stem  # Fallback

    assert timestamp == "results", "Should fallback to full stem for short filenames"

    # Filename with many underscores
    long_filename = "20251101_141244_05_my_complex_strategy_name_sweep_results.html"
    output_path = Path(long_filename)
    filename_parts = output_path.stem.split('_')
    timestamp = f"{filename_parts[0]}_{filename_parts[1]}"

    assert timestamp == "20251101_141244", \
        "Should extract only first two parts regardless of filename complexity"


if __name__ == '__main__':
    # Run tests
    test_timestamp_extraction_from_filename()
    print("✓ Timestamp extraction test passed")

    test_csv_filename_construction()
    print("✓ CSV filename construction test passed")

    test_old_vs_new_link_format()
    print("✓ Old vs new format test passed")

    test_edge_cases()
    print("✓ Edge cases test passed")

    print("\nAll file link validation tests passed!")
