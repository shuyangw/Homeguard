"""
Test that sweep backtest CSVs have the same format as portfolio backtest CSVs.
"""

import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from backtesting.engine.trade_logger import TradeLogger
from pathlib import Path
import tempfile


def test_csv_format_consistency():
    """Verify that both Portfolio and MultiAssetPortfolio produce same CSV format."""

    print("\nTesting CSV format consistency between Portfolio and MultiAssetPortfolio...\n")

    # ============================================================
    # Test 1: Custom Portfolio (used in sweep mode)
    # ============================================================
    print("1. Testing Custom Portfolio format (sweep mode):")

    class MockPortfolio:
        def __init__(self):
            # Simulate Portfolio.trades as list of dicts with 'type' field
            self.trades = [
                # Entry (buy)
                {
                    'type': 'entry',
                    'timestamp': pd.Timestamp('2023-01-03 10:00:00', tz='UTC'),
                    'price': 100.0,
                    'shares': 10,
                    'pnl': None,
                    'pnl_pct': None
                },
                # Exit (sell)
                {
                    'type': 'exit',
                    'timestamp': pd.Timestamp('2023-01-03 15:00:00', tz='UTC'),
                    'price': 105.0,
                    'shares': 10,
                    'pnl': 50.0,
                    'pnl_pct': 0.05,
                    'exit_reason': 'strategy_signal'
                }
            ]

    portfolio = MockPortfolio()

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        portfolio_path = Path(f.name)

    try:
        TradeLogger.export_trades_csv(portfolio, portfolio_path, symbol='AAPL')
        portfolio_df = pd.read_csv(portfolio_path)

        print(f"   Columns: {list(portfolio_df.columns)}")
        print(f"   Shape: {portfolio_df.shape}")
        print(f"   Directions: {portfolio_df['Direction'].unique()}")
        print("\n   Sample rows:")
        print(portfolio_df.to_string(index=False))
        print()

    finally:
        if portfolio_path.exists():
            portfolio_path.unlink()

    # ============================================================
    # Test 2: MultiAssetPortfolio (used in portfolio mode)
    # ============================================================
    print("\n2. Testing MultiAssetPortfolio format (portfolio mode):")

    class MockMultiAssetPortfolio:
        def __init__(self):
            # Simulate MultiAssetPortfolio.trades as DataFrame with entry_timestamp
            self.trades = pd.DataFrame({
                'symbol': ['AAPL'],
                'entry_timestamp': [pd.Timestamp('2023-01-03 10:00:00', tz='UTC')],
                'exit_timestamp': [pd.Timestamp('2023-01-03 15:00:00', tz='UTC')],
                'entry_price': [100.0],
                'exit_price': [105.0],
                'shares': [10],
                'pnl': [50.0],
                'pnl_pct': [0.05],
                'exit_reason': ['strategy_signal']
            })

    multi_portfolio = MockMultiAssetPortfolio()

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        multi_path = Path(f.name)

    try:
        TradeLogger.export_trades_csv(multi_portfolio, multi_path)
        multi_df = pd.read_csv(multi_path)

        print(f"   Columns: {list(multi_df.columns)}")
        print(f"   Shape: {multi_df.shape}")
        print(f"   Directions: {multi_df['Direction'].unique()}")
        print("\n   Sample rows:")
        print(multi_df.to_string(index=False))
        print()

    finally:
        if multi_path.exists():
            multi_path.unlink()

    # ============================================================
    # Test 3: Verify formats are identical
    # ============================================================
    print("\n3. Verifying format consistency:")

    assert list(portfolio_df.columns) == list(multi_df.columns), \
        f"Column mismatch!\n  Portfolio: {list(portfolio_df.columns)}\n  MultiAsset: {list(multi_df.columns)}"

    assert set(portfolio_df['Direction'].unique()) == set(multi_df['Direction'].unique()), \
        "Direction values don't match!"

    # Both should have Buy and Sell
    assert 'Buy' in portfolio_df['Direction'].values, "Portfolio missing 'Buy' rows"
    assert 'Sell' in portfolio_df['Direction'].values, "Portfolio missing 'Sell' rows"
    assert 'Buy' in multi_df['Direction'].values, "MultiAssetPortfolio missing 'Buy' rows"
    assert 'Sell' in multi_df['Direction'].values, "MultiAssetPortfolio missing 'Sell' rows"

    # Both should have same number of rows (2 each: 1 buy + 1 sell)
    assert len(portfolio_df) == len(multi_df), \
        f"Row count mismatch! Portfolio: {len(portfolio_df)}, MultiAsset: {len(multi_df)}"

    print("   ✓ Columns match")
    print("   ✓ Direction values match (Buy/Sell)")
    print("   ✓ Row counts match")
    print("   ✓ Both formats are IDENTICAL")
    print("\n✓ All CSV format consistency tests passed!")


if __name__ == '__main__':
    test_csv_format_consistency()
