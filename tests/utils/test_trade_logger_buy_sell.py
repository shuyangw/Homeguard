"""
Test that trade_logger correctly splits trades into buy/sell rows.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from backtesting.engine.trade_logger import TradeLogger


def test_buy_sell_split():
    """Test that trades are split into separate buy and sell rows."""

    # Create mock portfolio with trades in MultiAssetPortfolio format
    class MockPortfolio:
        def __init__(self):
            # Simulate completed trades
            self.trades = pd.DataFrame({
                'symbol': ['AAPL', 'MSFT'],
                'entry_timestamp': [
                    pd.Timestamp('2023-01-03 14:30:00', tz='UTC'),
                    pd.Timestamp('2023-01-03 15:00:00', tz='UTC')
                ],
                'exit_timestamp': [
                    pd.Timestamp('2023-01-03 16:00:00', tz='UTC'),
                    pd.Timestamp('2023-01-03 17:00:00', tz='UTC')
                ],
                'entry_price': [100.0, 200.0],
                'exit_price': [105.0, 195.0],
                'shares': [10, 5],
                'pnl': [50.0, -25.0],
                'pnl_pct': [0.05, -0.025],
                'exit_reason': ['strategy_signal', 'strategy_signal']
            })

    portfolio = MockPortfolio()

    # Export to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_path = Path(f.name)

    try:
        TradeLogger.export_trades_csv(portfolio, temp_path)

        # Read the exported CSV
        result = pd.read_csv(temp_path)

        print("\nExported trades CSV:")
        print(result.to_string())

        # Verify we have 4 rows (2 trades * 2 rows each)
        assert len(result) == 4, f"Expected 4 rows, got {len(result)}"

        # Verify we have both Buy and Sell directions
        directions = result['Direction'].unique()
        assert 'Buy' in directions, "Missing 'Buy' direction"
        assert 'Sell' in directions, "Missing 'Sell' direction"

        # Verify we have 2 buys and 2 sells
        buy_count = len(result[result['Direction'] == 'Buy'])
        sell_count = len(result[result['Direction'] == 'Sell'])
        assert buy_count == 2, f"Expected 2 Buy rows, got {buy_count}"
        assert sell_count == 2, f"Expected 2 Sell rows, got {sell_count}"

        # Verify Buy rows don't have PnL (NaN)
        buy_rows = result[result['Direction'] == 'Buy']
        assert buy_rows['PnL'].isna().all(), "Buy rows should have NaN for PnL"

        # Verify Sell rows have PnL
        sell_rows = result[result['Direction'] == 'Sell']
        assert not sell_rows['PnL'].isna().any(), "Sell rows should have PnL values"

        # Verify trades are sorted by date
        dates = pd.to_datetime(result['Date'])
        assert (dates == dates.sort_values()).all(), "Trades should be sorted by date"

        # Verify specific values
        aapl_buy = result[(result['Symbol'] == 'AAPL') & (result['Direction'] == 'Buy')].iloc[0]
        assert aapl_buy['Price'] == 100.0, f"AAPL buy price should be 100.0, got {aapl_buy['Price']}"
        assert aapl_buy['Size'] == 10, f"AAPL buy size should be 10, got {aapl_buy['Size']}"

        aapl_sell = result[(result['Symbol'] == 'AAPL') & (result['Direction'] == 'Sell')].iloc[0]
        assert aapl_sell['Price'] == 105.0, f"AAPL sell price should be 105.0, got {aapl_sell['Price']}"
        assert aapl_sell['PnL'] == 50.0, f"AAPL sell PnL should be 50.0, got {aapl_sell['PnL']}"
        assert aapl_sell['Return %'] == 5.0, f"AAPL sell return should be 5.0%, got {aapl_sell['Return %']}"

        msft_sell = result[(result['Symbol'] == 'MSFT') & (result['Direction'] == 'Sell')].iloc[0]
        assert msft_sell['PnL'] == -25.0, f"MSFT sell PnL should be -25.0, got {msft_sell['PnL']}"
        assert msft_sell['Return %'] == -2.5, f"MSFT sell return should be -2.5%, got {msft_sell['Return %']}"

        print("\n✓ All tests passed!")

    finally:
        # Clean up
        if temp_path.exists():
            temp_path.unlink()


if __name__ == '__main__':
    test_buy_sell_split()
