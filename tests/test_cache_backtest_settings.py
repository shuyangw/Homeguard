"""
Test that cache manager stores and retrieves complete backtest settings.
"""

import sys
from pathlib import Path
import tempfile
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.cache_manager import CacheManager


def test_cache_backtest_settings():
    """Test that all backtest settings are cached and retrievable."""

    print("\nTesting cache manager with complete backtest settings...\n")

    # Create temporary cache directory
    with tempfile.TemporaryDirectory() as temp_dir:
        cache = CacheManager(cache_dir=Path(temp_dir))

        # Create a mock strategy class
        class MockStrategy:
            pass

        # Create complete backtest configuration
        config = {
            'strategy_class': MockStrategy,
            'strategy_params': {'fast_period': 10, 'slow_period': 20},
            'symbols': ['AAPL', 'MSFT', 'GOOGL'],
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'initial_capital': 100000,
            'fees': 0.001,
            'risk_profile': 'Aggressive',
            'portfolio_mode': 'Multi-Symbol Portfolio',
            'position_sizing_method': 'volatility_based',
            'rebalancing_frequency': 'monthly',
            'rebalancing_threshold_pct': 0.10,
            'generate_full_output': True
        }

        # Create mock results
        results_df = pd.DataFrame({
            'Symbol': ['AAPL', 'MSFT', 'GOOGL'],
            'Total Return [%]': [15.5, 22.3, -5.2],
            'Sharpe Ratio': [1.5, 2.1, 0.8]
        })

        print("1. Caching backtest results with settings...")
        cache_key = cache.cache_results(
            config=config,
            results_df=results_df,
            description="Test backtest with all settings"
        )
        print(f"   ✓ Cached with key: {cache_key}")

        # Test 2: Retrieve cached config
        print("\n2. Retrieving cached config...")
        cached = cache.get_cached_results(config)
        assert cached is not None, "Failed to retrieve cached results!"
        print("   ✓ Cache retrieved successfully")

        # Test 3: Verify all settings are preserved
        print("\n3. Verifying all settings are preserved...")
        cached_config = cached['config']

        settings_to_verify = [
            ('strategy', 'MockStrategy'),
            ('initial_capital', 100000),
            ('fees', 0.001),
            ('risk_profile', 'Aggressive'),
            ('portfolio_mode', 'Multi-Symbol Portfolio'),
            ('position_sizing_method', 'volatility_based'),
            ('rebalancing_frequency', 'monthly'),
            ('rebalancing_threshold_pct', 0.10),
            ('generate_full_output', True),
            ('start_date', '2023-01-01'),
            ('end_date', '2023-12-31')
        ]

        for setting, expected_value in settings_to_verify:
            actual_value = cached_config.get(setting)
            assert actual_value == expected_value, \
                f"Setting '{setting}' mismatch: expected {expected_value}, got {actual_value}"
            print(f"   ✓ {setting}: {actual_value}")

        # Test 4: Verify strategy params
        print("\n4. Verifying strategy parameters...")
        strategy_params = cached_config.get('strategy_params', {})
        assert strategy_params.get('fast_period') == 10, "fast_period not preserved"
        assert strategy_params.get('slow_period') == 20, "slow_period not preserved"
        print(f"   ✓ Strategy params: {strategy_params}")

        # Test 5: Verify symbols
        print("\n5. Verifying symbols list...")
        symbols = cached_config.get('symbols', [])
        assert symbols == ['AAPL', 'MSFT', 'GOOGL'], f"Symbols mismatch: {symbols}"
        print(f"   ✓ Symbols: {symbols}")

        # Test 6: Retrieve last run settings
        print("\n6. Testing get_last_run_settings()...")
        last_settings = cache.get_last_run_settings()
        assert last_settings is not None, "Failed to get last run settings!"
        assert last_settings['risk_profile'] == 'Aggressive', "Risk profile not in last run settings"
        assert last_settings['portfolio_mode'] == 'Multi-Symbol Portfolio', "Portfolio mode not in last run settings"
        assert last_settings['rebalancing_frequency'] == 'monthly', "Rebalancing frequency not in last run settings"
        print("   ✓ Last run settings retrieved:")
        print(f"     - Risk Profile: {last_settings['risk_profile']}")
        print(f"     - Portfolio Mode: {last_settings['portfolio_mode']}")
        print(f"     - Position Sizing: {last_settings['position_sizing_method']}")
        print(f"     - Rebalancing: {last_settings['rebalancing_frequency']}")

        # Test 7: Verify cache differentiation
        print("\n7. Testing cache differentiation with different settings...")
        config_v2 = config.copy()
        config_v2['risk_profile'] = 'Conservative'  # Changed setting

        # Should NOT be cached (different config)
        assert not cache.is_cached(config_v2), "Different config should not be cached!"
        print("   ✓ Different settings correctly identified as not cached")

        # Original config should still be cached
        assert cache.is_cached(config), "Original config should still be cached!"
        print("   ✓ Original settings still cached")

        # Test 8: List cached runs
        print("\n8. Testing list_cached_runs()...")
        runs = cache.list_cached_runs()
        assert len(runs) >= 1, "Should have at least one cached run"
        print(f"   ✓ Found {len(runs)} cached run(s)")

        latest_run = runs[0]
        print(f"   Latest run:")
        print(f"     - Strategy: {latest_run['strategy']}")
        print(f"     - Symbols: {latest_run['symbols']}")
        print(f"     - Date range: {latest_run['date_range']}")

        print("\n✓ All cache settings tests passed!")
        print("✓ Cache now stores complete backtest configuration including:")
        print("  - Risk profile")
        print("  - Portfolio mode")
        print("  - Position sizing method")
        print("  - Rebalancing settings")
        print("  - All strategy parameters")


if __name__ == '__main__':
    test_cache_backtest_settings()
