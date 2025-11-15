"""
Validation Script: OMR Symbol Universe Compliance

This script validates that the OMR trading system is using the
authoritative symbol list from the production configuration file.

Run this before deploying to ensure correct symbols are in use.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.trading.config import load_omr_config, get_production_symbols, validate_symbols
from src.strategies.universe import ETFUniverse
from src.utils.logger import logger


def test_config_loader():
    """Test 1: Verify config loader works correctly."""
    print("\n" + "="*80)
    print("TEST 1: Config Loader Functionality")
    print("="*80)

    try:
        config = load_omr_config()
        print(f"[OK] Config loaded successfully")
        print(f"     Name: {config.name}")
        print(f"     Version: {config.version}")
        print(f"     Symbols: {len(config.symbols)}")
        print(f"     Entry: {config.entry_time} | Exit: {config.exit_time}")
        return True
    except Exception as e:
        print(f"[FAILED] Config loader error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_symbol_count():
    """Test 2: Verify symbol count is 20."""
    print("\n" + "="*80)
    print("TEST 2: Symbol Count Validation")
    print("="*80)

    symbols = get_production_symbols()
    expected_count = 20

    if len(symbols) == expected_count:
        print(f"[OK] Symbol count correct: {len(symbols)} symbols")
        return True
    else:
        print(f"[FAILED] Symbol count mismatch!")
        print(f"         Expected: {expected_count}")
        print(f"         Got: {len(symbols)}")
        print(f"         Symbols: {symbols}")
        return False


def test_symbol_uniqueness():
    """Test 3: Verify all symbols are unique (no duplicates)."""
    print("\n" + "="*80)
    print("TEST 3: Symbol Uniqueness")
    print("="*80)

    symbols = get_production_symbols()
    unique_symbols = set(symbols)

    if len(symbols) == len(unique_symbols):
        print(f"[OK] All {len(symbols)} symbols are unique")
        return True
    else:
        duplicates = [s for s in symbols if symbols.count(s) > 1]
        print(f"[FAILED] Found duplicate symbols!")
        print(f"         Duplicates: {set(duplicates)}")
        return False


def test_leveraged_3x_mismatch():
    """Test 4: Verify production list differs from LEVERAGED_3X default."""
    print("\n" + "="*80)
    print("TEST 4: Production vs Default Comparison")
    print("="*80)

    production = set(get_production_symbols())
    default = set(ETFUniverse.LEVERAGED_3X)

    if production != default:
        print(f"[OK] Production config differs from default (as expected)")
        print(f"     Production: {len(production)} symbols")
        print(f"     Default: {len(default)} symbols")

        missing = default - production
        added = production - default

        if missing:
            print(f"\n     Excluded from production ({len(missing)}):")
            for sym in sorted(missing):
                print(f"       - {sym}")

        if added:
            print(f"\n     Added to production ({len(added)}):")
            for sym in sorted(added):
                print(f"       - {sym}")

        return True
    else:
        print(f"[WARNING] Production config matches LEVERAGED_3X default!")
        print(f"           This might be intentional, but verify manually.")
        return True  # Not necessarily a failure


def test_expected_symbols():
    """Test 5: Verify specific symbols from validation report are present."""
    print("\n" + "="*80)
    print("TEST 5: Expected Symbol Presence")
    print("="*80)

    production = get_production_symbols()

    # Critical symbols from walk-forward validation
    expected_symbols = [
        'TQQQ', 'SQQQ',  # Nasdaq 3x
        'UPRO', 'SPXU',  # S&P 500 3x
        'SOXL', 'TECL',  # Tech/Semi 3x
        'FAS', 'FAZ',    # Financials
        'TNA'            # Small Cap
    ]

    missing = []
    for symbol in expected_symbols:
        if symbol not in production:
            missing.append(symbol)

    if not missing:
        print(f"[OK] All {len(expected_symbols)} critical symbols present")
        return True
    else:
        print(f"[FAILED] Missing critical symbols: {missing}")
        return False


def test_validate_function():
    """Test 6: Test the validate_symbols() function."""
    print("\n" + "="*80)
    print("TEST 6: Validate Symbols Function")
    print("="*80)

    production = get_production_symbols()

    # Test 1: Should pass with production symbols
    print("Testing with production symbols...")
    result1 = validate_symbols(production)

    # Test 2: Should fail with LEVERAGED_3X
    print("\nTesting with LEVERAGED_3X (should fail)...")
    result2 = validate_symbols(ETFUniverse.LEVERAGED_3X)

    if result1 and not result2:
        print("\n[OK] validate_symbols() works correctly")
        return True
    else:
        print(f"\n[FAILED] validate_symbols() not working as expected")
        print(f"         Production validation: {result1} (should be True)")
        print(f"         LEVERAGED_3X validation: {result2} (should be False)")
        return False


def print_symbol_list():
    """Display the full production symbol list."""
    print("\n" + "="*80)
    print("PRODUCTION SYMBOL LIST")
    print("="*80)

    symbols = get_production_symbols()

    print(f"\nTotal: {len(symbols)} symbols\n")
    for i, symbol in enumerate(symbols, 1):
        print(f"  {i:2d}. {symbol}")


def main():
    """Run all validation tests."""
    print("\n" + "="*80)
    print("OMR SYMBOL UNIVERSE VALIDATION")
    print("="*80)
    print("This script validates that production OMR configuration")
    print("uses the correct 20-symbol universe.")
    print()

    tests = [
        ("Config Loader", test_config_loader),
        ("Symbol Count", test_symbol_count),
        ("Symbol Uniqueness", test_symbol_uniqueness),
        ("Production vs Default", test_leveraged_3x_mismatch),
        ("Expected Symbols", test_expected_symbols),
        ("Validate Function", test_validate_function),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n[ERROR] Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n[SUCCESS] All validation tests passed!")
        print_symbol_list()
        return 0
    else:
        print(f"\n[FAILED] {total - passed} test(s) failed")
        print("\nPlease review failures above and fix configuration.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
