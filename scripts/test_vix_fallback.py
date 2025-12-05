"""
Test VIX fallback chain comprehensively.

Tests:
1. Normal operation (yfinance succeeds)
2. yfinance failure -> FRED fallback
3. All sources fail -> cache fallback
4. Cache persistence and recovery
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import shutil
from unittest.mock import patch, MagicMock
from src.utils.vix_provider import VIXProvider
from src.utils.logger import logger


def test_normal_operation():
    """Test 1: Normal operation - yfinance succeeds."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 1: Normal operation (yfinance succeeds)")
    logger.info("=" * 60)

    provider = VIXProvider()
    data = provider.get_vix_data(lookback_days=30)

    if data is not None and provider.last_source == "yfinance":
        logger.success(f"PASS: Got {len(data)} days from yfinance")
        logger.info(f"  Latest VIX: {data['close'].iloc[-1]:.2f}")
        return True
    else:
        logger.error(f"FAIL: Expected yfinance, got {provider.last_source}")
        return False


def test_yfinance_failure_fred_fallback():
    """Test 2: yfinance fails -> FRED fallback."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: yfinance failure -> FRED fallback")
    logger.info("=" * 60)

    provider = VIXProvider()

    # Mock yfinance to fail
    with patch.object(provider, '_fetch_yfinance', return_value=None):
        data = provider.get_vix_data(lookback_days=30)

        if data is not None and provider.last_source == "FRED":
            logger.success(f"PASS: Fell back to FRED, got {len(data)} days")
            logger.info(f"  Latest VIX: {data['close'].iloc[-1]:.2f}")
            return True
        else:
            logger.error(f"FAIL: Expected FRED fallback, got {provider.last_source}")
            return False


def test_all_fail_cache_fallback():
    """Test 3: All sources fail -> cache fallback."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: All sources fail -> cache fallback")
    logger.info("=" * 60)

    provider = VIXProvider()

    # First, ensure cache exists by fetching normally
    logger.info("Pre-populating cache...")
    provider.get_vix_data(lookback_days=30)

    if not provider.cache_file.exists():
        logger.error("FAIL: Cache not created")
        return False

    # Now mock both sources to fail
    with patch.object(provider, '_fetch_yfinance', return_value=None):
        with patch.object(provider, '_fetch_fred', return_value=None):
            data = provider.get_vix_data(lookback_days=30)

            if data is not None and provider.last_source == "cache":
                logger.success(f"PASS: Fell back to cache, got {len(data)} days")
                logger.info(f"  Latest VIX: {data['close'].iloc[-1]:.2f}")
                return True
            else:
                logger.error(f"FAIL: Expected cache fallback, got {provider.last_source}")
                return False


def test_cache_persistence():
    """Test 4: Cache is persisted and can be recovered."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Cache persistence and recovery")
    logger.info("=" * 60)

    provider = VIXProvider()

    # Fetch to populate cache
    data = provider.get_vix_data(lookback_days=30)
    if data is None:
        logger.error("FAIL: Could not fetch initial data")
        return False

    # Read cache file
    if not provider.cache_file.exists():
        logger.error("FAIL: Cache file not created")
        return False

    with open(provider.cache_file) as f:
        cache = json.load(f)

    logger.info(f"Cache file: {provider.cache_file}")
    logger.info(f"  Timestamp: {cache.get('timestamp')}")
    logger.info(f"  Source: {cache.get('source')}")
    logger.info(f"  Current VIX: {cache.get('current_vix'):.2f}")
    logger.info(f"  Data points: {len(cache.get('data', {}))}")

    # Verify cache data matches fetched data
    cached_vix = cache.get('current_vix')
    fetched_vix = float(data['close'].iloc[-1])

    if abs(cached_vix - fetched_vix) < 0.01:
        logger.success("PASS: Cache matches fetched data")
        return True
    else:
        logger.error(f"FAIL: Cache ({cached_vix}) != Fetched ({fetched_vix})")
        return False


def test_integration_with_omr():
    """Test 5: Integration with OMR adapter."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 5: Integration with OMR adapter")
    logger.info("=" * 60)

    try:
        from src.trading.adapters.omr_live_adapter import OMRLiveAdapter

        # Create mock broker
        mock_broker = MagicMock()
        mock_broker.is_market_open.return_value = False

        # Check that the import chain works
        logger.success("PASS: OMR adapter imports VIX provider correctly")
        return True

    except ImportError as e:
        logger.error(f"FAIL: Import error - {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    logger.info("=" * 60)
    logger.info("VIX FALLBACK CHAIN - COMPREHENSIVE TEST SUITE")
    logger.info("=" * 60)

    results = {
        "Normal operation": test_normal_operation(),
        "FRED fallback": test_yfinance_failure_fred_fallback(),
        "Cache fallback": test_all_fail_cache_fallback(),
        "Cache persistence": test_cache_persistence(),
        "OMR integration": test_integration_with_omr(),
    }

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 60)

    passed = 0
    failed = 0
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        icon = "[+]" if result else "[X]"
        logger.info(f"  {icon} {test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1

    logger.info("")
    logger.info(f"Total: {passed}/{len(results)} tests passed")

    if failed == 0:
        logger.success("All tests passed!")
    else:
        logger.error(f"{failed} test(s) failed")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
