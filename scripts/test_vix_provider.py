"""Test the VIX provider with fallback chain."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.vix_provider import VIXProvider
from src.utils.logger import logger

def test_vix_provider():
    """Test VIX provider functionality."""
    logger.info("=" * 60)
    logger.info("Testing VIX Provider")
    logger.info("=" * 60)

    p = VIXProvider()

    # Test 30-day lookback
    logger.info("\n1. Testing 30-day lookback...")
    d = p.get_vix_data(30)
    if d is not None:
        logger.success(f"Source: {p.last_source}")
        logger.info(f"Rows: {len(d)}")
        logger.info(f"Columns: {list(d.columns)}")
        logger.info(f"Latest VIX: {d['close'].iloc[-1]:.2f}")
        logger.info(f"Date range: {d.index[0]} to {d.index[-1]}")
    else:
        logger.error("FAILED - no data returned")
        return False

    # Test current VIX
    logger.info("\n2. Testing get_current_vix()...")
    current = p.get_current_vix()
    if current is not None:
        logger.success(f"Current VIX: {current:.2f}")
    else:
        logger.error("Failed to get current VIX")

    # Test 252-day lookback (regime detection requirement)
    logger.info("\n3. Testing 252-day lookback (regime detection)...")
    d252 = p.get_vix_data(252)
    if d252 is not None:
        logger.success(f"Source: {p.last_source}")
        logger.info(f"Rows: {len(d252)}")
        logger.info(f"Date range: {d252.index[0]} to {d252.index[-1]}")
    else:
        logger.error("FAILED - 252-day lookback failed")

    # Test cache was created
    logger.info("\n4. Testing cache persistence...")
    if p.cache_file.exists():
        logger.success(f"Cache file exists: {p.cache_file}")
        import json
        with open(p.cache_file) as f:
            cache = json.load(f)
        logger.info(f"Cache timestamp: {cache.get('timestamp')}")
        logger.info(f"Cache source: {cache.get('source')}")
        logger.info(f"Cached VIX: {cache.get('current_vix'):.2f}")
    else:
        logger.warning("Cache file not found")

    logger.info("\n" + "=" * 60)
    logger.success("VIX Provider test complete!")
    return True


if __name__ == "__main__":
    test_vix_provider()
