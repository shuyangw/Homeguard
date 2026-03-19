"""
Unit tests for scripts/download_crypto.py CLI constants.

Tests the default crypto pairs list used by the download script.
"""


class TestDefaultCryptoPairs:
    """Tests for default crypto pairs in CLI script."""

    def test_default_pairs_count(self):
        from scripts.data.download_crypto import DEFAULT_CRYPTO_PAIRS

        # Should have 18 pairs (no stablecoins)
        assert len(DEFAULT_CRYPTO_PAIRS) == 18

    def test_no_stablecoins(self):
        from scripts.data.download_crypto import DEFAULT_CRYPTO_PAIRS

        # USDC and USDT should not be in the list
        for pair in DEFAULT_CRYPTO_PAIRS:
            assert "USDC/USD" not in pair or pair != "USDC/USD"
            assert "USDT/USD" not in pair or pair != "USDT/USD"

    def test_all_usd_pairs(self):
        from scripts.data.download_crypto import DEFAULT_CRYPTO_PAIRS

        # All pairs should end with /USD
        for pair in DEFAULT_CRYPTO_PAIRS:
            assert pair.endswith("/USD"), f"{pair} should end with /USD"

    def test_expected_pairs_present(self):
        from scripts.data.download_crypto import DEFAULT_CRYPTO_PAIRS

        expected = ["BTC/USD", "ETH/USD", "DOGE/USD", "LINK/USD"]
        for pair in expected:
            assert pair in DEFAULT_CRYPTO_PAIRS, f"{pair} should be in default pairs"
