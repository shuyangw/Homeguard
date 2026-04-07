"""
Tests for IBKR ContractResolver -- OCC parsing is pure unit tests,
resolve_* methods require IBKR connection (@pytest.mark.ibkr).
"""

import pytest

from src.trading.brokers.ibkr.contracts import ContractResolver


# === OCC Symbology Parsing (pure unit tests) ======================

class TestOCCParser:
    """Test OCC option symbol parsing -- no IBKR connection needed."""

    def test_standard_occ_symbol(self):
        result = ContractResolver.parse_occ("AAPL  260417C00190000")
        assert result["symbol"] == "AAPL"
        assert result["expiration"] == "20260417"
        assert result["right"] == "C"
        assert result["strike"] == 190.0

    def test_put_option(self):
        result = ContractResolver.parse_occ("SPY   260320P00500000")
        assert result["symbol"] == "SPY"
        assert result["expiration"] == "20260320"
        assert result["right"] == "P"
        assert result["strike"] == 500.0

    def test_fractional_strike(self):
        result = ContractResolver.parse_occ("XSP   260117C00475500")
        assert result["strike"] == 475.5

    def test_low_strike(self):
        result = ContractResolver.parse_occ("F     261218P00012000")
        assert result["symbol"] == "F"
        assert result["strike"] == 12.0

    def test_high_strike(self):
        result = ContractResolver.parse_occ("BRK   260619C00690000")
        assert result["symbol"] == "BRK"
        assert result["strike"] == 690.0

    def test_short_symbol(self):
        result = ContractResolver.parse_occ("X     260815P00025000")
        assert result["symbol"] == "X"
        assert result["expiration"] == "20260815"

    def test_invalid_occ_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            ContractResolver.parse_occ("not_an_occ_symbol")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            ContractResolver.parse_occ("")

    def test_missing_right_raises(self):
        with pytest.raises(ValueError):
            ContractResolver.parse_occ("AAPL  260417X00190000")


class TestOCCRoundTrip:
    """Test OCC to components round-tripping."""

    def test_parse_then_construct(self):
        original = "AAPL  260417C00190000"
        parsed = ContractResolver.parse_occ(original)
        rebuilt = ContractResolver.to_occ(
            parsed["symbol"], parsed["expiration"],
            parsed["strike"], parsed["right"],
        )
        # Re-parse to verify equivalence (whitespace may differ)
        reparsed = ContractResolver.parse_occ(rebuilt)
        assert reparsed == parsed

    def test_construct_from_components(self):
        occ = ContractResolver.to_occ("SPY", "20260320", 500.0, "P")
        parsed = ContractResolver.parse_occ(occ)
        assert parsed["symbol"] == "SPY"
        assert parsed["expiration"] == "20260320"
        assert parsed["strike"] == 500.0
        assert parsed["right"] == "P"

    @pytest.mark.parametrize("symbol,expiry,strike,right", [
        ("AAPL", "20260417", 190.0, "C"),
        ("SPY", "20260320", 500.0, "P"),
        ("NVDA", "20261218", 150.5, "C"),
        ("F", "20260619", 12.0, "P"),
        ("TSLA", "20260117", 250.0, "C"),
    ])
    def test_round_trip_parametrized(self, symbol, expiry, strike, right):
        occ = ContractResolver.to_occ(symbol, expiry, strike, right)
        parsed = ContractResolver.parse_occ(occ)
        assert parsed["symbol"] == symbol
        assert parsed["expiration"] == expiry
        assert parsed["strike"] == strike
        assert parsed["right"] == right


# === Integration Tests (require Gateway) ==========================

@pytest.mark.ibkr
class TestContractResolution:
    """Test actual contract resolution against paper Gateway."""

    def test_resolve_stock(self, ibkr_connection):
        resolver = ContractResolver(ibkr_connection)
        contract = resolver.resolve_stock("AAPL")
        assert contract.conId > 0
        assert contract.symbol == "AAPL"
        assert contract.secType == "STK"

    def test_resolve_stock_caching(self, ibkr_connection):
        resolver = ContractResolver(ibkr_connection)
        c1 = resolver.resolve_stock("SPY")
        c2 = resolver.resolve_stock("SPY")
        assert c1.conId == c2.conId
        assert resolver.cache_size >= 1

    def test_resolve_option(self, ibkr_connection):
        resolver = ContractResolver(ibkr_connection)
        # Use a known near-term expiry -- this may need updating
        contract = resolver.resolve_option(
            "SPY", "20261218", 500.0, "C"
        )
        assert contract.conId > 0
        assert contract.secType == "OPT"

    def test_resolve_nonexistent_raises(self, ibkr_connection):
        from src.trading.brokers.interfaces.base import SymbolNotFoundError
        resolver = ContractResolver(ibkr_connection)
        with pytest.raises(SymbolNotFoundError):
            resolver.resolve_stock("ZZZZZNOTREAL")

    def test_option_expirations(self, ibkr_connection):
        resolver = ContractResolver(ibkr_connection)
        expirations = resolver.get_option_chain_expirations("AAPL")
        assert len(expirations) > 0
        assert all(len(e) == 8 for e in expirations)  # YYYYMMDD format
