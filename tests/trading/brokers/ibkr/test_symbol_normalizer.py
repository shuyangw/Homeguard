"""
Tests for IBKR symbol normalization.

IBKR uses space-separated class suffixes ('BRK B', 'BF B') while our
universes and strategies use dot format ('BRK.B', 'BF.B'). Normalization
happens at the IBKR layer boundary -- outbound on symbols heading into the
IB API, inbound on symbols returned from it.
"""

import pytest

from src.trading.brokers.ibkr.symbols import from_ibkr_symbol, to_ibkr_symbol


class TestToIbkrSymbol:
    """Domain-format symbol -> IBKR-format symbol."""

    def test_brk_b_gets_space(self):
        assert to_ibkr_symbol("BRK.B") == "BRK B"

    def test_bf_b_gets_space(self):
        assert to_ibkr_symbol("BF.B") == "BF B"

    def test_brk_a_gets_space(self):
        # Not in our universe but same pattern
        assert to_ibkr_symbol("BRK.A") == "BRK A"

    def test_aapl_passthrough(self):
        assert to_ibkr_symbol("AAPL") == "AAPL"

    def test_spy_passthrough(self):
        assert to_ibkr_symbol("SPY") == "SPY"

    def test_single_letter_passthrough(self):
        # 'V' (Visa) has no class suffix
        assert to_ibkr_symbol("V") == "V"

    def test_empty_string_passthrough(self):
        assert to_ibkr_symbol("") == ""

    def test_already_ibkr_format_passthrough(self):
        # If a caller accidentally passes IBKR format, don't mangle it
        assert to_ibkr_symbol("BRK B") == "BRK B"

    def test_lowercase_unchanged(self):
        # We don't case-fold; pattern requires uppercase
        assert to_ibkr_symbol("brk.b") == "brk.b"

    def test_multiple_dots_unchanged(self):
        # Pattern matches exactly one dot before a single letter suffix
        assert to_ibkr_symbol("A.B.C") == "A.B.C"

    def test_long_suffix_unchanged(self):
        # Class suffix must be exactly one letter
        assert to_ibkr_symbol("BRK.BB") == "BRK.BB"


class TestFromIbkrSymbol:
    """IBKR-format symbol -> domain-format symbol."""

    def test_brk_b_gets_dot(self):
        assert from_ibkr_symbol("BRK B") == "BRK.B"

    def test_bf_b_gets_dot(self):
        assert from_ibkr_symbol("BF B") == "BF.B"

    def test_brk_a_gets_dot(self):
        assert from_ibkr_symbol("BRK A") == "BRK.A"

    def test_aapl_passthrough(self):
        assert from_ibkr_symbol("AAPL") == "AAPL"

    def test_spy_passthrough(self):
        assert from_ibkr_symbol("SPY") == "SPY"

    def test_empty_string_passthrough(self):
        assert from_ibkr_symbol("") == ""

    def test_already_domain_format_passthrough(self):
        assert from_ibkr_symbol("BRK.B") == "BRK.B"


class TestRoundTrip:
    """Normalization must be symmetric for symbols that get translated."""

    @pytest.mark.parametrize("symbol", ["BRK.B", "BF.B", "BRK.A", "BF.A"])
    def test_multi_class_roundtrip(self, symbol):
        assert from_ibkr_symbol(to_ibkr_symbol(symbol)) == symbol

    @pytest.mark.parametrize(
        "symbol",
        ["AAPL", "MSFT", "GOOGL", "SPY", "V", "MRK", "UPRO", "SOXL"],
    )
    def test_regular_symbols_unchanged(self, symbol):
        assert to_ibkr_symbol(symbol) == symbol
        assert from_ibkr_symbol(symbol) == symbol
        assert from_ibkr_symbol(to_ibkr_symbol(symbol)) == symbol
