"""
ORB (Opening Range Breakout) Universe Definitions.

Optimized symbol lists for Opening Range Breakout strategy.
Focus on high-liquidity instruments with clear opening range patterns.

Best performers for ORB per research:
- High liquidity leveraged ETFs (clear breakouts, tight spreads)
- High beta instruments
- Instruments with predictable opening range patterns
"""

from typing import List


class ORBUniverse:
    """
    Symbol universe optimized for Opening Range Breakout strategy.

    Usage:
        ```python
        from src.strategies.universe.orb_universe import ORBUniverse

        # Get primary universe (most liquid 3x ETFs)
        symbols = ORBUniverse.PRIMARY

        # Get all tradeable symbols
        symbols = ORBUniverse.get_all()

        # Get long-only symbols (bull ETFs only)
        symbols = ORBUniverse.get_long_only()
        ```
    """

    # Primary universe: 3x leveraged ETFs with highest liquidity
    # These have the clearest breakouts and tightest spreads
    PRIMARY = [
        'TQQQ',  # ProShares UltraPro QQQ (3x Nasdaq) - Most liquid 3x
        'SQQQ',  # ProShares UltraPro Short QQQ (3x Nasdaq Bear)
        'SOXL',  # Direxion Semiconductor Bull 3X - High volatility
        'SOXS',  # Direxion Semiconductor Bear 3X
        'UPRO',  # ProShares UltraPro S&P500 (3x S&P)
        'SPXU',  # ProShares UltraPro Short S&P500
        'TNA',   # Direxion Small Cap Bull 3X
        'TZA',   # Direxion Small Cap Bear 3X
        'TECL',  # Direxion Technology Bull 3X
        'TECS',  # Direxion Technology Bear 3X
        'SPXL',  # Direxion S&P 500 Bull 3X
    ]

    # Secondary universe: 2x leveraged (lower volatility, smoother moves)
    # Good for more conservative ORB trading
    SECONDARY = [
        'QLD',   # ProShares Ultra QQQ (2x Nasdaq)
        'QID',   # ProShares UltraShort QQQ (2x Nasdaq Bear)
        'SSO',   # ProShares Ultra S&P500 (2x S&P)
        'SDS',   # ProShares UltraShort S&P500 (2x S&P Bear)
        'USD',   # ProShares Ultra Semiconductors (2x)
    ]

    # Conservative: Major index ETFs only (lowest volatility)
    # Suitable for testing or low-risk ORB
    INDEX = [
        'SPY',   # SPDR S&P 500
        'QQQ',   # Invesco QQQ (Nasdaq 100)
        'IWM',   # iShares Russell 2000
        'DIA',   # SPDR Dow Jones Industrial Average
    ]

    # Long-only subset (bull ETFs only)
    LONG_ONLY = [
        'TQQQ',  # 3x Nasdaq
        'SOXL',  # 3x Semiconductor
        'UPRO',  # 3x S&P
        'TNA',   # 3x Small Cap
        'TECL',  # 3x Technology
        'SPXL',  # 3x S&P (Direxion)
        'QLD',   # 2x Nasdaq
        'SSO',   # 2x S&P
    ]

    # Short-only subset (bear ETFs only)
    SHORT_ONLY = [
        'SQQQ',  # 3x Nasdaq Bear
        'SOXS',  # 3x Semiconductor Bear
        'SPXU',  # 3x S&P Bear
        'TZA',   # 3x Small Cap Bear
        'TECS',  # 3x Technology Bear
        'QID',   # 2x Nasdaq Bear
        'SDS',   # 2x S&P Bear
    ]

    # Pairs for hedged ORB (enter both sides simultaneously)
    PAIRS = [
        ('TQQQ', 'SQQQ'),   # Nasdaq 3x pair
        ('SOXL', 'SOXS'),   # Semiconductor 3x pair
        ('UPRO', 'SPXU'),   # S&P 3x pair
        ('TNA', 'TZA'),     # Small Cap 3x pair
        ('TECL', 'TECS'),   # Technology 3x pair
    ]

    @classmethod
    def get_primary(cls) -> List[str]:
        """Get primary ORB universe (most liquid 3x ETFs)."""
        return cls.PRIMARY.copy()

    @classmethod
    def get_secondary(cls) -> List[str]:
        """Get secondary ORB universe (2x ETFs)."""
        return cls.SECONDARY.copy()

    @classmethod
    def get_index(cls) -> List[str]:
        """Get conservative index ETF universe."""
        return cls.INDEX.copy()

    @classmethod
    def get_all(cls) -> List[str]:
        """Get all ORB-suitable symbols."""
        return cls.PRIMARY + cls.SECONDARY + cls.INDEX

    @classmethod
    def get_long_only(cls) -> List[str]:
        """Get bull ETFs only (for long-only ORB)."""
        return cls.LONG_ONLY.copy()

    @classmethod
    def get_short_only(cls) -> List[str]:
        """Get bear ETFs only (for short-only ORB)."""
        return cls.SHORT_ONLY.copy()

    @classmethod
    def get_pairs(cls) -> List[tuple]:
        """Get bull/bear pairs for hedged ORB."""
        return cls.PAIRS.copy()

    @classmethod
    def get_inverse(cls, symbol: str) -> str:
        """
        Get the inverse ETF for a given symbol.

        Args:
            symbol: ETF symbol

        Returns:
            Inverse ETF symbol, or None if not found
        """
        inverse_map = {
            # 3x pairs
            'TQQQ': 'SQQQ', 'SQQQ': 'TQQQ',
            'SOXL': 'SOXS', 'SOXS': 'SOXL',
            'UPRO': 'SPXU', 'SPXU': 'UPRO',
            'TNA': 'TZA', 'TZA': 'TNA',
            'TECL': 'TECS', 'TECS': 'TECL',
            'SPXL': 'SPXS', 'SPXS': 'SPXL',
            # 2x pairs
            'QLD': 'QID', 'QID': 'QLD',
            'SSO': 'SDS', 'SDS': 'SSO',
        }
        return inverse_map.get(symbol)

    @classmethod
    def is_bull(cls, symbol: str) -> bool:
        """Check if symbol is a bull (long) ETF."""
        return symbol in cls.LONG_ONLY

    @classmethod
    def is_bear(cls, symbol: str) -> bool:
        """Check if symbol is a bear (short) ETF."""
        return symbol in cls.SHORT_ONLY
