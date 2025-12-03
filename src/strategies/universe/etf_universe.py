"""
ETF Universe Definitions.

Organized lists of ETF symbols for different categories.
Replaces hardcoded symbol lists throughout the codebase.
"""

from typing import List


class ETFUniverse:
    """
    ETF symbol lists organized by category.

    Usage:
        ```python
        # Get leveraged 3x ETFs
        symbols = ETFUniverse.LEVERAGED_3X

        # Get all leveraged ETFs
        symbols = ETFUniverse.get_all_leveraged()

        # Get sector ETFs
        symbols = ETFUniverse.SECTOR
        ```
    """

    # Leveraged 3x Bull/Bear ETFs
    LEVERAGED_3X = [
        # Nasdaq 3x
        'TQQQ',  # ProShares UltraPro QQQ (3x Long)
        'SQQQ',  # ProShares UltraPro Short QQQ (3x Short)

        # S&P 500 3x
        'UPRO',  # ProShares UltraPro S&P500 (3x Long)
        'SPXU',  # ProShares UltraPro Short S&P500 (3x Short)

        # Dow Jones 3x
        'UDOW',  # ProShares UltraPro Dow30 (3x Long)
        'SDOW',  # ProShares UltraPro Short Dow30 (3x Short)

        # Treasury 3x
        'TMF',   # Direxion Daily 20+ Year Treasury Bull 3X
        'TMV',   # Direxion Daily 20+ Year Treasury Bear 3X

        # Technology 3x
        'TECL',  # Direxion Daily Technology Bull 3X
        'TECS',  # Direxion Daily Technology Bear 3X

        # Financial 3x
        'FAS',   # Direxion Daily Financial Bull 3X
        'FAZ',   # Direxion Daily Financial Bear 3X

        # Small Cap 3x
        'TNA',   # Direxion Daily Small Cap Bull 3X
        'TZA',   # Direxion Daily Small Cap Bear 3X

        # Energy 3x
        'ERX',   # Direxion Daily Energy Bull 3X
        'ERY',   # Direxion Daily Energy Bear 3X

        # Semiconductor 3x
        'SOXL',  # Direxion Daily Semiconductor Bull 3X
        'SOXS',  # Direxion Daily Semiconductor Bear 3X

        # Gold 3x
        'NUGT',  # Direxion Daily Gold Miners Bull 3X
        'DUST',  # Direxion Daily Gold Miners Bear 3X
    ]

    # Optimal OMR Symbols - Production Bayesian Model Universe
    # These 18 symbols are what the production model is trained on
    # Model trained: 2025-11-26, excludes SPY/VIX/USD (non-tradeable)
    OPTIMAL_OMR = [
        # Core Index 3x
        'TQQQ',   # Nasdaq 3x Bull - Most liquid
        'SQQQ',   # Nasdaq 3x Bear
        'UPRO',   # S&P 3x Bull
        'SPXU',   # S&P 3x Bear
        'UDOW',   # Dow 3x Bull

        # Core Index 2x
        'QLD',    # Nasdaq 2x Bull
        'SSO',    # S&P 2x Bull

        # Sector 3x
        'TECL',   # Technology 3x Bull
        'SOXL',   # Semiconductor 3x Bull
        'FAS',    # Financial 3x Bull
        'FAZ',    # Financial 3x Bear
        'TNA',    # Small Cap 3x Bull
        'DFEN',   # Defense 3x Bull
        'LABU',   # Biotech 3x Bull
        'NAIL',   # Homebuilders 3x Bull
        'WEBL',   # Internet 3x Bull

        # Commodity/Volatility
        'UCO',    # Oil 2x Bull
        'SVXY',   # VIX Short (-0.5x)
    ]

    # OMR Conservative (7 symbols - proven + high liquidity)
    OMR_CONSERVATIVE = [
        'SDOW', 'SOXS',  # Top 2 proven
        'SQQQ', 'SPXU',  # Inverse core
        'TQQQ', 'UPRO',  # Bull core
        'TECS'           # #3 proven
    ]

    # OMR Balanced (12 symbols - proven + sectors)
    OMR_BALANCED = [
        'SDOW', 'SOXS', 'SQQQ', 'SPXU', 'TQQQ', 'UPRO',
        'TECS', 'TECL', 'SOXL', 'FAZ', 'FAS', 'TZA'
    ]

    # Leveraged 2x ETFs
    LEVERAGED_2X = [
        # Nasdaq 2x
        'QLD',   # ProShares Ultra QQQ (2x Long)
        'QID',   # ProShares UltraShort QQQ (2x Short)

        # S&P 500 2x
        'SSO',   # ProShares Ultra S&P500 (2x Long)
        'SDS',   # ProShares UltraShort S&P500 (2x Short)

        # Dow Jones 2x
        'DDM',   # ProShares Ultra Dow30 (2x Long)
        'DXD',   # ProShares UltraShort Dow30 (2x Short)

        # Russell 2000 2x
        'UWM',   # ProShares Ultra Russell2000 (2x Long)
        'TWM',   # ProShares UltraShort Russell2000 (2x Short)

        # Treasury 2x
        'UBT',   # ProShares Ultra 20+ Year Treasury (2x Long)
        'TBT',   # ProShares UltraShort 20+ Year Treasury (2x Short)

        # Financials 2x
        'UYG',   # ProShares Ultra Financials (2x Long)
        'SKF',   # ProShares UltraShort Financials (2x Short)
    ]

    # Sector ETFs (non-leveraged)
    SECTOR = [
        'XLF',   # Financial Select Sector SPDR
        'XLK',   # Technology Select Sector SPDR
        'XLE',   # Energy Select Sector SPDR
        'XLV',   # Health Care Select Sector SPDR
        'XLI',   # Industrial Select Sector SPDR
        'XLP',   # Consumer Staples Select Sector SPDR
        'XLY',   # Consumer Discretionary Select Sector SPDR
        'XLU',   # Utilities Select Sector SPDR
        'XLB',   # Materials Select Sector SPDR
        'XLRE',  # Real Estate Select Sector SPDR
        'XLC',   # Communication Services Select Sector SPDR
    ]

    # Major Index ETFs
    MAJOR_INDICES = [
        'SPY',   # SPDR S&P 500
        'QQQ',   # Invesco QQQ (Nasdaq 100)
        'DIA',   # SPDR Dow Jones Industrial Average
        'IWM',   # iShares Russell 2000
        'VTI',   # Vanguard Total Stock Market
        'EFA',   # iShares MSCI EAFE (International)
        'EEM',   # iShares MSCI Emerging Markets
    ]

    # Bond ETFs
    BONDS = [
        'TLT',   # iShares 20+ Year Treasury Bond
        'IEF',   # iShares 7-10 Year Treasury Bond
        'SHY',   # iShares 1-3 Year Treasury Bond
        'LQD',   # iShares iBoxx $ Investment Grade Corporate Bond
        'HYG',   # iShares iBoxx $ High Yield Corporate Bond
        'MUB',   # iShares National Muni Bond
    ]

    # Commodity ETFs
    COMMODITIES = [
        'GLD',   # SPDR Gold Shares
        'SLV',   # iShares Silver Trust
        'USO',   # United States Oil Fund
        'UNG',   # United States Natural Gas Fund
        'DBA',   # Invesco DB Agriculture Fund
    ]

    # Volatility ETFs
    VOLATILITY = [
        'VXX',   # iPath Series B S&P 500 VIX Short-Term Futures ETN
        'UVXY',  # ProShares Ultra VIX Short-Term Futures ETF (2x)
        'SVXY',  # ProShares Short VIX Short-Term Futures ETF (-0.5x)
    ]

    @classmethod
    def get_all_leveraged(cls) -> List[str]:
        """Get all leveraged ETFs (2x and 3x)."""
        return cls.LEVERAGED_2X + cls.LEVERAGED_3X

    @classmethod
    def get_leveraged_bull(cls) -> List[str]:
        """Get only bullish (long) leveraged ETFs."""
        bull_etfs = [
            'TQQQ', 'UPRO', 'TMF', 'TECL', 'FAS', 'TNA', 'ERX', 'SOXL', 'NUGT',  # 3x
            'QLD', 'SSO', 'DDM', 'UWM', 'UBT', 'UYG'  # 2x
        ]
        return [etf for etf in cls.get_all_leveraged() if etf in bull_etfs]

    @classmethod
    def get_leveraged_bear(cls) -> List[str]:
        """Get only bearish (short) leveraged ETFs."""
        all_leveraged = set(cls.get_all_leveraged())
        bull = set(cls.get_leveraged_bull())
        return list(all_leveraged - bull)

    @classmethod
    def get_by_sector(cls, sector: str) -> List[str]:
        """
        Get ETFs by sector.

        Args:
            sector: Sector name ('Technology', 'Financial', 'Energy', etc.)

        Returns:
            List of ETF symbols for that sector
        """
        sector_map = {
            'Technology': ['XLK', 'TECL', 'TECS', 'SOXL', 'SOXS'],
            'Financial': ['XLF', 'FAS', 'FAZ', 'UYG', 'SKF'],
            'Energy': ['XLE', 'ERX', 'ERY', 'USO'],
            'Healthcare': ['XLV'],
            'Consumer': ['XLP', 'XLY'],
            'Industrial': ['XLI'],
            'Utilities': ['XLU'],
            'Materials': ['XLB'],
            'RealEstate': ['XLRE'],
            'Communication': ['XLC'],
            'Nasdaq': ['QQQ', 'TQQQ', 'SQQQ', 'QLD', 'QID'],
            'SP500': ['SPY', 'UPRO', 'SPXU', 'SSO', 'SDS'],
            'SmallCap': ['IWM', 'TNA', 'TZA', 'UWM', 'TWM'],
        }

        return sector_map.get(sector, [])

    @classmethod
    def get_inverse_etf(cls, symbol: str) -> str:
        """
        Get the inverse/opposite ETF for a given symbol.

        Args:
            symbol: ETF symbol

        Returns:
            Inverse ETF symbol, or None if not found

        Example:
            >>> ETFUniverse.get_inverse_etf('TQQQ')
            'SQQQ'
        """
        pairs = {
            # 3x pairs
            'TQQQ': 'SQQQ', 'SQQQ': 'TQQQ',
            'UPRO': 'SPXU', 'SPXU': 'UPRO',
            'TMF': 'TMV', 'TMV': 'TMF',
            'TECL': 'TECS', 'TECS': 'TECL',
            'FAS': 'FAZ', 'FAZ': 'FAS',
            'TNA': 'TZA', 'TZA': 'TNA',
            'ERX': 'ERY', 'ERY': 'ERX',
            'SOXL': 'SOXS', 'SOXS': 'SOXL',
            'NUGT': 'DUST', 'DUST': 'NUGT',

            # 2x pairs
            'QLD': 'QID', 'QID': 'QLD',
            'SSO': 'SDS', 'SDS': 'SSO',
            'DDM': 'DXD', 'DXD': 'DDM',
            'UWM': 'TWM', 'TWM': 'UWM',
            'UBT': 'TBT', 'TBT': 'UBT',
            'UYG': 'SKF', 'SKF': 'UYG',
        }

        return pairs.get(symbol)

    @classmethod
    def is_leveraged(cls, symbol: str) -> bool:
        """Check if symbol is a leveraged ETF."""
        return symbol in cls.get_all_leveraged()

    @classmethod
    def get_leverage_factor(cls, symbol: str) -> int:
        """
        Get leverage factor for an ETF.

        Returns:
            Leverage factor (2 or 3), or 1 if not leveraged
        """
        if symbol in cls.LEVERAGED_3X:
            return 3
        elif symbol in cls.LEVERAGED_2X:
            return 2
        else:
            return 1
