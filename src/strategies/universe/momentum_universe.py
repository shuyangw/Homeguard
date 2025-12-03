"""
Momentum Universe Definitions.

Symbol universes for the LightGBM momentum strategy, organized by
asset class and liquidity requirements.
"""

from typing import List, Set


class MomentumUniverse:
    """
    Symbol universes for momentum strategy.

    Designed for cross-sectional momentum prediction using LightGBM.
    Includes ETFs (core) and liquid stocks (expansion).

    Usage:
        ```python
        from src.strategies.universe import MomentumUniverse

        # Get core ETF universe
        symbols = MomentumUniverse.get_core_universe()

        # Get all momentum symbols
        all_symbols = MomentumUniverse.get_full_universe()
        ```
    """

    # Core ETFs - highly liquid, diverse sectors
    CORE_ETFS: Set[str] = {
        # Broad market indices
        'SPY', 'QQQ', 'IWM', 'DIA', 'VTI',

        # Sector ETFs
        'XLF', 'XLE', 'XLK', 'XLV', 'XLI',
        'XLC', 'XLY', 'XLP', 'XLB', 'XLU', 'XLRE',

        # International
        'EEM', 'EFA', 'FXI', 'EWJ', 'EWZ',

        # Fixed income
        'TLT', 'IEF', 'HYG', 'LQD',

        # Commodities
        'GLD', 'SLV', 'USO',
    }

    # Leveraged ETFs (overlap with OMR universe)
    LEVERAGED_ETFS: Set[str] = {
        # 3x Bull
        'TQQQ', 'UPRO', 'TNA', 'SOXL', 'TECL', 'FAS', 'ERX',
        # 3x Bear
        'SQQQ', 'SPXU', 'TZA', 'SOXS', 'TECS', 'FAZ', 'ERY',
    }

    # Sector rotation candidates (non-leveraged)
    SECTOR_ROTATION: Set[str] = {
        'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC',
        'GDX', 'XOP', 'XBI', 'XHB', 'XRT', 'KRE', 'ITB', 'IYR',
    }

    # Thematic ETFs
    THEMATIC_ETFS: Set[str] = {
        'ARKK', 'ARKG', 'ARKW', 'ARKF',
        'ICLN', 'TAN', 'LIT', 'QCLN',
        'SKYY', 'CIBR', 'BOTZ', 'ROBO',
    }

    # Liquid mega-cap stocks (Phase 2 expansion)
    MEGA_CAPS: Set[str] = {
        # FAANG+
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
        # Other mega caps
        'JPM', 'V', 'JNJ', 'UNH', 'HD', 'MA', 'PG',
        'BAC', 'XOM', 'DIS', 'NFLX', 'ADBE', 'CRM', 'AMD',
    }

    # Volatility regime indicators (for context, not trading)
    VOLATILITY_INDICATORS: Set[str] = {
        'VXX', 'UVXY', 'SVXY',
    }

    @classmethod
    def get_core_universe(cls) -> List[str]:
        """
        Get core momentum universe (ETFs only).

        Recommended for initial training and backtesting.

        Returns:
            List of core ETF symbols
        """
        return sorted(cls.CORE_ETFS)

    @classmethod
    def get_leveraged_universe(cls) -> List[str]:
        """
        Get leveraged ETF universe.

        High risk/reward, suitable for swing trading.

        Returns:
            List of leveraged ETF symbols
        """
        return sorted(cls.LEVERAGED_ETFS)

    @classmethod
    def get_sector_rotation_universe(cls) -> List[str]:
        """
        Get sector rotation universe.

        Non-leveraged sector ETFs for relative rotation.

        Returns:
            List of sector ETF symbols
        """
        return sorted(cls.SECTOR_ROTATION)

    @classmethod
    def get_full_universe(cls) -> List[str]:
        """
        Get full momentum universe (all assets).

        Includes ETFs and mega-cap stocks.

        Returns:
            List of all momentum universe symbols
        """
        all_symbols = (
            cls.CORE_ETFS |
            cls.LEVERAGED_ETFS |
            cls.SECTOR_ROTATION |
            cls.MEGA_CAPS
        )
        return sorted(all_symbols)

    @classmethod
    def get_etf_only_universe(cls) -> List[str]:
        """
        Get ETF-only universe (no individual stocks).

        More stable, lower idiosyncratic risk.

        Returns:
            List of all ETF symbols
        """
        all_etfs = (
            cls.CORE_ETFS |
            cls.LEVERAGED_ETFS |
            cls.SECTOR_ROTATION |
            cls.THEMATIC_ETFS
        )
        return sorted(all_etfs)

    @classmethod
    def get_conservative_universe(cls) -> List[str]:
        """
        Get conservative momentum universe.

        Core ETFs only - highest liquidity, lowest slippage.

        Returns:
            List of conservative ETF symbols
        """
        conservative = {
            'SPY', 'QQQ', 'IWM', 'DIA',
            'XLF', 'XLK', 'XLE', 'XLV',
            'TLT', 'GLD',
            'EEM', 'EFA',
        }
        return sorted(conservative)

    @classmethod
    def get_aggressive_universe(cls) -> List[str]:
        """
        Get aggressive momentum universe.

        Leveraged ETFs and high-beta stocks.

        Returns:
            List of aggressive symbols
        """
        aggressive = cls.LEVERAGED_ETFS | {
            'NVDA', 'TSLA', 'AMD', 'ARKK',
        }
        return sorted(aggressive)

    @classmethod
    def get_universe_by_name(cls, name: str) -> List[str]:
        """
        Get universe by predefined name.

        Args:
            name: Universe name ('core', 'leveraged', 'sector', 'full',
                  'etf_only', 'conservative', 'aggressive')

        Returns:
            List of symbols for the specified universe

        Raises:
            ValueError: If universe name is not recognized
        """
        universe_map = {
            'core': cls.get_core_universe,
            'leveraged': cls.get_leveraged_universe,
            'sector': cls.get_sector_rotation_universe,
            'full': cls.get_full_universe,
            'etf_only': cls.get_etf_only_universe,
            'conservative': cls.get_conservative_universe,
            'aggressive': cls.get_aggressive_universe,
        }

        if name.lower() not in universe_map:
            valid_names = ', '.join(universe_map.keys())
            raise ValueError(f"Unknown universe '{name}'. Valid: {valid_names}")

        return universe_map[name.lower()]()

    @classmethod
    def is_leveraged(cls, symbol: str) -> bool:
        """Check if symbol is a leveraged ETF."""
        return symbol in cls.LEVERAGED_ETFS

    @classmethod
    def get_symbol_category(cls, symbol: str) -> str:
        """
        Get category for a symbol.

        Args:
            symbol: Symbol to categorize

        Returns:
            Category name or 'unknown'
        """
        if symbol in cls.LEVERAGED_ETFS:
            return 'leveraged'
        elif symbol in cls.SECTOR_ROTATION:
            return 'sector'
        elif symbol in cls.CORE_ETFS:
            return 'core_etf'
        elif symbol in cls.MEGA_CAPS:
            return 'mega_cap'
        elif symbol in cls.THEMATIC_ETFS:
            return 'thematic'
        else:
            return 'unknown'
