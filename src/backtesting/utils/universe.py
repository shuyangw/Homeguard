"""
Universe management for multi-symbol backtesting.
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional, Set
from src.utils import logger


class UniverseManager:
    """
    Manages stock universes for backtesting across multiple symbols.
    """

    PREDEFINED_UNIVERSES = {
        'SP500': None,
        'SP100': None,
        'NASDAQ100': None,
        'DOW30': None,
        'FAANG': ['META', 'AAPL', 'AMZN', 'NFLX', 'GOOGL'],
        'MAGNIFICENT7': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META'],
        'TECH_GIANTS': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'ORCL', 'CSCO'],
        'SEMICONDUCTORS': ['NVDA', 'AMD', 'INTC', 'TSM', 'QCOM', 'AVGO', 'MU', 'AMAT', 'LRCX', 'KLAC'],
        'ENERGY': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'MPC', 'VLO', 'PSX', 'OXY'],
        'FINANCE': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'USB'],
        'HEALTHCARE': ['UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'TMO', 'ABT', 'PFE', 'DHR', 'BMY'],
        'CONSUMER': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'TJX', 'DG'],
    }

    DOW30 = [
        'AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'DOW',
        'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM',
        'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WMT', 'WBA'
    ]

    NASDAQ100_TOP50 = [
        'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'GOOG', 'TSLA', 'AVGO', 'COST',
        'NFLX', 'ASML', 'TMUS', 'AMD', 'ADBE', 'PEP', 'CSCO', 'LIN', 'CMCSA', 'INTC',
        'TXN', 'INTU', 'QCOM', 'AMGN', 'AMAT', 'HON', 'ISRG', 'BKNG', 'VRTX', 'ADP',
        'SBUX', 'GILD', 'ADI', 'MU', 'PANW', 'REGN', 'MDLZ', 'LRCX', 'PYPL', 'SNPS',
        'KLAC', 'CDNS', 'MELI', 'CRWD', 'MAR', 'CSX', 'CTAS', 'FTNT', 'ADSK', 'NXPI'
    ]

    SP100_TOP50 = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'BRK.B', 'LLY', 'TSLA', 'JPM',
        'V', 'UNH', 'XOM', 'MA', 'JNJ', 'AVGO', 'PG', 'HD', 'CVX', 'COST',
        'ABBV', 'MRK', 'NFLX', 'KO', 'BAC', 'CRM', 'ORCL', 'PEP', 'AMD', 'WMT',
        'CSCO', 'TMO', 'ACN', 'MCD', 'LIN', 'ABT', 'ADBE', 'DHR', 'DIS', 'TMUS',
        'GE', 'VZ', 'INTC', 'PM', 'TXN', 'CAT', 'QCOM', 'INTU', 'AMGN', 'CMCSA'
    ]

    PREDEFINED_UNIVERSES['DOW30'] = DOW30
    PREDEFINED_UNIVERSES['NASDAQ100'] = NASDAQ100_TOP50
    PREDEFINED_UNIVERSES['SP100'] = SP100_TOP50

    @classmethod
    def get_universe(cls, universe_name: str) -> List[str]:
        """
        Get symbol list for a predefined universe.

        Args:
            universe_name: Name of universe (e.g., 'SP500', 'DOW30', 'FAANG')

        Returns:
            List of symbols

        Raises:
            ValueError: If universe name is not recognized
        """
        universe_name = universe_name.upper()

        if universe_name not in cls.PREDEFINED_UNIVERSES:
            available = ', '.join(sorted(cls.PREDEFINED_UNIVERSES.keys()))
            raise ValueError(
                f"Unknown universe '{universe_name}'. Available: {available}"
            )

        symbols = cls.PREDEFINED_UNIVERSES[universe_name]

        if symbols is None:
            raise NotImplementedError(
                f"Universe '{universe_name}' is defined but not yet populated. "
                f"Use a custom file or one of: {', '.join([k for k, v in cls.PREDEFINED_UNIVERSES.items() if v is not None])}"
            )

        logger.info(f"Loaded universe '{universe_name}' with {len(symbols)} symbols")

        return symbols.copy()

    @classmethod
    def load_from_file(cls, file_path: Path | str) -> List[str]:
        """
        Load symbol list from file.

        Supported formats:
        - Plain text: One symbol per line
        - CSV: First column is symbols (with or without header)

        Args:
            file_path: Path to file containing symbols

        Returns:
            List of symbols

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is empty or invalid format
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Universe file not found: {file_path}")

        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
            symbols = df.iloc[:, 0].astype(str).str.strip().tolist()
        else:
            with open(file_path, 'r') as f:
                symbols = [line.strip() for line in f if line.strip()]

        symbols = [s.upper() for s in symbols if s and not s.startswith('#')]

        if not symbols:
            raise ValueError(f"No symbols found in file: {file_path}")

        logger.info(f"Loaded {len(symbols)} symbols from file: {file_path}")

        return symbols

    @classmethod
    def validate_symbols(cls, symbols: List[str]) -> List[str]:
        """
        Validate and clean symbol list.

        Args:
            symbols: List of symbols

        Returns:
            Cleaned list of unique symbols
        """
        cleaned = []
        seen = set()

        for symbol in symbols:
            symbol = symbol.strip().upper()

            if not symbol:
                continue

            if symbol in seen:
                logger.warning(f"Duplicate symbol removed: {symbol}")
                continue

            if len(symbol) > 10:
                logger.warning(f"Suspicious symbol (too long): {symbol}")

            cleaned.append(symbol)
            seen.add(symbol)

        logger.info(f"Validated {len(cleaned)} symbols")

        return cleaned

    @classmethod
    def get_symbols(
        cls,
        universe: Optional[str] = None,
        symbols_file: Optional[str] = None,
        symbols_list: Optional[List[str]] = None
    ) -> List[str]:
        """
        Get symbols from various sources (convenience method).

        Priority order:
        1. symbols_list (explicit list)
        2. symbols_file (file path)
        3. universe (predefined universe name)

        Args:
            universe: Predefined universe name
            symbols_file: Path to file with symbols
            symbols_list: Explicit list of symbols

        Returns:
            List of symbols

        Raises:
            ValueError: If no source provided or all sources fail
        """
        if symbols_list:
            return cls.validate_symbols(symbols_list)

        if symbols_file:
            return cls.validate_symbols(cls.load_from_file(symbols_file))

        if universe:
            return cls.validate_symbols(cls.get_universe(universe))

        raise ValueError("Must provide universe, symbols_file, or symbols_list")

    @classmethod
    def list_universes(cls) -> List[str]:
        """
        Get list of available universe names.

        Returns:
            List of universe names
        """
        available = [k for k, v in cls.PREDEFINED_UNIVERSES.items() if v is not None]
        return sorted(available)

    @classmethod
    def info(cls, universe_name: str) -> dict:
        """
        Get information about a universe.

        Args:
            universe_name: Name of universe

        Returns:
            Dictionary with universe metadata
        """
        universe_name = universe_name.upper()

        if universe_name not in cls.PREDEFINED_UNIVERSES:
            raise ValueError(f"Unknown universe: {universe_name}")

        symbols = cls.PREDEFINED_UNIVERSES[universe_name]

        return {
            'name': universe_name,
            'count': len(symbols) if symbols else 0,
            'available': symbols is not None,
            'symbols': symbols[:10] if symbols else None
        }
