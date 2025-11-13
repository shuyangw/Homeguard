"""
Equity (Stock) Universe Definitions.

Organized lists of stock symbols for different categories.
"""

from typing import List


class EquityUniverse:
    """
    Stock symbol lists organized by category.

    Usage:
        ```python
        # Get FAANG stocks
        symbols = EquityUniverse.FAANG

        # Get all mega cap stocks
        symbols = EquityUniverse.MEGA_CAP

        # Dynamically load S&P 500 constituents
        symbols = EquityUniverse.load_sp500()
        ```
    """

    # FAANG+ Stocks
    FAANG = [
        'META',   # Meta (Facebook)
        'AAPL',   # Apple
        'AMZN',   # Amazon
        'NFLX',   # Netflix
        'GOOGL',  # Alphabet (Google)
    ]

    # Magnificent 7
    MAG7 = [
        'AAPL',   # Apple
        'MSFT',   # Microsoft
        'GOOGL',  # Alphabet
        'AMZN',   # Amazon
        'NVDA',   # NVIDIA
        'META',   # Meta
        'TSLA',   # Tesla
    ]

    # Mega Cap Tech
    MEGA_CAP_TECH = [
        'AAPL',   # Apple
        'MSFT',   # Microsoft
        'GOOGL',  # Alphabet
        'AMZN',   # Amazon
        'NVDA',   # NVIDIA
        'META',   # Meta
        'TSLA',   # Tesla
        'AVGO',   # Broadcom
        'ORCL',   # Oracle
        'ADBE',   # Adobe
        'CRM',    # Salesforce
        'CSCO',   # Cisco
        'INTC',   # Intel
        'AMD',    # AMD
    ]

    # Top 20 by Market Cap (approximate)
    MEGA_CAP = [
        'AAPL',   # Apple
        'MSFT',   # Microsoft
        'GOOGL',  # Alphabet
        'AMZN',   # Amazon
        'NVDA',   # NVIDIA
        'META',   # Meta
        'TSLA',   # Tesla
        'BRK.B',  # Berkshire Hathaway
        'V',      # Visa
        'UNH',    # UnitedHealth
        'JNJ',    # Johnson & Johnson
        'WMT',    # Walmart
        'JPM',    # JPMorgan Chase
        'MA',     # Mastercard
        'PG',     # Procter & Gamble
        'XOM',    # Exxon Mobil
        'HD',     # Home Depot
        'CVX',    # Chevron
        'LLY',    # Eli Lilly
        'ABBV',   # AbbVie
    ]

    # Semiconductor Stocks
    SEMICONDUCTORS = [
        'NVDA',   # NVIDIA
        'AMD',    # AMD
        'INTC',   # Intel
        'TSM',    # Taiwan Semiconductor
        'AVGO',   # Broadcom
        'QCOM',   # Qualcomm
        'TXN',    # Texas Instruments
        'MU',     # Micron
        'AMAT',   # Applied Materials
        'LRCX',   # Lam Research
        'KLAC',   # KLA Corporation
        'ASML',   # ASML Holding
    ]

    # Electric Vehicle & Clean Energy
    EV_CLEAN_ENERGY = [
        'TSLA',   # Tesla
        'RIVN',   # Rivian
        'LCID',   # Lucid
        'NIO',    # NIO
        'XPEV',   # XPeng
        'LI',     # Li Auto
        'F',      # Ford
        'GM',     # General Motors
        'ENPH',   # Enphase Energy
        'SEDG',   # SolarEdge
    ]

    # Cloud & SaaS
    CLOUD_SAAS = [
        'MSFT',   # Microsoft
        'AMZN',   # Amazon (AWS)
        'GOOGL',  # Alphabet (Google Cloud)
        'CRM',    # Salesforce
        'NOW',    # ServiceNow
        'SNOW',   # Snowflake
        'DDOG',   # Datadog
        'MDB',    # MongoDB
        'NET',    # Cloudflare
        'ZS',     # Zscaler
        'CRWD',   # CrowdStrike
        'OKTA',   # Okta
    ]

    # E-commerce & Retail
    ECOMMERCE = [
        'AMZN',   # Amazon
        'SHOP',   # Shopify
        'MELI',   # MercadoLibre
        'BABA',   # Alibaba
        'JD',     # JD.com
        'PDD',    # Pinduoduo
        'SE',     # Sea Limited
        'ETSY',   # Etsy
        'W',      # Wayfair
        'CHWY',   # Chewy
    ]

    # Financial Services
    FINANCIALS = [
        'JPM',    # JPMorgan Chase
        'BAC',    # Bank of America
        'WFC',    # Wells Fargo
        'C',      # Citigroup
        'GS',     # Goldman Sachs
        'MS',     # Morgan Stanley
        'BLK',    # BlackRock
        'SCHW',   # Charles Schwab
        'AXP',    # American Express
        'V',      # Visa
        'MA',     # Mastercard
        'PYPL',   # PayPal
    ]

    # Healthcare & Biotech
    HEALTHCARE = [
        'UNH',    # UnitedHealth
        'JNJ',    # Johnson & Johnson
        'LLY',    # Eli Lilly
        'ABBV',   # AbbVie
        'PFE',    # Pfizer
        'MRK',    # Merck
        'TMO',    # Thermo Fisher
        'ABT',    # Abbott Labs
        'AMGN',   # Amgen
        'GILD',   # Gilead
        'BIIB',   # Biogen
        'VRTX',   # Vertex Pharmaceuticals
    ]

    # Meme Stocks (High Volatility)
    MEME_STOCKS = [
        'GME',    # GameStop
        'AMC',    # AMC Entertainment
        'BB',     # BlackBerry
        'BBBY',   # Bed Bath & Beyond
        'NOK',    # Nokia
        'PLTR',   # Palantir
    ]

    @classmethod
    def get_by_sector(cls, sector: str) -> List[str]:
        """
        Get stocks by sector.

        Args:
            sector: Sector name ('Technology', 'Financial', 'Healthcare', etc.)

        Returns:
            List of stock symbols for that sector
        """
        sector_map = {
            'Technology': cls.MEGA_CAP_TECH,
            'Semiconductors': cls.SEMICONDUCTORS,
            'Cloud': cls.CLOUD_SAAS,
            'Financial': cls.FINANCIALS,
            'Healthcare': cls.HEALTHCARE,
            'ECommerce': cls.ECOMMERCE,
            'EV': cls.EV_CLEAN_ENERGY,
            'MegaCap': cls.MEGA_CAP,
        }

        return sector_map.get(sector, [])

    @classmethod
    def load_sp500(cls) -> List[str]:
        """
        Dynamically load S&P 500 constituents from Wikipedia.

        Returns:
            List of S&P 500 stock symbols

        Note:
            Requires internet connection. For offline use, cache the result.
        """
        try:
            import pandas as pd

            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            table = pd.read_html(url)[0]
            symbols = table['Symbol'].tolist()

            # Clean symbols (remove periods for Berkshire, etc.)
            symbols = [s.replace('.', '-') for s in symbols]

            return symbols

        except Exception as e:
            # Fallback to a static list if internet is unavailable
            print(f"Warning: Failed to load S&P 500 from Wikipedia: {e}")
            print("Using fallback mega cap list")
            return cls.MEGA_CAP

    @classmethod
    def load_nasdaq100(cls) -> List[str]:
        """
        Dynamically load Nasdaq 100 constituents.

        Returns:
            List of Nasdaq 100 stock symbols
        """
        try:
            import pandas as pd

            url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
            tables = pd.read_html(url)

            # Find the table with ticker symbols
            for table in tables:
                if 'Ticker' in table.columns:
                    symbols = table['Ticker'].tolist()
                    return [s for s in symbols if isinstance(s, str)]

            # Fallback
            return cls.MEGA_CAP_TECH

        except Exception as e:
            print(f"Warning: Failed to load Nasdaq 100: {e}")
            return cls.MEGA_CAP_TECH

    @classmethod
    def create_custom_universe(
        cls,
        min_price: float = None,
        max_price: float = None,
        min_market_cap: float = None,
        sectors: List[str] = None
    ) -> List[str]:
        """
        Create a custom universe based on filters.

        Args:
            min_price: Minimum stock price
            max_price: Maximum stock price
            min_market_cap: Minimum market cap (in billions)
            sectors: List of sectors to include

        Returns:
            Filtered list of symbols

        Note:
            This is a placeholder. Full implementation would require
            real-time market data API integration.
        """
        # Placeholder implementation
        # In production, this would query a market data API
        universe = cls.MEGA_CAP.copy()

        if sectors:
            sector_stocks = []
            for sector in sectors:
                sector_stocks.extend(cls.get_by_sector(sector))
            universe = list(set(sector_stocks))

        return universe
