"""
IBKR Contract Resolution -- Symbol strings to qualified IBKR Contracts.

IBKR requires fully-qualified Contract objects (with conId) for
unambiguous identification.  This module:

    - Resolves stock/option/future symbols to qualified Contracts
    - Caches conIds to avoid repeated qualifyContracts() round-trips
    - Translates OCC option symbology (used by Alpaca) to IBKR format
    - Handles contract ambiguity (multiple exchanges, etc.)

OCC Symbology:
    'AAPL  260417C00190000'
     ^^^^  ^^^^^^ ^ ^^^^^^^^
     sym   YYMMDD R strike*1000  (R = C or P, strike zero-padded to 8 digits)
"""

from __future__ import annotations


import re
import time
from typing import Dict, List, Optional, Tuple

from ib_async import Contract, Future, Index, Option, Stock

from src.trading.brokers.ibkr.connection import IBKRConnectionManager
from src.trading.brokers.ibkr.symbols import to_ibkr_symbol
from src.trading.brokers.interfaces.base import SymbolNotFoundError

from src.utils.logger import get_logger
logger = get_logger(__name__)


class ContractResolver:
    """
    Resolves symbol strings to fully-qualified IBKR Contracts.

    Caches resolved contracts for 24 hours to minimize qualifyContracts()
    calls.  Thread-safe (all mutations go through the connection manager's
    event loop).
    """

    CACHE_TTL = 86400  # 24 hours

    def __init__(self, connection: IBKRConnectionManager):
        self._conn = connection
        self._cache: Dict[str, Contract] = {}
        self._cache_expiry: Dict[str, float] = {}

    # -- Public API ------------------------------------------------

    def resolve_stock(
        self,
        symbol: str,
        exchange: str = "SMART",
        currency: str = "USD",
    ) -> Contract:
        """
        Resolve a stock ticker to a qualified Contract.

        Uses SMART routing by default (best execution across exchanges).
        Caches the resolved conId for subsequent calls.
        """
        ibkr_symbol = to_ibkr_symbol(symbol)
        key = f"STK:{ibkr_symbol}:{exchange}:{currency}"
        cached = self._get_cached(key)
        if cached:
            return cached

        contract = Stock(ibkr_symbol, exchange, currency)
        qualified = self._qualify(contract)
        self._set_cached(key, qualified)
        return qualified

    def resolve_option(
        self,
        symbol: str,
        expiration: str,  # 'YYYYMMDD'
        strike: float,
        right: str,  # 'C' or 'P'
        exchange: str = "SMART",
        currency: str = "USD",
    ) -> Contract:
        """
        Resolve an option to a qualified Contract.

        Args:
            symbol: Underlying symbol (e.g., 'AAPL')
            expiration: Expiry in YYYYMMDD format
            strike: Strike price
            right: 'C' for call, 'P' for put
            exchange: Exchange (SMART for best routing)
            currency: Currency
        """
        key = f"OPT:{symbol}:{expiration}:{strike}:{right}:{exchange}"
        cached = self._get_cached(key)
        if cached:
            return cached

        contract = Option(symbol, expiration, strike, right, exchange, currency)
        qualified = self._qualify(contract)
        self._set_cached(key, qualified)
        return qualified

    def resolve_index(
        self,
        symbol: str,
        exchange: str = "CBOE",
        currency: str = "USD",
    ) -> Contract:
        """Resolve an index (e.g., 'VIX' on 'CBOE')."""
        key = f"IND:{symbol}:{exchange}:{currency}"
        cached = self._get_cached(key)
        if cached:
            return cached

        contract = Index(symbol, exchange, currency)
        qualified = self._qualify(contract)
        self._set_cached(key, qualified)
        return qualified

    def resolve_from_occ(self, occ_symbol: str) -> Contract:
        """
        Parse an OCC option symbol and resolve to an IBKR Contract.

        OCC format: 'AAPL  260417C00190000'
            - Chars 0-5:  underlying symbol (left-padded with spaces)
            - Chars 6-11: YYMMDD expiration
            - Char  12:   C or P
            - Chars 13-20: strike price * 1000 (zero-padded)

        This bridge is critical for Alpaca/IBKR interop since Alpaca
        uses OCC symbology for all options.
        """
        parsed = self.parse_occ(occ_symbol)
        return self.resolve_option(
            symbol=parsed["symbol"],
            expiration=parsed["expiration"],
            strike=parsed["strike"],
            right=parsed["right"],
        )

    @staticmethod
    def parse_occ(occ_symbol: str) -> dict:
        """
        Parse an OCC option symbol into its components.

        Returns:
            {
                'symbol': 'AAPL',
                'expiration': '20260417',  # YYYYMMDD
                'right': 'C',
                'strike': 190.0,
            }

        Raises:
            ValueError: If the OCC symbol cannot be parsed.
        """
        # OCC symbols are exactly 21 characters
        occ = occ_symbol.strip()

        # Handle both fixed-width (21 char) and variable formats
        # Pattern: SYMBOL + YYMMDD + C/P + 8-digit strike
        pattern = r"^(\w{1,6})\s*(\d{6})([CP])(\d{8})$"
        match = re.match(pattern, occ)
        if not match:
            raise ValueError(f"Cannot parse OCC symbol: '{occ_symbol}'")

        symbol = match.group(1).strip()
        date_str = match.group(2)  # YYMMDD
        right = match.group(3)
        strike_raw = match.group(4)

        # Convert YYMMDD -> YYYYMMDD
        year_2digit = int(date_str[:2])
        year_4digit = 2000 + year_2digit  # Valid through 2099
        expiration = f"{year_4digit}{date_str[2:]}"

        # Strike: last 8 digits / 1000
        strike = int(strike_raw) / 1000.0

        return {
            "symbol": symbol,
            "expiration": expiration,
            "right": right,
            "strike": strike,
        }

    @staticmethod
    def to_occ(symbol: str, expiration: str, strike: float, right: str) -> str:
        """
        Construct an OCC option symbol from components.

        Args:
            symbol: Underlying (e.g., 'AAPL')
            expiration: 'YYYYMMDD'
            strike: Strike price
            right: 'C' or 'P'

        Returns:
            OCC symbol like 'AAPL  260417C00190000'
        """
        sym_padded = symbol.ljust(6)
        date_yymmdd = expiration[2:]  # YYYYMMDD -> YYMMDD
        strike_int = int(strike * 1000)
        strike_padded = f"{strike_int:08d}"
        return f"{sym_padded}{date_yymmdd}{right}{strike_padded}"

    def get_option_chain_expirations(self, symbol: str) -> List[str]:
        """
        Get all available option expiration dates for an underlying.

        Returns list of 'YYYYMMDD' strings, sorted ascending.
        """
        contract = self.resolve_stock(symbol)
        chains = self._conn.run_sync(
            self._conn.ib.reqSecDefOptParamsAsync(
                contract.symbol,
                "",  # futFopExchange
                contract.secType,
                contract.conId,
            )
        )
        expirations = set()
        for chain in chains:
            expirations.update(chain.expirations)
        return sorted(expirations)

    def get_option_chain_strikes(
        self, symbol: str, expiration: str
    ) -> Tuple[List[float], List[float]]:
        """
        Get available strikes for a given underlying and expiration.

        Returns:
            (call_strikes, put_strikes) -- both sorted ascending.
        """
        contract = self.resolve_stock(symbol)
        chains = self._conn.run_sync(
            self._conn.ib.reqSecDefOptParamsAsync(
                contract.symbol, "", contract.secType, contract.conId
            )
        )
        strikes = set()
        for chain in chains:
            if expiration in chain.expirations:
                strikes.update(chain.strikes)
        sorted_strikes = sorted(strikes)
        return sorted_strikes, sorted_strikes  # Both calls and puts share strikes

    # -- Cache Management ------------------------------------------

    def _get_cached(self, key: str) -> Optional[Contract]:
        if key in self._cache and time.time() < self._cache_expiry.get(key, 0):
            return self._cache[key]
        return None

    def _set_cached(self, key: str, contract: Contract) -> None:
        self._cache[key] = contract
        self._cache_expiry[key] = time.time() + self.CACHE_TTL

    def clear_cache(self) -> None:
        """Clear the entire contract cache."""
        self._cache.clear()
        self._cache_expiry.clear()

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    # -- Internal --------------------------------------------------

    def _qualify(self, contract: Contract) -> Contract:
        """
        Qualify a contract via IBKR (resolve conId, verify it exists).
        Raises SymbolNotFoundError if the contract can't be resolved.
        """
        contracts = self._conn.run_sync(
            self._conn.ib.qualifyContractsAsync(contract)
        )
        if not contracts or contracts[0] is None:
            # qualifyContractsAsync returns [None] for ambiguous contracts
            # (multiple matches across exchanges) -- treat the same as empty.
            raise SymbolNotFoundError(
                f"Cannot resolve contract: {contract.symbol} "
                f"secType={contract.secType}"
            )
        qualified = contracts[0]
        logger.debug(f"[IBKR] Qualified {contract.symbol} -> conId={qualified.conId}")
        return qualified
