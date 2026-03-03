from dataclasses import dataclass
from datetime import date


@dataclass
class CSPPosition:
    symbol: str
    strike: float
    expiry: date
    entry_date: date
    entry_price: float
    num_contracts: int
    collateral: float

    current_price: float = 0.0
    current_delta: float = 0.0
    current_dte: int = 0

    @property
    def premium_collected(self) -> float:
        return self.entry_price * 100 * self.num_contracts

    @property
    def unrealized_pnl(self) -> float:
        return (self.entry_price - self.current_price) * 100 * self.num_contracts

    @property
    def pnl_pct_of_premium(self) -> float:
        if self.premium_collected == 0:
            return 0.0
        return self.unrealized_pnl / self.premium_collected


@dataclass
class CSPTrade:
    symbol: str
    strike: float
    expiry: date
    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    num_contracts: int
    exit_reason: str
    regime_at_entry: str
    regime_at_exit: str
    momentum_rank_at_entry: int

    @property
    def realized_pnl(self) -> float:
        return (self.entry_price - self.exit_price) * 100 * self.num_contracts

    @property
    def holding_days(self) -> int:
        return (self.exit_date - self.entry_date).days

    @property
    def return_on_collateral(self) -> float:
        collateral = self.strike * 100 * self.num_contracts
        if collateral == 0:
            return 0.0
        return self.realized_pnl / collateral
