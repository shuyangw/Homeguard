"""CSP backtest engine - event-driven day-by-day simulation with callback architecture."""

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from src.strategies.options.csp.contract_selector import CSPContractSelector
from src.strategies.options.csp.mark_to_market import CSPMarkToMarket
from src.strategies.options.csp.position import CSPPosition, CSPTrade
from src.utils.logger import get_logger

logger = get_logger()

EMERGENCY_REGIMES = {"BEAR", "UNPREDICTABLE"}
ENTRY_REGIME = "STRONG_BULL"


@dataclass
class CSPBacktestResult:
    closed_trades: List[CSPTrade]
    daily_snapshots: List[Dict]
    equity_curve: Optional[pd.Series] = None
    initial_capital: float = 100_000

    def __post_init__(self) -> None:
        if self.daily_snapshots:
            dates = [s["date"] for s in self.daily_snapshots]
            equities = [s["equity"] for s in self.daily_snapshots]
            self.equity_curve = pd.Series(equities, index=pd.DatetimeIndex(dates))


class CSPBacktestEngine:

    def __init__(
        self,
        initial_capital: float = 100_000,
        max_csp_allocation: float = 0.30,
        max_positions: int = 5,
        profit_target_pct: float = 0.50,
        loss_limit_multiple: float = 2.0,
        min_dte_exit: int = 5,
        slippage_pct: float = 0.01,
        contract_fee: float = 0.02,
        selector: Optional[CSPContractSelector] = None,
    ) -> None:
        self.initial_capital = initial_capital
        self.max_csp_allocation = max_csp_allocation
        self.max_positions = max_positions
        self.profit_target_pct = profit_target_pct
        self.loss_limit_multiple = loss_limit_multiple
        self.min_dte_exit = min_dte_exit
        self.slippage_pct = slippage_pct
        self.contract_fee = contract_fee
        self.selector = selector or CSPContractSelector()
        self.mtm = CSPMarkToMarket()

        self.cash: float = initial_capital
        self.positions: List[CSPPosition] = []
        self.closed_trades: List[CSPTrade] = []
        self.daily_snapshots: List[Dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        trading_days: List[date],
        get_regime: Callable[[date], Tuple[str, float]],
        get_crash_protection: Callable[[date], bool],
        get_top_n_symbols: Callable[[date], List[str]],
        get_chain: Callable[[str, date], pd.DataFrame],
        get_underlying_price: Callable[[str, date], float],
        options_symbols: List[str],
    ) -> CSPBacktestResult:
        logger.info(
            f"CSP backtest: {len(trading_days)} trading days, "
            f"capital=${self.initial_capital:,.0f}, "
            f"max_alloc={self.max_csp_allocation:.0%}"
        )

        for day in trading_days:
            regime, confidence = get_regime(day)
            crash_active = get_crash_protection(day)
            top_n = get_top_n_symbols(day)

            self._manage_positions(
                day, regime, crash_active, top_n, get_chain, get_underlying_price,
            )

            if regime == ENTRY_REGIME and not crash_active:
                self._scan_entries(
                    day, regime, confidence, top_n, options_symbols,
                    get_chain, get_underlying_price,
                )

            self._record_daily(day, regime)

        self._close_remaining(trading_days[-1] if trading_days else date.today())

        result = CSPBacktestResult(
            closed_trades=list(self.closed_trades),
            daily_snapshots=list(self.daily_snapshots),
            initial_capital=self.initial_capital,
        )

        logger.info(
            f"CSP backtest complete: {len(result.closed_trades)} trades, "
            f"final equity=${self._calc_equity():,.2f}"
        )
        return result

    # ------------------------------------------------------------------
    # Position management (exits)
    # ------------------------------------------------------------------

    def _manage_positions(
        self,
        current_date: date,
        regime: str,
        crash_active: bool,
        top_n: List[str],
        get_chain: Callable[[str, date], pd.DataFrame],
        get_underlying_price: Callable[[str, date], float],
    ) -> None:
        to_close: List[Tuple[CSPPosition, str]] = []

        for pos in self.positions:
            chain = get_chain(pos.symbol, current_date)
            if not chain.empty:
                self.mtm.update_position(pos, chain, current_date)
            else:
                underlying = get_underlying_price(pos.symbol, current_date)
                estimated = self.mtm.estimate_price(
                    pos, current_date, underlying, pos.last_iv,
                )
                pos.current_price = estimated
                pos.current_dte = max((pos.expiry - current_date).days, 0)

            exit_reason = self._check_exit(pos, regime, crash_active, top_n)
            if exit_reason:
                to_close.append((pos, exit_reason))

        for pos, reason in to_close:
            self._close_position(pos, current_date, reason, regime)

    def _check_exit(
        self,
        pos: CSPPosition,
        regime: str,
        crash_active: bool,
        top_n: List[str],
    ) -> Optional[str]:
        if regime in EMERGENCY_REGIMES or crash_active:
            return "regime_change"

        if pos.pnl_pct_of_premium >= self.profit_target_pct:
            return "profit_target"

        if pos.pnl_pct_of_premium <= -self.loss_limit_multiple:
            return "loss_limit"

        if pos.current_dte <= self.min_dte_exit:
            return "dte_exit"

        if pos.symbol not in top_n:
            return "left_top_n"

        return None

    def _close_position(
        self,
        pos: CSPPosition,
        exit_date: date,
        reason: str,
        regime: str,
    ) -> None:
        buy_back_price = pos.current_price * (1 + self.slippage_pct)
        fee = self.contract_fee * pos.num_contracts

        trade = CSPTrade(
            symbol=pos.symbol,
            strike=pos.strike,
            expiry=pos.expiry,
            entry_date=pos.entry_date,
            exit_date=exit_date,
            entry_price=pos.entry_price,
            exit_price=buy_back_price,
            num_contracts=pos.num_contracts,
            exit_reason=reason,
            regime_at_entry=pos.regime_at_entry,
            regime_at_exit=regime,
            momentum_rank_at_entry=pos.momentum_rank,
            entry_fee=pos.entry_fee,
            exit_fee=fee,
        )
        self.closed_trades.append(trade)

        cost = buy_back_price * 100 * pos.num_contracts + fee
        self.cash += pos.collateral - cost

        self.positions.remove(pos)

        logger.debug(
            f"Closed {pos.symbol} K={pos.strike:.1f}: reason={reason}, "
            f"pnl={trade.realized_pnl:+.2f}"
        )

    # ------------------------------------------------------------------
    # Entry scanning
    # ------------------------------------------------------------------

    def _scan_entries(
        self,
        current_date: date,
        regime: str,
        confidence: float,
        top_n: List[str],
        options_symbols: List[str],
        get_chain: Callable[[str, date], pd.DataFrame],
        get_underlying_price: Callable[[str, date], float],
    ) -> None:
        held_symbols = {p.symbol for p in self.positions}
        candidates = [
            s for s in top_n
            if s in options_symbols and s not in held_symbols
        ]

        total_equity = self._calc_equity()
        max_alloc = total_equity * self.max_csp_allocation
        current_collateral = sum(p.collateral for p in self.positions)
        remaining_alloc = max(max_alloc - current_collateral, 0.0)

        if remaining_alloc <= 0:
            return

        open_slots = self.max_positions - len(self.positions)
        if open_slots <= 0:
            return
        per_position_limit = remaining_alloc / open_slots

        for rank, symbol in enumerate(candidates):
            if len(self.positions) >= self.max_positions:
                break

            chain = get_chain(symbol, current_date)
            if chain.empty:
                continue

            contract = self.selector.select_contract(chain)
            if contract is None:
                continue

            strike = float(contract["strike"])
            bid = float(contract["bid"])
            entry_price = bid * (1 - self.slippage_pct)

            if entry_price <= 0:
                continue

            collateral_per_contract = strike * 100
            if collateral_per_contract <= 0:
                continue

            budget = min(per_position_limit, self.cash)
            if budget < collateral_per_contract:
                budget = min(remaining_alloc, self.cash)

            num_contracts = int(
                math.floor(budget / collateral_per_contract)
            )
            if num_contracts <= 0:
                continue

            collateral = collateral_per_contract * num_contracts
            premium = entry_price * 100 * num_contracts
            fee = self.contract_fee * num_contracts

            pos = CSPPosition(
                symbol=symbol,
                strike=strike,
                expiry=contract["expiry"],
                entry_date=current_date,
                entry_price=entry_price,
                num_contracts=num_contracts,
                collateral=collateral,
                current_price=entry_price,
                current_dte=int(contract["days_to_expiry"]),
                regime_at_entry=regime,
                momentum_rank=rank,
                last_iv=float(contract.get("implied_vol", 0.30)),
                entry_fee=fee,
            )

            self.cash -= collateral
            self.cash += premium - fee
            self.positions.append(pos)
            remaining_alloc -= collateral

            logger.debug(
                f"Opened {symbol} K={strike:.1f} x{num_contracts}: "
                f"premium={premium:.2f}, collateral={collateral:.2f}"
            )

    # ------------------------------------------------------------------
    # Daily snapshot
    # ------------------------------------------------------------------

    def _record_daily(self, current_date: date, regime: str) -> None:
        collateral = sum(p.collateral for p in self.positions)
        unrealized = sum(p.unrealized_pnl for p in self.positions)
        equity = self.cash + collateral + unrealized

        self.daily_snapshots.append({
            "date": current_date,
            "equity": equity,
            "cash": self.cash,
            "collateral": collateral,
            "unrealized_pnl": unrealized,
            "num_positions": len(self.positions),
            "regime": regime,
        })

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _calc_equity(self) -> float:
        collateral = sum(p.collateral for p in self.positions)
        unrealized = sum(p.unrealized_pnl for p in self.positions)
        return self.cash + collateral + unrealized

    def _close_remaining(self, final_date: date) -> None:
        remaining = list(self.positions)
        for pos in remaining:
            self._close_position(pos, final_date, "backtest_end", "END")
