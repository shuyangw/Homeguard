import math
from datetime import date

import pandas as pd
from scipy.stats import norm

from src.strategies.options.csp.position import CSPPosition
from src.utils.logger import get_logger

logger = get_logger()


class CSPMarkToMarket:

    def update_position(
        self, position: CSPPosition, chain: pd.DataFrame, current_date: date
    ) -> bool:
        if chain.empty:
            return False

        match = chain[
            (chain["expiry"] == position.expiry)
            & (chain["strike"] == position.strike)
            & (chain["option_type"] == "P")
        ]

        if match.empty:
            return False

        row = match.iloc[0]
        position.current_price = float(row["mid_price"])
        position.current_delta = float(row["delta"])
        position.current_dte = int(row["days_to_expiry"])

        logger.debug(
            f"MTM update {position.symbol}: price={position.current_price:.2f}, "
            f"delta={position.current_delta:.2f}, dte={position.current_dte}"
        )

        return True

    def estimate_price(
        self,
        position: CSPPosition,
        current_date: date,
        underlying_price: float,
        last_known_iv: float,
        risk_free_rate: float = 0.05,
    ) -> float:
        days_remaining = (position.expiry - current_date).days
        time_to_expiry = max(days_remaining, 0) / 365.0

        price = self.bs_put_price(
            spot=underlying_price,
            strike=position.strike,
            time_to_expiry_years=time_to_expiry,
            volatility=last_known_iv,
            risk_free_rate=risk_free_rate,
        )

        logger.debug(
            f"B-S estimate {position.symbol}: spot={underlying_price:.2f}, "
            f"K={position.strike:.1f}, T={time_to_expiry:.4f}, "
            f"iv={last_known_iv:.2f} -> price={price:.4f}"
        )

        return price

    @staticmethod
    def bs_put_price(
        spot: float,
        strike: float,
        time_to_expiry_years: float,
        volatility: float,
        risk_free_rate: float = 0.05,
    ) -> float:
        intrinsic = max(strike - spot, 0.0)

        if time_to_expiry_years <= 0:
            return intrinsic

        if volatility <= 0:
            return intrinsic

        sqrt_t = math.sqrt(time_to_expiry_years)
        d1 = (
            math.log(spot / strike)
            + (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry_years
        ) / (volatility * sqrt_t)
        d2 = d1 - volatility * sqrt_t

        put_price = (
            strike * math.exp(-risk_free_rate * time_to_expiry_years) * norm.cdf(-d2)
            - spot * norm.cdf(-d1)
        )

        return max(put_price, 0.0)
