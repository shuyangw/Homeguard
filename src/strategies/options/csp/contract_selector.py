"""CSP contract selector - filters options chain to find the best put to sell."""

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger()


@dataclass
class CSPContractSelector:
    target_delta_min: float = -0.35
    target_delta_max: float = -0.25
    min_dte: int = 21
    max_dte: int = 35
    min_open_interest: int = 100
    max_spread_pct: float = 0.15

    def select_contract(self, chain: pd.DataFrame) -> Optional[pd.Series]:
        if chain.empty:
            return None

        df = chain.copy()

        df = df[df["option_type"] == "P"]
        if df.empty:
            return None

        df = df[
            (df["days_to_expiry"] >= self.min_dte)
            & (df["days_to_expiry"] <= self.max_dte)
        ]
        if df.empty:
            return None

        df = df[
            (df["delta"] >= self.target_delta_min)
            & (df["delta"] <= self.target_delta_max)
        ]
        if df.empty:
            return None

        df = df[df["open_interest"] >= self.min_open_interest]
        if df.empty:
            return None

        safe_mid = df["mid_price"].replace(0, float("nan"))
        spread_pct = (df["ask"] - df["bid"]) / safe_mid
        df = df[spread_pct <= self.max_spread_pct]
        if df.empty:
            return None

        df = df.sort_values("mid_price", ascending=False)
        selected = df.iloc[0]

        logger.debug(
            f"Selected put: strike={selected['strike']:.1f}, "
            f"delta={selected['delta']:.2f}, "
            f"DTE={selected['days_to_expiry']}, "
            f"mid={selected['mid_price']:.2f}"
        )

        return selected
