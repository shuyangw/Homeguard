"""
Gamma Exposure (GEX) Calculator.

Calculates dealer gamma exposure from options chain data to identify:
- Net GEX: Overall market maker positioning (stabilizing vs destabilizing)
- Pin Strike: The strike price stocks are "pulled" toward
- GEX Profile: How hedging pressure changes as price moves

Key Assumptions:
- Dealers are NET SHORT options (sold to retail/institutions)
- Short calls = negative gamma exposure (buy on rally, sell on dip)
- Short puts = negative gamma exposure (same effect)
- Positive net GEX = stabilizing (dealers absorb moves)
- Negative net GEX = destabilizing (dealers amplify moves)

Formula:
    GEX_contract = gamma * open_interest * 100 * spot_price * direction
    where direction = -1 for dealer short positions
"""

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.utils.logger import logger
from src.data.options.options_schema import (
    OptionsChain,
    OptionSnapshot,
    StrikeGEX,
    DailyGEXSummary,
    OptionRight,
)


@dataclass
class GEXProfile:
    """
    GEX profile across a price range.

    Shows how net GEX changes as the underlying price moves,
    useful for identifying support/resistance from gamma.
    """
    prices: List[float]          # Price levels
    net_gex: List[float]         # Net GEX at each price
    call_gex: List[float]        # Call GEX at each price
    put_gex: List[float]         # Put GEX at each price
    zero_gamma_price: Optional[float]  # Price where GEX crosses zero

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for analysis."""
        return pd.DataFrame({
            "price": self.prices,
            "net_gex": self.net_gex,
            "call_gex": self.call_gex,
            "put_gex": self.put_gex,
        })


class GEXCalculator:
    """
    Calculator for Gamma Exposure (GEX) from options chains.

    Analyzes options open interest and gamma to determine:
    1. Net GEX: Are dealers long or short gamma?
    2. Pin Strike: Where is price likely to be "pinned"?
    3. Support/Resistance: Where does gamma change direction?

    Usage:
        calc = GEXCalculator()

        # Calculate GEX by strike
        strike_gex = calc.calculate_strike_gex(chain)

        # Find likely pin strike
        pin = calc.find_pin_strike(chain)

        # Get total market GEX
        total = calc.calculate_total_gex(chain)

        # Get daily summary
        summary = calc.calculate_daily_summary(chain)
    """

    # Contract multiplier (standard options = 100 shares)
    CONTRACT_MULTIPLIER = 100

    # Default assumption: dealers are net short
    DEFAULT_DEALER_POSITION = "short"

    def __init__(
        self,
        dealer_position: str = "short",
        near_strike_pct: float = 0.10,
    ):
        """
        Initialize GEX calculator.

        Args:
            dealer_position: Assumed dealer position ("short" or "long")
            near_strike_pct: Strikes within this % of spot are "near money"
        """
        self.dealer_position = dealer_position
        self.near_strike_pct = near_strike_pct

        # Direction multiplier based on dealer position
        # Short gamma = -1 (dealers hedge opposite to gamma)
        self._direction = -1 if dealer_position == "short" else 1

    def calculate_contract_gex(
        self,
        gamma: float,
        open_interest: int,
        spot_price: float,
    ) -> float:
        """
        Calculate GEX for a single contract.

        Formula:
            GEX = gamma * OI * 100 * spot * direction

        Args:
            gamma: Option gamma
            open_interest: Number of contracts
            spot_price: Current underlying price

        Returns:
            GEX in dollars (notional delta change per $1 move)
        """
        return (
            gamma *
            open_interest *
            self.CONTRACT_MULTIPLIER *
            spot_price *
            self._direction
        )

    def calculate_strike_gex(
        self,
        chain: OptionsChain,
    ) -> Dict[float, StrikeGEX]:
        """
        Calculate GEX aggregated by strike price.

        Args:
            chain: Options chain with contracts

        Returns:
            Dict mapping strike -> StrikeGEX
        """
        if not chain.contracts:
            return {}

        spot = chain.underlying_price
        strike_data: Dict[float, Dict] = {}

        for contract in chain.contracts:
            strike = contract.strike

            if strike not in strike_data:
                strike_data[strike] = {
                    "call_gex": 0.0,
                    "put_gex": 0.0,
                    "call_oi": 0,
                    "put_oi": 0,
                }

            gex = self.calculate_contract_gex(
                gamma=contract.gamma,
                open_interest=contract.open_interest,
                spot_price=spot,
            )

            if contract.is_call:
                strike_data[strike]["call_gex"] += gex
                strike_data[strike]["call_oi"] += contract.open_interest
            else:
                strike_data[strike]["put_gex"] += gex
                strike_data[strike]["put_oi"] += contract.open_interest

        # Convert to StrikeGEX objects
        result = {}
        for strike, data in strike_data.items():
            net_gex = data["call_gex"] + data["put_gex"]
            result[strike] = StrikeGEX(
                strike=strike,
                call_gex=data["call_gex"],
                put_gex=data["put_gex"],
                net_gex=net_gex,
                call_oi=data["call_oi"],
                put_oi=data["put_oi"],
            )

        return result

    def calculate_total_gex(self, chain: OptionsChain) -> float:
        """
        Calculate total net GEX across all strikes.

        Positive = dealers long gamma = stabilizing
        Negative = dealers short gamma = destabilizing

        Args:
            chain: Options chain

        Returns:
            Total net GEX
        """
        strike_gex = self.calculate_strike_gex(chain)
        return sum(sg.net_gex for sg in strike_gex.values())

    def find_pin_strike(
        self,
        chain: OptionsChain,
        method: str = "max_gex",
    ) -> Optional[float]:
        """
        Identify the most likely pin strike.

        Methods:
        - max_gex: Strike with highest absolute GEX near spot
        - max_oi: Strike with highest total OI near spot
        - weighted: Combination of GEX magnitude and proximity

        Args:
            chain: Options chain
            method: Pin identification method

        Returns:
            Pin strike price, or None if cannot determine
        """
        if not chain.contracts:
            return None

        spot = chain.underlying_price
        strike_gex = self.calculate_strike_gex(chain)

        if not strike_gex:
            return None

        # Filter to near-money strikes
        min_strike = spot * (1 - self.near_strike_pct)
        max_strike = spot * (1 + self.near_strike_pct)
        near_money = {
            k: v for k, v in strike_gex.items()
            if min_strike <= k <= max_strike
        }

        if not near_money:
            # Fall back to closest strikes
            strikes = sorted(strike_gex.keys())
            closest = min(strikes, key=lambda s: abs(s - spot))
            return closest

        if method == "max_gex":
            # Strike with highest absolute GEX
            return max(near_money.keys(), key=lambda s: abs(near_money[s].net_gex))

        elif method == "max_oi":
            # Strike with highest total OI
            return max(near_money.keys(), key=lambda s: near_money[s].total_oi)

        elif method == "weighted":
            # Weight by GEX magnitude and inverse distance
            scores = {}
            for strike, gex in near_money.items():
                distance = abs(strike - spot) / spot
                distance_weight = 1 / (1 + distance * 10)  # Closer = higher weight
                gex_weight = abs(gex.net_gex)
                scores[strike] = gex_weight * distance_weight

            return max(scores.keys(), key=lambda s: scores[s])

        else:
            raise ValueError(f"Unknown method: {method}")

    def calculate_gex_profile(
        self,
        chain: OptionsChain,
        price_range: Tuple[float, float] = (0.95, 1.05),
        n_points: int = 21,
    ) -> GEXProfile:
        """
        Calculate GEX profile across a price range.

        Shows how net GEX changes as the underlying moves,
        useful for identifying gamma "walls" and flip points.

        Args:
            chain: Options chain
            price_range: (min_pct, max_pct) of spot price
            n_points: Number of price points to calculate

        Returns:
            GEXProfile with GEX at each price level
        """
        spot = chain.underlying_price
        min_price = spot * price_range[0]
        max_price = spot * price_range[1]

        prices = np.linspace(min_price, max_price, n_points)
        net_gex_list = []
        call_gex_list = []
        put_gex_list = []

        for price in prices:
            total_call_gex = 0.0
            total_put_gex = 0.0

            for contract in chain.contracts:
                gex = self.calculate_contract_gex(
                    gamma=contract.gamma,
                    open_interest=contract.open_interest,
                    spot_price=price,  # Use hypothetical price
                )

                if contract.is_call:
                    total_call_gex += gex
                else:
                    total_put_gex += gex

            net_gex_list.append(total_call_gex + total_put_gex)
            call_gex_list.append(total_call_gex)
            put_gex_list.append(total_put_gex)

        # Find zero-gamma crossing
        zero_gamma = None
        for i in range(1, len(net_gex_list)):
            if net_gex_list[i-1] * net_gex_list[i] < 0:  # Sign change
                # Linear interpolation
                x1, x2 = prices[i-1], prices[i]
                y1, y2 = net_gex_list[i-1], net_gex_list[i]
                zero_gamma = x1 - y1 * (x2 - x1) / (y2 - y1)
                break

        return GEXProfile(
            prices=list(prices),
            net_gex=net_gex_list,
            call_gex=call_gex_list,
            put_gex=put_gex_list,
            zero_gamma_price=zero_gamma,
        )

    def calculate_daily_summary(
        self,
        chain: OptionsChain,
        historical_gex: Optional[List[float]] = None,
    ) -> DailyGEXSummary:
        """
        Calculate daily GEX summary for storage/analysis.

        Args:
            chain: Options chain for the day
            historical_gex: Historical GEX values for percentile calc

        Returns:
            DailyGEXSummary with all metrics
        """
        spot = chain.underlying_price
        strike_gex = self.calculate_strike_gex(chain)

        # Total GEX
        total_gex = sum(sg.net_gex for sg in strike_gex.values())
        call_gex = sum(sg.call_gex for sg in strike_gex.values())
        put_gex = sum(sg.put_gex for sg in strike_gex.values())

        # Pin strike
        pin_strike = self.find_pin_strike(chain, method="max_gex")
        pin_strike_gex = 0.0
        distance_to_pin = 0.0

        if pin_strike is not None and pin_strike in strike_gex:
            pin_strike_gex = strike_gex[pin_strike].net_gex
            distance_to_pin = (spot - pin_strike) / pin_strike

        # GEX percentile
        gex_percentile = 0.0
        if historical_gex:
            gex_percentile = (
                sum(1 for g in historical_gex if g < abs(total_gex))
                / len(historical_gex) * 100
            )

        return DailyGEXSummary(
            symbol=chain.symbol,
            date=chain.date,
            expiry=chain.expiry,
            underlying_price=spot,
            total_gex=total_gex,
            call_gex=call_gex,
            put_gex=put_gex,
            pin_strike=pin_strike or 0.0,
            pin_strike_gex=pin_strike_gex,
            distance_to_pin=distance_to_pin,
            total_oi=chain.total_oi,
            put_call_ratio=chain.put_call_ratio,
            gex_percentile=gex_percentile,
        )

    def get_gex_regime(self, total_gex: float) -> str:
        """
        Classify GEX regime.

        Args:
            total_gex: Net GEX value

        Returns:
            Regime string: "positive", "negative", or "neutral"
        """
        # Threshold for meaningful GEX (can be tuned)
        threshold = 0  # Simple sign-based

        if total_gex > threshold:
            return "positive"
        elif total_gex < -threshold:
            return "negative"
        else:
            return "neutral"

    def analyze_pin_probability(
        self,
        chain: OptionsChain,
    ) -> Dict[float, float]:
        """
        Estimate probability of pinning at each strike.

        Based on relative GEX magnitude and proximity to spot.

        Args:
            chain: Options chain

        Returns:
            Dict mapping strike -> probability (0-1)
        """
        spot = chain.underlying_price
        strike_gex = self.calculate_strike_gex(chain)

        if not strike_gex:
            return {}

        # Calculate raw scores
        scores = {}
        for strike, gex in strike_gex.items():
            distance = abs(strike - spot) / spot
            # Higher GEX and closer distance = higher score
            if distance < self.near_strike_pct:
                distance_factor = 1 - (distance / self.near_strike_pct)
                gex_factor = abs(gex.net_gex)
                scores[strike] = gex_factor * distance_factor
            else:
                scores[strike] = 0.0

        # Normalize to probabilities
        total_score = sum(scores.values())
        if total_score > 0:
            return {k: v / total_score for k, v in scores.items()}
        else:
            return scores

    def to_dataframe(
        self,
        strike_gex: Dict[float, StrikeGEX],
    ) -> pd.DataFrame:
        """
        Convert strike GEX dict to DataFrame.

        Args:
            strike_gex: Dict from calculate_strike_gex

        Returns:
            DataFrame with one row per strike
        """
        if not strike_gex:
            return pd.DataFrame()

        records = []
        for strike, gex in sorted(strike_gex.items()):
            records.append({
                "strike": strike,
                "call_gex": gex.call_gex,
                "put_gex": gex.put_gex,
                "net_gex": gex.net_gex,
                "call_oi": gex.call_oi,
                "put_oi": gex.put_oi,
                "total_oi": gex.total_oi,
            })

        return pd.DataFrame(records)
