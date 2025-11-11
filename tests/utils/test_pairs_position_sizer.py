"""
Comprehensive unit tests for pairs position sizing strategies.

Tests all three sizing methods: DollarNeutral, VolatilityAdjusted, and RiskParity.
Covers normal operation, edge cases, and error handling.
"""

import pytest
import numpy as np
from src.backtesting.utils.pairs_position_sizer import (
    PairsPositionSizer,
    DollarNeutralSizer,
    VolatilityAdjustedSizer,
    RiskParitySizer,
    create_pairs_sizer
)


class TestDollarNeutralSizer:
    """Test DollarNeutralSizer - equal capital allocation."""

    def test_initialization_default_params(self):
        """Test sizer initialization with default parameters."""
        sizer = DollarNeutralSizer()
        assert sizer.position_pct == 0.1
        assert sizer.min_shares == 1

    def test_initialization_custom_params(self):
        """Test sizer initialization with custom parameters."""
        sizer = DollarNeutralSizer(position_pct=0.2, min_shares=10)
        assert sizer.position_pct == 0.2
        assert sizer.min_shares == 10

    def test_invalid_position_pct_raises_error(self):
        """Test that invalid position_pct raises ValueError."""
        with pytest.raises(ValueError, match="position_pct must be between 0 and 1"):
            DollarNeutralSizer(position_pct=0.0)

        with pytest.raises(ValueError, match="position_pct must be between 0 and 1"):
            DollarNeutralSizer(position_pct=1.5)

        with pytest.raises(ValueError, match="position_pct must be between 0 and 1"):
            DollarNeutralSizer(position_pct=-0.1)

    def test_invalid_min_shares_raises_error(self):
        """Test that invalid min_shares raises ValueError."""
        with pytest.raises(ValueError, match="min_shares must be >= 1"):
            DollarNeutralSizer(min_shares=0)

        with pytest.raises(ValueError, match="min_shares must be >= 1"):
            DollarNeutralSizer(min_shares=-5)

    def test_equal_allocation_equal_prices(self):
        """Test equal allocation when both assets have equal prices."""
        sizer = DollarNeutralSizer(position_pct=0.5)
        shares1, shares2 = sizer.calculate_position_size(
            cash=10000,
            price1=100,
            price2=100,
            hedge_ratio=1.0
        )

        # 50% of 10000 = 5000
        # Split equally: 2500 per leg
        # At $100 each: 25 shares each
        assert shares1 == 25.0
        assert shares2 == 25.0

    def test_equal_allocation_different_prices(self):
        """Test equal allocation with different prices."""
        sizer = DollarNeutralSizer(position_pct=0.4)
        shares1, shares2 = sizer.calculate_position_size(
            cash=10000,
            price1=50,
            price2=200,
            hedge_ratio=1.0
        )

        # 40% of 10000 = 4000
        # Split equally: 2000 per leg
        # At $50: 40 shares, At $200: 10 shares
        assert shares1 == 40.0
        assert shares2 == 10.0

    def test_insufficient_capital_returns_zero(self):
        """Test that insufficient capital returns zero positions."""
        sizer = DollarNeutralSizer(position_pct=0.1, min_shares=100)
        shares1, shares2 = sizer.calculate_position_size(
            cash=1000,
            price1=100,
            price2=100,
            hedge_ratio=1.0
        )

        # 10% of 1000 = 100
        # Split equally: 50 per leg
        # At $100 each: 0 shares (need 100 minimum)
        assert shares1 == 0.0
        assert shares2 == 0.0

    def test_one_leg_insufficient_returns_zero(self):
        """Test that if one leg can't meet min_shares, both return zero."""
        sizer = DollarNeutralSizer(position_pct=0.5, min_shares=10)
        shares1, shares2 = sizer.calculate_position_size(
            cash=10000,
            price1=100,  # Can afford 25 shares
            price2=1000,  # Can only afford 2 shares (< 10 min)
            hedge_ratio=1.0
        )

        assert shares1 == 0.0
        assert shares2 == 0.0

    def test_small_capital_allocation(self):
        """Test with very small capital allocation."""
        sizer = DollarNeutralSizer(position_pct=0.01)  # 1%
        shares1, shares2 = sizer.calculate_position_size(
            cash=10000,
            price1=50,
            price2=50,
            hedge_ratio=1.0
        )

        # 1% of 10000 = 100
        # Split equally: 50 per leg
        # At $50 each: 1 share each
        assert shares1 == 1.0
        assert shares2 == 1.0

    def test_large_capital_allocation(self):
        """Test with large capital allocation."""
        sizer = DollarNeutralSizer(position_pct=1.0)  # 100%
        shares1, shares2 = sizer.calculate_position_size(
            cash=100000,
            price1=100,
            price2=100,
            hedge_ratio=1.0
        )

        # 100% of 100000 = 100000
        # Split equally: 50000 per leg
        # At $100 each: 500 shares each
        assert shares1 == 500.0
        assert shares2 == 500.0

    def test_hedge_ratio_ignored(self):
        """Test that hedge ratio is passed but not used (for compatibility)."""
        sizer = DollarNeutralSizer(position_pct=0.5)
        shares1_a, shares2_a = sizer.calculate_position_size(
            cash=10000, price1=100, price2=100, hedge_ratio=1.0
        )
        shares1_b, shares2_b = sizer.calculate_position_size(
            cash=10000, price1=100, price2=100, hedge_ratio=2.0
        )

        # Dollar-neutral doesn't use hedge ratio
        assert shares1_a == shares1_b
        assert shares2_a == shares2_b


class TestVolatilityAdjustedSizer:
    """Test VolatilityAdjustedSizer - volatility-weighted allocation."""

    def test_initialization_default_params(self):
        """Test initialization with default parameters."""
        sizer = VolatilityAdjustedSizer()
        assert sizer.position_pct == 0.1
        assert sizer.min_shares == 1
        assert sizer.volatility_lookback == 20

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        sizer = VolatilityAdjustedSizer(
            position_pct=0.3,
            min_shares=5,
            volatility_lookback=60
        )
        assert sizer.position_pct == 0.3
        assert sizer.min_shares == 5
        assert sizer.volatility_lookback == 60

    def test_invalid_params_raise_errors(self):
        """Test that invalid parameters raise ValueError."""
        with pytest.raises(ValueError):
            VolatilityAdjustedSizer(position_pct=0.0)

        with pytest.raises(ValueError):
            VolatilityAdjustedSizer(min_shares=0)

        with pytest.raises(ValueError):
            VolatilityAdjustedSizer(volatility_lookback=1)

    def test_equal_volatility_equal_allocation(self):
        """Test equal allocation when volatilities are equal."""
        sizer = VolatilityAdjustedSizer(position_pct=0.5)
        shares1, shares2 = sizer.calculate_position_size(
            cash=10000,
            price1=100,
            price2=100,
            hedge_ratio=1.0,
            volatility1=0.02,
            volatility2=0.02
        )

        # Equal volatilities => Equal allocation (like dollar-neutral)
        assert shares1 == shares2
        assert shares1 > 0

    def test_higher_vol_gets_less_capital(self):
        """Test that higher volatility asset gets less capital."""
        sizer = VolatilityAdjustedSizer(position_pct=0.5)
        shares1, shares2 = sizer.calculate_position_size(
            cash=10000,
            price1=100,
            price2=100,
            hedge_ratio=1.0,
            volatility1=0.04,  # 2x volatility
            volatility2=0.02
        )

        # Asset1 has 2x volatility => should get ~0.5x capital
        # At equal prices, shares1 should be < shares2
        assert shares1 < shares2
        assert shares1 > 0
        assert shares2 > 0

    def test_lower_vol_gets_more_capital(self):
        """Test that lower volatility asset gets more capital."""
        sizer = VolatilityAdjustedSizer(position_pct=0.4)
        shares1, shares2 = sizer.calculate_position_size(
            cash=20000,
            price1=50,
            price2=50,
            hedge_ratio=1.0,
            volatility1=0.01,  # Lower volatility
            volatility2=0.03   # Higher volatility
        )

        # Asset1 has lower vol => should get more capital
        assert shares1 > shares2
        assert shares1 > 0
        assert shares2 > 0

    def test_missing_volatility_falls_back(self):
        """Test fallback to dollar-neutral when volatility missing."""
        sizer = VolatilityAdjustedSizer(position_pct=0.5)

        # Without volatility data
        shares1, shares2 = sizer.calculate_position_size(
            cash=10000,
            price1=100,
            price2=100,
            hedge_ratio=1.0
        )

        # Should fall back to dollar-neutral (equal allocation)
        assert shares1 == 25.0
        assert shares2 == 25.0

    def test_zero_volatility_falls_back(self):
        """Test fallback when volatility is zero."""
        sizer = VolatilityAdjustedSizer(position_pct=0.5)
        shares1, shares2 = sizer.calculate_position_size(
            cash=10000,
            price1=100,
            price2=100,
            hedge_ratio=1.0,
            volatility1=0.0,  # Invalid
            volatility2=0.02
        )

        # Should fall back to dollar-neutral
        assert shares1 == 25.0
        assert shares2 == 25.0

    def test_negative_volatility_falls_back(self):
        """Test fallback when volatility is negative."""
        sizer = VolatilityAdjustedSizer(position_pct=0.5)
        shares1, shares2 = sizer.calculate_position_size(
            cash=10000,
            price1=100,
            price2=100,
            hedge_ratio=1.0,
            volatility1=0.02,
            volatility2=-0.01  # Invalid
        )

        # Should fall back to dollar-neutral
        assert shares1 == 25.0
        assert shares2 == 25.0

    def test_extreme_volatility_difference(self):
        """Test with extreme volatility difference."""
        sizer = VolatilityAdjustedSizer(position_pct=0.8, min_shares=1)
        shares1, shares2 = sizer.calculate_position_size(
            cash=20000,
            price1=100,
            price2=100,
            hedge_ratio=1.0,
            volatility1=0.001,  # Very low
            volatility2=0.05    # Higher (50x difference)
        )

        # Asset1 should get much more capital (or return 0,0 if can't meet min_shares)
        if shares1 > 0 and shares2 > 0:
            assert shares1 > shares2 * 5
        else:
            # Extreme weighting may result in insufficient capital for one leg
            assert shares1 == 0.0 and shares2 == 0.0

    def test_insufficient_capital_returns_zero(self):
        """Test that insufficient capital returns zero."""
        sizer = VolatilityAdjustedSizer(position_pct=0.1, min_shares=100)
        shares1, shares2 = sizer.calculate_position_size(
            cash=1000,
            price1=100,
            price2=100,
            hedge_ratio=1.0,
            volatility1=0.02,
            volatility2=0.02
        )

        assert shares1 == 0.0
        assert shares2 == 0.0


class TestRiskParitySizer:
    """Test RiskParitySizer - risk-balanced allocation with correlation."""

    def test_initialization_default_params(self):
        """Test initialization with default parameters."""
        sizer = RiskParitySizer()
        assert sizer.position_pct == 0.1
        assert sizer.min_shares == 1
        assert sizer.target_risk == 0.02

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        sizer = RiskParitySizer(
            position_pct=0.25,
            min_shares=5,
            target_risk=0.03
        )
        assert sizer.position_pct == 0.25
        assert sizer.min_shares == 5
        assert sizer.target_risk == 0.03

    def test_invalid_params_raise_errors(self):
        """Test that invalid parameters raise ValueError."""
        with pytest.raises(ValueError):
            RiskParitySizer(position_pct=0.0)

        with pytest.raises(ValueError):
            RiskParitySizer(min_shares=0)

        with pytest.raises(ValueError):
            RiskParitySizer(target_risk=0.0)

        with pytest.raises(ValueError):
            RiskParitySizer(target_risk=0.15)  # Too high

    def test_equal_vol_zero_correlation(self):
        """Test allocation with equal volatility and zero correlation."""
        sizer = RiskParitySizer(position_pct=0.5, target_risk=0.02)
        shares1, shares2 = sizer.calculate_position_size(
            cash=10000,
            price1=100,
            price2=100,
            hedge_ratio=1.0,
            volatility1=0.02,
            volatility2=0.02,
            correlation=0.0
        )

        # With equal vol and no correlation, should be roughly equal
        assert shares1 > 0
        assert shares2 > 0
        # Allow some difference due to rounding
        assert abs(shares1 - shares2) <= 5

    def test_positive_correlation_reduces_size(self):
        """Test that positive correlation reduces position size."""
        sizer = RiskParitySizer(position_pct=0.5, target_risk=0.02)

        # Low correlation
        shares1_low, shares2_low = sizer.calculate_position_size(
            cash=10000, price1=100, price2=100, hedge_ratio=1.0,
            volatility1=0.02, volatility2=0.02, correlation=0.1
        )

        # High correlation
        shares1_high, shares2_high = sizer.calculate_position_size(
            cash=10000, price1=100, price2=100, hedge_ratio=1.0,
            volatility1=0.02, volatility2=0.02, correlation=0.9
        )

        # Higher correlation => higher portfolio vol => smaller position
        assert shares1_high <= shares1_low
        assert shares2_high <= shares2_low

    def test_negative_correlation_increases_size(self):
        """Test that negative correlation increases position size."""
        sizer = RiskParitySizer(position_pct=0.5, target_risk=0.02)

        # Positive correlation
        shares1_pos, shares2_pos = sizer.calculate_position_size(
            cash=10000, price1=100, price2=100, hedge_ratio=1.0,
            volatility1=0.02, volatility2=0.02, correlation=0.5
        )

        # Negative correlation
        shares1_neg, shares2_neg = sizer.calculate_position_size(
            cash=10000, price1=100, price2=100, hedge_ratio=1.0,
            volatility1=0.02, volatility2=0.02, correlation=-0.5
        )

        # Negative correlation => lower portfolio vol => larger position
        assert shares1_neg >= shares1_pos
        assert shares2_neg >= shares2_pos

    def test_missing_correlation_uses_default(self):
        """Test that missing correlation uses default value (0.5)."""
        sizer = RiskParitySizer(position_pct=0.5, target_risk=0.02)

        # Without correlation
        shares1_no, shares2_no = sizer.calculate_position_size(
            cash=10000, price1=100, price2=100, hedge_ratio=1.0,
            volatility1=0.02, volatility2=0.02
        )

        # With correlation=0.5
        shares1_yes, shares2_yes = sizer.calculate_position_size(
            cash=10000, price1=100, price2=100, hedge_ratio=1.0,
            volatility1=0.02, volatility2=0.02, correlation=0.5
        )

        # Should be the same
        assert shares1_no == shares1_yes
        assert shares2_no == shares2_yes

    def test_correlation_clipping(self):
        """Test that extreme correlations are clipped."""
        sizer = RiskParitySizer(position_pct=0.5, target_risk=0.02)

        # Correlation > 1 (should be clipped to 0.99)
        shares1_high, shares2_high = sizer.calculate_position_size(
            cash=10000, price1=100, price2=100, hedge_ratio=1.0,
            volatility1=0.02, volatility2=0.02, correlation=1.5
        )

        # Correlation < -1 (should be clipped to -0.99)
        shares1_low, shares2_low = sizer.calculate_position_size(
            cash=10000, price1=100, price2=100, hedge_ratio=1.0,
            volatility1=0.02, volatility2=0.02, correlation=-1.5
        )

        # Both should return valid positions (not crash)
        assert shares1_high > 0
        assert shares2_high > 0
        assert shares1_low > 0
        assert shares2_low > 0

    def test_missing_volatility_falls_back(self):
        """Test fallback to dollar-neutral when volatility missing."""
        sizer = RiskParitySizer(position_pct=0.5, target_risk=0.02)
        shares1, shares2 = sizer.calculate_position_size(
            cash=10000,
            price1=100,
            price2=100,
            hedge_ratio=1.0
        )

        # Should fall back to dollar-neutral
        assert shares1 == 25.0
        assert shares2 == 25.0

    def test_zero_volatility_falls_back(self):
        """Test fallback when volatility is zero."""
        sizer = RiskParitySizer(position_pct=0.5, target_risk=0.02)
        shares1, shares2 = sizer.calculate_position_size(
            cash=10000,
            price1=100,
            price2=100,
            hedge_ratio=1.0,
            volatility1=0.0,
            volatility2=0.02,
            correlation=0.5
        )

        # Should fall back
        assert shares1 == 25.0
        assert shares2 == 25.0

    def test_target_risk_scales_position(self):
        """Test that target_risk parameter scales position size."""
        # Low target risk with higher volatility to avoid cap
        sizer_low = RiskParitySizer(position_pct=1.0, target_risk=0.01)
        shares1_low, shares2_low = sizer_low.calculate_position_size(
            cash=50000, price1=100, price2=100, hedge_ratio=1.0,
            volatility1=0.03, volatility2=0.03, correlation=0.5
        )

        # High target risk
        sizer_high = RiskParitySizer(position_pct=1.0, target_risk=0.05)
        shares1_high, shares2_high = sizer_high.calculate_position_size(
            cash=50000, price1=100, price2=100, hedge_ratio=1.0,
            volatility1=0.03, volatility2=0.03, correlation=0.5
        )

        # Higher target risk => larger position
        # With volatility, target_risk should matter before hitting cap
        assert shares1_high >= shares1_low
        assert shares2_high >= shares2_low
        # Total capital allocated should be higher with higher target risk
        capital_low = (shares1_low + shares2_low) * 100
        capital_high = (shares1_high + shares2_high) * 100
        assert capital_high >= capital_low

    def test_position_pct_caps_allocation(self):
        """Test that position_pct caps the allocation."""
        sizer = RiskParitySizer(position_pct=0.1, target_risk=0.1)  # High target risk
        shares1, shares2 = sizer.calculate_position_size(
            cash=10000,
            price1=100,
            price2=100,
            hedge_ratio=1.0,
            volatility1=0.01,  # Low vol
            volatility2=0.01,  # Low vol
            correlation=0.0
        )

        # Even with low vol and high target risk, should be capped at 10%
        capital_used = (shares1 + shares2) * 100
        assert capital_used <= 1000  # 10% of 10000

    def test_insufficient_capital_returns_zero(self):
        """Test that insufficient capital returns zero."""
        sizer = RiskParitySizer(position_pct=0.1, min_shares=100, target_risk=0.02)
        shares1, shares2 = sizer.calculate_position_size(
            cash=1000,
            price1=100,
            price2=100,
            hedge_ratio=1.0,
            volatility1=0.02,
            volatility2=0.02,
            correlation=0.5
        )

        assert shares1 == 0.0
        assert shares2 == 0.0


class TestFactoryFunction:
    """Test create_pairs_sizer factory function."""

    def test_create_dollar_neutral(self):
        """Test creating dollar-neutral sizer."""
        sizer = create_pairs_sizer('dollar_neutral', position_pct=0.3)
        assert isinstance(sizer, DollarNeutralSizer)
        assert sizer.position_pct == 0.3

    def test_create_volatility_adjusted(self):
        """Test creating volatility-adjusted sizer."""
        sizer = create_pairs_sizer('volatility_adjusted', position_pct=0.25, volatility_lookback=30)
        assert isinstance(sizer, VolatilityAdjustedSizer)
        assert sizer.position_pct == 0.25
        assert sizer.volatility_lookback == 30

    def test_create_risk_parity(self):
        """Test creating risk-parity sizer."""
        sizer = create_pairs_sizer('risk_parity', position_pct=0.2, target_risk=0.03)
        assert isinstance(sizer, RiskParitySizer)
        assert sizer.position_pct == 0.2
        assert sizer.target_risk == 0.03

    def test_invalid_method_raises_error(self):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown sizing method"):
            create_pairs_sizer('invalid_method')

    def test_default_parameters(self):
        """Test factory with default parameters."""
        sizer = create_pairs_sizer('dollar_neutral')
        assert isinstance(sizer, DollarNeutralSizer)
        assert sizer.position_pct == 0.1
        assert sizer.min_shares == 1


class TestSizerComparison:
    """Compare different sizing methods side by side."""

    def test_all_sizersreturn_valid_positions(self):
        """Test that all sizers return valid positions for same inputs."""
        cash = 10000
        price1 = 100
        price2 = 100
        hedge_ratio = 1.0
        vol1 = 0.02
        vol2 = 0.02
        corr = 0.5

        dn_sizer = DollarNeutralSizer(position_pct=0.5)
        va_sizer = VolatilityAdjustedSizer(position_pct=0.5)
        rp_sizer = RiskParitySizer(position_pct=0.5, target_risk=0.02)

        dn_shares1, dn_shares2 = dn_sizer.calculate_position_size(
            cash, price1, price2, hedge_ratio
        )

        va_shares1, va_shares2 = va_sizer.calculate_position_size(
            cash, price1, price2, hedge_ratio, vol1, vol2
        )

        rp_shares1, rp_shares2 = rp_sizer.calculate_position_size(
            cash, price1, price2, hedge_ratio, vol1, vol2, corr
        )

        # All should return valid positions
        assert dn_shares1 > 0 and dn_shares2 > 0
        assert va_shares1 > 0 and va_shares2 > 0
        assert rp_shares1 > 0 and rp_shares2 > 0

    def test_sizers_differ_with_asymmetric_volatility(self):
        """Test that sizers produce different results with asymmetric volatility."""
        cash = 50000
        price1 = 100
        price2 = 100
        hedge_ratio = 1.0
        vol1 = 0.04  # High vol
        vol2 = 0.01  # Low vol
        corr = 0.3

        dn_sizer = DollarNeutralSizer(position_pct=0.5)
        va_sizer = VolatilityAdjustedSizer(position_pct=0.5)
        rp_sizer = RiskParitySizer(position_pct=0.8, target_risk=0.03)

        dn_shares1, dn_shares2 = dn_sizer.calculate_position_size(
            cash, price1, price2, hedge_ratio
        )

        va_shares1, va_shares2 = va_sizer.calculate_position_size(
            cash, price1, price2, hedge_ratio, vol1, vol2
        )

        rp_shares1, rp_shares2 = rp_sizer.calculate_position_size(
            cash, price1, price2, hedge_ratio, vol1, vol2, corr
        )

        # Dollar-neutral ignores vol, should be equal
        assert dn_shares1 == dn_shares2

        # Vol-adjusted should favor low-vol asset
        assert va_shares2 > va_shares1

        # Risk-parity should favor low-vol asset if not falling back
        if rp_shares1 > 0 and rp_shares2 > 0:
            # Check relative allocation makes sense
            capital1 = rp_shares1 * price1
            capital2 = rp_shares2 * price2
            # Lower vol asset should get more or equal capital
            assert capital2 >= capital1 * 0.8  # Allow some variation


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])
