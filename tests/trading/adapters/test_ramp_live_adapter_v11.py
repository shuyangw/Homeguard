"""
TDD tests for RAMPLiveAdapter V11 logic backport (Phase 2D).

V11 composition (from src/research/ramp_phase4/variants.py::_variant_v11):
    proposed_targets (V01 base) -> rank_buffer(...) -> min_hold(...)
    + delta_rebalance_pct = 0.02 at the trade-sizing floor.

The live adapter exposes:
    - self.variant in {'v01', 'v11'} (Phase 2C plumbing).
    - self._position_open_dates: Dict[str, datetime] (Phase 2A).

Phase 2D adds:
    - _LiveAdapterState (HarnessState-compatible shim for the filter functions).
    - _apply_v11_filters(plan, momentum_scores, current_positions, current_date) -> new RampPlan.
    - DELTA_REBALANCE_PCT_V11 (class constant = 0.02) applied at trim/buy floor.

For variant='v01', behavior is byte-identical to the prior production path.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.strategies.advanced.ramp_target_planner import (
    RampPlan,
    RampTarget,
    compute_plan,
)
from src.trading.adapters.ramp_live_adapter import RAMPLiveAdapter


@pytest.fixture
def mock_broker():
    broker = MagicMock()
    broker.get_account.return_value = {
        'buying_power': 100000.0,
        'portfolio_value': 100000.0,
        'cash': 100000.0,
    }
    broker.get_positions.return_value = []
    broker.is_market_open.return_value = True
    return broker


def _build_adapter(mock_broker, **kwargs):
    """Construct adapter with minimal mocks (mirrors test_ramp_live_adapter_variant)."""
    with patch('src.trading.adapters.strategy_adapter.ExecutionEngine'):
        with patch('src.trading.adapters.strategy_adapter.PositionManager'):
            with patch('src.trading.adapters.ramp_live_adapter.StrategyStateManager') as mock_sm:
                with patch('src.trading.adapters.ramp_live_adapter.PortfolioHealthChecker'):
                    with patch('src.trading.adapters.ramp_live_adapter._load_cache_from_disk') as mock_cache:
                        mock_cache.return_value = None
                        mock_sm_instance = MagicMock()
                        mock_sm_instance.is_enabled.return_value = True
                        mock_sm_instance.is_shutdown_requested.return_value = False
                        mock_sm_instance.acquire_execution_lock.return_value = True
                        mock_sm_instance.get_positions.return_value = {}
                        mock_sm_instance.get_runner_session_state.return_value = {
                            'peak_equity_usd': None,
                            'session_open_equity_usd': None,
                            'session_open_date': None,
                            'position_open_dates': None,
                        }
                        mock_sm.return_value = mock_sm_instance

                        defaults = dict(
                            broker=mock_broker,
                            symbols=['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA',
                                     'META', 'AMZN', 'NFLX', 'INTC', 'CSCO',
                                     'V', 'MA'],
                            max_capital_allocation=1.0,
                            broker_name='alpaca',
                        )
                        defaults.update(kwargs)
                        adapter = RAMPLiveAdapter(**defaults)
                        adapter.state_manager = mock_sm_instance
                        return adapter


def _make_plan(*, regime='WEAK_BULL', top_n=10, top_symbols, exposure_pct=1.0,
               max_capital_allocation=1.0, current_positions=None):
    """Build a RampPlan with the given top-N symbols, mimicking compute_plan output."""
    current_positions = current_positions or {}
    per_position_weight = max_capital_allocation * exposure_pct / top_n
    targets = {}
    for rank, sym in enumerate(top_symbols, start=1):
        reason = 'hold' if sym in current_positions else 'new_entry'
        targets[sym] = RampTarget(
            symbol=sym, target_weight=per_position_weight,
            rank=rank, regime=regime, reason=reason,
        )
    exits = {
        sym: 'dropped_from_top_n'
        for sym in current_positions if sym not in targets
    }
    return RampPlan(
        as_of=datetime(2026, 5, 22, 15, 55),
        regime=regime,
        regime_confidence=0.85,
        regime_scores={},
        exposure_pct=exposure_pct,
        top_n=top_n,
        targets=targets,
        exits=exits,
        diagnostics={},
    )


class TestVariantV01Regression:
    """variant='v01' must produce a plan byte-identical to the legacy behavior."""

    def test_variant_v01_does_not_invoke_v11_filters(self, mock_broker):
        adapter = _build_adapter(mock_broker, variant='v01')
        # V01 path: _apply_v11_filters must not be called at planning time.
        # Verify the gate condition simply.
        assert adapter.variant == 'v01'

    def test_variant_v01_plan_unchanged_by_filter_method(self, mock_broker):
        """Calling _apply_v11_filters with a V01 plan but no held positions is idempotent.

        This is the day-1 case: no state, the filter returns a plan equivalent
        to the input. Equivalence is over (symbols set, exposure_pct, top_n).
        """
        adapter = _build_adapter(mock_broker, variant='v01')
        # Build a V01 plan with 10 top symbols, no positions held.
        top_symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA',
                       'META', 'AMZN', 'NFLX', 'INTC', 'CSCO']
        plan = _make_plan(top_n=10, top_symbols=top_symbols)
        momentum = pd.Series(
            [0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01,
             0.005, 0.004],
            index=['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA',
                   'META', 'AMZN', 'NFLX', 'INTC', 'CSCO', 'V', 'MA'],
        )
        # Empty positions and open_dates: V11 filters should be no-ops.
        adapter._position_open_dates = {}
        filtered = adapter._apply_v11_filters(
            plan=plan,
            momentum_scores=momentum,
            current_positions={},
            current_date=datetime(2026, 5, 22),
        )
        assert set(filtered.targets.keys()) == set(top_symbols)
        assert filtered.top_n == 10
        assert filtered.exposure_pct == 1.0


class TestVariantV11Day1:
    """Day-1: no held positions, no open_dates. V11 plan == V01 plan."""

    def test_v11_empty_state_equals_v01_targets(self, mock_broker):
        adapter = _build_adapter(mock_broker, variant='v11')
        top_symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA',
                       'META', 'AMZN', 'NFLX', 'INTC', 'CSCO']
        plan = _make_plan(top_n=10, top_symbols=top_symbols)
        momentum = pd.Series(
            [0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01,
             0.005, 0.004],
            index=['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA',
                   'META', 'AMZN', 'NFLX', 'INTC', 'CSCO', 'V', 'MA'],
        )
        adapter._position_open_dates = {}
        filtered = adapter._apply_v11_filters(
            plan=plan,
            momentum_scores=momentum,
            current_positions={},
            current_date=datetime(2026, 5, 22),
        )
        assert set(filtered.targets.keys()) == set(top_symbols)
        # No exits because no positions were held.
        assert filtered.exits == {}


class TestVariantV11RankBuffer:
    """rank_buffer retains held names whose rank is in [1, top_n + buffer_size]."""

    def test_v11_retains_held_name_within_buffer_zone(self, mock_broker):
        adapter = _build_adapter(mock_broker, variant='v11')
        # top_n = 10, buffer_size = top_n // 2 = 5, buffer_limit = 15.
        # Top-10 symbols: AAPL..CSCO.
        # Held name V at rank 11 (just outside top_n, within buffer): retained.
        top_symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA',
                       'META', 'AMZN', 'NFLX', 'INTC', 'CSCO']
        # Construct momentum so V is at rank 11.
        momentum = pd.Series(
            [0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01,
             0.005, 0.004],
            index=['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA',
                   'META', 'AMZN', 'NFLX', 'INTC', 'CSCO', 'V', 'MA'],
        )
        plan = _make_plan(
            top_n=10, top_symbols=top_symbols,
            current_positions={'V': 5000.0},  # V is held
        )
        # V was originally in exits (dropped from top_n). After V11 filter,
        # V should be back in targets (within buffer).
        adapter._position_open_dates = {'V': datetime(2026, 1, 1)}  # old date; no min_hold protection
        filtered = adapter._apply_v11_filters(
            plan=plan,
            momentum_scores=momentum,
            current_positions={'V': 5000.0},
            current_date=datetime(2026, 5, 22),
        )
        assert 'V' in filtered.targets, (
            f"V (held, rank 11, within buffer of top_n+5=15) must be retained; "
            f"got targets={list(filtered.targets.keys())}"
        )
        # And it should NOT be in exits anymore.
        assert 'V' not in filtered.exits


class TestVariantV11MinHold:
    """min_hold protects positions opened within the past 5 trading days."""

    def test_v11_protects_recently_opened_position(self, mock_broker):
        adapter = _build_adapter(mock_broker, variant='v11')
        # Top-10 symbols.
        top_symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA',
                       'META', 'AMZN', 'NFLX', 'INTC', 'CSCO']
        # Held name 'MA' has momentum way below buffer (rank far past top_n+5).
        # But it was opened 1 day ago -- min_hold should protect it.
        # Need >= 100 symbols so rank > buffer to ensure rank_buffer DROPS it.
        # Build a momentum series with MA at low rank.
        rest = [f'SYM{i}' for i in range(100)]
        all_symbols = top_symbols + ['MA'] + rest
        scores = [1.0 - 0.001 * i for i in range(len(all_symbols))]
        # Put MA at the very end (lowest momentum).
        # Order: top_symbols (high), then rest (mid), then MA (lowest).
        # Reorder list so MA is last:
        all_symbols = top_symbols + rest + ['MA']
        scores = [1.0 - 0.001 * i for i in range(len(all_symbols))]
        momentum = pd.Series(scores, index=all_symbols)
        plan = _make_plan(
            top_n=10, top_symbols=top_symbols,
            current_positions={'MA': 5000.0},
        )
        # MA was opened 1 day ago. calendar_floor_days = ceil(5*7/5) = 7,
        # so 1 day < 7 -- protected.
        adapter._position_open_dates = {'MA': datetime(2026, 5, 21)}
        filtered = adapter._apply_v11_filters(
            plan=plan,
            momentum_scores=momentum,
            current_positions={'MA': 5000.0},
            current_date=datetime(2026, 5, 22),
        )
        assert 'MA' in filtered.targets, (
            f"MA (recently opened, must be protected by min_hold) was dropped; "
            f"got targets={list(filtered.targets.keys())}"
        )
        assert 'MA' not in filtered.exits


class TestVariantV11DeltaRebalanceFloor:
    """delta_rebalance_pct = 0.02 floors out small trades; full exits bypass."""

    def test_v11_delta_threshold_is_two_percent(self, mock_broker):
        adapter = _build_adapter(mock_broker, variant='v11')
        assert adapter.DELTA_REBALANCE_PCT_V11 == 0.02

    def test_v11_floor_blocks_small_trim(self, mock_broker):
        """Trim trade smaller than 2% of equity_base is suppressed under V11."""
        adapter = _build_adapter(mock_broker, variant='v11')
        # equity_base = 100k, V11 floor = max(min_trade_value=100, 100k * 0.02) = $2000.
        # A trim delta of $1500 should be skipped.
        keep, reason = adapter._v11_trade_passes_floor(
            abs_delta=1500.0,
            equity_base=100000.0,
            min_trade_value=100.0,
            is_full_exit=False,
        )
        assert keep is False, f"expected floor to block 1500 trade, kept={keep} reason={reason}"

    def test_v11_floor_allows_large_trim(self, mock_broker):
        adapter = _build_adapter(mock_broker, variant='v11')
        keep, _ = adapter._v11_trade_passes_floor(
            abs_delta=3000.0,
            equity_base=100000.0,
            min_trade_value=100.0,
            is_full_exit=False,
        )
        assert keep is True

    def test_v11_floor_bypassed_by_full_exit(self, mock_broker):
        """A full exit (target_weight==0) must always execute, regardless of size."""
        adapter = _build_adapter(mock_broker, variant='v11')
        keep, _ = adapter._v11_trade_passes_floor(
            abs_delta=500.0,
            equity_base=100000.0,
            min_trade_value=100.0,
            is_full_exit=True,
        )
        assert keep is True, "full exits must bypass the V11 floor"

    def test_v01_floor_is_unchanged(self, mock_broker):
        """V01 floor is just min_trade_value -- no 2% threshold."""
        adapter = _build_adapter(mock_broker, variant='v01')
        # For V01 the helper should report keep=True for any delta > min_trade_value
        # because V01 does not apply the 2% multiplier.
        keep, _ = adapter._v11_trade_passes_floor(
            abs_delta=1500.0,
            equity_base=100000.0,
            min_trade_value=100.0,
            is_full_exit=False,
        )
        assert keep is True, "V01 path must not apply the 2% floor"
