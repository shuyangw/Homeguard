"""F3 crash-protection parity diagnostic.

Tests the live adapter's BUY sizing against expected target gross exposure
for each (regime, VIX, SPY drawdown) combination.

Expected target gross = max_capital_allocation * exposure_pct

Where exposure_pct = 0.5 if VIX > 25 OR SPY-DD < -5%, else 1.0.

Pre-F2: the adapter ignores risk_exposure in BUY sizing. Several scenarios
FAIL by design. The failure magnitudes quantify production drift.

Post-F2 (with use_target_planner=True): all scenarios pass.

Deviations from plan:
- Import is 'from src.strategies.core import Signal' (not src.trading.signal)
- Signal requires timestamp, confidence, price positional fields
- Adapter class is RAMPLiveAdapter (not RAMPAdapter)
- StrategyStateManager, PortfolioHealthChecker, ExecutionEngine, PositionManager
  are patched out (same pattern as test_ramp_live_adapter.py)
- state_manager.symbol_owned_by_other returns None (no cross-strategy conflict)
- execution_engine.execute_order is replaced via _capture_buy_orders after init
"""
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.strategies.core import Signal


PORTFOLIO_VALUE = 100_000.0
INITIAL_CAPITAL = 100_000.0
MAX_CAPITAL_ALLOCATION = 1.0
PRICE_PER_SHARE = 100.0

# Each scenario: (regime, vix, spy_dd, top_n, expected_gross)
SCENARIOS = [
    ("STRONG_BULL", 18, -0.02, 20, 1.0),   # no trigger
    ("STRONG_BULL", 26, -0.02, 20, 0.5),   # VIX trigger
    ("STRONG_BULL", 18, -0.07, 20, 0.5),   # SPY DD trigger
    ("STRONG_BULL", 26, -0.07, 20, 0.5),   # both triggers (no double-reduce)
    ("SIDEWAYS",    18, -0.02,  5, 1.0),
    ("SIDEWAYS",    26, -0.02,  5, 0.5),
    ("BEAR",        22, -0.08, 10, 1.0),   # current production behavior
]


def _make_buy_signal(symbol: str, rank: int, regime: str,
                     risk_exposure: float, top_n: int) -> Signal:
    """Build a BUY signal as RAMPSignals.generate_signals() would emit."""
    return Signal(
        timestamp=datetime(2026, 5, 15, 15, 55, 0),
        symbol=symbol,
        direction="BUY",
        confidence=0.8,
        price=PRICE_PER_SHARE,
        metadata={
            "rank": rank,
            "regime": regime,
            "risk_exposure": risk_exposure,
            "momentum_score": 0.05,
            "top_n": top_n,
        },
    )


def _make_adapter(top_n: int, regime: str):
    """Create RAMPLiveAdapter with all infrastructure patched out."""
    from src.trading.adapters.ramp_live_adapter import RAMPLiveAdapter
    from src.trading.brokers.broker_interface import BrokerInterface

    broker = MagicMock(spec=BrokerInterface)
    broker.get_account.return_value = {
        "portfolio_value": PORTFOLIO_VALUE,
        "cash": PORTFOLIO_VALUE,
    }
    broker.get_positions.return_value = []
    broker.get_latest_quote.side_effect = lambda sym: {
        "ask": PRICE_PER_SHARE,
        "bid": PRICE_PER_SHARE - 0.10,
    }

    mock_ee_instance = MagicMock()
    mock_pm_instance = MagicMock()
    mock_pm_instance.get_open_positions.return_value = []

    mock_sm_instance = MagicMock()
    mock_sm_instance.is_enabled.return_value = True
    mock_sm_instance.is_shutdown_requested.return_value = False
    mock_sm_instance.acquire_execution_lock.return_value = True
    mock_sm_instance.get_positions.return_value = {}
    # No cross-strategy conflict: return None means RAMP owns the symbol
    mock_sm_instance.symbol_owned_by_other.return_value = None

    mock_hc_instance = MagicMock()
    health_result = MagicMock()
    health_result.passed = True
    health_result.warnings = []
    health_result.errors = []
    mock_hc_instance.check_before_entry.return_value = health_result

    with patch("src.trading.adapters.strategy_adapter.ExecutionEngine",
               return_value=mock_ee_instance):
        with patch("src.trading.adapters.strategy_adapter.PositionManager",
                   return_value=mock_pm_instance):
            with patch("src.trading.adapters.ramp_live_adapter.StrategyStateManager",
                       return_value=mock_sm_instance):
                with patch("src.trading.adapters.ramp_live_adapter.PortfolioHealthChecker",
                           return_value=mock_hc_instance):
                    with patch("src.trading.adapters.ramp_live_adapter._load_cache_from_disk",
                               return_value=None):
                        symbols = [f"SYM{i:02d}" for i in range(top_n)]
                        adapter = RAMPLiveAdapter(
                            broker=broker,
                            symbols=symbols,
                            initial_capital=INITIAL_CAPITAL,
                            max_capital_allocation=MAX_CAPITAL_ALLOCATION,
                            reduced_exposure=0.5,
                            vix_threshold=25.0,
                            spy_dd_threshold=-0.05,
                            broker_name="alpaca",
                        )
                        adapter.execution_engine = mock_ee_instance
                        adapter.position_manager = mock_pm_instance
                        adapter.state_manager = mock_sm_instance
                        adapter.health_checker = mock_hc_instance

                        # Force the regime and params to match the scenario
                        adapter._ramp_signals._current_regime = regime
                        adapter._ramp_signals._current_params = {"top_n": top_n}

    return adapter


def _capture_buy_orders(adapter):
    """Replace execution_engine.execute_order with a capturing stub.

    Returns (captured_list, side_effect_fn). The captured list is mutated
    in-place as orders arrive.
    """
    captured = []

    def _fake_execute_order(symbol, quantity, side, order_type):
        captured.append({"symbol": symbol, "quantity": quantity, "side": side})
        # Return the shape execute_order normally returns so BUY path
        # state-manager calls don't crash.
        return {
            "order": {
                "filled_avg_price": PRICE_PER_SHARE,
                "filled_qty": quantity,
                "order_id": f"test-{symbol}",
            }
        }

    adapter.execution_engine.execute_order = MagicMock(side_effect=_fake_execute_order)
    return captured


@pytest.mark.parametrize("regime,vix,spy_dd,top_n,expected_gross", SCENARIOS)
def test_crash_protection_target_gross_legacy(regime, vix, spy_dd, top_n, expected_gross):
    """Diagnostic: verify the legacy adapter sizes BUYs to the expected target gross.

    Pre-F2: scenarios with expected_gross < 1.0 FAIL because the adapter
    ignores risk_exposure in BUY sizing. The failure magnitude shows
    realized_gross > expected_gross (full exposure instead of reduced).
    """
    # Derive risk_exposure from triggers (mirrors RAMPSignals.calculate_risk_signals)
    crash_triggered = (vix > 25) or (spy_dd < -0.05)
    risk_exposure = 0.5 if crash_triggered else 1.0

    adapter = _make_adapter(top_n=top_n, regime=regime)
    captured = _capture_buy_orders(adapter)

    # Build top_n BUY signals with the computed risk_exposure
    signals = [
        _make_buy_signal(
            symbol=f"SYM{i:02d}",
            rank=i + 1,
            regime=regime,
            risk_exposure=risk_exposure,
            top_n=top_n,
        )
        for i in range(top_n)
    ]

    # Run rebalance (legacy path; use_target_planner=False is the default)
    adapter._execute_rebalance(signals=signals, current_positions={})

    # Compute realized target gross from filled orders
    total_buy_value = sum(o["quantity"] * PRICE_PER_SHARE for o in captured)
    realized_gross = total_buy_value / PORTFOLIO_VALUE

    # Assertion: WILL FAIL pre-F2 when expected_gross < 1.0 because the
    # adapter does not read risk_exposure from signal metadata.
    assert abs(realized_gross - expected_gross) < 0.01, (
        f"Regime={regime}, VIX={vix}, DD={spy_dd:.0%}: "
        f"expected target gross={expected_gross:.0%}, got={realized_gross:.0%}. "
        f"Drift={realized_gross - expected_gross:+.0%}."
    )
