"""F2 target-aware execution tests.

Verifies the adapter trades current -> target value for every position,
sells before buys, applies crash-protection exposure to all holdings.

Tests require use_target_planner=True. The plan is injected via
adapter._latest_plan rather than computed in the test.

Fixture pattern mirrors test_ramp_crash_protection_parity.py and
test_ramp_live_adapter.py: all infrastructure (ExecutionEngine,
PositionManager, StrategyStateManager, PortfolioHealthChecker) is
patched out; execution_engine.execute_order is replaced by a capturing
stub after adapter construction.
"""
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.strategies.core import Signal
from src.strategies.advanced.ramp_target_planner import RampTarget, RampPlan
from src.trading.brokers.broker_interface import BrokerInterface, OrderSide


PORTFOLIO_VALUE = 100_000.0
INITIAL_CAPITAL = 100_000.0
MAX_CAPITAL_ALLOCATION = 1.0
PRICE_PER_SHARE = 100.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plan_for(
    targets_dict: dict,
    exits_dict: dict,
    regime: str = "STRONG_BULL",
    exposure_pct: float = 1.0,
    top_n: int = 20,
) -> RampPlan:
    """Build a minimal RampPlan from weight dicts."""
    targets = {
        sym: RampTarget(
            symbol=sym,
            target_weight=w,
            rank=i + 1,
            regime=regime,
            reason="new_entry",
        )
        for i, (sym, w) in enumerate(targets_dict.items())
    }
    return RampPlan(
        as_of=datetime(2026, 5, 15),
        regime=regime,
        regime_confidence=0.9,
        regime_scores={regime: 0.9},
        exposure_pct=exposure_pct,
        top_n=top_n,
        targets=targets,
        exits=exits_dict,
        diagnostics={},
    )


def _make_adapter(
    symbols: list,
    positions: list,
    top_n: int = 20,
    regime: str = "STRONG_BULL",
    use_target_planner: bool = True,
    cash: float = PORTFOLIO_VALUE,
):
    """Create RAMPLiveAdapter with all infrastructure patched out.

    Args:
        symbols: universe symbols
        positions: list of broker position dicts
                   (keys: symbol, quantity, market_value, current_price, avg_entry_price)
        top_n: regime top-n override
        regime: regime string override
        use_target_planner: passed to constructor
        cash: available cash in the mock account
    """
    from src.trading.adapters.ramp_live_adapter import RAMPLiveAdapter

    broker = MagicMock(spec=BrokerInterface)
    broker.get_account.return_value = {
        "portfolio_value": PORTFOLIO_VALUE,
        "cash": cash,
    }
    broker.get_positions.return_value = list(positions)
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
                        adapter = RAMPLiveAdapter(
                            broker=broker,
                            symbols=symbols,
                            initial_capital=INITIAL_CAPITAL,
                            max_capital_allocation=MAX_CAPITAL_ALLOCATION,
                            reduced_exposure=0.5,
                            vix_threshold=25.0,
                            spy_dd_threshold=-0.05,
                            broker_name="alpaca",
                            use_target_planner=use_target_planner,
                        )
                        adapter.execution_engine = mock_ee_instance
                        adapter.position_manager = mock_pm_instance
                        adapter.state_manager = mock_sm_instance
                        adapter.health_checker = mock_hc_instance

                        adapter._ramp_signals._current_regime = regime
                        adapter._ramp_signals._current_params = {"top_n": top_n}

    return adapter


def _capture_orders(adapter):
    """Replace execution_engine.execute_order with a capturing stub.

    Returns the captured list (mutated in-place as orders arrive).
    Each entry: {symbol, quantity, side, call_index}.
    """
    captured = []

    def _fake_execute_order(symbol, quantity, side, order_type):
        captured.append({
            "symbol": symbol,
            "quantity": quantity,
            "side": side,
            "call_index": len(captured),
        })
        return {
            "order": {
                "filled_avg_price": PRICE_PER_SHARE,
                "filled_qty": quantity,
                "order_id": f"test-{symbol}",
            }
        }

    adapter.execution_engine.execute_order = MagicMock(side_effect=_fake_execute_order)
    return captured


def _pos(symbol: str, qty: int, price: float = PRICE_PER_SHARE) -> dict:
    """Build a broker position dict."""
    return {
        "symbol": symbol,
        "quantity": qty,
        "market_value": qty * price,
        "current_price": price,
        "avg_entry_price": price,
    }


def _current_positions_from_broker_positions(positions: list) -> dict:
    """Build current_positions dict (symbol -> market_value) from broker positions."""
    return {p["symbol"]: p["market_value"] for p in positions}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTargetAwareExecution:

    def test_full_rebalance_sells_before_buys(self):
        """Plan with 5 exits and 5 new entries: all SELLs precede all BUYs."""
        old_syms = [f"OLD{i}" for i in range(5)]
        new_syms = [f"NEW{i}" for i in range(5)]
        all_syms = old_syms + new_syms

        # Broker holds the 5 OLD positions at $10k each
        old_positions = [_pos(s, 100) for s in old_syms]
        current_positions = _current_positions_from_broker_positions(old_positions)

        adapter = _make_adapter(symbols=all_syms, positions=old_positions)
        captured = _capture_orders(adapter)

        plan = _plan_for(
            targets_dict={s: 0.05 for s in new_syms},
            exits_dict={s: "dropped_from_top_n" for s in old_syms},
            top_n=5,
        )
        adapter._latest_plan = plan

        adapter._execute_rebalance_target_aware(
            signals=[],
            current_positions=current_positions,
        )

        sells = [o for o in captured if o["side"] == OrderSide.SELL]
        buys = [o for o in captured if o["side"] == OrderSide.BUY]

        assert len(sells) == 5, f"Expected 5 SELLs, got {len(sells)}"
        assert len(buys) == 5, f"Expected 5 BUYs, got {len(buys)}"

        # Every sell call_index must be less than every buy call_index
        max_sell_idx = max(o["call_index"] for o in sells)
        min_buy_idx = min(o["call_index"] for o in buys)
        assert max_sell_idx < min_buy_idx, (
            f"Some BUYs preceded SELLs: max_sell_idx={max_sell_idx}, "
            f"min_buy_idx={min_buy_idx}"
        )

    def test_block_new_entries_skips_buys_but_keeps_sells(self):
        """Over position cap -> sell-only: exits/trims execute, NEW entries skipped.

        Heals the deadlock where being over max_positions blocks the whole
        rebalance: the rebalance must still SELL down toward top_n. With
        block_new_entries=True, all exits fire but no BUY orders are placed.
        """
        old_syms = [f"OLD{i}" for i in range(5)]
        new_syms = [f"NEW{i}" for i in range(3)]
        old_positions = [_pos(s, 100) for s in old_syms]
        current_positions = _current_positions_from_broker_positions(old_positions)

        adapter = _make_adapter(
            symbols=old_syms + new_syms, positions=old_positions, cash=PORTFOLIO_VALUE
        )
        captured = _capture_orders(adapter)

        plan = _plan_for(
            targets_dict={s: 0.05 for s in new_syms},
            exits_dict={s: "dropped_from_top_n" for s in old_syms},
            top_n=5,
        )
        adapter._latest_plan = plan

        adapter._execute_rebalance_target_aware(
            signals=[],
            current_positions=current_positions,
            block_new_entries=True,
        )

        sells = [o for o in captured if o["side"] == OrderSide.SELL]
        buys = [o for o in captured if o["side"] == OrderSide.BUY]

        assert len(sells) == 5, f"Expected 5 SELLs in sell-only mode, got {len(sells)}"
        assert len(buys) == 0, f"Expected 0 BUYs in sell-only mode, got {len(buys)}"

    def test_target_aware_exit_removes_position_from_state(self):
        """A full exit MUST call state_manager.remove_position, else the closed
        position lingers in state as a phantom and trips the startup reconciler
        (root cause of the 2026-06-09 preflight crash-loop: WELL was sold via
        this path but never removed from state)."""
        syms = [f"OLD{i}" for i in range(3)]
        positions = [_pos(s, 100) for s in syms]
        current_positions = _current_positions_from_broker_positions(positions)

        adapter = _make_adapter(symbols=syms, positions=positions)
        _capture_orders(adapter)

        plan = _plan_for(
            targets_dict={},
            exits_dict={s: "dropped_from_top_n" for s in syms},
            top_n=5,
        )
        adapter._latest_plan = plan

        adapter._execute_rebalance_target_aware(
            signals=[],
            current_positions=current_positions,
        )

        removed = {
            c.args[1] for c in adapter.state_manager.remove_position.call_args_list
        }
        assert removed == set(syms), (
            f"Expected remove_position for exited {syms}, got {removed}"
        )

    def test_hold_overweight_position_is_trimmed(self):
        """Held at $12k, target 10% of $100k = $10k -> SELL ~20 shares ($2k @ $100)."""
        # 120 shares at $100 = $12k held; target weight 0.10 -> $10k
        sym = "AAPL"
        positions = [_pos(sym, 120)]
        current_positions = {sym: 12_000.0}

        adapter = _make_adapter(symbols=[sym], positions=positions)
        captured = _capture_orders(adapter)

        plan = _plan_for(
            targets_dict={sym: 0.10},
            exits_dict={},
            top_n=10,
        )
        adapter._latest_plan = plan

        adapter._execute_rebalance_target_aware(
            signals=[],
            current_positions=current_positions,
        )

        sells = [o for o in captured if o["side"] == OrderSide.SELL]
        assert len(sells) == 1, f"Expected 1 SELL (trim), got {len(sells)}"
        # $2k / $100 = 20 shares
        assert sells[0]["quantity"] == 20, (
            f"Expected trim qty=20, got {sells[0]['quantity']}"
        )
        assert sells[0]["symbol"] == sym

    def test_hold_underweight_position_is_topped_up(self):
        """Held at $8k, target $10k -> BUY ~20 shares ($2k @ $100)."""
        sym = "MSFT"
        positions = [_pos(sym, 80)]
        current_positions = {sym: 8_000.0}

        # Need enough cash available
        adapter = _make_adapter(
            symbols=[sym], positions=positions, cash=PORTFOLIO_VALUE
        )
        captured = _capture_orders(adapter)

        plan = _plan_for(
            targets_dict={sym: 0.10},
            exits_dict={},
            top_n=10,
        )
        adapter._latest_plan = plan

        adapter._execute_rebalance_target_aware(
            signals=[],
            current_positions=current_positions,
        )

        buys = [o for o in captured if o["side"] == OrderSide.BUY]
        assert len(buys) == 1, f"Expected 1 BUY (top-up), got {len(buys)}"
        assert buys[0]["quantity"] == 20, (
            f"Expected top-up qty=20, got {buys[0]['quantity']}"
        )
        assert buys[0]["symbol"] == sym

    def test_below_threshold_delta_is_skipped(self):
        """Held at $9,949.50 (99.495 shares), target $10k, min_trade_value=$100 -> no order."""
        sym = "GOOG"
        # Market value just $50.50 below target; below the $100 threshold
        positions = [{"symbol": sym, "quantity": 99, "market_value": 9900.0,
                      "current_price": PRICE_PER_SHARE, "avg_entry_price": PRICE_PER_SHARE}]
        # Held at $9,900. Target = $10,000. Delta = $100. With min_trade_value=100
        # we need strictly greater than $100 to trigger. Set held value such that
        # delta < $100: held = $9,949.50 -> delta = $50.50
        current_positions = {sym: 9_949.50}

        adapter = _make_adapter(symbols=[sym], positions=positions)
        # Explicitly set min_trade_value_usd to 100.0 (same as default)
        adapter.min_trade_value_usd = 100.0
        captured = _capture_orders(adapter)

        plan = _plan_for(
            targets_dict={sym: 0.10},
            exits_dict={},
            top_n=10,
        )
        adapter._latest_plan = plan

        adapter._execute_rebalance_target_aware(
            signals=[],
            current_positions=current_positions,
        )

        assert len(captured) == 0, (
            f"Expected no orders for sub-threshold delta, got {len(captured)}: {captured}"
        )

    def test_crash_protection_propagates_to_holds(self):
        """10 symbols held at $10k each. Plan with exposure_pct=0.5 -> target $5k each -> 10 SELLs."""
        syms = [f"SYM{i:02d}" for i in range(10)]
        # Each held at $10k (100 shares at $100)
        positions = [_pos(s, 100) for s in syms]
        # Target weight = 1.0 * 0.5 / 10 = 0.05 -> $5k per position
        current_positions = {s: 10_000.0 for s in syms}

        adapter = _make_adapter(symbols=syms, positions=positions)
        captured = _capture_orders(adapter)

        # exposure_pct=0.5 means target_weight = 1.0 * 0.5 / 10 = 0.05
        plan = _plan_for(
            targets_dict={s: 0.05 for s in syms},
            exits_dict={},
            regime="STRONG_BULL",
            exposure_pct=0.5,
            top_n=10,
        )
        adapter._latest_plan = plan

        adapter._execute_rebalance_target_aware(
            signals=[],
            current_positions=current_positions,
        )

        sells = [o for o in captured if o["side"] == OrderSide.SELL]
        buys = [o for o in captured if o["side"] == OrderSide.BUY]

        assert len(sells) == 10, (
            f"Expected 10 SELLs (crash-protection trim), got {len(sells)}"
        )
        assert len(buys) == 0, f"Expected 0 BUYs, got {len(buys)}"

        # Each trim should be ~50 shares ($5k / $100)
        for sell_order in sells:
            assert sell_order["quantity"] == 50, (
                f"{sell_order['symbol']}: expected trim qty=50, got {sell_order['quantity']}"
            )

    def test_cash_buffer_respected_under_slippage(self):
        """Sequential cash tracking: BUYs stop when simulated cash is exhausted.

        Setup: adapter has $10k cash total, wants to buy 5 positions at $2k each
        (total $10k). With cash_buffer_pct=0.0025, spendable per call ~$9,975.
        After first 4 buys ($8k spent), remaining ~$2k still >= $100 threshold so
        the 5th buy should also fire (all 5 fit within $10k budget).

        More importantly, we test that broker.get_account() is called before each
        BUY to re-read available cash (sequential cash tracking pattern).
        """
        syms = [f"BUY{i}" for i in range(5)]
        positions = []
        current_positions = {}

        # Start with $10k cash; each target is 2% of $100k = $2k
        adapter = _make_adapter(
            symbols=syms, positions=positions, cash=10_000.0
        )
        adapter.cash_buffer_pct = 0.0  # disable buffer for deterministic math

        # Mock broker to decrease cash by $2k per get_account() call after first
        cash_sequence = [10_000.0, 8_000.0, 6_000.0, 4_000.0, 2_000.0, 0.0]
        call_counter = {"n": 0}

        def _get_account_seq():
            val = cash_sequence[min(call_counter["n"], len(cash_sequence) - 1)]
            call_counter["n"] += 1
            return {"portfolio_value": PORTFOLIO_VALUE, "cash": val}

        adapter.broker.get_account.side_effect = _get_account_seq
        captured = _capture_orders(adapter)

        plan = _plan_for(
            targets_dict={s: 0.02 for s in syms},
            exits_dict={},
            top_n=5,
        )
        adapter._latest_plan = plan

        adapter._execute_rebalance_target_aware(
            signals=[],
            current_positions=current_positions,
        )

        buys = [o for o in captured if o["side"] == OrderSide.BUY]

        # get_account() should have been called at least once per BUY
        # (sequential cash tracking)
        assert adapter.broker.get_account.call_count >= len(buys), (
            "Expected get_account() called at least once per BUY for sequential cash tracking"
        )


class TestTargetAwareTradeLogging:
    """Trims and buys MUST reach the trade log and state.

    Background: f7ea22f fixed the full-exit loop and left the trim and buy
    branches silent. compute_lifetime_realized_pnl sums only rows with
    trade_type == 'exit', so an unlogged trim's realized PnL is invisible with no
    warning. The 2026-07-28 audit measured 80 unlogged trims over ~3 months,
    $292k of notional, against a reported lifetime realized PnL of -$3.2k.
    See docs/progress/20260728_RAMP_REALIZED_PNL_AUDIT.md
    """

    def test_trim_logs_exit_with_trimmed_qty(self):
        """A trim realizes PnL and must be logged, with the PARTIAL qty."""
        sym = "AAPL"
        positions = [_pos(sym, 120)]              # $12k held
        current_positions = {sym: 12_000.0}       # target 10% of $100k -> $10k

        adapter = _make_adapter(symbols=[sym], positions=positions)
        _capture_orders(adapter)
        adapter.state_manager.get_positions.return_value = {
            sym: {"qty": 120, "entry_price": 90.0, "entry_time": "2026-07-01T15:55:00"}
        }

        plan = _plan_for(targets_dict={sym: 0.10}, exits_dict={}, top_n=5)
        adapter._latest_plan = plan

        with patch(
            "src.trading.adapters.ramp_live_adapter.get_trade_log_writer"
        ) as mock_writer:
            adapter._execute_rebalance_target_aware(
                signals=[], current_positions=current_positions
            )
            log_exit = mock_writer.return_value.log_exit

        assert log_exit.called, (
            "A trim MUST call log_exit; otherwise its realized PnL never reaches "
            "compute_lifetime_realized_pnl and is silently lost"
        )
        kw = log_exit.call_args.kwargs
        assert kw["symbol"] == sym
        # 20 shares trimmed ($2k / $100), NOT the full 120-share position.
        assert kw["qty"] == 20, f"expected the trimmed qty 20, got {kw['qty']}"
        assert kw["entry_price"] == 90.0, "entry_price must come from state"
        assert kw["metadata"]["exit_reason"] == "trim"

    def test_trim_decrements_state_rather_than_removing(self):
        """A trim is partial: state qty must drop, and the position must survive."""
        sym = "AAPL"
        adapter = _make_adapter(symbols=[sym], positions=[_pos(sym, 120)])
        _capture_orders(adapter)
        adapter.state_manager.get_positions.return_value = {
            sym: {"qty": 120, "entry_price": 90.0}
        }
        adapter._latest_plan = _plan_for(
            targets_dict={sym: 0.10}, exits_dict={}, top_n=5
        )

        with patch("src.trading.adapters.ramp_live_adapter.get_trade_log_writer"):
            adapter._execute_rebalance_target_aware(
                signals=[], current_positions={sym: 12_000.0}
            )

        adapter.state_manager.update_position_qty.assert_called_once_with(
            "ramp", sym, 100
        )
        assert not adapter.state_manager.remove_position.called, (
            "A partial trim must NOT remove the position from state"
        )

    def test_buy_logs_entry(self):
        """The buy path logged nothing, so the trade log had zero entry rows
        across 15 trading days while 225 BUY orders were placed."""
        sym = "NEW"
        adapter = _make_adapter(symbols=[sym], positions=[])
        _capture_orders(adapter)
        adapter.state_manager.get_positions.return_value = {}
        adapter._latest_plan = _plan_for(
            targets_dict={sym: 0.10}, exits_dict={}, top_n=5
        )

        with patch(
            "src.trading.adapters.ramp_live_adapter.get_trade_log_writer"
        ) as mock_writer:
            adapter._execute_rebalance_target_aware(
                signals=[], current_positions={}
            )
            log_entry = mock_writer.return_value.log_entry

        assert log_entry.called, "A BUY MUST call log_entry"
        kw = log_entry.call_args.kwargs
        assert kw["symbol"] == sym
        assert kw["qty"] > 0
        assert kw["price"] == PRICE_PER_SHARE, "must use the actual fill price"

    def test_buy_updates_state_via_add_or_update(self):
        """Must use add_or_update_position so a top-up accumulates rather than
        resetting qty and losing the original cost basis."""
        sym = "NEW"
        adapter = _make_adapter(symbols=[sym], positions=[])
        _capture_orders(adapter)
        adapter.state_manager.get_positions.return_value = {}
        adapter._latest_plan = _plan_for(
            targets_dict={sym: 0.10}, exits_dict={}, top_n=5
        )

        with patch("src.trading.adapters.ramp_live_adapter.get_trade_log_writer"):
            adapter._execute_rebalance_target_aware(
                signals=[], current_positions={}
            )

        assert adapter.state_manager.add_or_update_position.called, (
            "BUY must persist to state via add_or_update_position"
        )
        args = adapter.state_manager.add_or_update_position.call_args
        assert args.args[0] == "ramp" and args.args[1] == sym
        assert args.kwargs.get("broker") is not None, (
            "add_or_update_position requires broker for new positions"
        )

    def test_failed_order_does_not_touch_log_or_state(self):
        """If execute_order returns nothing, we must not claim a fill happened."""
        sym = "AAPL"
        adapter = _make_adapter(symbols=[sym], positions=[_pos(sym, 120)])
        adapter.execution_engine.execute_order = MagicMock(return_value=None)
        adapter.state_manager.get_positions.return_value = {
            sym: {"qty": 120, "entry_price": 90.0}
        }
        adapter._latest_plan = _plan_for(
            targets_dict={sym: 0.10}, exits_dict={}, top_n=5
        )

        with patch(
            "src.trading.adapters.ramp_live_adapter.get_trade_log_writer"
        ) as mock_writer:
            adapter._execute_rebalance_target_aware(
                signals=[], current_positions={sym: 12_000.0}
            )
            assert not mock_writer.return_value.log_exit.called
        assert not adapter.state_manager.update_position_qty.called
