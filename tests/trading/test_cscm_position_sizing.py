"""
Position-sizing regression tests for CSCMLiveAdapter.rebalance().

Reproduces the 2026-04-26 bug where 6/7 fills succeeded but the 7th failed
with InsufficientFundsError because each fill consumed ~12.5 bps more cash
than the strategy planned (slippage + fees), and the strategy never re-read
remaining cash between fills.

Fix: sequential cash tracking + per-fill safety factor in rebalance().
These tests run against a real DemoBroker with deterministic prices and
non-randomized worst-case slippage to verify the fix holds.
"""
from datetime import datetime
from decimal import Decimal
from unittest.mock import patch, MagicMock

import pytest

from src.streaming.types import Bar
from src.trading.demo.demo_broker import DemoBroker


UNIVERSE = [
    'BTC/USD', 'ETH/USD', 'SOL/USD', 'AVAX/USD',
    'LINK/USD', 'DOGE/USD', 'SUSHI/USD',
]
PRICES = {
    'BTC/USD': 77500.0,
    'ETH/USD': 2310.0,
    'SOL/USD': 86.0,
    'AVAX/USD': 9.30,
    'LINK/USD': 9.30,
    'DOGE/USD': 0.10,
    'SUSHI/USD': 0.95,
}


def _seed_broker_prices(broker: DemoBroker, prices: dict) -> None:
    """Pre-populate the bar buffer with fixed prices so tests don't hit network."""
    ts = datetime(2026, 4, 26, 0, 0, 0)
    for symbol, price in prices.items():
        bar = Bar(
            symbol=symbol,
            timestamp=ts,
            open=price, high=price, low=price, close=price,
            volume=1000.0,
        )
        broker._bar_buffer.add_bar(bar)


def _make_adapter(broker, buy_safety_factor=None):
    """Construct a CSCMLiveAdapter wired to a pre-seeded broker."""
    with patch('src.trading.adapters.cscm_live_adapter.CSCMLiveAdapter._load_state'):
        from src.trading.adapters.cscm_live_adapter import CSCMLiveAdapter
        adapter = CSCMLiveAdapter(
            universe=UNIVERSE,
            top_n=7,
            broker=broker,
            paper=True,
            max_capital_usd=100000,
            buy_safety_factor=buy_safety_factor,
        )
    # Patch the network paths so tests are deterministic
    adapter._fetch_historical_data = MagicMock(return_value={'BTC/USD': MagicMock()})
    adapter.signals.update_historical_data = MagicMock()
    return adapter


def _stub_signals(adapter, target_weights, regime='bullish'):
    """Stub signals.get_risk_signals + get_target_positions."""
    risk_signals = MagicMock(
        regime=regime,
        btc_price=PRICES['BTC/USD'],
        btc_sma=70000.0,
        is_rebalance_day=True,
        reduce_exposure=False,
        exposure_pct=1.0,
        top_symbols=list(target_weights.keys()),
        momentum_scores={s: 0.1 for s in target_weights},
    )
    adapter.signals.get_risk_signals = MagicMock(return_value=risk_signals)
    adapter.signals.get_target_positions = MagicMock(return_value=target_weights)
    adapter.signals.mark_rebalanced = MagicMock()


@pytest.fixture
def broker_with_prices():
    """DemoBroker with $100k cash, deterministic worst-case slippage, seeded bars."""
    broker = DemoBroker(initial_cash=100000.0, slippage_bps=5.0, fee_bps=10.0)
    # Reset to wipe any persisted state from previous test runs (state file
    # at ~/.homeguard/demo/state.json is shared across DemoBroker instances).
    broker.reset_portfolio(initial_cash=100000.0)
    # Force worst-case slippage on every fill -- no randomization.
    broker._execution_sim.randomize_slippage = False
    _seed_broker_prices(broker, PRICES)
    return broker


def test_rebalance_all_targets_fill_under_slippage_and_fees(broker_with_prices):
    """All 7 equal-weight targets must fill cleanly, no InsufficientFundsError on the last."""
    adapter = _make_adapter(broker_with_prices)
    weights = {sym: 1.0 / 7 for sym in UNIVERSE}
    _stub_signals(adapter, weights)

    adapter.rebalance()

    # Inspect the execution records via the decision-log emit path.
    # `_maybe_emit_decision` writes after rebalance; we instead inspect the
    # broker's portfolio + order log to confirm fills.
    positions = broker_with_prices.get_crypto_positions()
    assert len(positions) == 7, (
        f"Expected 7 filled positions, got {len(positions)}: "
        f"{[p['symbol'] for p in positions]}"
    )
    # Final cash must be non-negative; the bug was negative-cash overshoot
    account = broker_with_prices.get_account()
    assert account['cash'] >= 0, f"cash went negative: {account['cash']}"


def test_rebalance_skips_below_threshold_for_existing_positions(broker_with_prices):
    """If a current position is within 5% of target, it must be recorded as a hold/skip."""
    # Pre-load BTC at exactly the equal-weight target -- this should trip the 5% threshold
    broker = broker_with_prices
    target_value = 100000.0 / 7
    btc_price = PRICES['BTC/USD']
    target_qty = Decimal(str(target_value / btc_price))
    broker._portfolio.add_or_update_position(
        symbol='BTC/USD',
        quantity_delta=target_qty,
        price=btc_price,
    )
    # Deduct cash for that pre-loaded position so account state is consistent
    broker._portfolio.adjust_cash(-float(target_qty) * btc_price)

    adapter = _make_adapter(broker)
    weights = {sym: 1.0 / 7 for sym in UNIVERSE}
    _stub_signals(adapter, weights)

    adapter.rebalance()

    positions = broker.get_crypto_positions()
    held_symbols = {p['symbol'] for p in positions}
    # BTC should still be held (skipped), and the other 6 should now be filled
    assert 'BTC/USD' in held_symbols
    assert len(held_symbols) == 7


def test_rebalance_uses_safety_factor_to_size_under_cash(broker_with_prices):
    """Each BUY's target_value should be available_cash * safety_factor / remaining_buys."""
    adapter = _make_adapter(broker_with_prices, buy_safety_factor=0.99)
    weights = {sym: 1.0 / 7 for sym in UNIVERSE}
    _stub_signals(adapter, weights)

    # Spy on place_crypto_order to capture each request
    orig_place = broker_with_prices.place_crypto_order
    captured = []

    def spy(*args, **kwargs):
        captured.append(kwargs)
        return orig_place(*args, **kwargs)

    broker_with_prices.place_crypto_order = spy

    adapter.rebalance()

    # First BUY should be sized at $100,000 * 0.99 / 7 = $14,142.86
    # which is 1% under the naive $14,285.71 target -- proving the safety factor applied.
    buy_calls = [c for c in captured if str(c.get('side', '')).lower().endswith('buy')]
    assert len(buy_calls) >= 1
    first_buy = buy_calls[0]
    first_qty = float(first_buy['quantity'])
    first_symbol = first_buy['symbol']
    first_price = PRICES[first_symbol]
    first_notional = first_qty * first_price
    naive_target = 100000.0 / 7
    safe_target = naive_target * 0.99
    # Allow 0.5% tolerance for sequential-tracking dynamics on the first fill
    assert first_notional == pytest.approx(safe_target, rel=0.005), (
        f"first BUY notional {first_notional:.2f} should be near {safe_target:.2f} "
        f"(safety factor 0.99 applied to naive target {naive_target:.2f})"
    )
