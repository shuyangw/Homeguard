"""
Tests for MAE/MFE per-trade tracking (methodology Section 11.6).

Verifies that the trade log produced by Portfolio / PortfolioV2 carries the
fields required by methodology Section 11.6 for downstream Section 12.1
diagnostics:

    mae_pct, mfe_pct      -- signed worst/best paper P&L during the trade
    mae_time, mfe_time    -- timestamps at which MAE/MFE occurred
    hit_stop              -- bool: did intra-trade MAE breach the configured stop?
    hit_target            -- bool: did intra-trade MFE breach the configured target?

For longs:
    mae = (running_low  - entry) / entry   (negative when price went down)
    mfe = (running_high - entry) / entry   (positive when price went up)

For shorts (position < 0):
    mae = (entry - running_high) / entry   (negative when price went up)
    mfe = (entry - running_low)  / entry   (positive when price went down)

MAE/MFE only appear on EXIT records (exit, cover_short). Entry records do not
carry these fields.
"""

import pandas as pd
import pytest

from src.backtesting.engine.portfolio_simulator import Portfolio
from src.backtesting.utils.risk_config import RiskConfig


def _exit_records(portfolio: Portfolio):
    return [t for t in portfolio.trades if t.get('type') in ('exit', 'cover_short')]


class TestLongTradeMAEMFE:
    """Long-trade MAE/MFE values and timestamps."""

    def test_records_mae_mfe_for_long_signal_exit(self):
        # Down 5% on bar 1 (MAE), up 10% on bar 2 (MFE), exit at bar 4
        idx = pd.date_range('2023-01-02', periods=5, freq='D')
        price = pd.Series([100.0, 95.0, 110.0, 108.0, 105.0], index=idx)
        entries = pd.Series([True, False, False, False, False], index=idx)
        exits = pd.Series([False, False, False, False, True], index=idx)

        portfolio = Portfolio(
            price=price,
            entries=entries,
            exits=exits,
            init_cash=10000,
            fees=0.0,
            slippage=0.0,
            market_hours_only=False,
            risk_config=RiskConfig(position_size_pct=1.0, use_stop_loss=False),
        )

        exits_log = _exit_records(portfolio)
        assert len(exits_log) == 1
        exit_rec = exits_log[0]

        assert exit_rec['mae_pct'] == pytest.approx(-0.05, abs=1e-9)
        assert exit_rec['mfe_pct'] == pytest.approx(0.10, abs=1e-9)
        assert exit_rec['mae_time'] == idx[1]
        assert exit_rec['mfe_time'] == idx[2]
        assert exit_rec['hit_stop'] is False
        assert exit_rec['hit_target'] is False

    def test_long_trade_that_never_dips_has_zero_mae(self):
        # Monotonic up: MAE bounded at 0, MFE at the high
        idx = pd.date_range('2023-01-02', periods=4, freq='D')
        price = pd.Series([100.0, 102.0, 105.0, 110.0], index=idx)
        entries = pd.Series([True, False, False, False], index=idx)
        exits = pd.Series([False, False, False, True], index=idx)

        portfolio = Portfolio(
            price=price,
            entries=entries,
            exits=exits,
            init_cash=10000,
            fees=0.0,
            slippage=0.0,
            market_hours_only=False,
            risk_config=RiskConfig(position_size_pct=1.0, use_stop_loss=False),
        )

        exit_rec = _exit_records(portfolio)[0]
        assert exit_rec['mae_pct'] == pytest.approx(0.0, abs=1e-9)
        assert exit_rec['mfe_pct'] == pytest.approx(0.10, abs=1e-9)


class TestShortTradeMAEMFE:
    """Short-trade MAE/MFE values (signed by long-convention)."""

    def test_records_mae_mfe_for_short_signal_exit(self):
        # Short entered at 100. Bar 1=105 (adverse, MAE).
        # Bar 2=90 (favorable, MFE). Bar 4=95: cover.
        idx = pd.date_range('2023-01-02', periods=5, freq='D')
        price = pd.Series([100.0, 105.0, 90.0, 92.0, 95.0], index=idx)
        # Exit signal opens short; entry signal closes it
        exits = pd.Series([True, False, False, False, False], index=idx)
        entries = pd.Series([False, False, False, False, True], index=idx)

        portfolio = Portfolio(
            price=price,
            entries=entries,
            exits=exits,
            init_cash=10000,
            fees=0.0,
            slippage=0.0,
            market_hours_only=False,
            risk_config=RiskConfig(position_size_pct=0.5, use_stop_loss=False),
            allow_shorts=True,
        )

        covers = [t for t in portfolio.trades if t.get('type') == 'cover_short']
        assert len(covers) == 1
        rec = covers[0]

        # Long-convention signing: adverse for short = price up = mae negative
        assert rec['mae_pct'] == pytest.approx(-0.05, abs=1e-9)
        assert rec['mfe_pct'] == pytest.approx(0.10, abs=1e-9)
        assert rec['mae_time'] == idx[1]
        assert rec['mfe_time'] == idx[2]
        assert rec['hit_stop'] is False
        assert rec['hit_target'] is False


class TestHitStopHitTarget:
    """hit_stop / hit_target are derived from MAE/MFE vs configured thresholds."""

    def test_hit_stop_true_when_stop_fires(self):
        # 10% stop. Price -11% on bar 3 forces stop.
        idx = pd.date_range('2023-01-02', periods=5, freq='D')
        price = pd.Series([100.0, 99.0, 98.0, 89.0, 80.0], index=idx)
        entries = pd.Series([True, False, False, False, False], index=idx)
        exits = pd.Series([False, False, False, False, False], index=idx)

        portfolio = Portfolio(
            price=price,
            entries=entries,
            exits=exits,
            init_cash=10000,
            fees=0.0,
            slippage=0.0,
            market_hours_only=False,
            risk_config=RiskConfig(
                position_size_pct=1.0,
                use_stop_loss=True,
                stop_loss_type='percentage',
                stop_loss_pct=0.10,
            ),
        )

        rec = _exit_records(portfolio)[0]
        assert rec['exit_reason'] == 'stop_loss'
        assert rec['hit_stop'] is True
        assert rec['hit_target'] is False
        # MAE was -11% on bar 3 (the bar that fired the stop)
        assert rec['mae_pct'] == pytest.approx(-0.11, abs=1e-9)
        assert rec['mae_time'] == idx[3]

    def test_hit_target_true_when_target_fires(self):
        # 10% take-profit. Price +15% on bar 2 forces target hit.
        idx = pd.date_range('2023-01-02', periods=5, freq='D')
        price = pd.Series([100.0, 105.0, 115.0, 116.0, 117.0], index=idx)
        entries = pd.Series([True, False, False, False, False], index=idx)
        exits = pd.Series([False, False, False, False, False], index=idx)

        portfolio = Portfolio(
            price=price,
            entries=entries,
            exits=exits,
            init_cash=10000,
            fees=0.0,
            slippage=0.0,
            market_hours_only=False,
            risk_config=RiskConfig(
                position_size_pct=1.0,
                use_stop_loss=True,
                stop_loss_type='profit_target',
                stop_loss_pct=0.20,  # wide stop so target fires first
                take_profit_pct=0.10,
            ),
        )

        rec = _exit_records(portfolio)[0]
        assert rec['exit_reason'] == 'profit_target'
        assert rec['hit_target'] is True
        assert rec['hit_stop'] is False
        assert rec['mfe_pct'] == pytest.approx(0.15, abs=1e-9)
        assert rec['mfe_time'] == idx[2]

    def test_hit_flags_false_when_no_stop_no_target_configured(self):
        # No stop and no target configured -- both flags False regardless of MAE/MFE.
        idx = pd.date_range('2023-01-02', periods=4, freq='D')
        price = pd.Series([100.0, 80.0, 130.0, 110.0], index=idx)
        entries = pd.Series([True, False, False, False], index=idx)
        exits = pd.Series([False, False, False, True], index=idx)

        portfolio = Portfolio(
            price=price,
            entries=entries,
            exits=exits,
            init_cash=10000,
            fees=0.0,
            slippage=0.0,
            market_hours_only=False,
            risk_config=RiskConfig(position_size_pct=1.0, use_stop_loss=False),
        )

        rec = _exit_records(portfolio)[0]
        assert rec['hit_stop'] is False
        assert rec['hit_target'] is False
        assert rec['mae_pct'] == pytest.approx(-0.20, abs=1e-9)
        assert rec['mfe_pct'] == pytest.approx(0.30, abs=1e-9)


class TestEntryRecordsAreNotAnnotated:
    """Entry-side trade records do not carry MAE/MFE (they describe entries only)."""

    def test_entry_record_has_no_mae_mfe_keys(self):
        idx = pd.date_range('2023-01-02', periods=3, freq='D')
        price = pd.Series([100.0, 105.0, 110.0], index=idx)
        entries = pd.Series([True, False, False], index=idx)
        exits = pd.Series([False, False, True], index=idx)

        portfolio = Portfolio(
            price=price,
            entries=entries,
            exits=exits,
            init_cash=10000,
            fees=0.0,
            slippage=0.0,
            market_hours_only=False,
            risk_config=RiskConfig(position_size_pct=1.0, use_stop_loss=False),
        )

        entry_records = [t for t in portfolio.trades if t.get('type') == 'entry']
        assert len(entry_records) == 1
        for forbidden in ('mae_pct', 'mfe_pct', 'mae_time', 'mfe_time', 'hit_stop', 'hit_target'):
            assert forbidden not in entry_records[0]


class TestMultipleTradesIsolateMAEMFE:
    """Each round-trip's MAE/MFE is independent of the previous one."""

    def test_two_consecutive_long_trades(self):
        # Trade 1: entry at 100, dips to 95, exits at 102 (bar 3)
        # Trade 2: entry at 102 (bar 4), spikes to 115, exits at 110 (bar 7)
        idx = pd.date_range('2023-01-02', periods=8, freq='D')
        price = pd.Series(
            [100.0, 95.0, 99.0, 102.0, 102.0, 115.0, 113.0, 110.0],
            index=idx,
        )
        entries = pd.Series(
            [True,  False, False, False, True,  False, False, False],
            index=idx,
        )
        exits = pd.Series(
            [False, False, False, True,  False, False, False, True],
            index=idx,
        )

        portfolio = Portfolio(
            price=price,
            entries=entries,
            exits=exits,
            init_cash=10000,
            fees=0.0,
            slippage=0.0,
            market_hours_only=False,
            risk_config=RiskConfig(position_size_pct=1.0, use_stop_loss=False),
        )

        exits_log = _exit_records(portfolio)
        assert len(exits_log) == 2

        # First trade: MAE = -5% at bar 1, MFE = +2% at bar 3 (exit bar is included)
        first = exits_log[0]
        assert first['mae_pct'] == pytest.approx(-0.05, abs=1e-9)
        assert first['mae_time'] == idx[1]
        assert first['mfe_pct'] == pytest.approx(0.02, abs=1e-9)

        # Second trade: opened at 102 (bar 4), peak 115 at bar 5 -> MFE = 13/102
        second = exits_log[1]
        assert second['mae_pct'] == pytest.approx(0.0, abs=1e-9)
        assert second['mfe_pct'] == pytest.approx(13.0 / 102.0, abs=1e-9)
        assert second['mfe_time'] == idx[5]
