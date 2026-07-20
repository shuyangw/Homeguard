"""Documents CSCM fill-logging coverage (Task 13).

CSCM (Cross-Sectional Crypto Momentum) has NO bespoke backtest path. It runs
through the standard equity/crypto config-driven runner
(`src/backtest_runner.py::run_single_from_config`), which returns a
vectorbt/MultiAssetPortfolio and persists its fills via
`TradeLogger.export_trades_csv` (gated on `output.save_trades`, default True;
`config/backtesting/cscm_baseline.yaml` sets it true).

Per the fill-logging-everywhere design (Section: "Top-level verdict artifact
stays plain") the single-run config path is deliberately left on TradeLogger.
CSCM is therefore ALREADY COVERED and needs no FillSink wiring. These tests
pin that coverage and prove CSCM's fill shape routes correctly through the sink.
"""

from pathlib import Path

import pandas as pd

from src.backtesting.engine.fill_sink import FillSink
from src.strategies.registry import _STRATEGY_REGISTRY


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_cscm_registered_on_standard_config_runner():
    # CSCM resolves through the shared registry the config runner uses -> it is
    # not a bespoke entry point but the standard equity/crypto single-run path.
    assert "CSCMStrategy" in _STRATEGY_REGISTRY
    module_path, class_name = _STRATEGY_REGISTRY["CSCMStrategy"]
    assert module_path == "src.strategies.advanced.cscm_strategy"
    assert class_name == "CSCMStrategy"


def test_cscm_baseline_config_routes_single_mode_with_trade_logging():
    # The CSCM backtest config drives the standard single-mode runner and keeps
    # trade persistence enabled -- the already-covered path.
    cfg = (REPO_ROOT / "config" / "backtesting" / "cscm_baseline.yaml").read_text()
    assert "CSCMStrategy" in cfg
    assert "mode: single" in cfg
    assert "save_trades: true" in cfg


def test_single_run_config_path_persists_fills_via_tradelogger():
    # The covered persistence: run_single_from_config exports fills through
    # TradeLogger when output.save_trades is set. Pin the source so a future
    # refactor that drops the export is caught here.
    src = (REPO_ROOT / "src" / "backtest_runner.py").read_text()
    assert "def run_single_from_config" in src
    assert "config.output.save_trades" in src
    assert "TradeLogger.export_trades_csv" in src


def _cscm_shaped_portfolio():
    # CSCM is multi-symbol -> the engine returns a MultiAssetPortfolio whose
    # `trades` is a round-trip DataFrame (entry_timestamp/exit_timestamp cols).
    # This is the exact shape TradeLogger (and thus FillSink.write_portfolio)
    # consumes; build a minimal synthetic instance.
    class FakeMultiAssetPortfolio:
        trades = pd.DataFrame(
            {
                "symbol": ["BTC/USD", "ETH/USD"],
                "entry_timestamp": ["2021-01-03", "2021-01-03"],
                "entry_price": [30000.0, 800.0],
                "exit_timestamp": ["2021-01-10", "2021-01-10"],
                "exit_price": [33000.0, 820.0],
                "shares": [0.5, 10.0],
                "pnl": [1500.0, 200.0],
                "pnl_pct": [0.1, 0.025],
                "exit_reason": ["rebalance", "rebalance"],
            }
        )

    return FakeMultiAssetPortfolio()


def test_cscm_vectorbt_fill_shape_routes_through_sink(tmp_path):
    sink = FillSink("CSCMStrategy", "20260720T000000Z_cscm01", {"kind": "verdict"},
                    root=tmp_path)
    path = sink.write_portfolio(_cscm_shaped_portfolio(), window=1, cfg_hash="cscm01")
    assert path.name == "w01_cscm01_trades.csv.gz"

    manifest_path = sink.finalize(oos_windows=[1], oos_cfg_hash="cscm01")
    assert manifest_path.name == "manifest.csv"
    assert manifest_path.exists()

    manifest = pd.read_csv(manifest_path)
    trade_rows = manifest[manifest["file"] == "w01_cscm01_trades.csv.gz"].iloc[0]
    assert trade_rows["kind"] == "trades"
    # 2 round-trips -> 2 buy rows + 2 sell rows.
    assert trade_rows["row_count"] == 4
