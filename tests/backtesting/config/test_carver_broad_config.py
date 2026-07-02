from pathlib import Path
import yaml
from src.data.futures.contract_specs import SPECS

CONFIG = Path("config/backtesting/carver_tsmom_broad.yaml")
EXPECTED = {
    "ES", "NQ", "YM", "ZT", "ZF", "ZN", "TN", "ZB", "UB",
    "6E", "6J", "6B", "6A", "6C", "6S", "6M", "6N",
    "CL", "BZ", "NG", "HO", "RB", "GC", "SI", "HG", "PL",
    "ZC", "ZW", "ZS", "ZL", "ZM", "LE", "HE",
}

def test_broad_config_shape_and_roots():
    cfg = yaml.safe_load(CONFIG.read_text())
    assert cfg["asset_class"] == "futures"
    universe = set(cfg["strategy"]["universe"])
    assert len(cfg["strategy"]["universe"]) == 33
    assert universe == EXPECTED
    assert universe <= set(SPECS.keys())  # no typos; every root is speced
    assert cfg["backtest"]["initial_capital"] == 10_000_000
    assert cfg["backtest"]["vol_target_per_instrument"] == 0.20
    assert cfg["backtest"]["rebalance"] == "weekly"
    assert cfg["backtest"]["cost_mult"] == 1.0
    assert cfg["dates"]["start"] == "2010-06-07"
    assert cfg["dates"]["end"] == "2026-02-20"
