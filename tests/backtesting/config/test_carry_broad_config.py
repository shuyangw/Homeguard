# tests/backtesting/config/test_carry_broad_config.py
from pathlib import Path
import yaml
from src.data.futures.asset_class import asset_class_for

CONFIG = Path("config/backtesting/carry_broad.yaml")


def test_carry_broad_config():
    cfg = yaml.safe_load(CONFIG.read_text())
    assert cfg["asset_class"] == "futures"
    assert cfg["strategy"]["name"] == "FuturesCarry"
    u = cfg["strategy"]["universe"]
    assert len(u) == 33
    for r in u:
        assert asset_class_for(r)  # every root is carry-mappable
    assert cfg["backtest"]["initial_capital"] == 10_000_000
    assert cfg["dates"]["start"] == "2010-06-07"
    assert cfg["dates"]["end"] == "2026-02-20"
