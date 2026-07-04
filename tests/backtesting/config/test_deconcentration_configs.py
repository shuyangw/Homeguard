"""Test that the 3 carry de-concentration trial configs are correct."""
import yaml
from pathlib import Path


def load_config(config_name: str) -> dict:
    """Load a YAML config from config/backtesting/."""
    config_path = Path(__file__).parent.parent.parent.parent / "config" / "backtesting" / f"{config_name}.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def test_carry_xs_broad_config():
    """Test carry_xs_broad.yaml: FuturesCarryXS, no idm, 33 roots."""
    cfg = load_config("carry_xs_broad")

    assert cfg["strategy"]["name"] == "FuturesCarryXS", "Strategy name should be FuturesCarryXS"
    assert "idm" not in cfg["backtest"], "idm should not be present in backtest block"
    assert cfg["backtest"].get("idm", False) is False, "idm should not exist or be False"
    assert len(cfg["strategy"]["universe"]) == 33, "Universe should have 33 roots"


def test_carry_idm_broad_config():
    """Test carry_idm_broad.yaml: FuturesCarry, idm=true, 33 roots."""
    cfg = load_config("carry_idm_broad")

    assert cfg["strategy"]["name"] == "FuturesCarry", "Strategy name should be FuturesCarry"
    assert cfg["backtest"]["idm"] is True, "idm should be True in backtest block"
    assert len(cfg["strategy"]["universe"]) == 33, "Universe should have 33 roots"


def test_carry_xs_idm_broad_config():
    """Test carry_xs_idm_broad.yaml: FuturesCarryXS, idm=true, 33 roots."""
    cfg = load_config("carry_xs_idm_broad")

    assert cfg["strategy"]["name"] == "FuturesCarryXS", "Strategy name should be FuturesCarryXS"
    assert cfg["backtest"]["idm"] is True, "idm should be True in backtest block"
    assert len(cfg["strategy"]["universe"]) == 33, "Universe should have 33 roots"
