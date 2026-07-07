import yaml
from pathlib import Path
from src.backtesting.utils.pre_registration import validate_pre_registration


def test_curve_slope_config_valid():
    cfg = yaml.safe_load(Path("config/backtesting/curve_slope_xs.yaml").read_text())
    validate_pre_registration(cfg)
    assert cfg["strategy"]["name"] == "FuturesCarryXS"
    # commodity-only universe (no equity/fx/bond roots)
    assert set(cfg["strategy"]["universe"]) <= {
        "CL", "BZ", "NG", "HO", "RB", "GC", "SI", "HG", "PL",
        "ZC", "ZW", "ZS", "ZL", "ZM", "LE", "HE"}
