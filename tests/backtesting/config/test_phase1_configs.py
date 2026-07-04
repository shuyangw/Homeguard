from pathlib import Path
import yaml

def _load(name):
    return yaml.safe_load((Path("config/backtesting") / name).read_text())

def test_value_config():
    c = _load("value_broad.yaml")
    assert c["strategy"]["name"] == "FuturesValue"
    assert len(c["strategy"]["universe"]) == 33

def test_crypto_carry_config():
    c = _load("crypto_carry_broad.yaml")
    assert c["strategy"]["name"] == "FuturesCarry"
    assert c["strategy"]["universe"] == ["BTC", "ETH"]
    assert str(c["dates"]["start"]) == "2017-01-01"
