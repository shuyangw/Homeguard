"""
Tests for config loader (YAML loading, inheritance, overrides).
"""

import pytest
import tempfile
from pathlib import Path

from src.config.loader import (
    load_yaml,
    merge_dicts,
    apply_overrides,
    get_nested,
    load_config_dict,
    load_config,
)
from src.config.schema import BacktestConfig, BacktestMode


class TestMergeDicts:
    """Tests for merge_dicts function."""

    def test_simple_merge(self):
        """Test simple dictionary merge."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = merge_dicts(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        """Test nested dictionary merge."""
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 5, "z": 6}}
        result = merge_dicts(base, override)
        assert result == {"a": {"x": 1, "y": 5, "z": 6}, "b": 3}

    def test_does_not_modify_base(self):
        """Test that base dict is not modified."""
        base = {"a": 1}
        override = {"a": 2}
        merge_dicts(base, override)
        assert base == {"a": 1}


class TestApplyOverrides:
    """Tests for apply_overrides function."""

    def test_simple_override(self):
        """Test simple dot notation override."""
        config = {"backtest": {"initial_capital": 100000}}
        overrides = {"backtest.initial_capital": 50000}
        result = apply_overrides(config, overrides)
        assert result["backtest"]["initial_capital"] == 50000

    def test_nested_override(self):
        """Test deeply nested override."""
        config = {"a": {"b": {"c": 1}}}
        overrides = {"a.b.c": 99}
        result = apply_overrides(config, overrides)
        assert result["a"]["b"]["c"] == 99

    def test_creates_missing_keys(self):
        """Test that missing intermediate keys are created."""
        config = {}
        overrides = {"a.b.c": 123}
        result = apply_overrides(config, overrides)
        assert result["a"]["b"]["c"] == 123

    def test_does_not_modify_original(self):
        """Test that original config is not modified."""
        config = {"a": 1}
        overrides = {"a": 2}
        apply_overrides(config, overrides)
        assert config == {"a": 1}


class TestGetNested:
    """Tests for get_nested function."""

    def test_get_existing_key(self):
        """Test getting existing nested key."""
        config = {"a": {"b": {"c": 42}}}
        assert get_nested(config, "a.b.c") == 42

    def test_get_missing_key_default(self):
        """Test getting missing key returns default."""
        config = {"a": 1}
        assert get_nested(config, "b.c.d", default="missing") == "missing"

    def test_get_shallow_key(self):
        """Test getting shallow key."""
        config = {"a": 123}
        assert get_nested(config, "a") == 123


class TestLoadYaml:
    """Tests for load_yaml function."""

    def test_load_valid_yaml(self):
        """Test loading valid YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("key: value\nnested:\n  a: 1\n  b: 2\n")
            f.flush()

            result = load_yaml(Path(f.name))
            assert result == {"key": "value", "nested": {"a": 1, "b": 2}}

    def test_load_missing_file(self):
        """Test loading missing file raises error."""
        with pytest.raises(FileNotFoundError):
            load_yaml(Path("/nonexistent/file.yaml"))

    def test_load_empty_file(self):
        """Test loading empty file returns empty dict."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
            f.flush()

            result = load_yaml(Path(f.name))
            assert result == {}


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_valid_config(self):
        """Test loading valid config file."""
        yaml_content = """
strategy:
  name: MovingAverageCrossover
  parameters:
    fast_window: 10
    slow_window: 50

symbols:
  list: [SPY, QQQ]

dates:
  start: "2023-01-01"
  end: "2024-01-01"

backtest:
  initial_capital: 50000
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = load_config(f.name)

            assert isinstance(config, BacktestConfig)
            assert config.strategy.name == "MovingAverageCrossover"
            assert config.strategy.parameters["fast_window"] == 10
            assert config.symbols.list == ["SPY", "QQQ"]
            assert config.backtest.initial_capital == 50000

    def test_load_config_with_overrides(self):
        """Test loading config with CLI overrides."""
        yaml_content = """
strategy:
  name: MeanReversion

symbols:
  list: [AAPL]

dates:
  start: "2023-01-01"
  end: "2024-01-01"

backtest:
  initial_capital: 100000
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = load_config(
                f.name,
                overrides={"backtest.initial_capital": 25000}
            )

            assert config.backtest.initial_capital == 25000

    def test_load_config_applies_defaults(self):
        """Test that defaults are applied for missing values."""
        yaml_content = """
strategy:
  name: MeanReversion

symbols:
  list: [SPY]

dates:
  start: "2023-01-01"
  end: "2024-01-01"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = load_config(f.name)

            # Check defaults were applied
            assert config.backtest.initial_capital == 100000.0
            assert config.backtest.fees == 0.001
            assert config.risk.position_size_pct == 0.10


class TestLoadConfigDict:
    """Tests for load_config_dict function."""

    def test_load_config_dict(self):
        """Test loading config as dict (before validation)."""
        yaml_content = """
strategy:
  name: BreakoutStrategy

symbols:
  list: [NVDA]

dates:
  start: "2023-01-01"
  end: "2024-01-01"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config_dict = load_config_dict(f.name)

            assert isinstance(config_dict, dict)
            assert config_dict["strategy"]["name"] == "BreakoutStrategy"
            # Defaults should be merged
            assert "backtest" in config_dict
            assert config_dict["backtest"]["initial_capital"] == 100000.0
