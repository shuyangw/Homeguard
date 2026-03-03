"""Tests for RAMP-CSP integration runner and config loading."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.strategies.options.csp.ramp_integration import (
    CSPBacktestRunner,
    load_csp_config,
)


class TestLoadCspConfig:
    def test_loads_yaml_config(self, tmp_path):
        """Create a temp YAML file, load it, verify values are parsed."""
        config_data = {
            "strategy": {
                "initial_capital": 50000,
                "max_csp_allocation": 0.20,
                "max_positions": 3,
            },
            "ramp": {
                "vix_threshold": 30.0,
                "spy_dd_threshold": -0.08,
            },
        }
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml.dump(config_data))

        loaded = load_csp_config(config_file)

        assert loaded["strategy"]["initial_capital"] == 50000
        assert loaded["strategy"]["max_csp_allocation"] == 0.20
        assert loaded["strategy"]["max_positions"] == 3
        assert loaded["ramp"]["vix_threshold"] == 30.0
        assert loaded["ramp"]["spy_dd_threshold"] == -0.08

    def test_loads_default_config(self):
        """Default config path should load the committed YAML."""
        loaded = load_csp_config()

        assert "strategy" in loaded
        assert "ramp" in loaded
        assert loaded["strategy"]["initial_capital"] == 100000
        assert loaded["strategy"]["max_positions"] == 5
        assert loaded["ramp"]["vix_threshold"] == 25.0


class TestCSPBacktestRunnerInit:
    def test_initialization(self):
        """CSPBacktestRunner() creates without error with default config."""
        runner = CSPBacktestRunner()

        assert runner is not None
        assert runner._config is not None
        assert runner._config["strategy"]["initial_capital"] == 100000

    def test_initialization_custom_config(self):
        """CSPBacktestRunner accepts a custom config dict."""
        custom = {
            "strategy": {
                "initial_capital": 50000,
                "max_csp_allocation": 0.20,
                "max_positions": 2,
                "profit_target_pct": 0.50,
                "loss_limit_multiple": 2.0,
                "min_dte_exit": 5,
                "slippage_pct": 0.01,
                "contract_fee": 0.02,
            },
            "contract_selection": {
                "target_delta_min": -0.35,
                "target_delta_max": -0.25,
                "min_dte": 21,
                "max_dte": 35,
                "min_open_interest": 100,
                "max_spread_pct": 0.15,
            },
            "ramp": {
                "vix_threshold": 30.0,
                "spy_dd_threshold": -0.08,
            },
        }
        runner = CSPBacktestRunner(config=custom)

        assert runner._config["strategy"]["initial_capital"] == 50000
        assert runner._config["ramp"]["vix_threshold"] == 30.0


class TestGetOptionsSymbols:
    def test_get_options_symbols(self):
        """Returns non-empty list if options data exists on disk."""
        runner = CSPBacktestRunner()
        symbols = runner._get_options_symbols()

        # This test relies on real options data on disk.
        # If data is not present, the list may be empty -- skip in that case.
        if not symbols:
            pytest.skip("No options data found on disk")

        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert all(isinstance(s, str) for s in symbols)
