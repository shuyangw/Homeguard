"""
Regression tests for adapter factories in scripts/trading/run_live_paper_trading.py.

Catches the class of bug where an adapter's __init__ signature adds a required
kwarg (like `broker_name`) but a caller is not updated to pass it. On 2026-04-15
`OMRLiveAdapter.__init__` and `RAMPLiveAdapter.__init__` both added a required
keyword-only `broker_name` argument. The factory wrappers thread it through; the
pre-migration multi-runner did not and crashed every 30 seconds for 6 days.

These tests exercise the factory paths that the production
`homeguard-multi.service` unit now invokes via
`run_live_paper_trading.py --strategy multi`.
"""

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

from tests.trading.mock_broker import MockBroker


def _load_runner_module():
    """Import scripts/trading/run_live_paper_trading.py as a module."""
    project_root = Path(__file__).resolve().parents[2]
    script_path = project_root / "scripts" / "trading" / "run_live_paper_trading.py"
    spec = importlib.util.spec_from_file_location(
        "run_live_paper_trading_under_test", script_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return _load_runner_module()


@pytest.fixture
def broker():
    return MockBroker(initial_cash=100_000.0)


def test_create_ramp_adapter_threads_broker_name(runner, broker):
    adapter = runner.create_ramp_adapter(broker, broker_name="ibkr")
    assert adapter.broker_name == "ibkr"


def test_create_omr_adapter_threads_broker_name(runner, broker):
    omr_config = runner.load_omr_config()
    adapter = runner.create_omr_adapter(
        broker,
        omr_config=omr_config,
        broker_name="ibkr",
    )
    assert adapter.broker_name == "ibkr"


def test_create_mp_adapter_threads_broker_name(runner, broker):
    adapter = runner.create_mp_adapter(broker, broker_name="ibkr")
    assert adapter.broker_name == "ibkr"


def test_factories_reject_missing_broker_name(runner, broker):
    """Every factory must require broker_name. Guards against silent regressions."""
    with pytest.raises(TypeError, match="broker_name"):
        runner.create_ramp_adapter(broker)
    with pytest.raises(TypeError, match="broker_name"):
        runner.create_mp_adapter(broker)
