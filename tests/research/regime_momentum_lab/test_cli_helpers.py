"""Tests for run_momentum_variant CLI helpers (no market data needed)."""
import importlib.util
from pathlib import Path

import pytest

_CLI_PATH = Path('scripts/backtest_scripts/run_momentum_variant.py')


def _load_cli():
    spec = importlib.util.spec_from_file_location('run_momentum_variant', _CLI_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_daily_rebalance_is_allowed():
    cli = _load_cli()
    cli._validate_rebalance_frequency('daily')  # must not raise


@pytest.mark.parametrize('freq', ['weekly_friday', 'weekly_wednesday'])
def test_weekly_rebalance_raises_not_implemented(freq):
    cli = _load_cli()
    with pytest.raises(NotImplementedError) as exc:
        cli._validate_rebalance_frequency(freq)
    assert 'PR 6' in str(exc.value)


def test_format_turnover_line_shape():
    cli = _load_cli()
    line = cli._format_turnover_line('prod', 5.0, _turnover=0.1873)
    assert line.startswith('[turnover] prod @ 5.0bps:')
    assert 'avg_daily_turnover=0.1873' in line
    assert '18.73% of portfolio/day' in line
