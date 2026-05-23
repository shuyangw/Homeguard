"""
TDD tests for RAMPLiveAdapter variant plumbing (Phase 2C).

Phase 2C wires a 'variant' field from config/trading/strategy_toggle.yaml
into RAMPLiveAdapter.__init__. The adapter exposes self.variant and rejects
values other than 'v01' (current production behavior) or 'v11' (Phase 4
Wave 1 backport, gated by Phase 2D).

This task does NOT change any rebalance logic -- only plumbing.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.trading.adapters.ramp_live_adapter import RAMPLiveAdapter


@pytest.fixture
def mock_broker():
    broker = MagicMock()
    broker.get_account.return_value = {
        'buying_power': 100000.0,
        'portfolio_value': 100000.0,
        'cash': 100000.0,
    }
    broker.get_positions.return_value = []
    broker.is_market_open.return_value = True
    return broker


def _build_adapter(mock_broker, **kwargs):
    """Construct adapter with the minimum mocks needed (mirrors position_dates test)."""
    with patch('src.trading.adapters.strategy_adapter.ExecutionEngine'):
        with patch('src.trading.adapters.strategy_adapter.PositionManager'):
            with patch('src.trading.adapters.ramp_live_adapter.StrategyStateManager') as mock_sm:
                with patch('src.trading.adapters.ramp_live_adapter.PortfolioHealthChecker'):
                    with patch('src.trading.adapters.ramp_live_adapter._load_cache_from_disk') as mock_cache:
                        mock_cache.return_value = None
                        mock_sm_instance = MagicMock()
                        mock_sm_instance.is_enabled.return_value = True
                        mock_sm_instance.is_shutdown_requested.return_value = False
                        mock_sm_instance.acquire_execution_lock.return_value = True
                        mock_sm_instance.get_positions.return_value = {}
                        mock_sm_instance.get_runner_session_state.return_value = {
                            'peak_equity_usd': None,
                            'session_open_equity_usd': None,
                            'session_open_date': None,
                            'position_open_dates': None,
                        }
                        mock_sm.return_value = mock_sm_instance

                        defaults = dict(
                            broker=mock_broker,
                            symbols=['AAPL', 'MSFT', 'GOOGL'],
                            max_capital_allocation=1.0,
                            broker_name='alpaca',
                        )
                        defaults.update(kwargs)
                        return RAMPLiveAdapter(**defaults)


class TestRAMPLiveAdapterVariant:
    """Constructor plumbing for the 'variant' kwarg."""

    def test_ramp_live_adapter_variant_defaults_to_v01(self, mock_broker):
        """RAMPLiveAdapter defaults variant to 'v01' when not specified."""
        adapter = _build_adapter(mock_broker)
        assert adapter.variant == 'v01'

    def test_ramp_live_adapter_accepts_v11_variant(self, mock_broker):
        """RAMPLiveAdapter accepts variant='v11' explicitly."""
        adapter = _build_adapter(mock_broker, variant='v11')
        assert adapter.variant == 'v11'

    def test_ramp_live_adapter_accepts_v01_variant_explicit(self, mock_broker):
        """RAMPLiveAdapter accepts variant='v01' explicitly."""
        adapter = _build_adapter(mock_broker, variant='v01')
        assert adapter.variant == 'v01'

    def test_ramp_live_adapter_rejects_unknown_variant(self, mock_broker):
        """Invalid variant value raises ValueError at construction time."""
        with pytest.raises(ValueError, match='variant'):
            _build_adapter(mock_broker, variant='v99')


class TestToggleYamlVariantField:
    """The strategy_toggle.yaml on disk must include a variant field per entry."""

    def _project_root(self) -> Path:
        # tests/trading/adapters/<file>.py -> project root is 3 parents up
        return Path(__file__).resolve().parents[3]

    def test_toggle_yaml_includes_variant_field_with_v01_default(self):
        """strategy_toggle.yaml has variant: v01 (or v11) for each strategy."""
        path = self._project_root() / 'config' / 'trading' / 'strategy_toggle.yaml'
        assert path.exists(), f'expected toggle yaml at {path}'
        data = yaml.safe_load(path.read_text()) or {}
        strategies = data.get('strategies', {})
        assert strategies, 'strategy_toggle.yaml has no strategies entry'
        for name, cfg in strategies.items():
            assert 'variant' in cfg, f'{name} missing variant field'
            assert cfg['variant'] in ('v01', 'v11'), (
                f'{name} has invalid variant {cfg["variant"]!r}'
            )

    def test_toggle_yaml_example_includes_variant_field(self):
        """The committed example file must mirror the variant schema."""
        path = self._project_root() / 'config' / 'trading' / 'strategy_toggle.example.yaml'
        if not path.exists():
            pytest.skip('example toggle file not present')
        data = yaml.safe_load(path.read_text()) or {}
        strategies = data.get('strategies', {})
        assert strategies, 'strategy_toggle.example.yaml has no strategies entry'
        for name, cfg in strategies.items():
            assert 'variant' in cfg, f'{name} missing variant field in example'
            assert cfg['variant'] in ('v01', 'v11'), (
                f'{name} has invalid variant {cfg["variant"]!r} in example'
            )
