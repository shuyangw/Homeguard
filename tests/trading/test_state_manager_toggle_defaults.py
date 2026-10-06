"""A missing toggle file must fail closed: regenerate with every strategy off."""

import yaml

from src.trading.state.strategy_state_manager import StrategyStateManager


def test_missing_toggle_file_regenerates_with_every_strategy_disabled(tmp_path):
    toggle_file = tmp_path / "strategy_toggle.yaml"
    manager = StrategyStateManager(state_file=tmp_path / "strategy_positions.json", toggle_file=toggle_file)

    assert manager.get_enabled_strategies() == []
    for strategy in ("omr", "mp", "ramp", "cscm"):
        assert manager.is_enabled(strategy) is False


def test_missing_toggle_file_is_rewritten_to_disk_disabled(tmp_path):
    toggle_file = tmp_path / "strategy_toggle.yaml"
    StrategyStateManager(state_file=tmp_path / "strategy_positions.json", toggle_file=toggle_file)

    written = yaml.safe_load(toggle_file.read_text())
    assert set(written["strategies"]) == {"omr", "mp", "ramp", "cscm"}
    assert all(settings["enabled"] is False for settings in written["strategies"].values())
    assert all(settings["shutdown_requested"] is False for settings in written["strategies"].values())


def test_existing_toggle_file_is_left_alone(tmp_path):
    toggle_file = tmp_path / "strategy_toggle.yaml"
    toggle_file.write_text("strategies:\n  ramp:\n    enabled: true\n")
    manager = StrategyStateManager(state_file=tmp_path / "strategy_positions.json", toggle_file=toggle_file)

    assert manager.get_enabled_strategies() == ["ramp"]
