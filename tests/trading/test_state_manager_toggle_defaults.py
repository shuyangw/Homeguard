"""A missing toggle file must fail closed: regenerate with every strategy off."""

from pathlib import Path

import yaml

from src.trading.state import strategy_state_manager
from src.trading.state.strategy_state_manager import StrategyStateManager

EXAMPLE_TOGGLE = Path(__file__).resolve().parents[2] / "config" / "trading" / "strategy_toggle.example.yaml"


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


def test_example_template_enables_no_strategy():
    template = yaml.safe_load(EXAMPLE_TOGGLE.read_text())

    assert template["strategies"]
    assert all(settings["enabled"] is False for settings in template["strategies"].values())


def test_example_template_pins_every_variant_with_ramp_on_v11():
    template = yaml.safe_load(EXAMPLE_TOGGLE.read_text())

    assert all("variant" in settings for settings in template["strategies"].values())
    assert template["strategies"]["ramp"]["variant"] == "v11"


def test_missing_toggle_error_points_at_a_safe_restore_and_the_variant(tmp_path, monkeypatch):
    messages = []
    monkeypatch.setattr(strategy_state_manager.logger, "error", messages.append)

    StrategyStateManager(state_file=tmp_path / "strategy_positions.json", toggle_file=tmp_path / "strategy_toggle.yaml")

    missing = [message for message in messages if "Toggle file missing" in message]
    assert missing
    assert "strategy_toggle.example.yaml" in missing[0]
    assert "variant" in missing[0]
