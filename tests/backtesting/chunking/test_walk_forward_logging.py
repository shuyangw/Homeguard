import inspect

from src.backtesting.chunking.walk_forward import WalkForwardValidator


def test_validate_accepts_fill_sink_param():
    sig = inspect.signature(WalkForwardValidator.validate)
    assert "fill_sink" in sig.parameters


def test_validate_fill_sink_defaults_to_none():
    sig = inspect.signature(WalkForwardValidator.validate)
    assert sig.parameters["fill_sink"].default is None
