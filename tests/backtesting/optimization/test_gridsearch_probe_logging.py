import inspect

from src.backtesting.optimization import GridSearchOptimizer


def test_optimize_accepts_fill_sink_param():
    sig = inspect.signature(GridSearchOptimizer.optimize)
    assert "fill_sink" in sig.parameters
    assert sig.parameters["fill_sink"].default is None


def test_optimize_accepts_base_window_param():
    sig = inspect.signature(GridSearchOptimizer.optimize)
    assert "base_window" in sig.parameters
    assert sig.parameters["base_window"].default == 0
