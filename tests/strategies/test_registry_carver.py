from src.strategies.registry import get_strategy_class
from src.strategies.advanced.carver_momentum_strategy import CarverMomentumStrategy


def test_carver_registered_by_class_name():
    assert get_strategy_class("CarverMomentum") is CarverMomentumStrategy


def test_carver_registered_by_aliases():
    assert get_strategy_class("Carver") is CarverMomentumStrategy
    assert get_strategy_class("Carver TSMOM") is CarverMomentumStrategy
    assert get_strategy_class("Carver Momentum") is CarverMomentumStrategy
