"""Tests for ContinuousContractDataLoader."""

from src.data.continuous_contract_loader import ContinuousContractDataLoader


def test_class_importable():
    loader = ContinuousContractDataLoader()
    assert loader is not None
