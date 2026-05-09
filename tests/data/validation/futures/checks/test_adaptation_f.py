"""Tests for Adaptation F gating checks (lazy-loaded)."""
from src.data.validation.futures.checks import adaptation_f


def test_adaptation_f_module_exports():
    assert hasattr(adaptation_f, "ChainDensityCheck")
    assert hasattr(adaptation_f, "IvRankComputabilityCheck")
    assert hasattr(adaptation_f, "IvSmileConsistencyCheck")


def test_adaptation_f_checks_dont_auto_register():
    """Ensure these don't pollute the default registry."""
    from src.data.validation.core.base import _registry
    _registry.clear()
    # Importing the module should NOT register
    import importlib
    importlib.reload(adaptation_f)
    found = _registry.get(domain="futures", layer=4, name_match="chain_density")
    # They register only when explicitly added via register_check
    assert all("adaptation_f" not in c.name for c in found)
