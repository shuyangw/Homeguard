"""Tests for DataAcquisitionManager orchestrator."""

import pytest

from src.data.acquisition.manager import DataAcquisitionManager


class TestListSources:
    def test_list_sources(self):
        manager = DataAcquisitionManager()
        sources = manager.list_sources()
        assert "equities" in sources
        assert "crypto" in sources
        assert "futures" in sources
        assert "news" in sources

    def test_unknown_source_raises(self):
        manager = DataAcquisitionManager()
        with pytest.raises(ValueError, match="Unknown source"):
            manager.download(source="nonexistent", symbols=["X"])


class TestPluginRegistryLazyLoading:
    def test_importing_manager_does_not_import_databento(self):
        import sys

        mods_before = set(sys.modules.keys())
        from src.data.acquisition.manager import DataAcquisitionManager  # noqa: F811

        mods_after = set(sys.modules.keys())
        new_mods = mods_after - mods_before
        assert "databento" not in new_mods
