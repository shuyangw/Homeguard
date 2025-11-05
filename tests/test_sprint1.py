"""
Unit tests for Sprint 1 features:
- Configuration Presets
- Symbol Lists Management
- Quick Re-run (Last Run)
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gui.utils.config_manager import ConfigManager
from datetime import datetime


class TestConfigManager:
    """Test configuration manager functionality."""

    def setup_method(self):
        """Create a temporary config manager for testing."""
        test_dir = Path(__file__).parent / "test_gui_config"
        test_dir.mkdir(exist_ok=True)
        self.config_manager = ConfigManager(config_dir=test_dir)

    def teardown_method(self):
        """Clean up test files."""
        import shutil
        test_dir = Path(__file__).parent / "test_gui_config"
        if test_dir.exists():
            shutil.rmtree(test_dir)

    def test_save_and_load_preset(self):
        """Test saving and loading a configuration preset."""
        # Create a test configuration
        test_config = {
            'strategy': 'MovingAverageCrossover',
            'symbols': ['AAPL', 'MSFT'],
            'start_date': '2023-01-01',
            'end_date': '2024-01-01',
            'workers': 4
        }

        # Save preset
        self.config_manager.save_preset('TestPreset', test_config)

        # Load preset
        loaded_config = self.config_manager.load_preset('TestPreset')

        # Verify
        assert loaded_config is not None
        assert loaded_config['strategy'] == 'MovingAverageCrossover'
        assert loaded_config['symbols'] == ['AAPL', 'MSFT']
        assert loaded_config['workers'] == 4

    def test_get_preset_names(self):
        """Test getting list of preset names."""
        # Save multiple presets
        self.config_manager.save_preset('Preset1', {'test': 1})
        self.config_manager.save_preset('Preset2', {'test': 2})
        self.config_manager.save_preset('Preset3', {'test': 3})

        # Get names
        names = self.config_manager.get_preset_names()

        # Verify
        assert len(names) == 3
        assert 'Preset1' in names
        assert 'Preset2' in names
        assert 'Preset3' in names
        assert names == sorted(names)  # Should be alphabetically sorted

    def test_delete_preset(self):
        """Test deleting a preset."""
        # Save preset
        self.config_manager.save_preset('DeleteMe', {'test': 1})
        assert 'DeleteMe' in self.config_manager.get_preset_names()

        # Delete preset
        result = self.config_manager.delete_preset('DeleteMe')
        assert result is True
        assert 'DeleteMe' not in self.config_manager.get_preset_names()

        # Try deleting non-existent preset
        result = self.config_manager.delete_preset('DoesNotExist')
        assert result is False

    def test_save_and_load_symbol_list(self):
        """Test saving and loading symbol lists."""
        # Save symbol list
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
        self.config_manager.save_symbol_list('FAANG', symbols)

        # Load symbol list
        loaded_symbols = self.config_manager.load_symbol_list('FAANG')

        # Verify
        assert loaded_symbols is not None
        assert loaded_symbols == symbols
        assert len(loaded_symbols) == 5

    def test_get_symbol_list_info(self):
        """Test getting symbol list info with counts."""
        # Save multiple lists
        self.config_manager.save_symbol_list('Tech', ['AAPL', 'MSFT', 'GOOGL'])
        self.config_manager.save_symbol_list('Finance', ['JPM', 'BAC'])

        # Get info
        info = self.config_manager.get_symbol_list_info()

        # Verify
        assert info['Tech'] == 3
        assert info['Finance'] == 2

    def test_delete_symbol_list(self):
        """Test deleting a symbol list."""
        # Save list
        self.config_manager.save_symbol_list('DeleteMe', ['AAPL'])
        assert 'DeleteMe' in self.config_manager.get_symbol_list_names()

        # Delete list
        result = self.config_manager.delete_symbol_list('DeleteMe')
        assert result is True
        assert 'DeleteMe' not in self.config_manager.get_symbol_list_names()

    def test_save_and_load_last_run(self):
        """Test saving and loading last run configuration."""
        # Create a test configuration
        test_config = {
            'strategy': 'BreakoutStrategy',
            'symbols': ['TSLA', 'NVDA'],
            'start_date': '2023-06-01',
            'end_date': '2023-12-31'
        }

        # Save as last run
        self.config_manager.save_last_run(test_config)

        # Check if has last run
        assert self.config_manager.has_last_run() is True

        # Load last run
        loaded_config = self.config_manager.load_last_run()

        # Verify
        assert loaded_config is not None
        assert loaded_config['strategy'] == 'BreakoutStrategy'
        assert loaded_config['symbols'] == ['TSLA', 'NVDA']

    def test_load_nonexistent_preset(self):
        """Test loading a preset that doesn't exist."""
        result = self.config_manager.load_preset('DoesNotExist')
        assert result is None

    def test_load_nonexistent_symbol_list(self):
        """Test loading a symbol list that doesn't exist."""
        result = self.config_manager.load_symbol_list('DoesNotExist')
        assert result is None

    def test_preset_metadata(self):
        """Test that preset metadata is saved (created/modified timestamps)."""
        # Save preset
        self.config_manager.save_preset('MetaTest', {'test': 1})

        # Load raw preset data
        presets = self.config_manager._load_presets()
        assert 'MetaTest' in presets
        assert 'created' in presets['MetaTest']
        assert 'modified' in presets['MetaTest']

        # Verify timestamps are valid ISO format
        created = datetime.fromisoformat(presets['MetaTest']['created'])
        modified = datetime.fromisoformat(presets['MetaTest']['modified'])
        assert isinstance(created, datetime)
        assert isinstance(modified, datetime)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
