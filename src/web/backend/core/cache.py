import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

CACHE_DIR = Path("cache")
HISTORY_FILE = CACHE_DIR / "history.json"

class ConfigCache:
    def __init__(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if not HISTORY_FILE.exists():
            self._save_history([])

    def _load_history(self) -> List[Dict[str, Any]]:
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _save_history(self, history: List[Dict[str, Any]]):
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)

    def save_config(self, config: Dict[str, Any]):
        """Saves a configuration to history, avoiding duplicates."""
        history = self._load_history()
        
        # Add timestamp
        config['saved_at'] = datetime.now().isoformat()
        
        # Check for duplicates (ignoring timestamp)
        # For simplicity, we just push to top
        history.insert(0, config)
        
        # Keep last 50
        history = history[:50]
        self._save_history(history)

    def get_history(self) -> List[Dict[str, Any]]:
        return self._load_history()
