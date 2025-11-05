# GUI Folder Structure & Organization

**Date:** 2025-01-01
**Status:** ✅ Reorganized & Cleaned

---

## Overview

The GUI code has been reorganized into a self-contained, well-structured Python package within `src/gui/`. All GUI-related code, documentation, and tests are now centralized in one location.

## New Structure

```
src/gui/
├── __init__.py              # Package initialization with exports
├── __main__.py              # Entry point for: python -m gui
├── app.py                   # Main application & navigation logic
├── README.md                # GUI developer documentation
│
├── docs/                    # GUI-specific documentation
│   ├── USER_GUIDE.md        # Complete user manual (465 lines)
│   ├── IMPLEMENTATION_PLAN.md  # Original design doc
│   ├── PHASE2_COMPLETE.md   # Implementation summary
│   └── FOLDER_STRUCTURE.md  # This file
│
├── views/                   # UI view components
│   ├── __init__.py          # View exports
│   ├── setup_view.py        # Configuration screen (390 lines)
│   ├── execution_view.py    # Progress monitoring (260 lines)
│   └── results_view.py      # Results display (315 lines)
│
├── workers/                 # Background processing
│   ├── __init__.py          # Worker exports
│   └── gui_controller.py    # Backtest coordinator (~400 lines)
│
├── utils/                   # Helper utilities
│   ├── __init__.py          # Utility exports
│   └── strategy_utils.py    # Strategy introspection (140 lines)
│
├── components/              # Reusable UI components (future)
│   └── __init__.py
│
└── tests/                   # GUI tests
    ├── __init__.py
    └── test_imports.py      # Import validation test
```

## Launch Options

### Option 1: Simple Launcher (Recommended)

```bash
# From project root
python run_gui.py
```

Uses minimal launcher script at project root that adds `src/` to path and runs the GUI.

### Option 2: As Python Module

```bash
# From src/ directory
cd src
python -m gui
```

Uses `__main__.py` entry point. Cleaner but requires cd into src/.

### Option 3: Direct Execution

```bash
# From project root
python src/gui/app.py
```

Direct script execution. Requires proper PYTHONPATH or relative imports.

## Files Moved

### From Root Directory

**Before:**
- `run_gui.py` → ✅ Kept (minimal launcher)
- `test_gui_imports.py` → ✅ Moved to `src/gui/tests/test_imports.py`

### From docs/ Directory

**Before:**
- `docs/GUI_USER_GUIDE.md` → ✅ Moved to `src/gui/docs/USER_GUIDE.md`
- `docs/GUI_IMPLEMENTATION_PLAN.md` → ✅ Moved to `src/gui/docs/IMPLEMENTATION_PLAN.md`
- `docs/PHASE2_COMPLETE.md` → ✅ Moved to `src/gui/docs/PHASE2_COMPLETE.md`

## Files Added

### Package Infrastructure

- `src/gui/__init__.py` - Package initialization with version info
- `src/gui/__main__.py` - Module entry point
- `src/gui/README.md` - GUI-specific README

### __init__.py Files (Proper Package Structure)

- `src/gui/views/__init__.py` - View components export
- `src/gui/workers/__init__.py` - Worker export
- `src/gui/utils/__init__.py` - Utility functions export
- `src/gui/tests/__init__.py` - Test package marker

### Documentation

- `src/gui/docs/FOLDER_STRUCTURE.md` - This file

## Benefits of New Structure

### 1. Self-Contained Package

All GUI code in one place:
- Easy to find all GUI-related files
- Clear separation from CLI/backend code
- Can be distributed as standalone package if needed

### 2. Proper Python Package

With `__init__.py` and `__main__.py`:
- Can be imported: `from gui import BacktestApp`
- Can be run as module: `python -m gui`
- Clean public API via `__all__` exports

### 3. Documentation Co-located

GUI docs inside `src/gui/docs/`:
- Easy to find relevant documentation
- Reduces clutter in main `docs/` folder
- Clear ownership (GUI team)

### 4. Testable Structure

Tests in `src/gui/tests/`:
- GUI-specific tests separate from backend tests
- Clear test organization
- Can run GUI tests independently

### 5. Root Directory Cleanup

Root directory now contains:
- `run_gui.py` - Single, minimal launcher (20 lines)
- `verify_phase1.py` - Backend verification (kept)
- Main project files (README, requirements, etc.)

Much cleaner than having GUI tests and docs scattered around.

## Import Patterns

### From Within GUI Package

```python
# In src/gui/app.py
from gui.views import SetupView, ExecutionView, ResultsView
from gui.workers import GUIBacktestController
from gui.utils import get_strategy_registry
```

### From External Code

```python
# From other packages or root scripts
import sys
sys.path.insert(0, 'src')

from gui import main, BacktestApp
from gui.utils import get_strategy_registry
```

### In Tests

```python
# In src/gui/tests/test_imports.py
import sys
from pathlib import Path

# Add src to path (go up 2 levels from gui/tests/)
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from gui.utils.strategy_utils import get_strategy_registry
```

## Code Statistics

### Total GUI Code

| Component | Files | Lines | Purpose |
|-----------|-------|-------|---------|
| Views | 3 | ~965 | UI screens |
| Workers | 1 | ~410 | Background processing |
| Utils | 1 | ~140 | Helper functions |
| App | 1 | ~200 | Main application |
| Tests | 1 | ~50 | Import validation |
| **Total Code** | **7** | **~1,765** | - |

### Documentation

| Document | Lines | Purpose |
|----------|-------|---------|
| USER_GUIDE.md | 465 | User manual |
| IMPLEMENTATION_PLAN.md | 600 | Original design |
| PHASE2_COMPLETE.md | 550 | Implementation summary |
| README.md | 150 | Developer guide |
| FOLDER_STRUCTURE.md | 250 | This file |
| **Total Docs** | **~2,015** | - |

**Total GUI Package:** ~3,780 lines (code + docs)

## Verification

### Test Imports

```bash
cd Homeguard
python src/gui/tests/test_imports.py
```

**Expected Output:**
```
Testing GUI imports...
Python path: C:\...\Homeguard\src

[OK] Strategy utils imported
[OK] Strategy registry loaded: 10 strategies
[OK] Setup view imported
[OK] Execution view imported
[OK] Results view imported
[OK] GUI controller imported

============================================================
All imports successful!
============================================================

Run GUI:
  python run_gui.py
  OR
  cd src && python -m gui
============================================================
```

### Launch GUI

```bash
# Option 1
python run_gui.py

# Option 2
cd src
python -m gui
```

Both should open the GUI window with Setup View.

## Future Organization

### Potential Additions

```
src/gui/
├── components/          # Reusable UI components
│   ├── progress_bar.py
│   ├── metric_card.py
│   └── symbol_selector.py
│
├── themes/              # Theme definitions
│   ├── light_theme.py
│   └── dark_theme.py
│
├── assets/              # Images, icons
│   └── logo.png
│
└── config/              # GUI configuration
    └── settings.py
```

### Recommended Practices

1. **New Views:** Add to `views/` and export in `views/__init__.py`
2. **New Workers:** Add to `workers/` and export in `workers/__init__.py`
3. **New Utils:** Add to `utils/` and export in `utils/__init__.py`
4. **Reusable Components:** Create `components/` subfolder
5. **Documentation:** Add to `docs/` subfolder
6. **Tests:** Add to `tests/` subfolder

## Migration Notes

### Updating Documentation References

Old paths in docs have been updated:

**Before:**
- `docs/GUI_USER_GUIDE.md` → `src/gui/docs/USER_GUIDE.md`
- `docs/GUI_IMPLEMENTATION_PLAN.md` → `src/gui/docs/IMPLEMENTATION_PLAN.md`
- `docs/PHASE2_COMPLETE.md` → `src/gui/docs/PHASE2_COMPLETE.md`

**Main README Updated:**
```markdown
**Documentation:**
- [GUI User Guide](src/gui/docs/USER_GUIDE.md)
- [GUI README](src/gui/README.md)
```

### Backward Compatibility

**Preserved:**
- `run_gui.py` in root (still works)
- All import paths (no changes needed)
- Backend Phase 1 code (untouched)

**Removed:**
- `test_gui_imports.py` from root (moved to `src/gui/tests/`)
- GUI docs from main `docs/` (moved to `src/gui/docs/`)

## Summary

✅ **Root directory cleaned**
✅ **GUI code self-contained in src/gui/**
✅ **Proper Python package structure**
✅ **Documentation co-located**
✅ **Tests organized**
✅ **All imports working**
✅ **Launch options preserved**

The GUI is now a well-organized, self-contained package that's easy to develop, test, and maintain!
