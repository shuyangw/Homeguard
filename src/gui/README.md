# Backtest Runner GUI

A modern, cross-platform desktop application for backtesting trading strategies with real-time progress monitoring.

## Quick Start

```bash
# Activate environment
conda activate fintech

# Launch GUI (from project root)
python run_gui.py

# Or as a module (from src/)
cd src
python -m gui

# Or directly
python src/gui/app.py
```

## Features

- **Visual Strategy Selector**: Choose from 10 built-in strategies
- **Dynamic Parameter Editor**: Auto-generates UI controls from strategy parameters
- **Real-Time Progress**: Individual progress cards for each symbol
- **Color-Coded Results**: Green/red metrics for quick analysis
- **Multi-Worker Support**: 1-16 parallel workers for fast execution
- **Export Options**: Save results to CSV or HTML

## Project Structure

```
gui/
├── __init__.py              # Package initialization
├── __main__.py              # Module entry point (python -m gui)
├── app.py                   # Main application & navigation
├── README.md                # This file
│
├── docs/                    # GUI-specific documentation
│   ├── USER_GUIDE.md        # Complete user manual
│   ├── IMPLEMENTATION_PLAN.md
│   └── PHASE2_COMPLETE.md
│
├── views/                   # UI components
│   ├── __init__.py
│   ├── setup_view.py        # Configuration screen
│   ├── execution_view.py    # Progress monitoring
│   └── results_view.py      # Results display
│
├── workers/                 # Background processing
│   ├── __init__.py
│   └── gui_controller.py    # Backtest coordinator
│
├── utils/                   # Helper utilities
│   ├── __init__.py
│   └── strategy_utils.py    # Strategy introspection
│
└── tests/                   # GUI tests
    ├── __init__.py
    └── test_imports.py      # Import validation
```

## Technology Stack

- **Flet 0.28.3**: Flutter for Python (Material Design UI)
- **Python 3.8+**: Core language
- **VectorBT**: Backtesting engine (via parent modules)
- **Pandas**: Data manipulation
- **Threading**: Multi-worker execution

## Architecture

```
SetupView (Configuration)
    ↓
BacktestApp (Navigation & Polling)
    ↓
GUIBacktestController (Thread Coordination)
    ↓
SweepRunner (Parallel Execution)
    ↓
BacktestEngine (VectorBT)
```

## Development

### Adding New Strategies

Strategies are auto-detected from `src/strategies/`. To add a new strategy to the GUI:

1. Create strategy class in `src/strategies/`
2. Add to `src/strategies/__init__.py`
3. Add to `get_strategy_registry()` in `utils/strategy_utils.py`
4. Restart GUI - parameters will auto-generate

### Customizing Views

Each view is a self-contained `ft.Container`:

- **setup_view.py**: Modify `_build_ui()` for layout changes
- **execution_view.py**: Modify `SymbolCard` for card customization
- **results_view.py**: Modify `_build_results_table()` for table changes

### Testing

```bash
# Run import tests
cd src
python gui/tests/test_imports.py

# Manual testing
python -m gui
```

## Documentation

- **[User Guide](docs/USER_GUIDE.md)** - Complete usage instructions
- **[Implementation Plan](docs/IMPLEMENTATION_PLAN.md)** - Original design
- **[Phase 2 Report](docs/PHASE2_COMPLETE.md)** - Implementation details

## Performance

- **Startup Time**: < 2 seconds
- **UI Update Frequency**: 200ms (5 FPS)
- **Memory Overhead**: ~50-100 MB (GUI framework)
- **Scalability**: Tested up to 10 symbols, supports 100+

## Known Limitations

- **No log capture** (backtesting console output not shown)
- **No equity curves** (results table only, no charts)
- **No dark mode** (light theme only)
- **Single backtest at a time** (no queuing)

These are planned for future phases.

## Troubleshooting

### GUI won't start

```bash
# Check Flet installation
pip show flet

# Reinstall if needed
pip install --upgrade flet
```

### Import errors

```bash
# Test imports
cd src
python gui/tests/test_imports.py

# Check Python path
python -c "import sys; print(sys.path)"
```

### No strategies appear

Ensure `src/strategies/__init__.py` exports all strategies and they're added to `get_strategy_registry()`.

## Contributing

When adding GUI features:

1. Follow existing view structure
2. Use Flet's Material Design components
3. Maintain non-blocking UI (async for long operations)
4. Add docstrings to all public methods
5. Update this README if structure changes

## Version History

- **2.0.0** (2025-01-01): Phase 2 complete - Full GUI implementation
- **1.0.0** (2025-01-01): Phase 1 complete - Backend with callbacks

## License

Part of Homeguard Backtesting Framework.
