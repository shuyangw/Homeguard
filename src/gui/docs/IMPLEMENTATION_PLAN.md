# Flet GUI Backtesting Application - Implementation Plan

## Executive Summary

Build a lightweight GUI using **Flet** that wraps your **existing multithreaded backtesting infrastructure** (`SweepRunner`, `ResultsAggregator`, `BacktestEngine`). Make minimal, backward-compatible enhancements to `SweepRunner` for GUI callbacks, then create a thin GUI wrapper layer.

**Key Principle: Reuse, Don't Rebuild**

**Code Reduction: 46%** (1,300 lines vs 2,400 lines in original plan)

---

## 1. Architecture: Existing Infrastructure Analysis

### What You Already Have (No Changes Needed)

✅ **SweepRunner** - Production-ready parallel backtest coordinator
- ThreadPoolExecutor with configurable max_workers (supports 1-16)
- `run_sweep()` - orchestrates parallel/sequential execution
- `_run_parallel()` - parallel execution with `as_completed()`
- `run_and_report()` - generates CSV/HTML reports
- **Status:** Keep as-is, add optional callbacks

✅ **ResultsAggregator** - Comprehensive results analysis
- `aggregate_results()` - creates summary DataFrame
- `calculate_summary_stats()` - detailed statistics
- `display_summary_stats()` - formatted console output
- `export_to_csv()` / `export_to_html()` - proven exporters
- **Status:** Reuse 100%, no changes

✅ **BacktestEngine** - Core backtesting engine
- `run()` - executes backtests
- `run_and_report()` - generates QuantStats reports
- **Status:** Reuse 100%, no changes

✅ **Logger** - Rich-based colored logging
- **Status:** Reuse, add log capture wrapper

---

## 2. Revised Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FLET MAIN THREAD (UI)                    │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Execution View                                        │  │
│  │  • Polls GUIBacktestController.get_updates()         │  │
│  │  • Updates progress bars, logs, status               │  │
│  └────────────────────────┬──────────────────────────────┘  │
└───────────────────────────┼─────────────────────────────────┘
                            │
                            │ get_updates() every 100ms
                            ▼
┌─────────────────────────────────────────────────────────────┐
│       GUIBacktestController (NEW - Thin Wrapper)            │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ • Creates progress/log queues for each symbol        │  │
│  │ • Wraps SweepRunner with callbacks                   │  │
│  │ • Runs SweepRunner in background thread              │  │
│  │ • Provides get_updates() for UI                      │  │
│  │ • Stores Portfolio objects for chart generation      │  │
│  └────────────────────────┬──────────────────────────────┘  │
└───────────────────────────┼─────────────────────────────────┘
                            │
                            │ Callbacks push to queues
                            ▼
┌─────────────────────────────────────────────────────────────┐
│      SweepRunner (EXISTING - Minor Enhancements)            │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ • ThreadPoolExecutor (1-16 workers)                  │  │
│  │ • Calls optional callbacks:                          │  │
│  │   - on_symbol_start(symbol)                          │  │
│  │   - on_symbol_progress(symbol, stage, progress)      │  │
│  │   - on_symbol_complete(symbol, portfolio, stats)     │  │
│  │   - on_symbol_error(symbol, exception)               │  │
│  │ • If callbacks=None, behaves exactly as before       │  │
│  └────────────────────────┬──────────────────────────────┘  │
└───────────────────────────┼─────────────────────────────────┘
                            │
                            │ executor.submit()
                            ▼
┌─────────────────────────────────────────────────────────────┐
│   Worker Threads (EXISTING - Wrapped for Log Capture)      │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ LogCapture context manager:                          │  │
│  │   with LogCapture(symbol, log_queue):                │  │
│  │     portfolio = engine.run(...)                      │  │
│  │                                                       │  │
│  │ • Intercepts logger calls                            │  │
│  │ • Routes to symbol-specific queue                    │  │
│  │ • Restores logger after completion                   │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│   ResultsAggregator (EXISTING - Reuse 100%)                │
│  • aggregate_results()                                      │
│  • calculate_summary_stats()                                │
│  • export_to_csv() / export_to_html()                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Project Structure

```
src/
├── gui/                              # NEW: GUI Application
│   ├── __init__.py
│   ├── app.py                        # Flet entry point & routing
│   ├── main.py                       # Launch script
│   │
│   ├── views/                        # Screen views
│   │   ├── __init__.py
│   │   ├── setup_view.py             # Configuration screen
│   │   ├── execution_view.py         # Live monitoring
│   │   └── results_view.py           # Results summary
│   │
│   ├── components/                   # UI components
│   │   ├── __init__.py
│   │   ├── strategy_selector.py
│   │   ├── symbol_input.py
│   │   ├── backtest_card.py
│   │   └── results_table.py
│   │
│   ├── workers/                      # Thin coordination layer
│   │   ├── __init__.py
│   │   ├── gui_controller.py         # NEW: GUIBacktestController (~250 lines)
│   │   └── log_capture.py            # NEW: LogCapture (~100 lines)
│   │
│   └── utils/                        # Helper utilities
│       ├── __init__.py
│       ├── mini_charts.py            # Generate equity curves
│       └── export.py                 # Export helpers
│
├── backtesting/                      # EXISTING
│   └── engine/
│       ├── sweep_runner.py           # ENHANCED: +50 lines (callbacks)
│       ├── results_aggregator.py     # REUSED: No changes
│       └── backtest_engine.py        # REUSED: No changes
│
├── strategies/                       # EXISTING: No changes
├── utils/                            # EXISTING: No changes
└── backtest_runner.py                # EXISTING: No changes (CLI)
```

---

## 4. Implementation Phases

### Phase 1: Enhance Existing Infrastructure (Week 1)

**Tasks:**
- [ ] Add optional callbacks to `SweepRunner.__init__()` (~20 lines)
- [ ] Enhance `SweepRunner._run_parallel()` to call callbacks (~30 lines)
- [ ] Add `get_portfolios()` method to return Portfolio objects
- [ ] **Unit test:** Run SweepRunner with callbacks, verify they fire
- [ ] **Unit test:** Run SweepRunner without callbacks (CLI mode), verify unchanged
- [ ] Create `GUIBacktestController` wrapper class (~250 lines)
- [ ] **Integration test:** Controller with 5 symbols, verify queue updates

**Changes to SweepRunner:**

```python
from typing import Callable, Optional
from backtesting.engine.portfolio_simulator import Portfolio

class SweepRunner:
    def __init__(
        self,
        engine: BacktestEngine,
        max_workers: int = 4,
        show_progress: bool = True,
        # NEW: Optional GUI callbacks (all default to None)
        on_symbol_start: Optional[Callable[[str], None]] = None,
        on_symbol_progress: Optional[Callable[[str, str, float], None]] = None,
        on_symbol_complete: Optional[Callable[[str, Portfolio, pd.Series], None]] = None,
        on_symbol_error: Optional[Callable[[str, Exception], None]] = None
    ):
        # Store callbacks
        self.on_symbol_start = on_symbol_start
        self.on_symbol_progress = on_symbol_progress
        self.on_symbol_complete = on_symbol_complete
        self.on_symbol_error = on_symbol_error
```

### Phase 2: Flet UI Foundation (Week 2)

**Tasks:**
- [ ] Create Flet project structure in `src/gui/`
- [ ] Build `setup_view.py` (strategy selector, symbols, dates)
- [ ] Load strategies from STRATEGY_REGISTRY
- [ ] Build dynamic parameter editor
- [ ] Add worker count slider (1-16, with recommendation)
- [ ] Wire up "Run Backtests" button to GUIBacktestController
- [ ] **Manual test:** Start backtest from UI, verify controller starts

### Phase 3: Real-time Execution View (Week 2-3)

**Tasks:**
- [ ] Build `execution_view.py` with polling loop
- [ ] Create `BacktestCard` component (per-symbol status)
- [ ] Add progress bars updating from queue data
- [ ] Add overall progress summary (X/Y completed)
- [ ] Add active workers count display
- [ ] **Integration test:** Run 10 symbols, verify real-time updates
- [ ] **Performance test:** Run 50 symbols with 16 workers

### Phase 4: Results & Export (Week 3)

**Tasks:**
- [ ] Build `results_view.py` with summary cards
- [ ] Reuse `ResultsAggregator.aggregate_results()` for table
- [ ] Display summary statistics (median return, Sharpe, etc.)
- [ ] Add sortable/filterable DataTable
- [ ] Wire up CSV export (reuse `ResultsAggregator.export_to_csv()`)
- [ ] Wire up HTML export (reuse `ResultsAggregator.export_to_html()`)
- [ ] Add navigation to individual QuantStats reports
- [ ] **Manual test:** Complete backtest, verify exports work

### Phase 5: Polish & Packaging (Week 4)

**Tasks:**
- [ ] Add log capture (optional - can defer to later)
- [ ] Generate mini equity curve charts (Plotly)
- [ ] Add cancel/stop functionality
- [ ] Settings panel (save worker count default)
- [ ] Dark mode support
- [ ] macOS build (`flet build macos`)
- [ ] Windows build (`flet build windows`)
- [ ] Performance profiling (100+ symbols)
- [ ] User documentation

---

## 5. Code Volume Comparison

| Component | Original Plan | Revised Plan | Savings |
|-----------|--------------|--------------|---------|
| Thread coordinator | 400 lines (new) | 50 lines (enhance) | 350 lines |
| Worker implementation | 200 lines (new) | 0 lines (reuse) | 200 lines |
| Progress tracking | 100 lines (new) | 50 lines (dataclasses) | 50 lines |
| GUI controller | 0 lines | 250 lines (wrapper) | -250 lines |
| Results aggregation | 500 lines (new) | 0 lines (reuse) | 500 lines |
| Export functions | 200 lines (new) | 0 lines (reuse) | 200 lines |
| Flet UI | 800 lines | 800 lines | 0 lines |
| Utilities | 200 lines | 150 lines | 50 lines |
| **TOTAL** | **~2,400 lines** | **~1,300 lines** | **~1,100 lines saved** |

**46% code reduction!**

---

## 6. Benefits of This Approach

✅ **Reuses Battle-Tested Code**
- SweepRunner already proven in production CLI
- ResultsAggregator generates correct reports
- No risk of introducing threading bugs

✅ **Backward Compatible**
- CLI completely unchanged
- Optional callbacks (default=None)
- Existing tests still pass

✅ **46% Less Code**
- 1,300 lines vs 2,400 lines
- Less code = less bugs = faster development

✅ **Separation of Concerns**
- SweepRunner: Business logic (backtesting)
- GUIBacktestController: GUI integration (queues, callbacks)
- Flet views: Presentation (UI components)

✅ **Easy to Test**
- SweepRunner testable independently
- GUIBacktestController testable with mock callbacks
- UI testable with mock controller

✅ **Future-Proof**
- Improvements to SweepRunner benefit both CLI and GUI
- Can add more callbacks later without breaking changes

---

## 7. Performance Targets

✅ **16 concurrent workers** - Maximum parallelism
✅ **100ms UI updates** - Real-time feel
✅ **50+ symbols** - Efficiently queue large universes
✅ **< 1.5GB memory** - Peak with 16 workers
✅ **Instant cancel** - Stop all workers < 500ms
✅ **No UI freezing** - Main thread always responsive

### Expected Performance Benchmarks

| Symbols | Workers | Est. Time | Peak Memory |
|---------|---------|-----------|-------------|
| 5 (FAANG) | 4 | 15-20s | 400MB |
| 30 (DOW30) | 8 | 45-60s | 800MB |
| 30 (DOW30) | 16 | 25-30s | 1.3GB |
| 100 (NASDAQ100) | 8 | 2-3 min | 800MB |
| 100 (NASDAQ100) | 16 | 1-1.5 min | 1.5GB |

---

## 8. Dependencies

```txt
# Add to requirements.txt (GUI only)
flet>=0.24.0
plotly>=5.18.0

# Everything else already available:
# pandas, numpy, vectorbt, rich, quantstats, etc.
```

---

## 9. Build & Distribution

### macOS Build

```bash
# Development (hot reload)
python src/gui/main.py

# Build .app bundle
cd src/gui
flet build macos --name "Homeguard Backtester"

# Output: build/macos/Homeguard Backtester.app
# Size: ~18-25MB
```

### Windows Build

```bash
# On Windows machine or VM
cd src/gui
flet build windows --name "Homeguard Backtester"

# Output: build/windows/Homeguard Backtester.exe
# Size: ~20-30MB
```

---

## 10. Migration Path

### Existing CLI (Unchanged)

```bash
# Works exactly as before - no GUI dependencies
python src/backtest_runner.py --sweep --universe DOW30 --parallel ...
```

### New GUI

```bash
# Launch GUI application
python src/gui/main.py

# Or packaged app (macOS)
open "Homeguard Backtester.app"

# Or packaged app (Windows)
"Homeguard Backtester.exe"
```

Both use the same underlying engine!

---

## 11. Success Criteria

### Phase 1 Complete When:
✅ SweepRunner enhanced with optional callbacks
✅ CLI still works unchanged (existing tests pass)
✅ GUIBacktestController created and functional
✅ Unit tests pass (4 new tests)
✅ Can run 10 symbol backtest with real-time queue updates
✅ Documentation exported to `docs/`

### Full Project Complete When:
✅ All 5 phases implemented
✅ macOS .app builds and runs
✅ Windows .exe builds and runs
✅ Can backtest 100+ symbols with 16 workers
✅ Real-time progress updates work smoothly
✅ CSV/HTML export works
✅ All unit and integration tests pass
✅ User documentation complete

---

## 12. Timeline

- **Phase 1:** Week 1 (3 hours for initial implementation)
- **Phase 2:** Week 2
- **Phase 3:** Week 2-3
- **Phase 4:** Week 3
- **Phase 5:** Week 4

**Total: 4 weeks**

**LOC Estimate:** ~1,300 lines (46% reduction vs original)

**Max Parallelism:** 16 concurrent backtests

**Platforms:** macOS (primary), Windows (secondary)

**Reuse:** SweepRunner, ResultsAggregator, BacktestEngine, Logger

---

## Appendix: Key Code Snippets

### SweepRunner Enhancement Pattern

```python
def _run_parallel(self, strategy, symbols, start_date, end_date):
    def run_single_backtest(symbol: str) -> tuple:
        try:
            # NEW: Callback - symbol started
            if self.on_symbol_start:
                self.on_symbol_start(symbol)

            # NEW: Callback - progress update
            if self.on_symbol_progress:
                self.on_symbol_progress(symbol, "Loading data...", 0.2)

            portfolio = self.engine.run(...)
            stats = portfolio.stats()

            # NEW: Callback - completed
            if self.on_symbol_complete:
                self.on_symbol_complete(symbol, portfolio, stats)

            return (symbol, stats, portfolio)  # NEW: Return portfolio

        except Exception as e:
            # NEW: Callback - error
            if self.on_symbol_error:
                self.on_symbol_error(symbol, e)

            return (symbol, None, None)
```

### GUIBacktestController Usage Pattern

```python
# In Flet UI
controller = GUIBacktestController(max_workers=8)

# Start backtests in background
controller.start_backtests(
    strategy=MovingAverageCrossover(fast=10, slow=50),
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    start_date='2023-01-01',
    end_date='2024-01-01'
)

# Poll for updates (in async loop)
while controller.is_running():
    updates = controller.get_updates()

    for symbol, data in updates.items():
        # Update UI with progress
        for prog in data['progress']:
            update_progress_bar(symbol, prog.progress)

        # Update UI with logs
        for log in data['logs']:
            append_log(symbol, log.message)

    await asyncio.sleep(0.1)  # 100ms

# Get final results
results = controller.get_results()
portfolios = controller.get_portfolios()
```
