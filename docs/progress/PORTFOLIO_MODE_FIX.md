# Portfolio Mode GUI Fix - Duplicate Symbol Cards

**Date**: 2025-11-03
**Issue**: Multi-symbol portfolio mode showed 2 separate symbol cards (AAPL, MSFT) instead of 1 unified "Portfolio" card
**Status**: [+] FIXED

---

## Problem

When running a multi-symbol portfolio backtest with symbols=['AAPL', 'MSFT'], the GUI showed:

```
Symbol Progress:
├─ AAPL         Pending
└─ MSFT         Pending
```

Instead of:

```
Symbol Progress:
└─ Portfolio (AAPL, MSFT)   Running
```

---

## Root Cause

**Issue 1 - Backend Queue Creation ([+] Already Fixed)**

The `GUIController.run()` method correctly created a single "Portfolio" queue for portfolio mode:

```python
# Line 205-218 in gui_controller.py
if portfolio_mode == "Multi-Symbol Portfolio":
    portfolio_key = "Portfolio"
    self.progress_queues[portfolio_key] = Queue()  # [+] Single queue
    self.log_queues[portfolio_key] = Queue()
    self.status[portfolio_key] = "pending"
else:
    for symbol in symbols:  # Multiple queues for single-symbol mode
        self.progress_queues[symbol] = Queue()
        ...
```

**Issue 2 - Frontend UI Initialization ([-] WAS BROKEN)**

The frontend `app.py` created UI cards BEFORE the backend set up queues:

```python
# Line 508 in app.py (BEFORE FIX)
self.run_view.initialize_symbols(config['symbols'])  # [-] Always ['AAPL', 'MSFT']
```

This created UI cards for each symbol regardless of portfolio mode.

**Issue 3 - Export Log Messages ([-] WAS BROKEN)**

The export code tried to send logs to non-existent symbol queues:

```python
# Multiple locations in _export_portfolio_results() (BEFORE FIX)
for symbol in symbols:
    self.log_queues[symbol].put(...)  # [-] KeyError! Queue doesn't exist
```

This caused crashes during export, preventing JSON/HTML generation.

---

## Solution

### **Fix 1: Frontend UI Initialization** [+]

**File:** `src/gui/app.py` (Line 509-534)

```python
# Initialize symbols based on portfolio mode
portfolio_mode = config.get('portfolio_mode', 'Single-Symbol')
if portfolio_mode == "Multi-Symbol Portfolio":
    # Single portfolio entry
    symbols_display = f"Portfolio ({', '.join(config['symbols'])})"
    display_symbols = [symbols_display]
else:
    # Individual symbol entries
    display_symbols = config['symbols']

self.run_view.initialize_symbols(display_symbols)
```

**Result:**
- Portfolio mode: Creates 1 UI card labeled "Portfolio (AAPL, MSFT)"
- Single-symbol mode: Creates 2 UI cards labeled "AAPL" and "MSFT"

---

### **Fix 2: Export Log Messages** [+]

**File:** `src/gui/workers/gui_controller.py` (Lines 1130-1246)

Updated 3 locations to use the portfolio queue:

**Location 1:** Tearsheet generation (Line 1130-1139)
```python
# BEFORE:
for symbol in symbols:
    self.log_queues[symbol].put(...)  # [-] Crash

# AFTER:
portfolio_key = "Portfolio"
symbols_display = f"Portfolio ({', '.join(symbols)})"
if portfolio_key in self.log_queues:
    self.log_queues[portfolio_key].put(...)  # [+] Works
```

**Location 2:** Metrics calculation (Line 1177-1186)
**Location 3:** File export (Line 1239-1246)

**Result:** No more crashes during export, all files generate correctly.

---

### **Fix 3: Added Helper Method** [+]

**File:** `src/gui/workers/gui_controller.py` (Line 923-935)

```python
def get_tracked_items(self) -> List[str]:
    """
    Get list of tracked items (symbols or portfolio).

    For Single-Symbol mode: Returns list of symbols
    For Multi-Symbol Portfolio mode: Returns ["Portfolio"]
    """
    return list(self.progress_queues.keys())
```

**Usage:** Frontend can call this to dynamically determine what UI elements to create.

---

## Files Modified

1. **[src/gui/app.py](src/gui/app.py)**
   - Line 509-534: Check portfolio mode before initializing UI

2. **[src/gui/workers/gui_controller.py](src/gui/workers/gui_controller.py)**
   - Line 1130-1139: Fix tearsheet log message
   - Line 1177-1186: Fix metrics log message
   - Line 1239-1246: Fix export log message
   - Line 923-935: Add get_tracked_items() helper method

---

## Testing

### **Before Fix:**
```
Symbol Progress:
├─ AAPL         Pending  [-] Wrong
└─ MSFT         Pending  [-] Wrong

Files Generated:
- None (crashed during export)  [-]
```

### **After Fix:**
```
Symbol Progress:
└─ Portfolio (AAPL, MSFT)   Running -> Complete  [+]

Files Generated:
- portfolio_metrics.json          [+]
- portfolio_charts.json           [+]
- portfolio_analytics.html        [+]
- portfolio_report.html           [+]
- portfolio_tearsheet.html        [+]
- portfolio_stats.csv             [+]
- symbol_comparison.csv           [+]
```

---

## Summary

**Problem:** Frontend created UI for individual symbols even in portfolio mode
**Cause:** Frontend used original symbols list before backend created queues
**Fix:** Frontend now checks portfolio mode and creates appropriate UI elements

**Result:**
- [+] Single "Portfolio (AAPL, MSFT)" card in Symbol Progress
- [+] All log messages appear in single card
- [+] Complete report generation (JSON, HTML, CSV)
- [+] No crashes during export
- [+] Clean, unified user experience

---

**Author**: Claude (AI Assistant)
**Date**: 2025-11-03
