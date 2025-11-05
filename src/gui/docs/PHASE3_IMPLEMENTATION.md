# Phase 3: Live Worker Log Streaming - Implementation Guide

**Status:** Ready to Implement
**Complexity:** Moderate
**Estimated Time:** 4-6 hours
**Files to Modify:** 3
**Files to Create:** 1
**Code to Add:** ~300 lines

---

## Executive Summary

Enable real-time log streaming from parallel worker threads to individual WorkerLogViewer panels in the GUI. Each of the n worker threads will have its own log panel showing:
- Which symbol it's currently processing
- Real-time logs from BacktestEngine execution
- Color-coded log levels
- Worker idle/busy status

---

## Current State (Phase 2.5)

**✅ What's Already Done:**
1. UI components exist (`WorkerLogViewer` class)
2. ExecutionView has split layout with n worker panels
3. Methods ready: `add_worker_log()`, `set_worker_symbol()`, `set_worker_idle()`
4. app.py initializes correct number of worker viewers

**❌ What's Missing:**
1. No tracking of which worker is processing which symbol
2. No log capture from worker threads
3. No routing of logs to correct worker panel
4. Polling loop doesn't read worker logs

---

## Architecture Design

### Problem: Anonymous Workers

**Challenge:**
ThreadPoolExecutor creates anonymous worker threads. We need to:
- Assign stable IDs (0 to max_workers-1) to map to UI panels
- Track which worker is processing which symbol
- Capture logs from specific worker threads
- Handle workers finishing and picking up new symbols

### Solution: Worker ID Pool + Log Capture

```
┌─────────────────────────────────────────────────────────┐
│  UI Thread (Flet)                                        │
│  ┌────────────────────────────────────────────────┐     │
│  │ ExecutionView                                   │     │
│  │  ┌──────────────────────────────────────────┐  │     │
│  │  │ WorkerLogViewer[0] "Worker 1"            │  │     │
│  │  │  Processing: AAPL                         │  │     │
│  │  │  [10:32:15.123] Loading data...           │  │     │
│  │  │  [10:32:16.456] Running backtest...       │  │     │
│  │  └──────────────────────────────────────────┘  │     │
│  │  ┌──────────────────────────────────────────┐  │     │
│  │  │ WorkerLogViewer[1] "Worker 2"            │  │     │
│  │  │  Idle                                     │  │     │
│  │  └──────────────────────────────────────────┘  │     │
│  └────────────▲──────────────────────────────────┘     │
└─────────────────┼──────────────────────────────────────┘
                  │ poll every 200ms
                  │ get_worker_updates()
┌─────────────────┼──────────────────────────────────────┐
│  GUIBacktestController                                   │
│  ┌──────────────┴─────────────────────────────────┐     │
│  │ Worker Tracking State:                          │     │
│  │                                                  │     │
│  │ worker_log_queues: Dict[int, Queue]            │     │
│  │   {0: Queue(), 1: Queue(), ...}                │     │
│  │                                                  │     │
│  │ worker_assignments: Dict[int, Optional[str]]   │     │
│  │   {0: "AAPL", 1: None, 2: "MSFT", ...}        │     │
│  │                                                  │     │
│  │ available_worker_ids: Queue                    │     │
│  │   Queue([1, 3, 4, ...])  # IDs not in use     │     │
│  │                                                  │     │
│  │ worker_status_queue: Queue                     │     │
│  │   WorkerStatusUpdate objects                   │     │
│  │                                                  │     │
│  └────────────────────────────────────────────────┘     │
└─────────────────┼──────────────────────────────────────┘
                  │ callbacks
                  │ push to queues
┌─────────────────┼──────────────────────────────────────┐
│  ThreadPoolExecutor                                      │
│  ┌──────────────▼─────────────────────────────────┐     │
│  │ Worker Thread (processing AAPL):                │     │
│  │                                                  │     │
│  │  1. worker_id = claim_worker_id()  # Get ID 0  │     │
│  │  2. update_assignment(0, "AAPL")               │     │
│  │  3. push_status(0, "AAPL", "started")          │     │
│  │  4. with LogCapture(worker_id=0):              │     │
│  │       logger.info("Loading data...")  ───────► │     │
│  │       portfolio = engine.run(...)              │     │
│  │       logger.success("Complete")  ────────────► │     │
│  │  5. push_status(0, None, "idle")               │     │
│  │  6. release_worker_id(0)                       │     │
│  │                                                  │     │
│  └─────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────┘
              All logs go to worker_log_queues[0]
```

---

## Data Structures

### New Dataclass: WorkerStatusUpdate

```python
@dataclass
class WorkerStatusUpdate:
    """Status update for a worker thread."""
    worker_id: int          # 0 to max_workers-1
    symbol: Optional[str]   # Symbol being processed, None if idle
    status: str             # "started"|"idle"
    timestamp: datetime
```

### New Dataclass: WorkerLogMessage

```python
@dataclass
class WorkerLogMessage:
    """Log message from a specific worker."""
    worker_id: int
    message: str
    level: str  # "info"|"success"|"warning"|"error"
    timestamp: datetime
```

### Controller State Additions

```python
# In GUIBacktestController.__init__():

# Worker tracking (NEW - Phase 3)
self.worker_log_queues: Dict[int, Queue] = {}          # worker_id -> log queue
self.worker_assignments: Dict[int, Optional[str]] = {} # worker_id -> current symbol (None if idle)
self.available_worker_ids: Queue = Queue()             # Pool of available IDs
self.worker_status_queue: Queue = Queue()              # Status change updates
self._worker_lock = threading.Lock()                   # Thread-safe ID assignment
```

---

## Implementation Steps

### Step 1: Add Worker Tracking to Controller

**File:** `src/gui/workers/gui_controller.py`

**Changes in `__init__()`:**

```python
def __init__(self, max_workers: int = 8):
    # ... existing code ...

    # NEW: Worker tracking (Phase 3)
    self.worker_log_queues: Dict[int, Queue] = {}
    self.worker_assignments: Dict[int, Optional[str]] = {}
    self.available_worker_ids: Queue = Queue()
    self.worker_status_queue: Queue = Queue()
    self._worker_lock = threading.Lock()
```

**Changes in `start_backtests()`:**

```python
def start_backtests(...):
    try:
        log_info(f"GUIController: Starting backtests...")

        # Reset existing state
        self.progress_queues.clear()
        self.log_queues.clear()
        # ...

        # NEW: Initialize worker tracking
        self.worker_log_queues.clear()
        self.worker_assignments.clear()

        # Create worker log queues and initialize assignments
        for worker_id in range(self.max_workers):
            self.worker_log_queues[worker_id] = Queue()
            self.worker_assignments[worker_id] = None
            self.available_worker_ids.put(worker_id)

        log_info(f"GUIController: Initialized {self.max_workers} worker log streams")

        # ... rest of existing code ...
```

**New Method: `get_worker_updates()`**

```python
def get_worker_updates(self) -> Dict[int, Dict]:
    """
    Get updates from all worker threads.

    Returns:
        Dict mapping worker_id to update data:
        {
            0: {
                'symbol': 'AAPL' or None,
                'status': 'started'|'idle',
                'logs': [WorkerLogMessage, ...]
            },
            1: { ... },
            ...
        }
    """
    updates = {}

    # Process worker status changes
    while True:
        try:
            status_update = self.worker_status_queue.get_nowait()
            worker_id = status_update.worker_id

            if worker_id not in updates:
                updates[worker_id] = {'logs': []}

            updates[worker_id]['symbol'] = status_update.symbol
            updates[worker_id]['status'] = status_update.status

        except Empty:
            break

    # Process worker logs
    for worker_id, log_queue in self.worker_log_queues.items():
        worker_logs = []

        while True:
            try:
                log_msg = log_queue.get_nowait()
                worker_logs.append(log_msg)
            except Empty:
                break

        if worker_logs:
            if worker_id not in updates:
                updates[worker_id] = {}
            updates[worker_id]['logs'] = worker_logs

    return updates
```

**New Helper Methods:**

```python
def _claim_worker_id(self) -> int:
    """
    Claim an available worker ID (blocking).

    Returns:
        worker_id: Integer from 0 to max_workers-1
    """
    return self.available_worker_ids.get()  # Blocks until available


def _release_worker_id(self, worker_id: int):
    """
    Release a worker ID back to the pool.

    Args:
        worker_id: ID to release
    """
    with self._worker_lock:
        self.worker_assignments[worker_id] = None
    self.available_worker_ids.put(worker_id)


def _update_worker_assignment(self, worker_id: int, symbol: Optional[str]):
    """
    Update which symbol a worker is processing.

    Args:
        worker_id: Worker ID
        symbol: Symbol being processed (None if idle)
    """
    with self._worker_lock:
        self.worker_assignments[worker_id] = symbol

    # Push status update
    status = "started" if symbol else "idle"
    self.worker_status_queue.put(WorkerStatusUpdate(
        worker_id=worker_id,
        symbol=symbol,
        status=status,
        timestamp=datetime.now()
    ))
```

---

### Step 2: Create LogCapture Context Manager

**File:** `src/gui/workers/log_capture.py` (NEW)

```python
"""
Log capture context manager for worker threads.

Intercepts logger calls within a specific thread and routes them
to a worker-specific queue for GUI display.
"""

import threading
from queue import Queue
from datetime import datetime
from typing import Optional
from contextlib import contextmanager

from gui.workers.gui_controller import WorkerLogMessage


# Thread-local storage for current worker context
_worker_context = threading.local()


class LogCapture:
    """
    Context manager that captures logs from utils.logger within a worker thread.

    Usage:
        with LogCapture(worker_id=0, log_queue=my_queue):
            logger.info("This goes to the queue")
            portfolio = engine.run(...)
    """

    def __init__(self, worker_id: int, log_queue: Queue):
        """
        Initialize log capture.

        Args:
            worker_id: Worker thread ID (0 to max_workers-1)
            log_queue: Queue to push log messages to
        """
        self.worker_id = worker_id
        self.log_queue = log_queue
        self.original_handlers = None

    def __enter__(self):
        """Start capturing logs for this thread."""
        # Store worker context in thread-local storage
        _worker_context.worker_id = self.worker_id
        _worker_context.log_queue = self.log_queue
        _worker_context.capturing = True

        # Monkey-patch the logger functions
        from utils import logger

        # Save original functions
        self.original_logger_funcs = {
            'info': logger.info,
            'success': logger.success,
            'error': logger.error,
            'warning': logger.warning,
            'metric': logger.metric,
            'profit': logger.profit,
            'loss': logger.loss
        }

        # Replace with capturing versions
        logger.info = lambda msg, **kwargs: self._capture_log(msg, 'info')
        logger.success = lambda msg, **kwargs: self._capture_log(msg, 'success')
        logger.error = lambda msg, **kwargs: self._capture_log(msg, 'error')
        logger.warning = lambda msg, **kwargs: self._capture_log(msg, 'warning')
        logger.metric = lambda msg, **kwargs: self._capture_log(msg, 'info')
        logger.profit = lambda msg, **kwargs: self._capture_log(msg, 'success')
        logger.loss = lambda msg, **kwargs: self._capture_log(msg, 'warning')

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop capturing logs and restore original logger."""
        from utils import logger

        # Restore original logger functions
        for func_name, original_func in self.original_logger_funcs.items():
            setattr(logger, func_name, original_func)

        # Clear thread-local context
        _worker_context.capturing = False
        _worker_context.worker_id = None
        _worker_context.log_queue = None

        return False  # Don't suppress exceptions

    def _capture_log(self, message: str, level: str):
        """Capture a log message and push to queue."""
        if not hasattr(_worker_context, 'capturing') or not _worker_context.capturing:
            return

        log_msg = WorkerLogMessage(
            worker_id=self.worker_id,
            message=message,
            level=level,
            timestamp=datetime.now()
        )

        self.log_queue.put(log_msg)
```

**Alternative Simpler Approach (if monkey-patching is too fragile):**

Just log manually within the worker function instead of trying to intercept:

```python
def _run_single_symbol_with_logging(symbol, worker_id, log_queue):
    """Wrapper that manually logs key events."""

    def worker_log(message: str, level: str = "info"):
        log_queue.put(WorkerLogMessage(
            worker_id=worker_id,
            message=message,
            level=level,
            timestamp=datetime.now()
        ))

    try:
        worker_log(f"Loading data for {symbol}...", "info")
        portfolio = engine.run(...)
        worker_log(f"Backtest complete for {symbol}", "success")
        return portfolio
    except Exception as e:
        worker_log(f"Error: {str(e)}", "error")
        raise
```

---

### Step 3: Modify SweepRunner to Use Worker Tracking

**File:** `src/backtesting/engine/sweep_runner.py`

**Option A: Minimal Changes (Recommended)**

Don't modify SweepRunner at all. Instead, handle worker tracking entirely in `GUIBacktestController._run_backtests()`:

```python
# In GUIBacktestController
def _run_backtests(self, strategy, symbols, start_date, end_date):
    """
    Run SweepRunner in background with worker tracking.
    """
    try:
        log_info("GUIController: Starting worker-tracked sweep")

        # Create a wrapper function that handles worker ID management
        def run_symbol_with_worker_tracking(symbol: str):
            # Claim worker ID
            worker_id = self._claim_worker_id()

            try:
                # Update assignment
                self._update_worker_assignment(worker_id, symbol)

                # Log to worker panel
                self.worker_log_queues[worker_id].put(WorkerLogMessage(
                    worker_id=worker_id,
                    message=f"Starting backtest for {symbol}",
                    level="info",
                    timestamp=datetime.now()
                ))

                # Run backtest (standard SweepRunner flow)
                # The on_symbol_* callbacks will still fire
                portfolio = self.engine.run(...)

                # Success log
                self.worker_log_queues[worker_id].put(WorkerLogMessage(
                    worker_id=worker_id,
                    message=f"Completed {symbol}",
                    level="success",
                    timestamp=datetime.now()
                ))

                return portfolio

            finally:
                # Always release worker ID
                self._update_worker_assignment(worker_id, None)
                self._release_worker_id(worker_id)

        # Run standard SweepRunner (it will call our wrapper via callbacks)
        results = self._runner.run_sweep(...)

    except Exception as e:
        log_exception(e, "GUIController: Error in sweep")
    finally:
        self._running = False
```

**Option B: Modify SweepRunner (More invasive)**

Add worker ID tracking directly to SweepRunner. NOT RECOMMENDED - breaks Phase 1's backward compatibility goal.

---

### Step 4: Update app.py Polling Loop

**File:** `src/gui/app.py`

**Modify `_poll_updates()` method:**

```python
async def _poll_updates(self):
    """Async task to poll for backtest updates."""
    try:
        while self.poll_task and self.controller:
            try:
                # EXISTING: Get symbol updates
                updates = self.controller.get_updates()

                for symbol, update_data in updates.items():
                    status = self.controller.get_status(symbol)
                    self.execution_view.update_symbol_status(symbol, status)

                    if 'progress' in update_data and update_data['progress']:
                        latest_progress = update_data['progress'][-1]
                        self.execution_view.update_symbol_progress(
                            symbol,
                            latest_progress.progress,
                            latest_progress.message
                        )

                # NEW: Get worker updates (Phase 3)
                worker_updates = self.controller.get_worker_updates()

                for worker_id, worker_data in worker_updates.items():
                    # Update worker's current symbol
                    if 'symbol' in worker_data:
                        symbol = worker_data['symbol']
                        if symbol:
                            self.execution_view.set_worker_symbol(worker_id, symbol)
                        else:
                            self.execution_view.set_worker_idle(worker_id)

                    # Update worker's logs
                    if 'logs' in worker_data:
                        for log_msg in worker_data['logs']:
                            self.execution_view.add_worker_log(
                                worker_id,
                                log_msg.message,
                                log_msg.level
                            )

                # EXISTING: Update overall progress
                summary = self.controller.get_progress_summary()
                self.execution_view.update_overall_progress(
                    completed=summary['completed'],
                    total=summary['total'],
                    running=summary['running'],
                    failed=summary['failed']
                )

                # Check if done
                if summary['completed'] + summary['failed'] >= summary['total']:
                    self.execution_view.mark_complete()
                    self.poll_task = None
                    log_info("All backtests completed")
                    break

            except Exception as e:
                log_error(f"Error during polling update: {e}")

            await asyncio.sleep(0.2)

    except Exception as e:
        log_exception(e, "Fatal error in polling loop")
        self.poll_task = None
```

---

## Testing Plan

### Unit Test: Worker ID Pool

```python
def test_worker_id_pool():
    """Test that worker IDs are assigned and released correctly."""
    controller = GUIBacktestController(max_workers=4)
    controller.start_backtests(...)

    # Claim all IDs
    ids = [controller._claim_worker_id() for _ in range(4)]
    assert ids == [0, 1, 2, 3]

    # Pool should be empty now
    assert controller.available_worker_ids.empty()

    # Release an ID
    controller._release_worker_id(1)

    # Should be able to claim it again
    new_id = controller._claim_worker_id()
    assert new_id == 1
```

### Integration Test: Worker Logs

```python
def test_worker_log_streaming():
    """Test that logs from workers reach the correct queues."""
    controller = GUIBacktestController(max_workers=2)

    # Manually simulate worker logging
    worker_id = 0
    log_queue = controller.worker_log_queues[worker_id]

    log_queue.put(WorkerLogMessage(
        worker_id=0,
        message="Test log",
        level="info",
        timestamp=datetime.now()
    ))

    # Poll updates
    updates = controller.get_worker_updates()

    assert 0 in updates
    assert len(updates[0]['logs']) == 1
    assert updates[0]['logs'][0].message == "Test log"
```

### Manual Test: Live GUI

1. Start GUI: `python scripts/run_gui.py`
2. Configure backtest with 8 workers, 10 symbols
3. Run backtest
4. **Verify:**
   - Each worker panel shows a different worker ID
   - Worker panels show which symbol they're processing
   - Logs appear in real-time
   - Workers go idle when done
   - New symbols assigned to idle workers

---

## Code Volume Summary

| File | Status | Lines | Purpose |
|------|--------|-------|---------|
| `gui/workers/gui_controller.py` | Modify | +150 | Worker tracking, ID pool, get_worker_updates() |
| `gui/workers/log_capture.py` | Create | +100 | LogCapture context manager (OPTIONAL) |
| `gui/app.py` | Modify | +30 | Poll worker updates, update UI |
| `gui/workers/__init__.py` | Modify | +5 | Export new classes |

**Total:** ~285 lines (or ~185 without LogCapture)

---

## Recommended Implementation Order

1. **Day 1 (2-3 hours):**
   - Add worker tracking structures to `gui_controller.py`
   - Implement `get_worker_updates()` method
   - Write unit tests for worker ID pool

2. **Day 2 (2-3 hours):**
   - Update `app.py` polling loop
   - Test with manual log injection (no LogCapture yet)
   - Verify UI updates correctly

3. **Day 3 (2 hours) - OPTIONAL:**
   - Create `LogCapture` context manager
   - Integrate into backtest execution
   - OR just manually log key events

---

## Risks & Mitigation

**Risk 1: Race Conditions**
- **Mitigation:** Use threading.Lock for worker_assignments
- **Mitigation:** Queue operations are thread-safe by default

**Risk 2: Worker ID Pool Exhaustion**
- **Mitigation:** `_claim_worker_id()` blocks until ID available
- **Mitigation:** `finally:` block ensures IDs always released

**Risk 3: Log Capture Breaking Existing Code**
- **Mitigation:** Make LogCapture optional
- **Mitigation:** Fall back to manual logging if complex

**Risk 4: UI Performance with Many Logs**
- **Mitigation:** WorkerLogViewer limits to 200 entries
- **Mitigation:** Polling every 200ms is throttled

---

## Success Criteria

**Phase 3 Complete When:**

- ✅ Each worker panel shows correct worker ID (1-8)
- ✅ Worker panels show which symbol they're processing
- ✅ Logs appear in real-time in correct worker panel
- ✅ Workers show "Idle" when not processing
- ✅ No crashes with 8 workers, 50 symbols
- ✅ Performance acceptable (no UI lag)
- ✅ Worker IDs correctly assigned and released

---

## Next Steps After Phase 3

**Phase 4: Advanced Results & Charts**
- Mini equity curves
- QuantStats integration
- Trade-by-trade analysis

**Phase 5: Polish & Deployment**
- Settings persistence
- macOS/Windows builds
- Performance optimizations

---

## Conclusion

Phase 3 adds live visibility into what each worker thread is doing, making the backtesting process fully transparent. The implementation is straightforward but requires careful coordination between:

1. Worker ID assignment (controller)
2. Log capture/routing (new)
3. UI updates (app.py)
4. Worker panels (already exists)

**Recommendation:** Start with manual logging (simpler) before attempting full LogCapture.

**Estimated Completion:** 4-6 hours for basic implementation, +2 hours for advanced LogCapture.
