# Logging Requirements

**CRITICAL**: All logging that is intended to be displayed to users OR written to disk MUST use the centralized logging module at `src/utils/logger.py`.

## Logging Module Usage

The project uses a Rich-based logging system for colored console output and file logging.

**Import the logger:**
```python
from utils import logger
```

**Logging functions available:**
- `logger.success(message)` - [+] Green for successful operations, profits, positive returns
- `logger.profit(message)` - [^] Green for gains, positive metrics
- `logger.error(message)` - [X] Red for errors, failures
- `logger.loss(message)` - [v] Red for losses, negative returns
- `logger.warning(message)` - [!] Yellow for warnings, potential issues
- `logger.info(message)` - [i] Cyan for informational messages
- `logger.header(message)` - Magenta for section headers
- `logger.metric(message)` - Blue for metrics and statistics
- `logger.neutral(message)` - White for neutral messages
- `logger.dim(message)` - Dim white for secondary information
- `logger.separator(char="=", length=80)` - Print separator line
- `logger.blank()` - Print blank line

## Color Coding Standards

Use intuitive colors to convey meaning:

**Green** (success/profit):
- Successful operations completed
- Positive returns or profits
- Good performance metrics (e.g., Sharpe Ratio ≥ 1.0, Win Rate ≥ 50%)
- File save confirmations

**Red** (error/loss):
- Errors and failures
- Negative returns or losses
- Poor performance metrics
- Large drawdowns (< -15%)

**Yellow** (warning):
- Warnings and cautions
- Moderate drawdowns (-5% to -15%)
- Potential issues that aren't errors

**Cyan** (info):
- General information
- Status updates
- Data loading messages

**Blue** (metric):
- Neutral metrics and statistics
- Trade counts
- Portfolio values

## Examples

```python
# Color-code returns: profit() for positive, loss() for negative
if total_return >= 0:
    logger.profit(f"Total Return: {total_return:.2f}%")
else:
    logger.loss(f"Total Return: {total_return:.2f}%")

# File ops: success/warning/error as appropriate
logger.success(f"Data saved to: {filepath}")
logger.error(f"Failed to load file: {filepath}")
```

## Writing to Disk

Use `to_file` parameter: `logger.info("msg", to_file=False)` for display only, `to_file=True` to also write to log file.

## DO NOT Use print()

**NEVER** use `print()` for user-facing output. Use `logger.success()`, `logger.info()`, etc. instead.

## Exception Logging - CRITICAL

**NEVER silently swallow exceptions.** Always log with context.

```python
# [-] WRONG: except: pass  OR  except: return None
# [+] CORRECT:
try:
    result = api_call()
except Exception as e:
    logger.error(f"API call failed for {symbol}: {e}")  # Include what, where, why
    raise  # Or return with logged context
```

For stack traces: `logger.error(f"Unexpected error: {e}", exc_info=True)`

**Log levels:** `error()` = failed, `warning()` = recoverable, `info()` = status, `debug()` = detailed

## File Logging for Services

Use `RotatingFileHandler` (10MB, 5 backups). Paths: `~/logs/trading_YYYYMMDD.log`, `~/logs/discord_bot/`

## Temporary print() Only

`print()` allowed ONLY for temporary debugging - remove before committing.
