# Backend API

**Purpose**: FastAPI REST API backend for the Homeguard web backtesting interface.

---

## Architecture

```
src/web/backend/
  main.py           # FastAPI application entry point
  schemas.py        # Pydantic request/response models
  api/
    router.py       # API route definitions
  core/
    engine_wrapper.py   # BacktestEngine integration
    cache.py            # Configuration caching
```

---

## API Endpoints

### Strategies

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/strategies` | GET | Get all strategies grouped by Production/Research |

Response:
```json
{
  "Production": [
    {"name": "OvernightMeanReversion", "description": "...", "parameters": {...}}
  ],
  "Research": [...]
}
```

### Symbols

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/symbols/lists` | GET | List available CSV files in config/universes/ |
| `/api/symbols/lists/{filename}` | GET | Get symbols from a specific CSV file |

### Backtest

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/backtest` | POST | Start a new backtest (async) |
| `/api/backtest/{run_id}` | GET | Get status of a specific run |
| `/api/backtest/history` | GET | Get configuration history |

### WebSocket

| Endpoint | Description |
|----------|-------------|
| `/ws/progress/{run_id}` | Real-time progress updates and logs |

---

## Request/Response Models

### BacktestRequest

```python
class BacktestRequest(BaseModel):
    strategy_name: str
    symbols: List[str]
    start_date: str
    end_date: str
    initial_capital: float = 100000.0
    risk_config: Optional[RiskConfigModel] = None
    strategy_params: Dict[str, Any] = {}
    walk_forward: Optional[WalkForwardConfig] = None
    rolling_window: Optional[RollingWindowConfig] = None
    notes: Optional[str] = None
```

### RiskConfigModel

```python
class RiskConfigModel(BaseModel):
    use_stop_loss: bool = True
    stop_loss_pct: float = 0.02
    stop_loss_type: Literal['percentage', 'atr', 'time', 'profit_target']
    take_profit_pct: Optional[float] = None
    max_positions: int = 10
    max_single_position_pct: float = 0.25
```

### BacktestResponse

```python
class BacktestResponse(BaseModel):
    run_id: str
    status: str  # "queued", "running", "completed", "failed"
    message: Optional[str] = None
    progress: float = 0.0
```

---

## Running the Server

```bash
# Development (with auto-reload)
uvicorn src.web.backend.main:app --reload --port 8000

# Production
uvicorn src.web.backend.main:app --host 0.0.0.0 --port 8000
```

Health check: `GET /health` returns `{"status": "ok"}`

---

## CORS Configuration

Allowed origins (development):
- `http://localhost:5173` (Vite dev server)
- `http://127.0.0.1:5173`

---

## Integration

### Engine Wrapper (`core/engine_wrapper.py`)

Bridges FastAPI to BacktestEngine:
- Async task execution
- Progress tracking via `ACTIVE_RUNS` dict
- Result serialization for JSON response

### Configuration Cache (`core/cache.py`)

Caches recent backtest configurations:
- Stores last N configurations
- Enables "run again" functionality

---

## Related Documentation

- [WEB.md](../WEB.md) - Full stack overview
- [FRONTEND.md](../frontend/FRONTEND.md) - React frontend

---

**Last Updated**: 2025-12-21
