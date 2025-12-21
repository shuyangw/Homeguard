# Web API & Frontend

**Purpose**: Browser-based backtesting interface with REST API backend and React frontend.

---

## Architecture

```
React Frontend (Vite + Tailwind)
    |
    | HTTP REST API
    v
FastAPI Backend
    |
    v
BacktestEngine
    |
    v
Results + Metrics
```

---

## Backend (src/web/backend/)

### Technology Stack
- **FastAPI** - Modern async Python web framework
- **uvicorn** - ASGI server
- **Pydantic** - Request/response validation

### Module Reference

#### `main.py` - FastAPI Application

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Homeguard Backtest API")

# CORS enabled for frontend access
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)
```

#### `api/router.py` - API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/run` | POST | Execute backtest |
| `/strategies` | GET | List available strategies |
| `/symbols` | GET | Get symbol universes |
| `/status/{job_id}` | GET | Check backtest status |

#### `schemas.py` - Request/Response Models

```python
class BacktestRequest(BaseModel):
    strategy: str
    symbols: List[str]
    start_date: str
    end_date: str
    parameters: Dict[str, Any]

class BacktestResponse(BaseModel):
    job_id: str
    status: str
    metrics: Optional[Dict]
    equity_curve: Optional[List]
```

#### `core/engine_wrapper.py` - Engine Integration

Bridges FastAPI to BacktestEngine:
- Async execution wrapper
- Result serialization
- Error handling

#### `core/cache.py` - Response Caching

Caches backtest results to avoid redundant computation.

### Running the Backend

```bash
# Development
uvicorn src.web.backend.main:app --reload --port 8000

# Production
uvicorn src.web.backend.main:app --host 0.0.0.0 --port 8000
```

---

## Frontend (src/web/frontend/)

### Technology Stack
- **React 18** - UI library
- **Vite** - Fast build tool
- **Tailwind CSS** - Utility-first styling

### Components

#### `App.jsx` - Main Application

Root component with routing and state management.

#### `components/ConfigForm.jsx` - Configuration Form

Strategy and parameter configuration:
- Strategy selection dropdown
- Parameter inputs (dynamic based on strategy)
- Date range picker

#### `components/StrategySelector.jsx` - Strategy Selection

Dropdown to select from available strategies:
- OMR, RAMP, HV ORB, etc.
- Shows strategy description

#### `components/SymbolSelector.jsx` - Symbol Selection

Symbol universe selection:
- Preset universes (S&P 500, Russell 1000)
- Custom symbol input
- Multi-select support

#### `components/ResultsDashboard.jsx` - Results Display

Performance metrics and visualization:
- Key metrics table (Sharpe, Return, Drawdown)
- Equity curve chart
- Trade log summary

#### `components/ErrorBoundary.jsx` - Error Handling

Graceful error handling with fallback UI.

### Running the Frontend

```bash
cd src/web/frontend

# Install dependencies
npm install

# Development server
npm run dev

# Build for production
npm run build
```

### Environment Configuration

```javascript
// vite.config.js
export default {
  server: {
    proxy: {
      '/api': 'http://localhost:8000'
    }
  }
}
```

---

## API Usage Examples

### Run a Backtest

```bash
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "MovingAverageCrossover",
    "symbols": ["AAPL", "MSFT"],
    "start_date": "2023-01-01",
    "end_date": "2024-01-01",
    "parameters": {
      "fast_window": 20,
      "slow_window": 50
    }
  }'
```

### List Strategies

```bash
curl http://localhost:8000/strategies
```

Response:
```json
{
  "strategies": [
    {"name": "MovingAverageCrossover", "description": "..."},
    {"name": "OvernightMeanReversion", "description": "..."},
    ...
  ]
}
```

---

## Development

### Full Stack Development

```bash
# Terminal 1 - Backend
uvicorn src.web.backend.main:app --reload --port 8000

# Terminal 2 - Frontend
cd src/web/frontend && npm run dev
```

Frontend available at `http://localhost:5173`
Backend API at `http://localhost:8000`

---

## Related Documentation

- [ARCHITECTURE_OVERVIEW.md](../../docs/architecture/ARCHITECTURE_OVERVIEW.md#layer-5b-web-api--frontend) - System architecture
- [MODULE_REFERENCE.md](../../docs/architecture/MODULE_REFERENCE.md) - Full module reference

---

**Last Updated**: 2025-12-15
