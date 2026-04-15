# Web & Server Development Guidelines

## Critical Safety Rule

**NEVER kill all node processes indiscriminately when restarting web servers.**

### Why This Matters
- Claude Code runs as a Node.js process
- Running `taskkill /f /im node.exe` or `pkill node` will kill Claude Code itself
- This causes loss of conversation context and work in progress

## Safe Server Management

### Starting the Web UI
```batch
scripts\start_web_ui.bat
```

### Stopping Servers Safely
- Use `Ctrl+C` in the terminal running the servers
- Or close the specific terminal window
- **NEVER** use blanket process kills like `taskkill /f /im node.exe`

### If a Port is in Use
```batch
:: Find what's using port 8000 (backend)
netstat -ano | findstr :8000

:: Find what's using port 5173 (frontend)
netstat -ano | findstr :5173

:: Kill specific PID (not all node processes)
taskkill /f /pid <PID>
```

## Web UI Architecture

### Backend (FastAPI)
- **Port**: 8000
- **Location**: `src/web/backend/`
- **Entry point**: `src/web/backend/main.py`
- **API routes**: `src/web/backend/api/router.py`

**Key endpoints:**
- `GET /api/strategies` - List all strategies (grouped by Production/Research)
- `GET /api/symbols/lists` - List available symbol CSV files
- `GET /api/symbols/lists/{filename}` - Get symbols from a specific list
- `POST /api/backtest` - Start a backtest
- `WS /ws/progress/{run_id}` - WebSocket for real-time progress

**Data sources:**
- Strategies loaded from `src/strategies/registry.py`
- Symbol lists from `backtest_lists/*.csv`
- Must use fintech conda environment

### Frontend (Vite + React)
- **Port**: 5173
- **Location**: `src/web/frontend/`
- **Components**: `src/web/frontend/src/components/`

**Key components:**
- `App.jsx` - Main app with navigation
- `ConfigForm.jsx` - Backtest configuration form
- `StrategySelector.jsx` - Strategy dropdown with Production/Research groups
- `SymbolSelector.jsx` - Manual input or CSV list selection
- `ResultsDashboard.jsx` - Progress, metrics, and equity curve

**E2E tests:**
```bash
cd src/web/frontend
npm run test:e2e           # Run tests
npm run test:e2e:headed    # Run with browser visible
npm run test:e2e:debug     # Debug mode
```

## Common Issues

### "Failed to load strategies"
**Cause**: Backend not running or using wrong Python environment

**Fix**:
1. Ensure backend is running: `curl http://localhost:8000/health`
2. Check the backend terminal for errors
3. Verify fintech environment: The batch script should use `C:\Users\qwqw1\anaconda3\envs\fintech\python.exe`

### "No symbol lists available"
**Cause**: Path resolution issue in backend

**Fix**:
- Backend uses `PROJECT_ROOT / "backtest_lists"` where PROJECT_ROOT is calculated from the router.py file location
- Verify `backtest_lists/` directory exists with CSV files

### CORS Errors
**Cause**: Frontend origin not allowed

**Fix**:
- Check `allow_origins` in `src/web/backend/main.py`
- Should include `http://localhost:5173` and `*` for development

### Port Already in Use
```batch
:: Find and kill specific process
netstat -ano | findstr :8000
taskkill /f /pid <PID_FROM_ABOVE>
```

## Development Workflow

1. **Start servers**: `scripts\start_web_ui.bat`
2. **Make changes**: Edit frontend/backend code
3. **Auto-reload**: Both servers support hot reload
4. **Test**: Open http://localhost:5173
5. **Debug**: Check browser console (F12) and backend terminal
6. **Stop**: Press `Ctrl+C` in the terminal (NEVER kill all node processes)

## Testing

### Backend API Testing
```bash
# Health check
curl http://localhost:8000/health

# Get strategies
curl http://localhost:8000/api/strategies

# Get symbol lists
curl http://localhost:8000/api/symbols/lists
```

### Frontend E2E Testing
```bash
cd src/web/frontend
npm run test:e2e
```

Tests are in `src/web/frontend/tests/e2e/app.spec.js`
