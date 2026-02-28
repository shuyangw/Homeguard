from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd
import asyncio
import json

from src.strategies.registry import list_strategies, list_strategy_display_names, get_strategy_info, _STRATEGY_REGISTRY
from src.web.backend.schemas import BacktestRequest, BacktestResponse
from src.web.backend.core.engine_wrapper import EngineWrapper, ACTIVE_RUNS
from src.web.backend.core.cache import ConfigCache
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()
cache = ConfigCache()

# Use project root for paths (go up from src/web/backend/api/router.py)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
BACKTEST_LISTS_DIR = PROJECT_ROOT / "backtest_lists"

# --- Strategies ---

@router.get("/strategies")
async def get_strategies():
    """Get all available strategies grouped by type based on module path."""
    try:
        strategies = list_strategies()
    except Exception as e:
        logger.error(f"Error listing strategies: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list strategies: {str(e)}")

    # Categorize based on module path from registry
    grouped = {
        "Production": [],
        "Research": []
    }

    # Skip duplicate aliases using the actual module:class tuple from registry
    seen_actual_classes = set()

    for strat_name in strategies:
        try:
            # Get actual module and class name from registry (not the alias)
            if strat_name not in _STRATEGY_REGISTRY:
                continue
            module_path, actual_class_name = _STRATEGY_REGISTRY[strat_name]

            # Skip duplicates (same actual module + actual class)
            actual_key = f"{module_path}:{actual_class_name}"
            if actual_key in seen_actual_classes:
                continue
            seen_actual_classes.add(actual_key)

            # Get full info for the strategy
            info = get_strategy_info(strat_name)

            # Categorize based on module path
            if "advanced" in module_path:
                grouped["Production"].append(info)
            else:
                grouped["Research"].append(info)
        except Exception as e:
            # Skip strategies that fail to load but log the error
            logger.warning(f"Could not load strategy {strat_name}: {e}")
            continue

    logger.info(f"Loaded {len(grouped['Production'])} production and {len(grouped['Research'])} research strategies")
    return grouped

# --- Symbols ---

@router.get("/symbols/lists")
async def get_symbol_lists():
    """Get list of available CSV files in backtest_lists."""
    logger.debug(f"Looking for symbol lists in: {BACKTEST_LISTS_DIR}")
    if not BACKTEST_LISTS_DIR.exists():
        logger.warning(f"Directory does not exist: {BACKTEST_LISTS_DIR}")
        return []

    files = [f.name for f in BACKTEST_LISTS_DIR.glob("*.csv")]
    logger.debug(f"Found {len(files)} symbol list files")
    return sorted(files)

@router.get("/symbols/lists/{filename}")
async def get_symbols_from_list(filename: str):
    """Get symbols from a specific CSV file."""
    file_path = BACKTEST_LISTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="List not found")
    
    try:
        df = pd.read_csv(file_path)
        # Heuristics to find symbol column
        possible_cols = ['Symbol', 'Ticker', 'symbol', 'ticker']
        col = next((c for c in possible_cols if c in df.columns), None)
        
        if col:
            return df[col].tolist()
        else:
             # Fallback: assume first column
             return df.iloc[:, 0].tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Backtest ---

@router.post("/backtest", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """Start a backtest."""
    # Save config to cache
    cache.save_config(request.dict())
    
    # Start task
    run_id = EngineWrapper.start_backtest_task(request)
    
    # Run in background
    background_tasks.add_task(EngineWrapper.run_backtest, run_id, request)
    
    return BacktestResponse(
        run_id=run_id,
        status="queued",
        message="Backtest queued successfully"
    )

@router.get("/backtest/history")
async def get_history():
    """Get configuration history."""
    return cache.get_history()

@router.get("/backtest/{run_id}")
async def get_run_status(run_id: str):
    """Get status of a specific run."""
    if run_id not in ACTIVE_RUNS:
        raise HTTPException(status_code=404, detail="Run not found")
    
    run = ACTIVE_RUNS[run_id]
    return {
        "run_id": run_id,
        "status": run["status"],
        "progress": run["progress"],
        "result": run["result"],
        "error": run.get("error")
    }

# --- WebSocket ---

@router.websocket("/ws/progress/{run_id}")
async def websocket_endpoint(websocket: WebSocket, run_id: str):
    """WebSocket for real-time progress and logs."""
    await websocket.accept()
    
    if run_id not in ACTIVE_RUNS:
        await websocket.close(code=1000, reason="Run not found")
        return

    try:
        last_idx = 0
        while True:
            if run_id not in ACTIVE_RUNS:
                break
                
            run_data = ACTIVE_RUNS[run_id]
            messages = run_data["messages"]
            
            # Send new messages
            if len(messages) > last_idx:
                new_msgs = messages[last_idx:]
                for msg in new_msgs:
                    await websocket.send_json({
                        "type": "log",
                        "data": msg
                    })
                last_idx = len(messages)
            
            # Send progress
            await websocket.send_json({
                "type": "progress",
                "data": run_data["progress"]
            })
            
            # Check for completion
            if run_data["status"] in ["completed", "failed"]:
                # Send one last update
                await websocket.send_json({
                    "type": "status",
                    "data": run_data["status"],
                    "result": run_data.get("result"),
                    "error": run_data.get("error")
                })
                break
            
            await asyncio.sleep(0.5)
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
