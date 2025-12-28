import asyncio
import uuid
import traceback
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from datetime import datetime
import pandas as pd

from src.backtesting.engine.backtest_engine import BacktestEngine
from src.backtesting.utils.risk_config import RiskConfig
from src.strategies.registry import get_strategy_class
from src.web.backend.schemas import BacktestRequest, BacktestResult, RiskConfigModel
from src.utils import logger
from src.settings import get_backtest_results_dir

# Global registry of active runs to allow WebSocket attachment
# run_id -> { "status": str, "messages": [], "progress": 0.0, "result": None }
ACTIVE_RUNS: Dict[str, Dict[str, Any]] = {}

class WebSocketLogHandler:
    """Captures logs for a specific run_id"""
    def __init__(self, run_id: str):
        self.run_id = run_id

    def emit(self, message: str, level: str = "INFO"):
        if self.run_id in ACTIVE_RUNS:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "message": message,
                "level": level
            }
            ACTIVE_RUNS[self.run_id]["messages"].append(entry)

class EngineWrapper:
    @staticmethod
    def start_backtest_task(request: BacktestRequest) -> str:
        """Starts a backtest in a background thread/task."""
        run_id = str(uuid.uuid4())
        ACTIVE_RUNS[run_id] = {
            "status": "running",
            "messages": [],
            "progress": 0.0,
            "result": None,
            "config": request.dict()
        }
        
        # In a real app, use Celery or BackgroundTasks. 
        # For this setup, we'll rely on FastAPI's BackgroundTasks injected in the router,
        # but here we just return the ID and let the router call run_backtest.
        return run_id

    @staticmethod
    def run_backtest(run_id: str, request: BacktestRequest):
        """
        Executes the backtest. strict synchronous execution 
        (to be run in threadpool by FastAPI).
        """
        log_handler = WebSocketLogHandler(run_id)
        
        try:
            log_handler.emit(f"Starting backtest for {request.strategy_name}...", "HEADER")
            ACTIVE_RUNS[run_id]["progress"] = 0.05

            # 1. Setup Risk Config
            risk_config = None
            if request.risk_config:
                rc = request.risk_config
                risk_config = RiskConfig(
                    use_stop_loss=rc.use_stop_loss,
                    stop_loss_pct=rc.stop_loss_pct,
                    stop_loss_type=rc.stop_loss_type,
                    take_profit_pct=rc.take_profit_pct,
                    max_positions=rc.max_positions,
                    max_single_position_pct=rc.max_single_position_pct
                )
            else:
                risk_config = RiskConfig.moderate()

            # 2. Initialize Engine
            engine = BacktestEngine(
                initial_capital=request.initial_capital,
                risk_config=risk_config,
                fees=0.001,  # Default
                slippage=0.0 # Default
            )

            # 3. Load Strategy
            strategy_cls = get_strategy_class(request.strategy_name)
            
            # 4. Run Logic
            result_artifacts = {}
            
            if request.walk_forward and request.walk_forward.enabled:
                log_handler.emit("Mode: Walk-Forward Validation", "INFO")
                # Import here to avoid circular deps
                from src.backtesting.optimization.walk_forward import WalkForwardOptimizer
                from src.backtesting.optimization.grid_search import GridSearchOptimizer
                
                # We need a custom optimizer to hook into progress
                # For now, we'll just run it and catch logs via the global logger interception (if we implemented it)
                # Or simply wrapper calls.
                
                # NOTE: The current WalkForwardOptimizer prints to console. 
                # Ideally we'd patch it or pass a callback.
                
                base_opt = GridSearchOptimizer(engine)
                wf_opt = WalkForwardOptimizer(engine, base_opt)
                
                ACTIVE_RUNS[run_id]["progress"] = 0.1
                
                # Mock parameter space if not provided (safety check)
                params = request.strategy_params or {}
                # If params are single values, make them lists for optimization compatibility
                # But WF requires a param_space (grid).
                # Assumption: Frontend sends lists for optimization.
                # If frontend sends single values (e.g. standard run params), WF doesn't make sense unless we define a range.
                # For this implementation, we assume params are properly formatted or we wrap them.
                
                wf_results = wf_opt.analyze(
                    strategy_class=strategy_cls,
                    param_space=params, 
                    symbols=request.symbols,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    train_months=request.walk_forward.train_months,
                    test_months=request.walk_forward.test_months,
                    step_months=request.walk_forward.step_months
                )
                
                ACTIVE_RUNS[run_id]["progress"] = 0.9
                log_handler.emit("Walk-Forward Analysis Complete", "SUCCESS")
                
                # We can bundle result url = the folder path
                # wf_results is a dict
                
                results_path = get_backtest_results_dir() # Simplified
                
            else:
                # Prepare rolling window arguments
                run_mode = 'batch'
                r_window = 60
                r_step = None
                
                if request.rolling_window and request.rolling_window.enabled:
                    run_mode = 'rolling'
                    r_window = request.rolling_window.window_days
                    r_step = request.rolling_window.step_days
                    log_handler.emit(f"Mode: Rolling Window (W={r_window}, S={r_step})", "INFO")
                else:
                    log_handler.emit("Mode: Standard Backtest", "INFO")

                strategy = strategy_cls(**request.strategy_params)
                
                ACTIVE_RUNS[run_id]["progress"] = 0.2
                log_handler.emit("Loading data...", "INFO")
                
                # Provide progress updates via standard engine
                # Since engine.run is blocking, we jump to 0.8 afterwards
                result = engine.run(
                    strategy=strategy,
                    symbols=request.symbols,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    mode=run_mode,
                    window_days=r_window,
                    step_days=r_step
                )
                
                ACTIVE_RUNS[run_id]["progress"] = 0.8
                log_handler.emit("Backtest simulation complete. Generating report...", "INFO")
                
                # Generate Report
                from src.backtesting.engine.rolling_results import RollingWindowResults
                
                if isinstance(result, RollingWindowResults):
                    stats = result.combined_stats
                    if not stats:
                         stats = result.compute_combined_stats()
                else:
                    # Standard Portfolio
                    stats = result.stats()
                    
                result_path = engine.get_results_dir(strategy_cls.__name__, request.symbols)
                report_file = result_path / "report.html"
                
                # Use engine's internal reporting or QuantStats
                # The engine likely has a method or we use result.plot()
                # From usage guide: result.plot() opens browser. We want to save it.
                # result.plot(filename=...) if supported, else we rely on stats.
                
                # Manually generating simple HTML or JSON results
                import json
                metrics = {k: str(v) for k, v in stats.items()} # Convert numpy types
                
                ACTIVE_RUNS[run_id]["result"] = {
                    "metrics": metrics,
                    "report_path": str(report_file) if report_file.exists() else ""
                }

            ACTIVE_RUNS[run_id]["progress"] = 1.0
            ACTIVE_RUNS[run_id]["status"] = "completed"
            log_handler.emit("Task completed successfully", "SUCCESS")

        except Exception as e:
            ACTIVE_RUNS[run_id]["status"] = "failed"
            ACTIVE_RUNS[run_id]["error"] = str(e)
            log_handler.emit(f"Error: {str(e)}", "ERROR")
            traceback.print_exc()

