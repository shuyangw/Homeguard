from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from datetime import date

class RiskConfigModel(BaseModel):
    use_stop_loss: bool = True
    stop_loss_pct: float = 0.02
    stop_loss_type: Literal['percentage', 'atr', 'time', 'profit_target'] = 'percentage'
    take_profit_pct: Optional[float] = None
    max_positions: int = 10
    max_single_position_pct: float = 0.25

class WalkForwardConfig(BaseModel):
    enabled: bool = False
    train_months: int = 12
    test_months: int = 6
    step_months: int = 6

class RollingWindowConfig(BaseModel):
    enabled: bool = False
    window_days: int = 60
    step_days: Optional[int] = None

class BacktestRequest(BaseModel):
    strategy_name: str
    symbols: List[str]
    start_date: str
    end_date: str
    initial_capital: float = 100000.0
    risk_config: Optional[RiskConfigModel] = None
    strategy_params: Dict[str, Any] = Field(default_factory=dict)
    walk_forward: Optional[WalkForwardConfig] = None
    rolling_window: Optional[RollingWindowConfig] = None
    notes: Optional[str] = None

class BacktestResponse(BaseModel):
    run_id: str
    status: str
    message: Optional[str] = None
    progress: float = 0.0

class BacktestResult(BaseModel):
    run_id: str
    metrics: Dict[str, Any]
    equity_curve: List[Dict[str, Any]]
    trades: List[Dict[str, Any]]
    report_url: str
    config: BacktestRequest
