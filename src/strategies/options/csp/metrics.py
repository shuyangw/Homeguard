from collections import defaultdict
from typing import Dict, List

from src.strategies.options.csp.position import CSPTrade


def compute_csp_metrics(trades: List[CSPTrade]) -> Dict:
    if not trades:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "avg_premium": 0.0,
            "avg_return_on_collateral": 0.0,
            "avg_holding_days": 0.0,
            "total_pnl": 0.0,
            "pnl_by_exit_reason": {},
            "count_by_exit_reason": {},
            "pnl_by_regime": {},
        }

    total = len(trades)
    winners = [t for t in trades if t.realized_pnl > 0]
    losers = [t for t in trades if t.realized_pnl <= 0]

    pnl_by_exit_reason: Dict[str, float] = defaultdict(float)
    count_by_exit_reason: Dict[str, int] = defaultdict(int)
    pnl_by_regime: Dict[str, float] = defaultdict(float)

    for t in trades:
        pnl_by_exit_reason[t.exit_reason] += t.realized_pnl
        count_by_exit_reason[t.exit_reason] += 1
        pnl_by_regime[t.regime_at_entry] += t.realized_pnl

    return {
        "total_trades": total,
        "winning_trades": len(winners),
        "losing_trades": len(losers),
        "win_rate": len(winners) / total,
        "avg_premium": sum(
            t.entry_price * 100 * t.num_contracts for t in trades
        ) / total,
        "avg_return_on_collateral": sum(
            t.return_on_collateral for t in trades
        ) / total,
        "avg_holding_days": sum(t.holding_days for t in trades) / total,
        "total_pnl": sum(t.realized_pnl for t in trades),
        "pnl_by_exit_reason": dict(pnl_by_exit_reason),
        "count_by_exit_reason": dict(count_by_exit_reason),
        "pnl_by_regime": dict(pnl_by_regime),
    }
