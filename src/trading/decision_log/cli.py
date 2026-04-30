"""CLI entry point: python -m src.trading.decision_log <cmd>.

Subcommands: show, list, grep, explain, summary.
ASCII-only output per project convention.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from typing import List, Optional

from src.trading.decision_log.reader import (
    latest, by_id, for_date, iter_records, filter_records, summary,
)
from src.trading.decision_log.record import DecisionRecord


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.trading.decision_log")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_show = sub.add_parser("show", help="Pretty-print a record")
    p_show.add_argument("strategy", choices=["ramp", "omr", "mp", "cscm"])
    p_show.add_argument("--date", default=None, help="YYYY-MM-DD; defaults to latest")
    p_show.add_argument("--id", dest="decision_id", default=None)
    p_show.add_argument("--json", action="store_true", help="Raw JSON instead of pretty")
    p_show.set_defaults(func=cmd_show)

    p_list = sub.add_parser("list", help="Table summary of recent decisions")
    p_list.add_argument("strategy", choices=["ramp", "omr", "mp", "cscm"])
    p_list.add_argument("--days", type=int, default=7)
    p_list.set_defaults(func=cmd_list)

    p_grep = sub.add_parser("grep", help="Filter records by predicate")
    p_grep.add_argument("strategy", choices=["ramp", "omr", "mp", "cscm"])
    p_grep.add_argument("--where", required=True, help="DSL: 'regime=BEAR', 'executions:length=0', etc.")
    p_grep.add_argument("--days", type=int, default=30)
    p_grep.set_defaults(func=cmd_grep)

    p_explain = sub.add_parser("explain", help="Show why a symbol decision was made")
    p_explain.add_argument("strategy", choices=["ramp", "omr", "mp", "cscm"])
    p_explain.add_argument("--symbol", required=True)
    p_explain.add_argument("--date", required=True, help="YYYY-MM-DD")
    p_explain.set_defaults(func=cmd_explain)

    p_summary = sub.add_parser("summary", help="Cross-strategy daily aggregates")
    p_summary.add_argument("--days", type=int, default=7)
    p_summary.set_defaults(func=cmd_summary)

    args = parser.parse_args(argv)
    return args.func(args)


def cmd_show(args) -> int:
    if args.decision_id:
        rec = by_id(args.decision_id)
    elif args.date:
        day = date.fromisoformat(args.date)
        records = for_date(args.strategy, day)
        rec = records[-1] if records else None
    else:
        rec = latest(args.strategy)

    if rec is None:
        print(f"No records found for {args.strategy}", file=sys.stderr)
        return 1

    if args.json:
        print(rec.to_jsonl_line().strip())
    else:
        print(_render_record(rec))
    return 0


def cmd_list(args) -> int:
    today = date.today()
    since = today - timedelta(days=args.days - 1)
    records = list(iter_records(args.strategy, since=since, until=today))

    if not records:
        print(f"No records for {args.strategy} in last {args.days} days", file=sys.stderr)
        return 1

    print(f"{'DATE':<11} {'TIME':<9} {'TRIGGER':<14} {'REGIME':<13} {'TARGETS':<8} {'EXEC':<6} STATUS")
    for rec in records:
        ts = datetime.fromisoformat(rec.timestamp)
        regime = rec.inputs.regime or "-"
        targets = len(rec.logic_decisions.target_symbols) if rec.logic_decisions else 0
        filled = sum(1 for e in rec.executions if e.status == "filled")
        total = len(rec.executions)
        if rec.error is not None:
            status = f"errored: {rec.error.stage}"
        elif not rec.preconditions.all_passed:
            status = "blocked: precondition"
        elif total == 0:
            status = "no executions"
        elif filled == total:
            status = "clean"
        else:
            status = f"partial ({total - filled} skipped)"
        print(
            f"{ts.date()!s:<11} {ts.strftime('%H:%M:%S'):<9} "
            f"{rec.trigger.kind[:14]:<14} {regime[:13]:<13} "
            f"{targets:<8} {filled}/{total:<5} {status}"
        )
    return 0


def cmd_grep(args) -> int:
    today = date.today()
    since = today - timedelta(days=args.days - 1)
    records = iter_records(args.strategy, since=since, until=today)
    matches = list(filter_records(records, args.where))

    if not matches:
        print(f"No matches for {args.where!r}", file=sys.stderr)
        return 1

    for rec in matches:
        print(_render_record(rec))
        print()
    return 0


def cmd_explain(args) -> int:
    day = date.fromisoformat(args.date)
    records = for_date(args.strategy, day)
    if not records:
        print(f"No records for {args.strategy} on {args.date}", file=sys.stderr)
        return 1

    found_any = False
    for rec in records:
        target_present = (
            rec.logic_decisions
            and args.symbol in rec.logic_decisions.target_symbols
        )
        execution = next((e for e in rec.executions if e.symbol == args.symbol), None)
        if not target_present and execution is None:
            continue
        found_any = True
        ts = datetime.fromisoformat(rec.timestamp)
        print(f"[{rec.strategy.upper()} {ts}] {args.symbol}")
        if target_present and rec.logic_decisions:
            tw = rec.logic_decisions.target_weights.get(args.symbol, 0.0)
            tv = rec.logic_decisions.target_value_usd.get(args.symbol, 0.0)
            print(f"  Logic decision: target_weight={tw:.4f}, target_value_usd=${tv:,.2f}")
        if execution:
            print(f"  Execution: action={execution.action}, status={execution.status}")
            if execution.reason:
                print(f"    Reason: {execution.reason!r}")
            if execution.fill_price is not None:
                print(f"    Fill: qty={execution.filled_qty} @ ${execution.fill_price}")
        elif target_present:
            print("  Execution: NOT ATTEMPTED (no execution entry for this symbol)")
        # Preconditions context
        if not rec.preconditions.all_passed:
            failed = [
                name for name in (
                    "strategy_enabled", "shutdown_requested",
                    "execution_lock_acquired", "health_check", "data_freshness",
                ) if not getattr(rec.preconditions, name).passed
            ]
            print(f"  Preconditions: BLOCKED ({', '.join(failed)})")
        print()
    if not found_any:
        print(f"{args.symbol} not found in any decision on {args.date}", file=sys.stderr)
        return 1
    return 0


def cmd_summary(args) -> int:
    today = date.today()
    since = today - timedelta(days=args.days - 1)
    aggregates = summary(start=since, end=today)
    if not aggregates:
        print(f"No records in last {args.days} days", file=sys.stderr)
        return 1
    strategies = sorted(aggregates.keys())
    cols = ["triggers", "clean", "blocked", "errored", "orders_filled", "orders_total"]
    print(" " * 14 + "".join(f"{s:<11}" for s in strategies))
    for col in cols:
        row = f"{col:<14}"
        for s in strategies:
            row += f"{getattr(aggregates[s], col):<11}"
        print(row)
    return 0


def _render_record(rec: DecisionRecord) -> str:
    """Pretty-print a record in ASCII box format."""
    ts = datetime.fromisoformat(rec.timestamp)
    lines: List[str] = []
    title = f" {rec.strategy} | {ts.strftime('%Y-%m-%d %H:%M:%S')} | {rec.trigger.kind} "
    width = max(len(title) + 4, 60)
    lines.append("+" + "-" * (width - 2) + "+")
    lines.append("| " + title.center(width - 4) + " |")
    git = rec.metadata.git_sha[:7] if rec.metadata.git_sha else "(none)"
    sub = f"decision_id: {rec.decision_id[:8]} | git: {git} | broker: {rec.metadata.broker_name}"
    lines.append("| " + sub.center(width - 4) + " |")
    lines.append("+" + "-" * (width - 2) + "+")
    lines.append("")

    lines.append(
        f"Trigger fired {rec.trigger.delay_seconds:.2f}s after schedule "
        f"({rec.trigger.schedule_time or '-'})."
    )
    lines.append("")

    # Preconditions
    if rec.preconditions.all_passed:
        lines.append(f"{'Preconditions:':<48} ALL PASS")
    else:
        failing = [
            name for name in (
                "strategy_enabled", "shutdown_requested",
                "execution_lock_acquired", "health_check", "data_freshness",
            ) if not getattr(rec.preconditions, name).passed
        ]
        lines.append(f"{'Preconditions:':<48} BLOCKED ({', '.join(failing)})")
    for name in ("strategy_enabled", "shutdown_requested",
                 "execution_lock_acquired", "health_check", "data_freshness"):
        gate = getattr(rec.preconditions, name)
        sym = "[+]" if gate.passed else "[-]"
        detail = ""
        if name == "health_check" and gate.details.get("buying_power"):
            detail = f" (buying_power=${gate.details['buying_power']:,.0f})"
        elif name == "data_freshness" and gate.details.get("valid_pct"):
            detail = f" ({gate.details['valid_pct']:.1f}% valid, source={rec.inputs.cache_source})"
        lines.append(f"  {sym} {name}{detail}")
    lines.append("")

    # Inputs
    lines.append("Inputs:")
    if rec.inputs.regime:
        conf = (
            f" ({rec.inputs.regime_confidence * 100:.1f}% confidence)"
            if rec.inputs.regime_confidence else ""
        )
        lines.append(f"  regime         {rec.inputs.regime}{conf}")
    if rec.inputs.vix is not None:
        spy_dd = (
            f"spy_dd  {rec.inputs.spy_drawdown_pct:.1f}%"
            if rec.inputs.spy_drawdown_pct is not None else ""
        )
        lines.append(f"  vix            {rec.inputs.vix:<30} {spy_dd}")
    if rec.inputs.universe_size:
        lines.append(
            f"  universe       {rec.inputs.universe_size} symbols"
            f"                    completeness {rec.inputs.data_completeness_pct:.1f}%"
        )
    if rec.inputs.momentum_scores:
        top5 = sorted(
            (
                (s, v) for s, v in rec.inputs.momentum_scores.items()
                if v is not None
            ),
            key=lambda x: x[1], reverse=True,
        )[:5]
        if top5:
            top_str = " | ".join(f"{s} {v:+.2f}" for s, v in top5)
            lines.append(f"  top 5 momentum {top_str}")
    lines.append("")

    # Logic decisions
    if rec.logic_decisions:
        ld = rec.logic_decisions
        n = ld.top_n
        per_pos = sum(ld.target_value_usd.values()) / n if n else 0
        total_target = sum(ld.target_value_usd.values())
        pct = (ld.exposure_pct * 100) if n else 0
        lines.append("Logic:")
        lines.append(
            f"  Target {n} positions x ${per_pos:,.0f} each = "
            f"${total_target:,.0f} ({pct:.0f}% of initial_capital)"
        )
        if ld.target_symbols:
            lines.append(f"  Buy:  {' '.join(ld.target_symbols[:n])}")
        if ld.exit_signals:
            lines.append(f"  Sell: {' '.join(ld.exit_signals)}")
        lines.append("")

    # Executions
    filled = sum(1 for e in rec.executions if e.status == "filled")
    lines.append(f"{'Executions:':<48} {filled} / {len(rec.executions)} filled")
    for ex in rec.executions:
        sym = "[+]" if ex.status == "filled" else "[-]"
        if ex.status == "filled":
            lines.append(
                f"  {sym} {ex.symbol:<7} {ex.action} {ex.target_qty:<5} "
                f"@ ${ex.fill_price:.2f}  fill_price=${ex.fill_price:.2f}  "
                f"order={ex.order_id or '-'}"
            )
        else:
            reason = f": {ex.reason}" if ex.reason else ""
            lines.append(f"  {sym} {ex.symbol:<7} {ex.action} {ex.target_qty:<5} {ex.status}{reason}")
    lines.append("")

    # Post-state
    if rec.post_state:
        n_pos = len(rec.post_state.positions_after)
        lines.append("Post-state:")
        lines.append(f"  Positions:   {n_pos} {rec.strategy.upper()}-tagged")
        lines.append(
            f"  Cash:        ${rec.post_state.cash_after:,.2f} / "
            f"${rec.metadata.initial_capital_usd:,.2f} budget"
        )
        lines.append(f"  Strategy equity: ${rec.post_state.strategy_equity_after:,.2f}")

    # Error (if any)
    if rec.error:
        lines.append("")
        lines.append(f"ERROR ({rec.error.stage}): {rec.error.type}: {rec.error.message}")

    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
