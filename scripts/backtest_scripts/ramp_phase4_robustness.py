#!/usr/bin/env python3
"""Phase 6.5 parameter robustness runner for V28 and V31.

This is a ROBUSTNESS MAP, NOT an optimization search.

Purpose: characterize whether the a-priori center config (V28 or V31) sits on a
stable plateau or a knife-edge by running a neighborhood grid and measuring the
ratio min(neighbor_sharpe) / center_sharpe.

Verdict (methodology Section 5.5):
  STABLE  : all neighbors hold >= 0.9 of center Sharpe across the swept range
  BRITTLE : any neighbor drops below 0.9 of center Sharpe (cliff-edge = overfit risk)

Non-negotiable rules (see strategy-lead Section 2, Phase 6.5):
  1. The center config is FIXED a-priori. It is never updated by this run.
  2. The best-variation is reported as a flag only; it is NEVER adopted.
  3. Every variation is logged to the experiment registry (phase='robustness_sensitivity').
  4. Stability metric = full-window net Sharpe at near_close, 5 bps.
  5. Trial-count treatment: stability-sweep variations are NOT added to the
     project SELECTION-trial count (they are never promoted). If one is later
     promoted as its own a-priori candidate, it receives its own trial entry
     at that point.

Usage:
    python scripts/backtest_scripts/ramp_phase4_robustness.py --variant V28
    python scripts/backtest_scripts/ramp_phase4_robustness.py --variant V31
    python scripts/backtest_scripts/ramp_phase4_robustness.py --variant V28 --smoke
    python scripts/backtest_scripts/ramp_phase4_robustness.py --variant V31 --budget 25
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path for direct execution.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.experiments.registry import (
    DEFAULT_DB_PATH,
    _git_sha as _registry_git_sha,
    append_run,
    init_db,
)
from src.research.ramp_phase4.config import HarnessConfig
from src.research.ramp_phase4.engine import DailyRecord, run_variant
from src.research.ramp_phase4.metrics import (
    avg_daily_turnover,
    cagr,
    max_drawdown,
    regime_attribution,
    sharpe_ratio,
)
from src.research.ramp_phase4.variants import (
    REGISTRY,
    VariantSpec,
    _V28_CENTER_BLEND_21D,
    _V28_CENTER_BLEND_63D,
    _V28_CENTER_BLEND_126D,
    _V28_CENTER_H_21,
    _V28_CENTER_H_63,
    _V28_CENTER_H_126,
    _V28_CENTER_LAMBDA_REV,
    _V31_CENTER_BETA_WINDOW,
    _V31_CENTER_RESIDUAL_HORIZON,
    make_v28_plan_fn,
    make_v31_plan_fn,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

DATA_SNAPSHOT_DATE = datetime(2026, 5, 16).date()

# Reference center Sharpes from the family gate (near_close 5bps full window).
_V28_CENTER_SHARPE_REF = 0.811
_V31_CENTER_SHARPE_REF = 0.769

# Stability threshold: neighbor must hold >= this fraction of center Sharpe.
STABILITY_THRESHOLD = 0.9


@dataclass
class VariationResult:
    """Result for one variation point in the neighborhood grid."""
    label: str
    params: Dict[str, Any]
    sharpe: float
    cagr_val: float
    max_dd: float
    run_id: str
    is_center: bool = False
    elapsed_s: float = 0.0


def _build_v31_grid(budget: int = 25) -> List[Dict[str, Any]]:
    """Build the V31 robustness grid: 5x5 = 25 points.

    beta_window in {72, 81, 90, 99, 108} (center=90, +/-10%, +/-20%)
    residual_horizon in {17, 19, 21, 23, 25} (center=21, +/-10%, +/-20% approx)

    Returns list of param dicts with keys: beta_window, residual_horizon, label, is_center.
    """
    bw_vals = [72, 81, 90, 99, 108]
    rh_vals = [17, 19, 21, 23, 25]
    grid = []
    for bw in bw_vals:
        for rh in rh_vals:
            is_center = (bw == _V31_CENTER_BETA_WINDOW and rh == _V31_CENTER_RESIDUAL_HORIZON)
            label = f"bw{bw}_rh{rh}"
            if is_center:
                label = f"CENTER_{label}"
            grid.append({
                'beta_window': bw,
                'residual_horizon': rh,
                'label': label,
                'is_center': is_center,
            })
    assert len(grid) <= budget, f"V31 grid size {len(grid)} exceeds budget {budget}"
    return grid


def _build_v28_grid(budget: int = 25) -> List[Dict[str, Any]]:
    """Build the V28 robustness grid: exactly 25 points.

    Structure:
      1  center: (w21=0.5, w63=0.3, w126=0.2, lam=0.1, h21=21, h63=63, h126=126)
      4  w21 one-at-a-time: +/-10%, +/-20% of 0.5 (renormalize; w63/w126 proportional)
      4  w63 one-at-a-time: +/-10%, +/-20% of 0.3 (renormalize; w21/w126 proportional)
      4  w126 one-at-a-time: +/-10%, +/-20% of 0.2 (renormalize; w21/w63 proportional)
      4  lambda_rev one-at-a-time: +/-10%, +/-20% of 0.1
      4  h21 one-at-a-time: 19, 17, 23, 25
      4  h63 one-at-a-time: 57, 51, 69, 76 (63*0.9, 63*0.8, 63*1.1, 63*1.2)
    Total: 1 + 24 = 25

    Weights are stored pre-normalization but passed to make_v28_plan_fn as-is
    (the factory renormalizes internally). For clarity we record both the raw
    passed values and the normalized effective values in the label.
    """
    center_w21 = _V28_CENTER_BLEND_21D    # 0.5
    center_w63 = _V28_CENTER_BLEND_63D    # 0.3
    center_w126 = _V28_CENTER_BLEND_126D  # 0.2
    center_lam = _V28_CENTER_LAMBDA_REV   # 0.1
    center_h21 = _V28_CENTER_H_21         # 21
    center_h63 = _V28_CENTER_H_63         # 63
    center_h126 = _V28_CENTER_H_126       # 126

    def _pt(label, w21, w63, w126, lam, h21, h63, h126, is_center=False):
        return {
            'blend_21d': w21,
            'blend_63d': w63,
            'blend_126d': w126,
            'lambda_rev': lam,
            'h_21': h21,
            'h_63': h63,
            'h_126': h126,
            'label': label,
            'is_center': is_center,
        }

    grid = []

    # 1. Center point.
    grid.append(_pt(
        label=f"CENTER_w21={center_w21}_w63={center_w63}_w126={center_w126}_lam={center_lam}",
        w21=center_w21, w63=center_w63, w126=center_w126,
        lam=center_lam, h21=center_h21, h63=center_h63, h126=center_h126,
        is_center=True,
    ))

    # 2. w21 perturbations (4 points): shift w21, keep w63/w126 ratio constant.
    for delta_pct, tag in [(-0.10, 'm10'), (-0.20, 'm20'), (+0.10, 'p10'), (+0.20, 'p20')]:
        new_w21 = center_w21 * (1 + delta_pct)
        # Redistribute remainder proportionally to w63/w126.
        remainder = 1.0 - new_w21
        total_rest = center_w63 + center_w126
        new_w63 = center_w63 * remainder / total_rest
        new_w126 = center_w126 * remainder / total_rest
        grid.append(_pt(
            label=f"w21_{tag}_{new_w21:.3f}_w63={new_w63:.3f}_w126={new_w126:.3f}",
            w21=new_w21, w63=new_w63, w126=new_w126,
            lam=center_lam, h21=center_h21, h63=center_h63, h126=center_h126,
        ))

    # 3. w63 perturbations (4 points): shift w63, keep w21/w126 ratio constant.
    for delta_pct, tag in [(-0.10, 'm10'), (-0.20, 'm20'), (+0.10, 'p10'), (+0.20, 'p20')]:
        new_w63 = center_w63 * (1 + delta_pct)
        remainder = 1.0 - new_w63
        total_rest = center_w21 + center_w126
        new_w21 = center_w21 * remainder / total_rest
        new_w126 = center_w126 * remainder / total_rest
        grid.append(_pt(
            label=f"w63_{tag}_{new_w63:.3f}_w21={new_w21:.3f}_w126={new_w126:.3f}",
            w21=new_w21, w63=new_w63, w126=new_w126,
            lam=center_lam, h21=center_h21, h63=center_h63, h126=center_h126,
        ))

    # 4. w126 perturbations (4 points): shift w126, keep w21/w63 ratio constant.
    for delta_pct, tag in [(-0.10, 'm10'), (-0.20, 'm20'), (+0.10, 'p10'), (+0.20, 'p20')]:
        new_w126 = center_w126 * (1 + delta_pct)
        remainder = 1.0 - new_w126
        total_rest = center_w21 + center_w63
        new_w21 = center_w21 * remainder / total_rest
        new_w63 = center_w63 * remainder / total_rest
        grid.append(_pt(
            label=f"w126_{tag}_{new_w126:.3f}_w21={new_w21:.3f}_w63={new_w63:.3f}",
            w21=new_w21, w63=new_w63, w126=new_w126,
            lam=center_lam, h21=center_h21, h63=center_h63, h126=center_h126,
        ))

    # 5. lambda_rev perturbations (4 points).
    for delta_pct, tag in [(-0.10, 'm10'), (-0.20, 'm20'), (+0.10, 'p10'), (+0.20, 'p20')]:
        new_lam = center_lam * (1 + delta_pct)
        grid.append(_pt(
            label=f"lam_{tag}_{new_lam:.4f}",
            w21=center_w21, w63=center_w63, w126=center_w126,
            lam=new_lam, h21=center_h21, h63=center_h63, h126=center_h126,
        ))

    # 6. h21 horizon perturbations (4 points): -10%, -20%, +10%, +20%.
    for new_h21, tag in [(19, 'm10'), (17, 'm20'), (23, 'p10'), (25, 'p20')]:
        grid.append(_pt(
            label=f"h21_{tag}_{new_h21}d",
            w21=center_w21, w63=center_w63, w126=center_w126,
            lam=center_lam, h21=new_h21, h63=center_h63, h126=center_h126,
        ))

    # 7. h63 horizon perturbations (4 points): 63*0.9=57, 63*0.8=50, 63*1.1=69, 63*1.2=76.
    for new_h63, tag in [(57, 'm10'), (50, 'm20'), (69, 'p10'), (76, 'p20')]:
        grid.append(_pt(
            label=f"h63_{tag}_{new_h63}d",
            w21=center_w21, w63=center_w63, w126=center_w126,
            lam=center_lam, h21=center_h21, h63=new_h63, h126=center_h126,
        ))

    assert len(grid) == 25, f"V28 grid size {len(grid)} != 25"
    assert len(grid) <= budget, f"V28 grid size {len(grid)} exceeds budget {budget}"
    return grid


def _build_cfg(
    args,
    cost_bps: float = 5.0,
    timing_mode: str = 'near_close',
) -> HarnessConfig:
    return HarnessConfig(
        start_date=args.start,
        end_date=args.end,
        universe_csv=args.universe,
        initial_capital=args.initial_capital,
        cost_bps_per_side=cost_bps,
        timing_mode=timing_mode,
        delta_rebalance_pct=0.02,  # Same as V11/V28/V31 per readiness runner.
    )


def _compute_metrics(records: List[DailyRecord]) -> Dict:
    rets = pd.Series([r.daily_return for r in records])
    eq = (1 + rets).cumprod() * records[0].portfolio_value if records else pd.Series()
    sr = sharpe_ratio(rets)
    cagr_val = cagr(eq)
    max_dd = max_drawdown(eq)
    avg_to = avg_daily_turnover(records)
    return {
        'sharpe': float(sr),
        'cagr': float(cagr_val),
        'max_dd': float(max_dd),
        'avg_daily_turnover': float(avg_to),
        'n_records': int(len(records)),
    }


def _run_one_variation(
    args,
    variant_id: str,
    params: Dict[str, Any],
    label: str,
    cfg: HarnessConfig,
    git_sha: str,
    is_center: bool,
) -> VariationResult:
    """Run a single variation through the engine and log to registry."""
    if variant_id == 'V28':
        plan_fn = make_v28_plan_fn(
            blend_21d=params['blend_21d'],
            blend_63d=params['blend_63d'],
            blend_126d=params['blend_126d'],
            lambda_rev=params['lambda_rev'],
            h_21=params['h_21'],
            h_63=params['h_63'],
            h_126=params['h_126'],
        )
    elif variant_id == 'V31':
        plan_fn = make_v31_plan_fn(
            beta_window=params['beta_window'],
            residual_horizon=params['residual_horizon'],
        )
    else:
        raise ValueError(f"Unsupported variant: {variant_id}")

    variation_spec = VariantSpec(
        id=f'{variant_id}_robustness_{label}',
        description=f'Robustness sweep variation: {label}',
        plan_fn=plan_fn,
    )

    t0 = time.time()
    records = run_variant(cfg, variation_spec)
    elapsed = time.time() - t0

    if not records:
        raise RuntimeError(f'Variation {label} produced no records')

    metrics = _compute_metrics(records)
    regime_attr = regime_attribution(records)

    return_stream = pd.DataFrame({
        'date': [r.date for r in records],
        'return_pct': [r.daily_return for r in records],
        'position_count': [len(r.realized_weights) for r in records],
    })

    wall_start = datetime.now(tz=timezone.utc) - timedelta(seconds=elapsed)
    wall_end = datetime.now(tz=timezone.utc)

    # Registry write: MANDATORY per Section 9.3.
    # phase='robustness_sensitivity' as required by Phase 6.5 rules.
    # Trial-count treatment: these variations are NEVER promoted, so they do NOT
    # add to the project SELECTION-trial count (see Phase 6.5 rule 4).
    run_id = append_run(
        strategy_name=f'RAMP-{variant_id}',
        agent_name='robustness-runner',
        metrics=metrics,
        db_path=args.db_path,
        phase='robustness_sensitivity',
        universe_name='sp500-2025',
        asset_class='equity',
        data_frequency='daily',
        window_start=args.start,
        window_end=args.end,
        cost_bps=5.0,
        cost_tier_used='5bps',
        config_payload={
            'variant_id': variant_id,
            'variation_label': label,
            'is_center': is_center,
            'params': {k: v for k, v in params.items() if k not in ('label', 'is_center')},
            'timing_mode': 'near_close',
            'delta_rebalance_pct': 0.02,
            'universe_csv': str(args.universe),
        },
        data_snapshot_date=DATA_SNAPSHOT_DATE,
        regime_breakdown=regime_attr,
        wall_clock_start=wall_start,
        wall_clock_end=wall_end,
        verdict='ROBUSTNESS_VARIATION',
        notes=(
            f'{variant_id} robustness sweep: {label}; '
            f'is_center={is_center}; '
            f'Sharpe={metrics["sharpe"]:.4f}; '
            f'trial_count_treatment=NOT_PROMOTED_no_selection_trial_added'
        ),
        return_stream=return_stream,
    )

    logger.info(
        f'[+] {label}: Sharpe={metrics["sharpe"]:.4f} CAGR={metrics["cagr"]:.2%} '
        f'elapsed={elapsed:.1f}s run_id={run_id[:8]}'
    )

    return VariationResult(
        label=label,
        params={k: v for k, v in params.items() if k not in ('label', 'is_center')},
        sharpe=metrics['sharpe'],
        cagr_val=metrics['cagr'],
        max_dd=metrics['max_dd'],
        run_id=run_id,
        is_center=is_center,
        elapsed_s=elapsed,
    )


def _classify_verdict(
    center_sharpe: float,
    neighbor_sharpes: List[float],
    threshold: float = STABILITY_THRESHOLD,
) -> Tuple[str, float, float]:
    """Classify STABLE or BRITTLE.

    Returns (verdict, min_ratio, worst_neighbor_sharpe).
    STABLE: all neighbors >= threshold * center_sharpe.
    BRITTLE: any neighbor < threshold * center_sharpe.
    """
    if not neighbor_sharpes:
        return 'STABLE', 1.0, center_sharpe
    if center_sharpe <= 0:
        # If center Sharpe is non-positive, stability is undefined.
        # Treat as BRITTLE to be conservative.
        return 'BRITTLE', 0.0, min(neighbor_sharpes)
    min_sharpe = min(neighbor_sharpes)
    min_ratio = min_sharpe / center_sharpe
    verdict = 'STABLE' if min_ratio >= threshold else 'BRITTLE'
    return verdict, float(min_ratio), float(min_sharpe)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    tmp_path.write_text(content, encoding='utf-8')
    os.replace(tmp_path, path)


def _build_md_report(
    variant_id: str,
    args,
    git_sha: str,
    results: List[VariationResult],
    center_sharpe_measured: float,
    center_sharpe_ref: float,
    verdict: str,
    min_ratio: float,
    worst_sharpe: float,
    worst_label: str,
    red_flag_results: List[VariationResult],
    grid: List[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    date_str = datetime.now().strftime('%Y-%m-%d')
    lines.append(f'# RAMP Phase 6.5 Robustness Report: {variant_id} -- {date_str}')
    lines.append('')
    lines.append('## Methodology')
    lines.append('')
    lines.append('**Gate applied**: Phase 6.5 parameter robustness (strategy-lead Section 2)')
    lines.append(
        '**Stability metric**: full-window net Sharpe at near_close, 5 bps, '
        f'{args.start.date()} to {args.end.date()}'
    )
    lines.append('**Universe**: sp500-2025')
    lines.append(f'**Data snapshot**: {DATA_SNAPSHOT_DATE}')
    lines.append(f'**Code commit**: {git_sha}')
    lines.append('')
    lines.append('**This is a robustness MAP, NOT an optimization search.**')
    lines.append(
        'The center config is the a-priori candidate. Neighborhood variations test '
        'whether the edge is a stable plateau or a cliff-edge. No variation is adopted '
        'regardless of result -- promotion requires re-entering as its own a-priori '
        'candidate with its own gate + walk-forward + trial count.'
    )
    lines.append('')
    lines.append('**Stability criterion (Section 5.5):**')
    lines.append(
        f'STABLE if all neighbors hold >= {STABILITY_THRESHOLD:.0%} of center Sharpe '
        f'({STABILITY_THRESHOLD:.0%} x {center_sharpe_measured:.4f} = '
        f'{STABILITY_THRESHOLD * center_sharpe_measured:.4f}); else BRITTLE.'
    )
    lines.append('')
    lines.append('**Trial-count treatment (Section 9.4):**')
    lines.append(
        'All variations logged with phase=robustness_sensitivity. None are promoted, '
        'so none add to the project SELECTION-trial count. If a variation were later '
        'promoted as its own a-priori candidate, it would receive its own selection-trial '
        'entry at that time.'
    )
    lines.append('')

    lines.append('## Exact grid used')
    lines.append('')
    if variant_id == 'V31':
        lines.append('**V31 grid**: 5 x 5 = 25 points')
        lines.append(
            f'beta_window in {{72, 81, 90, 99, 108}} '
            f'(center={_V31_CENTER_BETA_WINDOW}, approx -20%/-10%/center/+10%/+20%)'
        )
        lines.append(
            f'residual_horizon in {{17, 19, 21, 23, 25}} '
            f'(center={_V31_CENTER_RESIDUAL_HORIZON}, approx -19%/-10%/center/+10%/+19%)'
        )
    else:
        lines.append('**V28 grid**: 1 center + 4x6 one-at-a-time = 25 points')
        lines.append(
            f'Center: w21={_V28_CENTER_BLEND_21D}, w63={_V28_CENTER_BLEND_63D}, '
            f'w126={_V28_CENTER_BLEND_126D}, lam={_V28_CENTER_LAMBDA_REV}, '
            f'h21={_V28_CENTER_H_21}, h63={_V28_CENTER_H_63}, h126={_V28_CENTER_H_126}'
        )
        lines.append(
            'Perturbations: each of {w21, w63, w126} varied at +/-10%/+/-20% '
            '(remaining weights renormalized proportionally); '
            'lambda_rev at +/-10%/+/-20%; h21 at {19,17,23,25}; h63 at {57,50,69,76}.'
        )
    lines.append('')

    lines.append('## Results summary')
    lines.append('')
    lines.append(f'**Center Sharpe measured** : {center_sharpe_measured:.4f}')
    lines.append(f'**Center Sharpe (ref)**    : {center_sharpe_ref:.4f}')
    delta = center_sharpe_measured - center_sharpe_ref
    lines.append(f'**Delta vs ref**           : {delta:+.4f}')
    lines.append(f'**Stability threshold**    : {STABILITY_THRESHOLD:.0%} x center = {STABILITY_THRESHOLD * center_sharpe_measured:.4f}')
    lines.append(f'**Worst neighbor Sharpe**  : {worst_sharpe:.4f} ({worst_label})')
    lines.append(f'**min(neighbor)/center**   : {min_ratio:.4f}')
    lines.append(f'**Verdict**                : **{verdict}**')
    lines.append('')

    if red_flag_results:
        lines.append('## RED FLAGS: variations that beat the center')
        lines.append('')
        lines.append(
            '> These are reported as flags only. Per Phase 6.5 rules, a variation '
            'that materially beats the center is evidence of an arbitrary center or '
            'lumpy surface. It is NOT adopted here.'
        )
        lines.append('')
        lines.append('| Label | Sharpe | Delta vs center | Params |')
        lines.append('|---|---:|---:|---|')
        for r in red_flag_results:
            delta_rf = r.sharpe - center_sharpe_measured
            params_str = ', '.join(f'{k}={v}' for k, v in r.params.items())
            lines.append(
                f'| {r.label} | {r.sharpe:.4f} | {delta_rf:+.4f} | {params_str} |'
            )
        lines.append('')

    lines.append('## Full grid results')
    lines.append('')
    lines.append('| Label | Sharpe | CAGR | Max DD | Ratio/Center | is_center |')
    lines.append('|---|---:|---:|---:|---:|:---:|')
    for r in sorted(results, key=lambda x: -x.sharpe):
        ratio = r.sharpe / center_sharpe_measured if center_sharpe_measured > 0 else float('nan')
        center_marker = 'YES' if r.is_center else ''
        lines.append(
            f'| {r.label} | {r.sharpe:.4f} | {r.cagr_val:.2%} | {r.max_dd:.2%} '
            f'| {ratio:.4f} | {center_marker} |'
        )
    lines.append('')

    lines.append('## Registry run IDs')
    lines.append('')
    lines.append(f'Total variations logged: {len(results)}')
    for r in results:
        lines.append(f'- {r.label}: run_id={r.run_id}')
    lines.append('')

    lines.append('## Interpretation')
    lines.append('')
    if verdict == 'STABLE':
        lines.append(
            f'{variant_id} is **STABLE**: all {len(results)-1} neighbor variations hold '
            f'>= {STABILITY_THRESHOLD:.0%} of the center Sharpe (min ratio = {min_ratio:.4f}). '
            'The edge sits on a stable plateau. This is a necessary condition for the '
            'HYBRID build (V28/V31 signal + V11 regime-cash overlay) to be worth pursuing.'
        )
        if red_flag_results:
            lines.append('')
            lines.append(
                f'NOTE: {len(red_flag_results)} variation(s) beat the center. '
                'This may indicate the center is not at the plateau optimum. '
                'Per rules, no variation is adopted here; if the pattern is '
                'systematic, consider re-entering the best neighbor as a new '
                'a-priori candidate in a separate gate.'
            )
    else:
        lines.append(
            f'{variant_id} is **BRITTLE**: at least one neighbor variation drops below '
            f'{STABILITY_THRESHOLD:.0%} of the center Sharpe (min ratio = {min_ratio:.4f}, '
            f'worst = {worst_label} at Sharpe {worst_sharpe:.4f}). '
            'Cliff-edge sensitivity indicates the parameter surface is lumpy, which is '
            'consistent with overfitting. The HYBRID build decision should treat this as '
            'a negative signal for {variant_id}.'
        )
    lines.append('')
    lines.append(
        '**Note**: this robustness verdict does NOT change V28/V31\'s existing '
        'walk-forward REJECT verdict (they remain OOS-rejected). It only informs '
        'whether the HYBRID lead (V28/V31 signal + V11 regime overlay) is worth building.'
    )
    lines.append('')

    lines.append('## G0.5 durability status')
    lines.append('')
    lines.append(f'- Total variations: {len(results)}')
    elapsed_total = sum(r.elapsed_s for r in results)
    lines.append(f'- Total elapsed: {elapsed_total / 60:.1f} min')
    lines.append(
        '- Every variation written to registry BEFORE continuing (per G0.5 protocol).'
    )
    lines.append('')

    return '\n'.join(lines)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description='Phase 6.5 robustness runner for V28/V31.')
    p.add_argument('--variant', required=True, choices=['V28', 'V31'])
    p.add_argument(
        '--start',
        type=lambda s: datetime.strptime(s, '%Y-%m-%d'),
        default=datetime(2017, 1, 1),
    )
    p.add_argument(
        '--end',
        type=lambda s: datetime.strptime(s, '%Y-%m-%d'),
        default=datetime(2026, 5, 16),
    )
    p.add_argument(
        '--universe',
        type=Path,
        default=Path('config/universes/sp500-2025.csv'),
    )
    p.add_argument('--initial-capital', type=float, default=100_000.0)
    p.add_argument('--budget', type=int, default=25)
    p.add_argument(
        '--smoke',
        action='store_true',
        help='3-year slice for wiring check only. NEVER use for recorded results.',
    )
    p.add_argument(
        '--output-dir',
        type=Path,
        default=Path('docs/reports/ramp'),
    )
    p.add_argument(
        '--db-path',
        type=Path,
        default=DEFAULT_DB_PATH,
    )
    return p.parse_args(argv)


def main() -> int:
    args = _parse_args()

    if args.smoke:
        args.end = args.start + timedelta(days=365 * 3)
        logger.info(
            f'[!] SMOKE mode: window narrowed to {args.start.date()}..{args.end.date()}. '
            'Results NOT suitable for recording.'
        )

    logger.info(
        f'[+] Robustness runner: variant={args.variant} '
        f'window={args.start.date()}..{args.end.date()} '
        f'budget={args.budget}'
    )

    git_sha = _registry_git_sha()

    if args.variant == 'V31':
        grid_full = _build_v31_grid(budget=25)
        center_sharpe_ref = _V31_CENTER_SHARPE_REF
    else:
        grid_full = _build_v28_grid(budget=25)
        center_sharpe_ref = _V28_CENTER_SHARPE_REF

    # In smoke mode or when budget < full grid, take the center + first (budget-1) neighbors.
    if args.smoke or args.budget < len(grid_full):
        # Always include center; then fill remaining budget from the rest.
        center_pts = [pt for pt in grid_full if pt['is_center']]
        other_pts = [pt for pt in grid_full if not pt['is_center']]
        n_others = min(args.budget - 1, len(other_pts))
        grid = center_pts + other_pts[:n_others]
        if args.smoke:
            logger.info(
                f'[!] Smoke mode: grid truncated to {len(grid)} variations '
                f'(center + {n_others} neighbors) -- not suitable for recording'
            )
    else:
        grid = grid_full

    logger.info(f'[+] Grid size: {len(grid)} variations')

    cfg = _build_cfg(args)
    t_global = time.time()
    results: List[VariationResult] = []

    for i, pt in enumerate(grid):
        label = pt['label']
        is_center = pt['is_center']
        params = {k: v for k, v in pt.items() if k not in ('label', 'is_center')}
        logger.info(
            f'[+] Running variation {i+1}/{len(grid)}: {label} (center={is_center})'
        )
        try:
            result = _run_one_variation(
                args=args,
                variant_id=args.variant,
                params=params,
                label=label,
                cfg=cfg,
                git_sha=git_sha,
                is_center=is_center,
            )
            results.append(result)
        except Exception as exc:
            logger.error(f'[-] Variation {label} FAILED: {exc}')
            raise

    elapsed_total = time.time() - t_global
    logger.info(f'[+] All {len(results)} variations done in {elapsed_total/60:.1f} min')

    # Compute verdict.
    center_results = [r for r in results if r.is_center]
    neighbor_results = [r for r in results if not r.is_center]

    if not center_results:
        logger.error('[-] No center result found -- cannot compute verdict')
        return 1

    center_result = center_results[0]
    center_sharpe_measured = center_result.sharpe

    neighbor_sharpes = [r.sharpe for r in neighbor_results]
    verdict, min_ratio, worst_sharpe = _classify_verdict(
        center_sharpe_measured, neighbor_sharpes
    )

    worst_result = min(neighbor_results, key=lambda r: r.sharpe) if neighbor_results else center_result
    worst_label = worst_result.label

    # Red flags: variations that beat the center by > 1%.
    red_flag_results = [
        r for r in neighbor_results
        if r.sharpe > center_sharpe_measured + 0.01
    ]

    logger.info(
        f'[+] VERDICT: {verdict} | center_sharpe={center_sharpe_measured:.4f} | '
        f'min_ratio={min_ratio:.4f} | worst={worst_label} ({worst_sharpe:.4f})'
    )
    if red_flag_results:
        logger.info(
            f'[!] RED FLAGS ({len(red_flag_results)} variations beat center): '
            + ', '.join(f'{r.label}={r.sharpe:.4f}' for r in red_flag_results)
        )

    # Build and write artifacts atomically.
    date_str = datetime.now().strftime('%Y%m%d')
    md_path = args.output_dir / f'{date_str}_robustness_{args.variant.lower()}.md'
    json_path = args.output_dir / f'{date_str}_robustness_{args.variant.lower()}.json'

    md_content = _build_md_report(
        variant_id=args.variant,
        args=args,
        git_sha=git_sha,
        results=results,
        center_sharpe_measured=center_sharpe_measured,
        center_sharpe_ref=center_sharpe_ref,
        verdict=verdict,
        min_ratio=min_ratio,
        worst_sharpe=worst_sharpe,
        worst_label=worst_label,
        red_flag_results=red_flag_results,
        grid=grid,
    )

    json_payload = {
        'variant_id': args.variant,
        'git_sha': git_sha,
        'window_start': str(args.start.date()),
        'window_end': str(args.end.date()),
        'universe': str(args.universe),
        'generated_at': datetime.now().isoformat(),
        'budget': args.budget,
        'n_variations': len(results),
        'center_sharpe_measured': center_sharpe_measured,
        'center_sharpe_ref': center_sharpe_ref,
        'stability_threshold': STABILITY_THRESHOLD,
        'verdict': verdict,
        'min_ratio': min_ratio,
        'worst_neighbor_sharpe': worst_sharpe,
        'worst_neighbor_label': worst_label,
        'red_flag_count': len(red_flag_results),
        'red_flag_labels': [r.label for r in red_flag_results],
        'trial_count_treatment': (
            'phase=robustness_sensitivity; none promoted; '
            'no selection-trial added to project count'
        ),
        'variations': [
            {
                'label': r.label,
                'is_center': r.is_center,
                'params': r.params,
                'sharpe': r.sharpe,
                'cagr': r.cagr_val,
                'max_dd': r.max_dd,
                'run_id': r.run_id,
                'ratio_to_center': (
                    r.sharpe / center_sharpe_measured
                    if center_sharpe_measured > 0 else None
                ),
            }
            for r in results
        ],
    }

    _atomic_write(md_path, md_content)
    _atomic_write(json_path, json.dumps(json_payload, indent=2, default=str))

    logger.info(f'[+] Report: {md_path}')
    logger.info(f'[+] JSON:   {json_path}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
