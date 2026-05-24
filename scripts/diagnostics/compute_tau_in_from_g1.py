"""Compute pre-registered tau_in for V14 from G1_BEAR median BEAR_score.

Runs ONCE at spec time before any V14 backtest. Output: v14_tau_constants.json.
The G1_BEAR labeler is pinned at commit 9c48245; the pinning is recorded
in the output JSON.
"""

from __future__ import annotations

import json
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from scripts.diagnostics.ground_truth_labelers import label_g1_drawdown_bear
from src.utils.logger import logger


PANEL_PATH = Path('diagnostics/data/spy_vix_2016_2026.parquet')
SCORES_PATH = Path('diagnostics/regime/v0_scores/labels.parquet')
OUTPUT_PATH = Path('config/research/v14_tau_constants.json')
LABELER_FILE = 'scripts/diagnostics/ground_truth_labelers.py'
HYSTERESIS_BAND = 0.1


def main() -> int:
    if not PANEL_PATH.exists():
        raise FileNotFoundError(PANEL_PATH)
    if not SCORES_PATH.exists():
        raise FileNotFoundError(SCORES_PATH)

    panel = pd.read_parquet(PANEL_PATH)
    scores = pd.read_parquet(SCORES_PATH)
    scores['date'] = pd.to_datetime(scores['date'])
    scores = scores.set_index('date')

    g1_bear = label_g1_drawdown_bear(panel)
    g1_bear_dates = g1_bear[g1_bear].index

    joined = scores.loc[scores.index.isin(g1_bear_dates), 'score_BEAR']
    if joined.empty:
        raise RuntimeError(
            f'No overlap between G1_BEAR days ({len(g1_bear_dates)}) and '
            f'soft-score replay ({len(scores)} days).'
        )

    tau_in = float(joined.median())
    p25 = float(joined.quantile(0.25))
    p75 = float(joined.quantile(0.75))
    tau_out = round(tau_in - HYSTERESIS_BAND, 6)

    if not (0.0 < tau_out < tau_in < 1.0):
        raise RuntimeError(
            f'Pre-registered tau predicate failed: '
            f'tau_out={tau_out}, tau_in={tau_in}'
        )

    labeler_commit = subprocess.run(
        ['git', 'log', '-1', '--format=%H', '--', LABELER_FILE],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if not labeler_commit:
        raise RuntimeError(
            f'git log returned no commit for {LABELER_FILE} -- '
            f'file may be uncommitted or path is wrong.'
        )

    out = {
        'tau_in': round(tau_in, 6),
        'tau_out': tau_out,
        'tau_band': HYSTERESIS_BAND,
        'g1_bear_score_p25': round(p25, 6),
        'g1_bear_score_p75': round(p75, 6),
        'n_g1_bear_days': int(len(joined)),
        'g1_labeler_commit': labeler_commit,
        'computed_at': datetime.now(timezone.utc).isoformat(),
        'computation_script': 'scripts/diagnostics/compute_tau_in_from_g1.py',
        'computation_method': (
            'median BEAR_score on G1_BEAR days '
            '(drawdown > 10% from 252-day trailing high)'
        ),
        'source_data': {
            'panel': PANEL_PATH.as_posix(),
            'scores': SCORES_PATH.as_posix(),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(out, indent=2) + '\n')
    logger.info(f'[+] tau_in={out["tau_in"]} tau_out={out["tau_out"]} '
                f'(n={out["n_g1_bear_days"]} G1_BEAR days)')
    logger.info(f'[+] Wrote {OUTPUT_PATH}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
