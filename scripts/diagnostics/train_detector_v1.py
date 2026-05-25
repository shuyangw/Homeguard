"""Train the WS-3d LightGBM detector via purged combinatorial CV.

Inputs:
- 4 leading indicators from src.data.leading_indicators
- SPY + VIX OHLC for backwards-compat features (above_20, above_50, above_200,
  vix_percentile)
- Binary label selected by --target:
    * g1_bear (default, round 1 confirmation label): G1_BEAR from
      label_g1_drawdown_bear
    * g1_shift10 (round 3 leading target, 2026-05-24): G1_BEAR.shift(-10).
      The model predicts whether G1_BEAR will fire 10 days from t; a model
      that correctly learns this should fire 10 days BEFORE G1_BEAR onset.
      The last 10 rows have unknown future labels and are filled with False
      to avoid lookahead.
    * g2_bear (forward-window leading label): label_g2_forward_window_bear,
      forward 30d return < -5% AND forward vol > 25%.

Purge/embargo:
- g1_bear keeps the original purge=5, embargo=5 (label is point-in-time).
- g1_shift10 uses purge=15, embargo=15 (10 days of shift + 5d buffer each side)
  to prevent lookahead through overlapping shifted labels.
- g2_bear uses purge=35, embargo=35 (30d forward window + 5d buffer).

Output:
- Trained LightGBM model artifact at
  H:/Stock_Data/alt_data/models/v20_detector/<git_sha>/model_<target>.pkl
- Training metadata JSON at model_<target>.metadata.json, recording target
  name, shift, purge/embargo, and CV scores.
- Hyperparameter sweep bounded: max 48 combinations (4 n_estimators x 4
  max_depth x 3 lr). The entire sweep counts as 1 trial in the WS-3d family.

Usage:
  PYTHONPATH=. python scripts/diagnostics/train_detector_v1.py \\
      [--target {g1_bear,g1_shift10,g2_bear}] [--no-sweep]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from scripts.diagnostics.ground_truth_labelers import (
    label_g1_drawdown_bear,
    label_g2_forward_window_bear,
)
from src.data.leading_indicators import load_leading_indicators
from src.settings import get_local_storage_dir
from src.strategies.advanced.market_regime_detector_v1 import FEATURE_COLUMNS
from src.utils.logger import get_logger

logger = get_logger(__name__)


SPY_VIX_PANEL = Path('diagnostics/data/spy_vix_2016_2026.parquet')
TRAIN_START = datetime(2017, 1, 1)
TRAIN_END = datetime(2026, 5, 22)

N_FOLDS = 6
# Default purge/embargo for the original point-in-time G1_BEAR target.
PURGE_DAYS = 5
EMBARGO_DAYS = 5

# Target name -> (purge_days, embargo_days, shift, description)
# Shift k means target_t = base_label_{t+k}. For g1_shift10 the last 10 rows
# have unknown future labels and are masked to False.
TARGET_CONFIG: Dict[str, Dict] = {
    'g1_bear': {
        'purge': 5,
        'embargo': 5,
        'shift': 0,
        'base_label': 'G1_BEAR (drawdown >= 10% from trailing 252d high)',
        'description': (
            'Round 1 confirmation label. The model predicts P(G1_BEAR_t | indicators_t). '
            'Known to lag because G1_BEAR fires after the drawdown is already ~10%.'
        ),
    },
    'g1_shift10': {
        'purge': 15,
        'embargo': 15,
        'shift': 10,
        'base_label': 'G1_BEAR.shift(-10)',
        'description': (
            'Round 3 leading target (2026-05-24). The model predicts whether G1_BEAR '
            'will fire 10 days from t; a model that learns this should fire 10 days '
            'BEFORE G1_BEAR onset. Last 10 rows of target are unknown future -> False. '
            'Purge/embargo enlarged to 15d to absorb the 10d shift + 5d buffer and '
            'prevent lookahead via overlapping shifted labels.'
        ),
    },
    'g2_bear': {
        'purge': 35,
        'embargo': 35,
        'shift': 0,
        'base_label': 'G2_BEAR (forward 30d return < -5% AND forward vol > 25%)',
        'description': (
            'Forward-window leading label. Fully forward-looking but in-sample only '
            '(training data is historical). Purge/embargo widened to 35d to cover the '
            '30d forward window + 5d buffer.'
        ),
    },
}

# Hyperparameter sweep -- AT MOST 48 combinations per spec.
SWEEP_N_ESTIMATORS = [50, 100, 200, 500]
SWEEP_MAX_DEPTH = [3, 4, 5, 6]
SWEEP_LEARNING_RATE = [0.01, 0.05, 0.1]

# Fixed hyperparameters from spec.
FIXED_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'is_unbalance': True,
    'verbose': -1,
    'random_state': 42,
}


def get_git_sha() -> str:
    """Return short HEAD SHA. Falls back to 'unknown' on error."""
    try:
        out = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=Path(__file__).resolve().parent.parent.parent,
        )
        return out.decode().strip()
    except Exception as e:
        logger.warning(f'get_git_sha: {e}')
        return 'unknown'


def _build_target(panel: pd.DataFrame, target: str) -> pd.Series:
    """Construct the target label according to --target.

    g1_bear:    G1_BEAR (point-in-time confirmation, round 1).
    g1_shift10: G1_BEAR.shift(-10) -- the label at index t equals
                G1_BEAR_{t+10}. Last 10 rows' future labels are unknown
                and are filled with False to avoid lookahead.
    g2_bear:    G2_BEAR forward-window leading label.
    """
    if target == 'g1_bear':
        y = label_g1_drawdown_bear(panel).astype(int)
        y.name = 'g1_bear'
        return y
    if target == 'g1_shift10':
        base = label_g1_drawdown_bear(panel).astype(int)
        shifted = base.shift(-10)
        # Fill the trailing 10 rows (future unknown) with False/0 to avoid
        # accidentally leaking NaNs into y. This is a conservative choice:
        # it slightly understates positive rate near the panel tail but
        # never lookaheads.
        y = shifted.fillna(0).astype(int)
        y.name = 'g1_shift10'
        return y
    if target == 'g2_bear':
        y = label_g2_forward_window_bear(panel).astype(int)
        y.name = 'g2_bear'
        return y
    raise ValueError(
        f'Unknown --target {target!r}; must be one of '
        f'{list(TARGET_CONFIG.keys())}'
    )


def build_training_panel(target: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Build the feature matrix X and label series y for training.

    Returns (X, y) indexed by trade date. X has FEATURE_COLUMNS columns.
    y is the binary target selected by `target` (see TARGET_CONFIG).
    """
    logger.info('[+] loading SPY/VIX panel from diagnostics/data/...')
    panel = pd.read_parquet(SPY_VIX_PANEL)
    panel = panel.loc[TRAIN_START:TRAIN_END]
    logger.info(f'[+] SPY/VIX panel: {len(panel)} rows, {panel.index.min().date()} to {panel.index.max().date()}')

    logger.info('[+] loading leading indicators...')
    leading = load_leading_indicators(TRAIN_START, TRAIN_END, cache=True)
    logger.info(f'[+] leading_indicators: {len(leading)} rows')

    spy_close = panel['spy_close']
    sma_20 = spy_close.rolling(20).mean()
    sma_50 = spy_close.rolling(50).mean()
    sma_200 = spy_close.rolling(200).mean()
    above_20 = (spy_close > sma_20).astype(float)
    above_50 = (spy_close > sma_50).astype(float)
    above_200 = (spy_close > sma_200).astype(float)

    vix_close = panel['vix_close']
    vix_pct = vix_close.rolling(252).apply(
        lambda w: (w[:-1] < w.iloc[-1]).sum() / max(len(w) - 1, 1) * 100.0,
        raw=False,
    )

    v0_features = pd.DataFrame({
        'above_20': above_20,
        'above_50': above_50,
        'above_200': above_200,
        'vix_percentile': vix_pct,
    })

    leading_subset = leading[[
        'vix_term_ratio', 'hy_proxy_ratio', 'breadth_pct', 'skew_close',
    ]]
    X = leading_subset.join(v0_features, how='inner')
    X = X[FEATURE_COLUMNS]
    X = X.dropna()

    y = _build_target(panel, target)

    common = X.index.intersection(y.index)
    X = X.loc[common]
    y = y.loc[common]

    logger.info(
        f'[+] training panel ({target}): {len(X)} rows, '
        f'{int(y.sum())} positives ({y.mean()*100:.2f}%)'
    )
    return X, y


def purged_kfold_indices(
    n_samples: int,
    n_folds: int,
    purge: int,
    embargo: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate purged K-fold splits.

    For each fold, train indices exclude a `purge` window before the test
    fold and an `embargo` window after, preventing label/feature leakage.

    Returns list of (train_idx, test_idx) pairs.
    """
    fold_size = n_samples // n_folds
    splits = []
    for k in range(n_folds):
        test_start = k * fold_size
        test_end = (k + 1) * fold_size if k < n_folds - 1 else n_samples
        test_idx = np.arange(test_start, test_end)

        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[max(0, test_start - purge):min(n_samples, test_end + embargo)] = False
        train_idx = np.where(train_mask)[0]

        splits.append((train_idx, test_idx))
    return splits


def evaluate_hp_combo(
    X: pd.DataFrame,
    y: pd.Series,
    hp: Dict,
    splits: List[Tuple[np.ndarray, np.ndarray]],
) -> Dict:
    """Train + evaluate one hyperparameter combo across all CV folds.

    Returns dict with mean_logloss, std_logloss, fold_loglosses, fold_positive_rate.
    """
    fold_losses = []
    fold_pos_rate = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        pos_train = int(y_train.sum())
        pos_test = int(y_test.sum())
        fold_pos_rate.append(pos_test / max(len(y_test), 1))

        if pos_train == 0 or pos_train == len(y_train):
            logger.warning(
                f'  fold {fold_idx}: degenerate train labels '
                f'(pos={pos_train}/{len(y_train)}), skipping'
            )
            fold_losses.append(np.nan)
            continue
        if pos_test == 0:
            # Test fold has no positives -- logloss still defined but uninformative.
            pass

        params = dict(FIXED_PARAMS)
        params.update(hp)
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        prob = np.clip(prob, 1e-7, 1 - 1e-7)
        ll = -np.mean(
            y_test.values * np.log(prob) + (1 - y_test.values) * np.log(1 - prob)
        )
        fold_losses.append(float(ll))

    valid = [ll for ll in fold_losses if not np.isnan(ll)]
    return {
        'mean_logloss': float(np.mean(valid)) if valid else float('inf'),
        'std_logloss': float(np.std(valid)) if valid else 0.0,
        'fold_loglosses': fold_losses,
        'fold_positive_rate': fold_pos_rate,
        'n_valid_folds': len(valid),
    }


def run_sweep(
    X: pd.DataFrame,
    y: pd.Series,
    purge_days: int = PURGE_DAYS,
    embargo_days: int = EMBARGO_DAYS,
) -> Tuple[Dict, List[Dict]]:
    """Run purged CV sweep over the bounded hyperparameter grid.

    Returns (best_hp_with_metadata, all_results_sorted_by_logloss).
    """
    splits = purged_kfold_indices(
        n_samples=len(X),
        n_folds=N_FOLDS,
        purge=purge_days,
        embargo=embargo_days,
    )
    logger.info(
        f'[+] purged CV: {N_FOLDS} folds, purge={purge_days}d, embargo={embargo_days}d'
    )

    combos = list(product(SWEEP_N_ESTIMATORS, SWEEP_MAX_DEPTH, SWEEP_LEARNING_RATE))
    logger.info(f'[+] hyperparameter sweep: {len(combos)} combinations (<= 48 per spec)')
    assert len(combos) <= 48, f'sweep size {len(combos)} > 48; pre-commitment violated'

    results = []
    for i, (n_est, depth, lr) in enumerate(combos):
        hp = {
            'n_estimators': n_est,
            'max_depth': depth,
            'learning_rate': lr,
            'num_leaves': max(2 ** depth - 1, 4),
        }
        cv = evaluate_hp_combo(X, y, hp, splits)
        results.append({
            'hp': hp,
            'mean_logloss': cv['mean_logloss'],
            'std_logloss': cv['std_logloss'],
            'fold_loglosses': cv['fold_loglosses'],
            'n_valid_folds': cv['n_valid_folds'],
        })
        if (i + 1) % 12 == 0 or i == len(combos) - 1:
            logger.info(
                f'  combo {i + 1}/{len(combos)}: '
                f'hp={hp} mean_logloss={cv["mean_logloss"]:.5f}'
            )

    results_sorted = sorted(results, key=lambda r: r['mean_logloss'])
    best = results_sorted[0]
    logger.info(
        f'[+] best hyperparameters: {best["hp"]} '
        f'mean_logloss={best["mean_logloss"]:.5f}'
    )
    return best, results_sorted


def fit_final_model(X: pd.DataFrame, y: pd.Series, hp: Dict) -> lgb.LGBMClassifier:
    """Refit on the full training set with the chosen hyperparameters."""
    params = dict(FIXED_PARAMS)
    params.update(hp)
    model = lgb.LGBMClassifier(**params)
    model.fit(X, y)
    return model


def save_artifact(
    model: lgb.LGBMClassifier,
    metadata: Dict,
    git_sha: str,
    target: str,
) -> Tuple[Path, Path]:
    """Persist the trained model + metadata under the git-SHA scoped path.

    Files are suffixed by target name so the round-1 (g1_bear) artifacts
    remain intact when round-3 (g1_shift10) runs.
    """
    base = get_local_storage_dir() / 'alt_data' / 'models' / 'v20_detector' / git_sha
    base.mkdir(parents=True, exist_ok=True)
    # Backward compat: g1_bear continues to write model.pkl + metadata.json.
    # Other targets write model_<target>.pkl + model_<target>.metadata.json.
    if target == 'g1_bear':
        model_path = base / 'model.pkl'
        meta_path = base / 'metadata.json'
    else:
        model_path = base / f'model_{target}.pkl'
        meta_path = base / f'model_{target}.metadata.json'

    joblib.dump(model, model_path)
    with open(meta_path, 'w', encoding='ascii') as f:
        json.dump(metadata, f, indent=2, default=str, ensure_ascii=True)
    logger.info(f'[+] model artifact: {model_path}')
    logger.info(f'[+] metadata: {meta_path}')
    return model_path, meta_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--target',
        type=str,
        default='g1_bear',
        choices=sorted(TARGET_CONFIG.keys()),
        help=(
            'Training target. g1_bear (default, round 1 confirmation label) '
            'reproduces the round-1/2 model. g1_shift10 (round 3 leading '
            'target) trains on G1_BEAR.shift(-10) so the model fires earlier '
            'by construction. g2_bear trains on the forward-window leading '
            'label. Each non-default target writes to a target-suffixed '
            'artifact path.'
        ),
    )
    parser.add_argument(
        '--no-sweep', action='store_true',
        help='Skip hyperparameter sweep; train once with defaults (100/4/0.05)',
    )
    args = parser.parse_args(argv)

    git_sha = get_git_sha()
    logger.info(f'[+] git SHA: {git_sha}')

    cfg = TARGET_CONFIG[args.target]
    purge_days = int(cfg['purge'])
    embargo_days = int(cfg['embargo'])
    shift = int(cfg['shift'])
    logger.info(
        f'[+] target={args.target} ({cfg["base_label"]}), '
        f'shift={shift}, purge={purge_days}d, embargo={embargo_days}d'
    )
    logger.info(f'[+] target description: {cfg["description"]}')

    X, y = build_training_panel(args.target)

    if args.no_sweep:
        best_hp = {
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.05,
            'num_leaves': 15,
        }
        sweep_results = []
        n_combos = 1
        logger.info(f'[+] skipping sweep; using default hp {best_hp}')
        splits = purged_kfold_indices(len(X), N_FOLDS, purge_days, embargo_days)
        cv = evaluate_hp_combo(X, y, best_hp, splits)
        best = {
            'hp': best_hp,
            'mean_logloss': cv['mean_logloss'],
            'std_logloss': cv['std_logloss'],
            'fold_loglosses': cv['fold_loglosses'],
            'n_valid_folds': cv['n_valid_folds'],
        }
    else:
        best, sweep_results = run_sweep(X, y, purge_days, embargo_days)
        n_combos = len(sweep_results)

    final_model = fit_final_model(X, y, best['hp'])
    feature_importance = dict(zip(
        FEATURE_COLUMNS,
        [int(v) for v in final_model.feature_importances_]
    ))

    metadata = {
        'git_sha': git_sha,
        'trained_at': datetime.now().isoformat(),
        'target': args.target,
        'target_shift_days': shift,
        'target_base_label': cfg['base_label'],
        'target_description': cfg['description'],
        'train_start': str(TRAIN_START.date()),
        'train_end': str(TRAIN_END.date()),
        'n_samples': int(len(X)),
        'n_positives': int(y.sum()),
        'positive_rate': float(y.mean()),
        'feature_columns': FEATURE_COLUMNS,
        'feature_importance': feature_importance,
        'cv_n_folds': N_FOLDS,
        'cv_purge_days': purge_days,
        'cv_embargo_days': embargo_days,
        'sweep_n_combos': n_combos,
        'best_hp': best['hp'],
        'best_mean_logloss': best['mean_logloss'],
        'best_std_logloss': best['std_logloss'],
        'best_fold_loglosses': best['fold_loglosses'],
        'fixed_params': {k: v for k, v in FIXED_PARAMS.items() if k != 'verbose'},
        'sweep_top10': sweep_results[:10] if sweep_results else [],
        'spec_reference': 'docs/superpowers/specs/2026-05-25-ws3d-detector-replacement-design.md',
    }
    save_artifact(final_model, metadata, git_sha, args.target)
    logger.info(
        f'[+] Done. target={args.target} '
        f'best CV mean logloss = {best["mean_logloss"]:.5f} '
        f'(std = {best["std_logloss"]:.5f})'
    )
    logger.info(f'[+] Feature importance: {feature_importance}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
