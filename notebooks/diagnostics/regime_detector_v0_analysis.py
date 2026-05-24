"""Phase 4 analysis: regime detector diagnostic.

Six analyses (A-F) testing H1-H5. Notebook outputs are the figures + tables
referenced in the Phase 5 synthesis.
"""

# %% [markdown]
# # Regime Detector Diagnostic - Phase 4 Analysis
#
# Inputs:
# - `diagnostics/regime/v0/labels.parquet` (Phase 2 driver output)
# - `diagnostics/regime/ground_truth.parquet` (Phase 3 labelers output)
#
# Outputs:
# - Six analyses (A-F)
# - Saved figures under `diagnostics/regime/v0/figures/`
# - Summary stats inline for later Phase 5 synthesis

# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

LABELS = pd.read_parquet('diagnostics/regime/v0/labels.parquet')
GT = pd.read_parquet('diagnostics/regime/ground_truth.parquet')
LABELS['date'] = pd.to_datetime(LABELS['date'])
LABELS = LABELS.set_index('date').sort_index()
GT['date'] = pd.to_datetime(GT['date'])
GT = GT.set_index('date').sort_index()
# Drop overlapping 'year' col from LABELS before join to avoid suffix conflicts
LABELS_FOR_JOIN = LABELS.drop(columns=[c for c in ['year'] if c in LABELS.columns])
JOIN = LABELS_FOR_JOIN.join(GT, how='inner')
print(f'Loaded {len(JOIN)} day-rows; columns: {JOIN.columns.tolist()}')

FIG_DIR = Path('diagnostics/regime/v0/figures')
FIG_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Analysis A: Regime distribution (tests H1)
#
# % time in each regime, by year.

# %%
joined_year = LABELS.copy()
joined_year['year_int'] = joined_year.index.year
dist = joined_year.groupby(['year_int', 'regime']).size().unstack(fill_value=0)
dist_pct = dist.div(dist.sum(axis=1), axis=0) * 100
print('Regime distribution by year (%):')
print(dist_pct.round(1))
fig, ax = plt.subplots(figsize=(12, 6))
dist_pct.plot(kind='bar', stacked=True, ax=ax)
ax.set_title('Regime Distribution by Year (% of trading days)')
ax.set_ylabel('% of days')
ax.set_xlabel('Year')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig(FIG_DIR / 'analysis_A_regime_dist.png', dpi=120)
plt.close(fig)

bear_pct_total = (LABELS['regime'] == 'BEAR').mean() * 100
print(f'\nTOTAL BEAR %: {bear_pct_total:.2f}%')
if 'BEAR' in dist_pct.columns:
    h1_supported = (dist_pct['BEAR'] < 5).all()
else:
    h1_supported = True  # No BEAR labels at all
print(f'H1 prediction "BEAR < 5% of any year": '
      f'{"SUPPORTED" if h1_supported else "REFUTED"}')

# %% [markdown]
# ## Analysis B: Run-length distribution (tests H4)

# %%
def run_lengths(series: pd.Series) -> pd.DataFrame:
    """Return run lengths grouped by value: DataFrame with cols [first, size]."""
    blocks = (series != series.shift()).cumsum()
    grouped = series.groupby(blocks).agg(['first', 'size'])
    return grouped

rl = run_lengths(LABELS['regime'])
print('Run length stats per regime:')
for regime in ['STRONG_BULL', 'WEAK_BULL', 'SIDEWAYS', 'UNPREDICTABLE', 'BEAR']:
    sizes = rl.loc[rl['first'] == regime, 'size']
    if len(sizes) == 0:
        print(f'  {regime}: n=0 runs')
        continue
    print(f'  {regime}: n={len(sizes)} runs, median={sizes.median():.1f}, '
          f'P25={sizes.quantile(0.25):.1f}, P75={sizes.quantile(0.75):.1f}, '
          f'max={sizes.max()}')

fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
for ax, regime in zip(axes, ['STRONG_BULL', 'WEAK_BULL', 'SIDEWAYS', 'UNPREDICTABLE', 'BEAR']):
    sizes = rl.loc[rl['first'] == regime, 'size']
    if len(sizes) > 0:
        ax.hist(sizes, bins=30)
    ax.set_title(regime)
    ax.set_xlabel('Run length (days)')
plt.tight_layout()
plt.savefig(FIG_DIR / 'analysis_B_runlengths.png', dpi=120)
plt.close(fig)

# %% [markdown]
# ## Analysis C: Empirical transition matrix (tests H4, connects to H1)

# %%
transitions = pd.crosstab(LABELS['regime'].shift(), LABELS['regime'],
                          normalize='index')
print('P(r_{t+1} | r_t):')
print(transitions.round(3))
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(transitions, annot=True, fmt='.3f', cmap='Blues', ax=ax)
ax.set_title('Empirical Transition Matrix')
ax.set_xlabel('r_{t+1}')
ax.set_ylabel('r_t')
plt.tight_layout()
plt.savefig(FIG_DIR / 'analysis_C_transitions.png', dpi=120)
plt.close(fig)

diag_mass = float(np.mean([transitions.loc[r, r] for r in transitions.index
                           if r in transitions.columns]))
print(f'\nMean diagonal mass (stickiness): {diag_mass:.3f}')

# %% [markdown]
# ## Analysis D: Lag-to-event (tests H5)
#
# For each G4 drawdown event, measure days from event start to first BEAR label.

# %%
events = pd.read_csv('config/diagnostics/regime_events_2017_2026.csv',
                     parse_dates=['start_date', 'end_date'])
drawdown_events = events[events['event_type'] == 'drawdown']
lag_results = []
for _, ev in drawdown_events.iterrows():
    window = LABELS.loc[ev['start_date']:ev['end_date']]
    bear_dates = window.index[window['regime'] == 'BEAR']
    if len(bear_dates) == 0:
        lag_results.append({'event': ev['event_name'], 'lag': None,
                            'bear_in_window': False})
    else:
        lag_days = (bear_dates[0] - ev['start_date']).days
        lag_results.append({'event': ev['event_name'], 'lag': lag_days,
                            'bear_in_window': True})

lag_df = pd.DataFrame(lag_results)
print(lag_df)
valid_lags = lag_df['lag'].dropna()
if len(valid_lags) > 0:
    print(f'\nMedian lag (days): {valid_lags.median():.1f}')
    print(f'P25 / P75: {valid_lags.quantile(0.25):.1f} / {valid_lags.quantile(0.75):.1f}')
else:
    print('\nNo events with BEAR fire -- median lag undefined')
print(f'Events with NO BEAR label: '
      f'{(~lag_df["bear_in_window"]).sum()} of {len(lag_df)}')

# %% [markdown]
# ## Analysis E: Input ablation (tests H1, H2; MOST ACTIONABLE)
#
# For days where G1_BEAR is True but detector did not label BEAR, decompose
# which of the BEAR criteria failed.

# %%
mismatch = JOIN[(JOIN['g1_bear']) & (JOIN['regime'] != 'BEAR')]
print(f'G1_BEAR days where detector did not label BEAR: {len(mismatch)}')

# BEAR criteria from REGIME_CRITERIA: momentum <= -0.02, VIX pct >= 70, below all 3 SMAs.
# "Fail" = criterion not satisfied for BEAR.
if len(mismatch) > 0:
    failures = pd.DataFrame({
        'momentum_fail': mismatch['momentum_slope'] > -0.02,
        'vix_pct_fail': mismatch['vix_percentile_252d'] < 70,
        'above_20_fail': mismatch['above_20'],
        'above_50_fail': mismatch['above_50'],
        'above_200_fail': mismatch['above_200'],
    })
    fail_pct = failures.mean() * 100
    print('% of mismatch days failing each BEAR criterion:')
    print(fail_pct.sort_values(ascending=False).round(1))
    fig, ax = plt.subplots(figsize=(10, 5))
    fail_pct.sort_values(ascending=False).plot(kind='barh', ax=ax)
    ax.set_title('Why detector missed G1_BEAR: % of criteria failures')
    ax.set_xlabel('% of G1_BEAR-but-not-detector-BEAR days')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'analysis_E_ablation.png', dpi=120)
    plt.close(fig)
else:
    fail_pct = pd.Series(dtype=float)
    print('No G1_BEAR-but-not-detector-BEAR days; ablation skipped.')

# %% [markdown]
# ## Analysis F: Lookback-window sensitivity (tests H3)
#
# Count days where vix_percentile_<w>d >= 70 for each lookback in {63, 126, 252, 504}.
# This is NOT a full re-classification (BEAR is one of 5 scored regimes), but
# it tells us how many days would have passed BEAR's VIX criterion under each
# lookback.

# %%
for w in [63, 126, 252, 504]:
    col = f'vix_percentile_{w}d'
    passes_vix = LABELS[col] >= 70
    print(f'  lookback={w}d: passes VIX pct >= 70 on {passes_vix.sum()} of {len(LABELS)} days '
          f'({passes_vix.mean()*100:.1f}%)')

fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
for ax, w in zip(axes, [63, 126, 252, 504]):
    col = f'vix_percentile_{w}d'
    ax.plot(LABELS.index, LABELS[col], lw=0.5)
    ax.axhline(70, color='r', linestyle='--', alpha=0.5, label='BEAR threshold')
    ax.set_title(f'VIX percentile, lookback={w}d')
    ax.set_ylabel('Percentile')
    ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / 'analysis_F_lookback_sensitivity.png', dpi=120)
plt.close(fig)

# %% [markdown]
# ## Summary inputs for Phase 5 synthesis
#
# Numbers to carry into the synthesis report:

# %%
print('=== Summary for Phase 5 ===')
print(f'Total replay days: {len(LABELS)}')
print(f'BEAR % overall: {bear_pct_total:.2f}%')
if 'BEAR' in dist_pct.columns:
    print(f'BEAR % by year: {dist_pct["BEAR"].round(1).to_dict()}')
else:
    print('BEAR % by year: {} (no BEAR labels)')

print('\nMedian run lengths:')
for regime in ['STRONG_BULL', 'WEAK_BULL', 'SIDEWAYS', 'UNPREDICTABLE', 'BEAR']:
    sizes = rl.loc[rl['first'] == regime, 'size']
    if len(sizes) > 0:
        print(f'  {regime}: {sizes.median():.1f}')

print(f'\nMean transition-matrix diagonal mass: {diag_mass:.3f}')

g1_total = int(JOIN['g1_bear'].sum())
if g1_total > 0:
    print(f'\nG1_BEAR days missed by detector: {len(mismatch)} '
          f'({len(mismatch) / g1_total * 100:.1f}% of G1_BEAR days)')
else:
    print('\nG1_BEAR days missed by detector: N/A (no G1_BEAR days)')
if len(fail_pct) > 0:
    print(f'Most common missed-BEAR failure mode: {fail_pct.idxmax()} '
          f'({fail_pct.max():.1f}%)')

print(f'\nDrawdown events where BEAR fired: '
      f'{lag_df["bear_in_window"].sum()} of {len(lag_df)}')
if valid_lags.size > 0:
    print(f'Median onset lag (days): {valid_lags.median():.1f}')

print('\nLookback sensitivity (days with VIX pct >= 70):')
for w in [63, 126, 252, 504]:
    col = f'vix_percentile_{w}d'
    passes_vix = LABELS[col] >= 70
    print(f'  {w}d: {int(passes_vix.sum())} days ({passes_vix.mean()*100:.1f}%)')
