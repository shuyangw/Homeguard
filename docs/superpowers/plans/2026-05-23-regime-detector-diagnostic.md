# Regime Detector Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a re-runnable diagnostic harness that replays the production `MarketRegimeDetector` across 2017-2026, generates four independent ground-truth labelings, runs six analyses against H1-H5, and produces a synthesis report ranking five remediation options.

**Architecture:** Read-only against production code. Five sequential phases (0-5) with explicit decision gates between them. Phase 0 verifies the spec's mental model against actual code. Phase 1 stages data. Phase 2 produces a Parquet of detector outputs + intermediate values + parametrized alternatives. Phase 3 produces independent ground-truth labels. Phase 4 is a Jupyter notebook with 6 analyses. Phase 5 is a markdown synthesis with hypothesis verdicts and option ranking.

**Tech Stack:** Python 3 (fintech conda env), pandas, pyarrow, scipy, matplotlib, Jupyter. Existing helpers: `src.strategies.advanced.market_regime_detector.MarketRegimeDetector`, `src.backtesting.regimes.detector.TrendDetector`, Alpaca SIP data layer, yfinance for VIX.

**Spec:** `docs/superpowers/specs/2026-05-23-regime-detector-diagnostic-design.md`

**Branch:** `regime-detector-diagnostic` (already created from main at `d60686e`).

---

## File structure

New files this plan creates (in order of creation):

| Path | Responsibility |
|---|---|
| `docs/reports/ramp/20260523_regime_detector_diagnostic.md` | Phase 0 writeup at the top (code archaeology summary) + Phase 5 synthesis at the bottom |
| `scripts/diagnostics/__init__.py` | Empty init marker |
| `scripts/diagnostics/fetch_spy_vix.py` | Phase 1 data pipeline |
| `scripts/diagnostics/regime_detector_replay.py` | Phase 2 diagnostic driver |
| `scripts/diagnostics/ground_truth_labelers.py` | Phase 3 ground-truth labelers |
| `config/diagnostics/regime_events_2017_2026.csv` | Phase 3 hand-curated G4 labels |
| `tests/diagnostics/__init__.py` | Empty init marker |
| `tests/diagnostics/test_regime_detector_replay.py` | TDD tests for Phase 2 driver |
| `tests/diagnostics/test_ground_truth_labelers.py` | TDD tests for Phase 3 labelers |
| `notebooks/diagnostics/regime_detector_v0_analysis.ipynb` | Phase 4 analysis notebook |
| `diagnostics/data/spy_vix_2016_2026.parquet` | Phase 1 output (not committed; runtime artifact) |
| `diagnostics/regime/v0/labels.parquet` | Phase 2 output (committed; reference artifact) |
| `diagnostics/regime/ground_truth.parquet` | Phase 3 output (committed; reference artifact) |

Existing files referenced (read-only, no modifications):

- `src/strategies/advanced/market_regime_detector.py` — subject under test
- `src/backtesting/regimes/detector.py` — reused for ground-truth labelers
- `src/research/ramp_phase4/data.py` — possible reuse for the SPY+VIX panel loader
- `src/utils/logger.py` — standard logging

`.gitignore` does NOT currently exclude `diagnostics/`, `notebooks/diagnostics/`, or `config/diagnostics/`. We will force-add the reference Parquet artifacts (Phases 2 + 3) and notebook because they ARE the deliverable; the raw data Parquet (Phase 1) is regenerable and stays untracked.

## Detector signature & cached state (read directly from code)

Confirmed via direct inspection of `src/strategies/advanced/market_regime_detector.py`. Phase 0 must NOT regress from these facts; if it discovers contradictions, halt and re-plan.

```python
# Public entry point
def classify_regime(
    spy_data: pd.DataFrame,   # must have 'close' column
    vix_data: pd.DataFrame,   # must have 'close' column
    timestamp: datetime,
    *,
    min_coverage_pct: float = 0.95,
    hard_block_pct: float = 0.80,
) -> Tuple[str, float]:
    ...
# Returns (regime_name in {STRONG_BULL, WEAK_BULL, SIDEWAYS, UNPREDICTABLE, BEAR}, confidence in [0, 1])

# Constructor
def __init__(self, lookback_window: int = 252):
    self.last_indicators: Optional[Dict] = None       # populated after classify_regime
    self.last_regime_scores: Optional[Dict[str, float]] = None  # populated after classify_regime
```

**Critical correction to spec's H1**: The detector is NOT a 5-AND hard conjunction. It is an **argmax of soft scores**: `_score_regime(indicators, criteria)` returns a fraction in [0, 1] for each regime; the regime with the highest score wins. Each criterion contributes 1/N to the score if satisfied. This is closer to the spec's Option E ("score-based reformulation") than to the spec's Option A ("recalibrate the 5-AND thresholds").

Phase 0's writeup must document this correction prominently. Phase 5's option ranking must reflect that Option E is already partially the current design (the remaining work is replacing hard cutoffs in `_score_regime` with continuous activation functions).

---

## Task 1 (Phase 0): Code archaeology + writeup

**Files:**
- Create: `docs/reports/ramp/20260523_regime_detector_diagnostic.md` (placeholder with Phase 0 writeup as section 1)

**Goal**: Document what the detector actually does, in writing, before designing the driver.

- [ ] **Step 1.1: Read the full detector source**

```bash
wc -l src/strategies/advanced/market_regime_detector.py
# Expected: ~540 lines
```

Then `Read src/strategies/advanced/market_regime_detector.py` from top to bottom. Pay attention to:
- `MarketRegimeDetector.__init__` (line 91)
- `classify_regime` (line 111)
- `_calculate_indicators` (line 190)
- `_calculate_vix_percentile` (line 245)
- `_score_regime` (line 260)
- `REGIME_CRITERIA` dict (line 57)

- [ ] **Step 1.2: Grep all call sites**

```bash
grep -rn "MarketRegimeDetector\|classify_regime\|detect_regime" src/ --include="*.py" | grep -v test_
```

Expected callsites:
- `src/research/ramp_phase4/variants.py` — research harness
- `src/strategies/advanced/bayesian_reversion_model.py` — OMR
- `src/backtesting_v2/adapters/{omr,ramp}_adapter.py` — backtesting v2 framework (uses a different `detect_regime` method on a different class; verify these are not the production detector)

- [ ] **Step 1.3: Grep production data sources**

```bash
grep -rn "spy_data\|vix_data" src/strategies/advanced/ src/trading/adapters/ --include="*.py" | head -30
```

Note which adapters pass which kind of DataFrame (raw OHLCV vs panel slice).

- [ ] **Step 1.4: Verify Phase 4 research harness data path**

```bash
sed -n '36,90p' src/research/ramp_phase4/variants.py
```

This is the closest existing usage pattern to what Phase 2's driver will need to mirror. Note that variants.py slices `spy = panel['SPY'].dropna()`, builds a `spy_df` with `close` column from that Series, then calls `_DETECTOR.classify_regime(spy_df, vix_df, t)`. Phase 2's driver should follow this pattern exactly so that production-parity checks compare apples to apples.

- [ ] **Step 1.5: Write the Phase 0 writeup**

Create `docs/reports/ramp/20260523_regime_detector_diagnostic.md` with the following structure (only the Phase 0 section is filled in this task; Phase 5 synthesis is a separate task):

```markdown
# Regime Detector Diagnostic Report

**Status**: WIP -- Phase 0 complete
**Date**: 2026-05-23
**Author**: [implementer]
**Spec**: docs/superpowers/specs/2026-05-23-regime-detector-diagnostic-design.md

## Phase 0: Code archaeology

### What the detector actually does (read from code, not from docs)

The production `MarketRegimeDetector` at `src/strategies/advanced/market_regime_detector.py:37` is a soft-score argmax classifier, NOT the 5-AND conjunction described in the OMR strategy doc.

**Signature:**
- `classify_regime(spy_data, vix_data, timestamp, *, min_coverage_pct=0.95, hard_block_pct=0.80) -> (regime_name, confidence)`
- spy_data and vix_data are DataFrames with a `close` column.
- timestamp is unused in the current code path -- classification operates on the last row of each DataFrame.

**Algorithm:**
1. Compute indicators (SMAs at 20/50/200, momentum slope, VIX percentile over `lookback_window`, realized vol, volatility spike flag, MA slopes).
2. For each of 5 regimes in REGIME_CRITERIA, call `_score_regime(indicators, criteria)`.
3. Each criterion contributes 1/N to the score (where N is the count of criteria for that regime) if the indicator passes the threshold.
4. Return the regime with the maximum score.

**REGIME_CRITERIA structure (from lines 57-89):**

| Regime | Criteria |
|---|---|
| STRONG_BULL | momentum >= 0.02; VIX pct <= 30; above 20/50/200 SMA |
| WEAK_BULL | 0 <= momentum <= 0.02; VIX pct <= 50; above 20/50 SMA |
| SIDEWAYS | -0.01 <= momentum <= 0.01; 30 <= VIX pct <= 60 |
| UNPREDICTABLE | VIX pct >= 60; volatility_spike == True |
| BEAR | momentum <= -0.02; VIX pct >= 70; below 20/50/200 SMA |

### Why this is significant

The spec's H1 ("BEAR conjunction is structurally too restrictive") is partially based on the wrong model. The detector is already a score-based system. A regime can fire even if not all its criteria are satisfied -- it just needs to be the best fit relative to the other four. BEAR may still rarely win because its criteria are individually hard to satisfy (VIX pct >= 70 + below all three SMAs + momentum < -2%) AND because UNPREDICTABLE and SIDEWAYS often score higher even on bear-ish days.

H1 should be reframed: "BEAR is not the argmax winner often enough" (a relative score deficit) rather than "BEAR's AND-conjunction is too strict" (an absolute hard cutoff).

Phase 5's option ranking must reflect that Option E (score-based reformulation) is already partially the current design. The remaining design space for Option E is replacing hard pass/fail per criterion with continuous activation (e.g., sigmoid of `(VIX_pct - 70) / 10` instead of `VIX_pct >= 70`).

### Cached state (used by Phase 2 driver)

After each `classify_regime` call, the detector populates two instance attributes:
- `self.last_indicators: Dict[str, float|bool]` -- all values computed in `_calculate_indicators` (current_price, sma_{20,50,200}, above_{20,50,200}, momentum_slope, vix, vix_percentile, realized_vol, volatility_spike, sma_20_slope, sma_50_slope)
- `self.last_regime_scores: Dict[str, float]` -- the 5-element score vector before argmax

Phase 2's driver can read these directly to populate the labels.parquet columns without recomputing.

### Data path in production (verified via variants.py:36-90)

The Phase 4 research harness pattern -- which is the closest existing offline usage -- is:

1. Load a wide panel keyed by symbol from Alpaca SIP daily-aggregated data: `panel.SPY`, `panel.VIX` are Series.
2. Build per-symbol DataFrames on the fly: `spy_df = pd.DataFrame({'close': spy_series, ...})`.
3. Call `_DETECTOR.classify_regime(spy_df, vix_df, t)` with `t` as the current iteration timestamp.

Phase 2's driver mirrors this pattern but reads from a flat Parquet (Phase 1 output) rather than the harness panel.

### Open questions resolved

1. **Kalman vs raw SMAs**: raw SMAs. The memory note referencing Kalman was inaccurate or referred to a different file.
2. **VIX percentile window**: hard-coded at 252 days via `__init__(lookback_window=252)`. Parametrizable via constructor.
3. **Missing data**: `DataInsufficientError` raised when SPY close coverage < 80% or < 95% (depending on hard_block_pct vs min_coverage_pct). At < 80% (hard_block), no fallback. At 80-95% (soft_block), planner may use safe_mode.
4. **Volatility spike**: implemented as `VIX > 1.5 x VIX_20-day-average`. NOT documented in the OMR strategy doc.
5. **Test coverage**: TBD -- check `tests/strategies/advanced/` for any test_regime_detector* files in Step 1.6.
```

- [ ] **Step 1.6: Check existing test coverage of the detector**

```bash
find tests/ -name "test*regime*" -o -name "test*detector*" 2>&1
```

Document the results in the writeup's "Open questions resolved" section under item 5.

- [ ] **Step 1.7: Commit Phase 0**

```bash
git add docs/reports/ramp/20260523_regime_detector_diagnostic.md
git commit -m "diagnostic(regime): Phase 0 code archaeology writeup

Documents what the production MarketRegimeDetector actually does, vs
what the spec assumed. Critical correction: detector is a score-based
argmax classifier, not a 5-AND hard conjunction. This reshapes how H1
is interpreted and the relative weighting of options A vs E in Phase 5.

Detector signature, cached state, data-path mirror pattern from
variants.py:36-90, and open-question resolutions documented for Phase 2
driver implementation."
```

**Decision gate**: If Step 1.5's writeup reveals contradictions to the spec deeper than the H1 reframing already documented, halt and surface to the user before proceeding to Task 2. Examples requiring re-plan: detector returns 4 or 6 regimes not 5; uses Kalman after all; lookback is calendar-time not trading-day; volatility_spike has a completely different formula. The current code matches everything in the spec except the H1 framing.

---

## Task 2 (Phase 1): Data pipeline

**Files:**
- Create: `scripts/diagnostics/__init__.py`
- Create: `scripts/diagnostics/fetch_spy_vix.py`
- Output (untracked): `diagnostics/data/spy_vix_2016_2026.parquet`

**Goal**: Stage 2016-01-01 through latest-trading-day SPY+VIX OHLCV as a single Parquet. 2016 prefix covers the 252-day VIX percentile + 200-day SMA warm-up. Sanity-checked against a second source.

- [ ] **Step 2.1: Create the diagnostics package marker**

```bash
mkdir -p scripts/diagnostics tests/diagnostics
touch scripts/diagnostics/__init__.py tests/diagnostics/__init__.py
```

- [ ] **Step 2.2: Write the fetch script**

Create `scripts/diagnostics/fetch_spy_vix.py`:

```python
"""Stage SPY + VIX daily OHLCV for the regime detector diagnostic.

Outputs `diagnostics/data/spy_vix_2016_2026.parquet` with columns:
  date, spy_open, spy_high, spy_low, spy_close, spy_volume,
  vix_open, vix_high, vix_low, vix_close

SPY: Alpaca SIP daily aggregation (matches production source).
VIX: yfinance (Alpaca free tier does not carry VIX as a direct symbol).

Sanity check: pulls yfinance SPY as a second source and verifies daily
closes agree within 0.1% on every trading day. Stop condition on
mismatch.

Usage:
    PYTHONPATH=. python scripts/diagnostics/fetch_spy_vix.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.utils.logger import logger
from src.settings import get_local_storage_dir


START_DATE = datetime(2016, 1, 1)
END_DATE = datetime.now()
OUTPUT_PATH = Path('diagnostics/data/spy_vix_2016_2026.parquet')
CLOSE_TOLERANCE_PCT = 0.001  # 0.1% agreement required


def load_spy_alpaca() -> pd.DataFrame:
    """Load SPY daily OHLCV from the production Alpaca SIP cache."""
    storage = Path(get_local_storage_dir())
    parquet = storage / 'equities_daily_cache.parquet'
    if not parquet.exists():
        raise FileNotFoundError(
            f'Alpaca SIP daily cache not found at {parquet}. '
            f'Run scripts/data/download_symbols.py for SPY first.'
        )
    df = pd.read_parquet(parquet)
    spy = df[df['symbol'] == 'SPY'].copy() if 'symbol' in df.columns else df.copy()
    spy = spy.rename(columns=str.lower)
    spy = spy[(spy.index >= START_DATE) & (spy.index <= END_DATE)]
    return spy[['open', 'high', 'low', 'close', 'volume']].add_prefix('spy_')


def load_vix_yfinance() -> pd.DataFrame:
    """Load VIX daily OHLC from yfinance. CLAUDE.md exception: VIX has no
    Alpaca symbol; yfinance is the canonical project source via
    src/utils/vix_provider.py.
    """
    vix = yf.download('^VIX', start=START_DATE, end=END_DATE,
                      interval='1d', progress=False, auto_adjust=False)
    if vix.empty:
        raise RuntimeError('yfinance returned empty VIX dataframe')
    vix.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                   for c in vix.columns]
    return vix[['open', 'high', 'low', 'close']].add_prefix('vix_')


def sanity_check_spy(spy_alpaca: pd.DataFrame) -> None:
    """Pull SPY from yfinance and verify close agreement with Alpaca."""
    yf_spy = yf.download('SPY', start=START_DATE, end=END_DATE,
                         interval='1d', progress=False, auto_adjust=False)
    if yf_spy.empty:
        raise RuntimeError('yfinance returned empty SPY dataframe')
    yf_spy.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                      for c in yf_spy.columns]
    yf_close = yf_spy['close']
    alp_close = spy_alpaca['spy_close']
    common = yf_close.index.intersection(alp_close.index)
    if len(common) == 0:
        raise RuntimeError('No overlapping dates between Alpaca and yfinance SPY')
    diff = (yf_close.loc[common] - alp_close.loc[common]).abs()
    rel_diff = diff / alp_close.loc[common]
    mismatches = rel_diff[rel_diff > CLOSE_TOLERANCE_PCT]
    if len(mismatches) > 0:
        logger.error(
            f'SPY close mismatch on {len(mismatches)} of {len(common)} days: '
            f'worst rel_diff={rel_diff.max():.4%} on {rel_diff.idxmax().date()}'
        )
        logger.error(mismatches.head(20))
        raise RuntimeError(
            f'SPY close disagreement exceeds {CLOSE_TOLERANCE_PCT:.1%} '
            f'tolerance on {len(mismatches)} day(s); investigate before proceeding'
        )
    logger.info(
        f'[+] SPY sanity check: {len(common)} days, all within '
        f'{CLOSE_TOLERANCE_PCT:.1%} (worst {rel_diff.max():.4%})'
    )


def assert_nyse_trading_days(panel: pd.DataFrame) -> None:
    """Assert no missing trading days vs the NYSE calendar."""
    try:
        from pandas.tseries.offsets import BDay
        expected_start = panel.index.min()
        expected_end = panel.index.max()
        expected = pd.bdate_range(expected_start, expected_end, freq='C',
                                  holidays=pd.Series([], dtype='datetime64[ns]'))
        missing = expected.difference(panel.index)
        # Use a loose tolerance because BDay doesn't account for NYSE holidays.
        # If there are more than ~12 holidays per year missing, that's expected;
        # if there are way more, something is wrong.
        years = (expected_end - expected_start).days / 365.0
        threshold = int(years * 15) + 5
        if len(missing) > threshold:
            logger.warning(
                f'[!] {len(missing)} days in NYSE business-day range absent '
                f'from data; threshold was {threshold}. Investigate.'
            )
    except ImportError:
        logger.warning('[!] Could not import pandas calendar tools; skipping NYSE day check')


def main() -> int:
    logger.info(f'[+] Fetching SPY+VIX from {START_DATE.date()} to {END_DATE.date()}')

    spy = load_spy_alpaca()
    logger.info(f'[+] Loaded {len(spy)} SPY rows from Alpaca')

    vix = load_vix_yfinance()
    logger.info(f'[+] Loaded {len(vix)} VIX rows from yfinance')

    sanity_check_spy(spy)

    panel = spy.join(vix, how='inner')
    logger.info(f'[+] Joined panel: {len(panel)} rows')

    assert_nyse_trading_days(panel)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUTPUT_PATH)
    logger.info(f'[+] Wrote {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1e6:.2f} MB)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
```

- [ ] **Step 2.3: Run the fetch**

```bash
source /c/Users/qwqw1/anaconda3/etc/profile.d/conda.sh && conda activate fintech && PYTHONPATH=. python scripts/diagnostics/fetch_spy_vix.py 2>&1 | tail -20
```

Expected output: `[+] SPY sanity check: ~2400 days, all within 0.1% (worst <something small>)` and `[+] Wrote diagnostics/data/spy_vix_2016_2026.parquet (~0.3-0.5 MB)`.

Stop condition on:
- `SPY close disagreement exceeds 0.1% tolerance` -> investigate data source mismatch before any later phase.
- `Alpaca SIP daily cache not found` -> the SPY cache must be pre-warmed via `scripts/data/download_symbols.py`. If that fails, fall back to yfinance for SPY (with a documented deviation in the Phase 0 writeup).

- [ ] **Step 2.4: Verify the Parquet**

```bash
PYTHONPATH=. python -c "import pandas as pd; df = pd.read_parquet('diagnostics/data/spy_vix_2016_2026.parquet'); print(df.shape); print(df.columns.tolist()); print(df.head(3)); print(df.tail(3))" 2>&1
```

Expected:
- Shape ~(2400-2500, 9): one row per trading day, 9 columns (5 SPY OHLCV + 4 VIX OHLC).
- Columns: `['spy_open', 'spy_high', 'spy_low', 'spy_close', 'spy_volume', 'vix_open', 'vix_high', 'vix_low', 'vix_close']`.
- First row date around 2016-01-04 (first trading day of 2016).
- Last row date is today or yesterday.

- [ ] **Step 2.5: Commit Phase 1**

The Parquet itself is regenerable and stays untracked (under `diagnostics/data/`). Only the script is committed:

```bash
git add scripts/diagnostics/__init__.py scripts/diagnostics/fetch_spy_vix.py tests/diagnostics/__init__.py
git commit -m "diagnostic(regime): Phase 1 SPY+VIX data pipeline

scripts/diagnostics/fetch_spy_vix.py stages
diagnostics/data/spy_vix_2016_2026.parquet from Alpaca SIP (SPY) +
yfinance (VIX). Sanity check: yfinance SPY closes agree within 0.1%
of Alpaca on every day, otherwise raises and halts.

Output is regenerable and gitignored (~0.5 MB)."
```

**Decision gate**: If Step 2.3 fails the sanity check or returns less than 2200 rows, halt before Task 3.

---

## Task 3 (Phase 2): Diagnostic driver

**Files:**
- Create: `scripts/diagnostics/regime_detector_replay.py`
- Create: `tests/diagnostics/test_regime_detector_replay.py`
- Output (committed): `diagnostics/regime/v0/labels.parquet`

**Goal**: Day-by-day replay of `MarketRegimeDetector.classify_regime` across the full 2017-2026 sample. Capture detector outputs + all `last_indicators` + parametrized alternatives + branch_taken.

TDD: write tests first, then the driver.

- [ ] **Step 3.1: Write the test stub (failing)**

Create `tests/diagnostics/test_regime_detector_replay.py`:

```python
"""TDD tests for scripts/diagnostics/regime_detector_replay.py.

Tests use synthetic SPY+VIX data with known regime characteristics so that
expected detector outputs are predictable. Production parity is verified
in an integration test that pulls 10 real days from the staged Parquet.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.diagnostics.regime_detector_replay import (
    replay_one_day,
    replay_range,
    compute_alternative_vix_percentiles,
)


def _synthetic_panel(n_days: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2016-01-04', periods=n_days)
    spy = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n_days))
    vix = np.clip(15 + rng.normal(0, 4, n_days), 10, 50)
    return pd.DataFrame({
        'spy_open': spy * 0.999, 'spy_high': spy * 1.005,
        'spy_low': spy * 0.995, 'spy_close': spy, 'spy_volume': 1e8,
        'vix_open': vix, 'vix_high': vix * 1.02,
        'vix_low': vix * 0.98, 'vix_close': vix,
    }, index=dates)


def test_replay_one_day_returns_expected_columns():
    """replay_one_day produces a dict with all schema columns."""
    panel = _synthetic_panel(400)
    t = panel.index[-1]
    record = replay_one_day(panel, t)
    expected_keys = {
        'date', 'regime', 'confidence',
        'above_20', 'above_50', 'above_200', 'momentum_slope',
        'vix_close', 'vix_percentile_252d',
        'vix_percentile_63d', 'vix_percentile_126d', 'vix_percentile_504d',
        'realized_vol_20d', 'realized_vol_60d', 'vix_5d_ma_ratio',
        'branch_taken', 'spy_close', 'spy_drawdown_from_252d_high',
    }
    assert set(record.keys()) == expected_keys, (
        f'Missing: {expected_keys - set(record.keys())}; '
        f'Extra: {set(record.keys()) - expected_keys}'
    )


def test_replay_one_day_no_lookahead():
    """replay_one_day on date t must not use data after t."""
    panel = _synthetic_panel(400)
    t = panel.index[100]
    # Modify panel beyond t with sentinel values. Output should be identical.
    panel_clean = panel.copy()
    panel_polluted = panel.copy()
    panel_polluted.loc[panel.index > t] = np.nan
    rec_clean = replay_one_day(panel_clean, t)
    rec_poll = replay_one_day(panel_polluted, t)
    assert rec_clean['regime'] == rec_poll['regime']
    assert rec_clean['confidence'] == rec_poll['confidence']
    assert rec_clean['vix_percentile_252d'] == rec_poll['vix_percentile_252d']


def test_replay_one_day_idempotent():
    """Two calls on identical input produce identical output."""
    panel = _synthetic_panel(400)
    t = panel.index[-1]
    rec1 = replay_one_day(panel, t)
    rec2 = replay_one_day(panel, t)
    assert rec1 == rec2


def test_compute_alternative_vix_percentiles_returns_four_values():
    """The 63/126/252/504-day VIX percentiles all populate."""
    panel = _synthetic_panel(600)
    t = panel.index[-1]
    pcts = compute_alternative_vix_percentiles(panel, t)
    assert set(pcts.keys()) == {63, 126, 252, 504}
    for w, pct in pcts.items():
        assert 0.0 <= pct <= 100.0, f'pct[{w}d] = {pct} out of range'


def test_replay_range_writes_parquet(tmp_path: Path):
    """replay_range writes a Parquet partitioned by year."""
    panel = _synthetic_panel(800)
    output = tmp_path / 'labels.parquet'
    replay_range(panel, panel.index[300], panel.index[-1], output)
    df = pd.read_parquet(output)
    assert len(df) > 400
    assert {'regime', 'confidence', 'date'}.issubset(df.columns)


def test_replay_range_idempotency_check(tmp_path: Path):
    """Two runs produce byte-identical Parquets."""
    panel = _synthetic_panel(800)
    out1 = tmp_path / 'labels1.parquet'
    out2 = tmp_path / 'labels2.parquet'
    replay_range(panel, panel.index[300], panel.index[-1], out1)
    replay_range(panel, panel.index[300], panel.index[-1], out2)
    df1 = pd.read_parquet(out1).reset_index(drop=True)
    df2 = pd.read_parquet(out2).reset_index(drop=True)
    pd.testing.assert_frame_equal(df1, df2)
```

- [ ] **Step 3.2: Run the failing tests**

```bash
source /c/Users/qwqw1/anaconda3/etc/profile.d/conda.sh && conda activate fintech && python -m pytest tests/diagnostics/test_regime_detector_replay.py -v 2>&1 | tail -15
```

Expected: all 6 tests fail with `ModuleNotFoundError: No module named 'scripts.diagnostics.regime_detector_replay'`.

- [ ] **Step 3.3: Write the driver**

Create `scripts/diagnostics/regime_detector_replay.py`:

```python
"""Day-by-day replay of MarketRegimeDetector across 2017-2026.

Reads diagnostics/data/spy_vix_2016_2026.parquet, calls
MarketRegimeDetector.classify_regime for each trading day in 2017-2026,
and emits a Parquet with detector outputs + intermediate values +
parametrized alternatives, suitable for ad-hoc analysis.

Usage:
    PYTHONPATH=. python scripts/diagnostics/regime_detector_replay.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.strategies.advanced.market_regime_detector import (
    DataInsufficientError, MarketRegimeDetector,
)
from src.utils.logger import logger


INPUT_PATH = Path('diagnostics/data/spy_vix_2016_2026.parquet')
OUTPUT_PATH = Path('diagnostics/regime/v0/labels.parquet')
REPLAY_START = datetime(2017, 1, 1)
WARMUP_DAYS = 400  # 252 VIX pct lookback + 200 SMA + buffer
ALT_LOOKBACKS = (63, 126, 252, 504)


def compute_alternative_vix_percentiles(
    panel: pd.DataFrame, t: pd.Timestamp,
) -> Dict[int, float]:
    """For each lookback window, compute the percentile of VIX at t against
    the prior `window` days of VIX closes.
    """
    out = {}
    vix_close = panel['vix_close']
    current_vix = vix_close.loc[t]
    history = vix_close.loc[vix_close.index < t]
    for window in ALT_LOOKBACKS:
        sample = history.iloc[-window:]
        if len(sample) < window // 2:
            out[window] = float('nan')
            continue
        pct = float((sample < current_vix).sum() / len(sample) * 100.0)
        out[window] = pct
    return out


def _identify_branch(scores: Dict[str, float]) -> str:
    """Which regime won, formatted for the branch_taken column."""
    if not scores:
        return 'NO_SCORES'
    return max(scores, key=scores.get)


def replay_one_day(panel: pd.DataFrame, t: pd.Timestamp) -> Dict:
    """Replay the detector on a single date t.

    Args:
        panel: Full SPY+VIX panel indexed by date.
        t: The classification date (must be in panel.index).

    Returns:
        Dict matching the labels.parquet schema for this date.

    Strict point-in-time: slices panel to [t-400d, t] inclusive of t.
    """
    if t not in panel.index:
        raise KeyError(f'{t} not in panel.index')

    slice_start = t - timedelta(days=WARMUP_DAYS)
    pt_panel = panel.loc[slice_start:t]

    spy_df = pt_panel[['spy_open', 'spy_high', 'spy_low', 'spy_close', 'spy_volume']].copy()
    spy_df.columns = ['open', 'high', 'low', 'close', 'volume']
    vix_df = pt_panel[['vix_open', 'vix_high', 'vix_low', 'vix_close']].copy()
    vix_df.columns = ['open', 'high', 'low', 'close']

    detector = MarketRegimeDetector(lookback_window=252)
    try:
        regime, confidence = detector.classify_regime(spy_df, vix_df, t.to_pydatetime())
    except DataInsufficientError as e:
        # Insufficient data this early in the window; mark as SAFE_MODE so
        # the downstream notebook can filter or treat it as a sentinel.
        regime, confidence = 'SAFE_MODE', float('nan')

    indicators = detector.last_indicators or {}
    scores = detector.last_regime_scores or {}

    alt_pcts = compute_alternative_vix_percentiles(panel, t)

    # Realized vol and VIX MA ratio (computed independently, NOT from detector).
    spy_close = panel['spy_close']
    returns = spy_close.pct_change()
    rv20 = float(returns.loc[:t].iloc[-20:].std() * np.sqrt(252)) if returns.loc[:t].size >= 20 else float('nan')
    rv60 = float(returns.loc[:t].iloc[-60:].std() * np.sqrt(252)) if returns.loc[:t].size >= 60 else float('nan')

    vix_close = panel['vix_close']
    vix_5d_ma = vix_close.loc[:t].iloc[-5:].mean()
    vix_ratio = float(vix_close.loc[t] / vix_5d_ma) if vix_5d_ma > 0 else float('nan')

    spy_history = spy_close.loc[:t]
    if len(spy_history) >= 252:
        peak = spy_history.iloc[-252:].max()
        dd = float(spy_history.iloc[-1] / peak - 1.0)
    else:
        dd = float('nan')

    return {
        'date': t.date() if hasattr(t, 'date') else t,
        'regime': regime,
        'confidence': float(confidence) if confidence == confidence else float('nan'),
        'above_20': bool(indicators.get('above_20', False)),
        'above_50': bool(indicators.get('above_50', False)),
        'above_200': bool(indicators.get('above_200', False)),
        'momentum_slope': float(indicators.get('momentum_slope', float('nan'))),
        'vix_close': float(panel['vix_close'].loc[t]),
        'vix_percentile_252d': float(indicators.get('vix_percentile', float('nan'))),
        'vix_percentile_63d': float(alt_pcts.get(63, float('nan'))),
        'vix_percentile_126d': float(alt_pcts.get(126, float('nan'))),
        'vix_percentile_504d': float(alt_pcts.get(504, float('nan'))),
        'realized_vol_20d': rv20,
        'realized_vol_60d': rv60,
        'vix_5d_ma_ratio': vix_ratio,
        'branch_taken': _identify_branch(scores),
        'spy_close': float(panel['spy_close'].loc[t]),
        'spy_drawdown_from_252d_high': dd,
    }


def replay_range(panel: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
                 output: Path) -> pd.DataFrame:
    """Replay the detector across [start, end] inclusive and write Parquet."""
    if hasattr(start, 'to_pydatetime'):
        start_ts = start
    else:
        start_ts = pd.Timestamp(start)
    if hasattr(end, 'to_pydatetime'):
        end_ts = end
    else:
        end_ts = pd.Timestamp(end)

    dates_in_range = panel.index[(panel.index >= start_ts) & (panel.index <= end_ts)]
    records = []
    for i, t in enumerate(dates_in_range):
        if i % 250 == 0:
            logger.info(f'[+] Replaying day {i + 1}/{len(dates_in_range)}: {t.date()}')
        records.append(replay_one_day(panel, t))

    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    df['year'] = df['date'].dt.year
    df.to_parquet(output, partition_cols=['year'])
    logger.info(f'[+] Wrote {output} ({len(df)} rows)')
    return df


def main() -> int:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f'{INPUT_PATH} not found. Run scripts/diagnostics/fetch_spy_vix.py first.'
        )
    panel = pd.read_parquet(INPUT_PATH)
    logger.info(f'[+] Loaded {len(panel)} rows from {INPUT_PATH}')

    end = panel.index.max()
    df = replay_range(panel, pd.Timestamp(REPLAY_START), end, OUTPUT_PATH)

    logger.info(f'[+] Done. {len(df)} replay days. Regime distribution:')
    logger.info(df['regime'].value_counts().to_string())
    return 0


if __name__ == '__main__':
    sys.exit(main())
```

- [ ] **Step 3.4: Run the tests and see them pass**

```bash
python -m pytest tests/diagnostics/test_regime_detector_replay.py -v 2>&1 | tail -15
```

Expected: 6/6 PASS.

- [ ] **Step 3.5: Run the driver on the full sample**

```bash
PYTHONPATH=. python scripts/diagnostics/regime_detector_replay.py 2>&1 | tee /tmp/replay.log | tail -20
```

Expected wall-clock: 10-20 minutes for 2017-2026 (~2300 trading days; each `classify_regime` call recomputes indicators from a 400-day slice).

Expected final log lines: regime distribution counts (`STRONG_BULL`, `WEAK_BULL`, `SIDEWAYS`, `UNPREDICTABLE`, `BEAR`, possibly `SAFE_MODE` for the first ~10 days).

- [ ] **Step 3.6: Production-parity spot check**

Run the production detector against the same inputs for 10 random dates from 2024-2026 and verify the regime + confidence match the labels.parquet:

```bash
PYTHONPATH=. python -c "
import pandas as pd
from pathlib import Path
from datetime import timedelta
from src.strategies.advanced.market_regime_detector import MarketRegimeDetector

panel = pd.read_parquet('diagnostics/data/spy_vix_2016_2026.parquet')
labels = pd.read_parquet('diagnostics/regime/v0/labels.parquet')
labels['date'] = pd.to_datetime(labels['date'])
labels = labels.set_index('date').sort_index()

import numpy as np
rng = np.random.default_rng(7)
sample_idx = rng.choice(labels.loc['2024-01-01':].index, size=10, replace=False)

for t in sorted(sample_idx):
    expected = labels.loc[t, ['regime', 'confidence']]
    slice_start = t - timedelta(days=400)
    pt = panel.loc[slice_start:t]
    spy = pt[['spy_open', 'spy_high', 'spy_low', 'spy_close', 'spy_volume']].copy()
    spy.columns = ['open', 'high', 'low', 'close', 'volume']
    vix = pt[['vix_open', 'vix_high', 'vix_low', 'vix_close']].copy()
    vix.columns = ['open', 'high', 'low', 'close']
    d = MarketRegimeDetector(lookback_window=252)
    actual_regime, actual_conf = d.classify_regime(spy, vix, t.to_pydatetime())
    ok = actual_regime == expected['regime'] and abs(actual_conf - expected['confidence']) < 1e-9
    print(f'{t.date()}: replay={expected[\"regime\"]}/{expected[\"confidence\"]:.4f} '
          f'live={actual_regime}/{actual_conf:.4f} {\"OK\" if ok else \"MISMATCH\"}')
" 2>&1
```

Expected: all 10 lines end with `OK`. Any `MISMATCH` halts before Phase 3.

- [ ] **Step 3.7: Commit Phase 2**

```bash
git add scripts/diagnostics/regime_detector_replay.py tests/diagnostics/test_regime_detector_replay.py
git add -f diagnostics/regime/v0/labels.parquet/year=*
git commit -m "diagnostic(regime): Phase 2 driver + replay labels parquet

scripts/diagnostics/regime_detector_replay.py replays
MarketRegimeDetector.classify_regime day-by-day across 2017-2026 with
strict point-in-time discipline (slices to [t-400d, t]). Captures the
detector's regime + confidence + cached last_indicators +
last_regime_scores, plus independently-computed parametrized
alternatives (VIX pct at 63/126/252/504d, realized vol 20/60d, VIX 5d
MA ratio, SPY drawdown from 252d high).

Output: diagnostics/regime/v0/labels.parquet, Hive-partitioned by year.

Correctness checks: no-lookahead (synthetic test), idempotent
(byte-identical re-runs), production parity (10 random dates from
2024-2026 match the live detector exactly).

6 TDD tests in tests/diagnostics/test_regime_detector_replay.py."
```

**Decision gate**: If Step 3.6's parity check fails on any date, halt before Phase 3. The downstream analysis is meaningless if the driver doesn't faithfully replay production.

---

## Task 4 (Phase 3): Ground-truth labelers

**Files:**
- Create: `scripts/diagnostics/ground_truth_labelers.py`
- Create: `tests/diagnostics/test_ground_truth_labelers.py`
- Create: `config/diagnostics/regime_events_2017_2026.csv`
- Output (committed): `diagnostics/regime/ground_truth.parquet`

**Goal**: Build 4 independent labelers (G1 drawdown-based, G2 forward-window, G3 vol-spike, G4 hand-curated) and emit a single Parquet with one row per trading day.

- [ ] **Step 4.1: Create the hand-curated events CSV**

```bash
mkdir -p config/diagnostics
```

Create `config/diagnostics/regime_events_2017_2026.csv`:

```csv
event_name,start_date,end_date,event_type
Q4_2018_selloff,2018-10-03,2018-12-26,drawdown
Volmageddon_Feb_2018,2018-02-02,2018-02-09,vol_spike
COVID_crash,2020-02-19,2020-03-23,drawdown
COVID_vol_spike,2020-03-12,2020-03-16,vol_spike
2022_bear_market,2022-01-04,2022-10-13,drawdown
2025_tariff_drawdown,2025-02-19,2025-04-08,drawdown
2025_april_vol_spike,2025-04-04,2025-04-09,vol_spike
2025_dec_drawdown,2025-12-15,2026-02-03,drawdown
```

These are the events I'd expect any reasonable detector to identify. Adjust dates if your data shows different inflection points -- this is the most conservative ground truth, so prefer narrower windows when in doubt.

- [ ] **Step 4.2: Write the test stub (failing)**

Create `tests/diagnostics/test_ground_truth_labelers.py`:

```python
"""TDD tests for scripts/diagnostics/ground_truth_labelers.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.diagnostics.ground_truth_labelers import (
    label_g1_drawdown_bear,
    label_g2_forward_window_bear,
    label_g3_vol_spike,
    label_g4_hand_curated,
    build_ground_truth,
)


def _synthetic_panel_with_drawdown() -> pd.DataFrame:
    """Build a panel with a known 15% drawdown event."""
    dates = pd.bdate_range('2020-01-01', '2020-12-31')
    n = len(dates)
    # Sharp drop in Feb-Mar (mimics COVID): -25% from peak.
    prices = np.full(n, 100.0)
    peak_idx = 30
    trough_idx = 60
    prices[peak_idx:trough_idx] = np.linspace(100, 75, trough_idx - peak_idx)
    prices[trough_idx:] = np.linspace(75, 90, n - trough_idx)
    vix = np.full(n, 15.0)
    vix[peak_idx:trough_idx] = np.linspace(15, 60, trough_idx - peak_idx)
    return pd.DataFrame({
        'spy_close': prices,
        'vix_close': vix,
    }, index=dates)


def test_g1_drawdown_bear_fires_on_known_drawdown():
    panel = _synthetic_panel_with_drawdown()
    labels = label_g1_drawdown_bear(panel, threshold_pct=10.0, lookback_days=252)
    # Drawdown reaches -25% around day 60; G1_BEAR should fire there.
    assert labels.dtype == bool
    assert labels.loc[panel.index[60]] == True
    # Day 0: no drawdown yet (single price).
    assert labels.iloc[0] == False


def test_g2_forward_window_bear_uses_future_returns():
    panel = _synthetic_panel_with_drawdown()
    labels = label_g2_forward_window_bear(panel, fwd_days=30,
                                          ret_threshold=-0.05,
                                          vol_threshold=0.25)
    assert labels.dtype == bool
    # Day around peak should label True (forward 30d sees big drop + high vol).
    peak_t = panel.index[28]
    assert labels.loc[peak_t] == True
    # Very last days have no 30-day forward window -> False or NaN.
    assert labels.iloc[-1] == False


def test_g3_vol_spike_fires_on_vix_threshold():
    panel = _synthetic_panel_with_drawdown()
    labels = label_g3_vol_spike(panel, vix_abs_threshold=30.0, vix_5d_pct_threshold=0.5)
    # During the constructed VIX spike, label should fire.
    assert labels.dtype == bool
    # Mid-drawdown, VIX exceeded 30.
    mid_t = panel.index[55]
    assert labels.loc[mid_t] == True


def test_g4_hand_curated_assigns_event_types(tmp_path: Path):
    """G4 labels are populated from a CSV; verify event windows correctly mapped."""
    csv_path = tmp_path / 'events.csv'
    csv_path.write_text(
        'event_name,start_date,end_date,event_type\n'
        'test_dd,2020-02-01,2020-02-29,drawdown\n'
    )
    panel = _synthetic_panel_with_drawdown()
    labels = label_g4_hand_curated(panel, csv_path)
    assert isinstance(labels, pd.DataFrame)
    assert {'g4_event', 'g4_event_type'}.issubset(labels.columns)
    in_event = (panel.index >= '2020-02-01') & (panel.index <= '2020-02-29')
    assert (labels.loc[in_event, 'g4_event_type'] == 'drawdown').all()
    assert (labels.loc[~in_event, 'g4_event_type'].isna()).all()


def test_build_ground_truth_combines_all_four(tmp_path: Path):
    """End-to-end labeler emits one Parquet with all 4 labelers' columns."""
    csv_path = tmp_path / 'events.csv'
    csv_path.write_text(
        'event_name,start_date,end_date,event_type\n'
        'test_dd,2020-02-01,2020-02-29,drawdown\n'
    )
    panel = _synthetic_panel_with_drawdown()
    out = tmp_path / 'ground_truth.parquet'
    df = build_ground_truth(panel, csv_path, out)
    expected_cols = {
        'date', 'g1_bear', 'g2_bear', 'g3_vol_spike',
        'g4_event', 'g4_event_type',
    }
    assert expected_cols.issubset(df.columns)
    assert len(df) == len(panel)
```

- [ ] **Step 4.3: Run failing tests**

```bash
python -m pytest tests/diagnostics/test_ground_truth_labelers.py -v 2>&1 | tail -10
```

Expected: 5/5 fail with `ModuleNotFoundError`.

- [ ] **Step 4.4: Write the labelers**

Create `scripts/diagnostics/ground_truth_labelers.py`:

```python
"""Ground-truth regime labelers (G1, G2, G3, G4) for the diagnostic.

Each labeler operates on a SPY+VIX panel indexed by date. None look ahead
unless explicitly noted (G2 is forward-looking and IN-SAMPLE ONLY -- not
used for any live decision).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logger import logger


def label_g1_drawdown_bear(
    panel: pd.DataFrame,
    threshold_pct: float = 10.0,
    lookback_days: int = 252,
) -> pd.Series:
    """G1: SPY drawdown from trailing N-day high exceeds threshold."""
    spy = panel['spy_close']
    rolling_peak = spy.rolling(lookback_days, min_periods=1).max()
    dd = (spy / rolling_peak - 1.0) * 100.0
    return (dd <= -threshold_pct).rename('g1_bear')


def label_g2_forward_window_bear(
    panel: pd.DataFrame,
    fwd_days: int = 30,
    ret_threshold: float = -0.05,
    vol_threshold: float = 0.25,
) -> pd.Series:
    """G2: forward 30-day SPY return < -5% AND forward vol > 25%.

    FORWARD-LOOKING. In-sample only. Not for live decisions.
    """
    spy = panel['spy_close']
    fwd_ret = spy.shift(-fwd_days) / spy - 1.0
    returns = spy.pct_change()
    fwd_vol = returns.shift(-fwd_days).rolling(fwd_days).std() * np.sqrt(252)
    labels = (fwd_ret < ret_threshold) & (fwd_vol > vol_threshold)
    return labels.fillna(False).rename('g2_bear')


def label_g3_vol_spike(
    panel: pd.DataFrame,
    vix_abs_threshold: float = 30.0,
    vix_5d_pct_threshold: float = 0.5,
) -> pd.Series:
    """G3: VIX > absolute threshold OR VIX rose > pct over trailing 5 days."""
    vix = panel['vix_close']
    above_abs = vix > vix_abs_threshold
    rolling_pct = (vix / vix.shift(5)) - 1.0
    rapid_rise = rolling_pct > vix_5d_pct_threshold
    return (above_abs | rapid_rise).fillna(False).rename('g3_vol_spike')


def label_g4_hand_curated(
    panel: pd.DataFrame,
    csv_path: Path,
) -> pd.DataFrame:
    """G4: read hand-curated event windows from CSV.

    Returns a DataFrame with columns ['g4_event', 'g4_event_type'].
    Days outside any event window have NaN values.
    """
    events = pd.read_csv(csv_path, parse_dates=['start_date', 'end_date'])
    g4 = pd.DataFrame(index=panel.index, columns=['g4_event', 'g4_event_type'])
    for _, row in events.iterrows():
        mask = (panel.index >= row['start_date']) & (panel.index <= row['end_date'])
        g4.loc[mask, 'g4_event'] = row['event_name']
        g4.loc[mask, 'g4_event_type'] = row['event_type']
    return g4


def build_ground_truth(
    panel: pd.DataFrame,
    csv_path: Path,
    output: Path,
) -> pd.DataFrame:
    """Compute all 4 labelers and write a combined Parquet."""
    g1 = label_g1_drawdown_bear(panel)
    g2 = label_g2_forward_window_bear(panel)
    g3 = label_g3_vol_spike(panel)
    g4 = label_g4_hand_curated(panel, csv_path)
    df = pd.DataFrame({
        'date': panel.index,
        'g1_bear': g1.values,
        'g2_bear': g2.values,
        'g3_vol_spike': g3.values,
        'g4_event': g4['g4_event'].values,
        'g4_event_type': g4['g4_event_type'].values,
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output)
    logger.info(f'[+] Wrote {output} ({len(df)} rows)')
    logger.info(f'    g1_bear: {df["g1_bear"].sum()} days')
    logger.info(f'    g2_bear: {df["g2_bear"].sum()} days')
    logger.info(f'    g3_vol_spike: {df["g3_vol_spike"].sum()} days')
    logger.info(f'    g4_event: {df["g4_event"].notna().sum()} days')
    return df


def main() -> int:
    panel_path = Path('diagnostics/data/spy_vix_2016_2026.parquet')
    csv_path = Path('config/diagnostics/regime_events_2017_2026.csv')
    output = Path('diagnostics/regime/ground_truth.parquet')

    if not panel_path.exists():
        raise FileNotFoundError(panel_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    panel = pd.read_parquet(panel_path)
    panel = panel.loc['2017-01-01':]

    build_ground_truth(panel, csv_path, output)
    return 0


if __name__ == '__main__':
    sys.exit(main())
```

- [ ] **Step 4.5: Run tests and see them pass**

```bash
python -m pytest tests/diagnostics/test_ground_truth_labelers.py -v 2>&1 | tail -10
```

Expected: 5/5 PASS.

- [ ] **Step 4.6: Run the labeler on real data**

```bash
PYTHONPATH=. python scripts/diagnostics/ground_truth_labelers.py 2>&1 | tail -10
```

Expected output:
- `g1_bear: ~150-300 days` (out of ~2300; 6-13% of sample).
- `g2_bear: ~100-200 days` (forward-window, slightly smaller).
- `g3_vol_spike: ~150-300 days`.
- `g4_event: ~250-400 days` (depends on event window widths).

If any labeler reports 0 or > 50% of sample, something's wrong; halt before Phase 4.

- [ ] **Step 4.7: Commit Phase 3**

```bash
git add scripts/diagnostics/ground_truth_labelers.py tests/diagnostics/test_ground_truth_labelers.py
git add -f config/diagnostics/regime_events_2017_2026.csv
git add -f diagnostics/regime/ground_truth.parquet
git commit -m "diagnostic(regime): Phase 3 ground-truth labelers

Four independent labelers triangulating regime ground truth:
- G1: SPY drawdown from trailing 252d high > 10% (concurrent, observable)
- G2: forward 30d SPY ret < -5% AND forward vol > 25% (in-sample only)
- G3: VIX > 30 OR rose > 50% over 5d (vol spike, tests H2)
- G4: hand-curated event windows for 2018/2020/2022/2025-26

Output: diagnostics/regime/ground_truth.parquet (one row per trading day).
5 TDD tests in tests/diagnostics/test_ground_truth_labelers.py."
```

---

## Task 5 (Phase 4): Analysis notebook

**Files:**
- Create: `notebooks/diagnostics/regime_detector_v0_analysis.ipynb`

**Goal**: Jupyter notebook with 6 analyses (A-F), each producing plots + statistical summaries that bear on H1-H5.

**Note**: Notebook cells contain exploratory code; the plan documents intent + key code skeletons but not every line.

- [ ] **Step 5.1: Create the notebook scaffold**

```bash
mkdir -p notebooks/diagnostics
```

Create `notebooks/diagnostics/regime_detector_v0_analysis.ipynb` as a Jupyter notebook with the following cell structure. Use `nbformat` or write it as JSON directly. Recommended approach: use `jupytext` or write the Python source as `.py` first then convert via `jupyter nbconvert`, OR write JSON directly.

For implementer simplicity, write the notebook content as `notebooks/diagnostics/regime_detector_v0_analysis.py` first (paired notebook format), then convert. The `.py` source:

```python
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
JOIN = LABELS.join(GT, how='inner')
print(f'Loaded {len(JOIN)} day-rows; columns: {JOIN.columns.tolist()}')

FIG_DIR = Path('diagnostics/regime/v0/figures')
FIG_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Analysis A: Regime distribution (tests H1)
#
# % time in each regime, by year.

# %%
joined_year = LABELS.copy()
joined_year['year'] = joined_year.index.year
dist = joined_year.groupby(['year', 'regime']).size().unstack(fill_value=0)
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
plt.show()

bear_pct_total = (LABELS['regime'] == 'BEAR').mean() * 100
print(f'\nTOTAL BEAR %: {bear_pct_total:.2f}%')
print(f'H1 prediction "BEAR < 5% of any year": '
      f'{"SUPPORTED" if (dist_pct["BEAR"] < 5).all() else "REFUTED"}')

# %% [markdown]
# ## Analysis B: Run-length distribution (tests H4)

# %%
def run_lengths(series: pd.Series) -> pd.Series:
    """Return run lengths grouped by value: dict[value] -> list[int]."""
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
    ax.hist(sizes, bins=30)
    ax.set_title(regime)
    ax.set_xlabel('Run length (days)')
plt.tight_layout()
plt.savefig(FIG_DIR / 'analysis_B_runlengths.png', dpi=120)
plt.show()

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
plt.show()

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
plt.show()

# %% [markdown]
# ## Analysis F: Lookback-window sensitivity (tests H3)
#
# Re-classify substituting alternative VIX percentile lookbacks into the
# detector's BEAR threshold check. Count BEAR-firing days for each lookback.
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
plt.show()

# %% [markdown]
# ## Summary inputs for Phase 5 synthesis
#
# Numbers to carry into the synthesis report:

# %%
print('=== Summary for Phase 5 ===')
print(f'Total replay days: {len(LABELS)}')
print(f'BEAR % overall: {bear_pct_total:.2f}%')
print(f'BEAR % by year: {dist_pct["BEAR"].round(1).to_dict()}')

print('\nMedian run lengths:')
for regime in ['STRONG_BULL', 'WEAK_BULL', 'SIDEWAYS', 'UNPREDICTABLE', 'BEAR']:
    sizes = rl.loc[rl['first'] == regime, 'size']
    if len(sizes) > 0:
        print(f'  {regime}: {sizes.median():.1f}')

print(f'\nG1_BEAR days missed by detector: {len(mismatch)} '
      f'({len(mismatch) / JOIN["g1_bear"].sum() * 100:.1f}% of G1_BEAR days)')
print(f'Most common missed-BEAR failure mode: {fail_pct.idxmax()} '
      f'({fail_pct.max():.1f}%)')
print(f'Drawdown events where BEAR fired: '
      f'{lag_df["bear_in_window"].sum()} of {len(lag_df)}')
if valid_lags.size > 0:
    print(f'Median onset lag (days): {valid_lags.median():.1f}')
```

Convert to notebook:

```bash
PYTHONPATH=. python -c "
import jupytext
nb = jupytext.read('notebooks/diagnostics/regime_detector_v0_analysis.py')
jupytext.write(nb, 'notebooks/diagnostics/regime_detector_v0_analysis.ipynb')
"
```

If `jupytext` isn't installed, fall back to writing the notebook as JSON directly. Alternatively, just commit the `.py` and run it as a script -- the figures still produce.

- [ ] **Step 5.2: Run the notebook end-to-end**

```bash
PYTHONPATH=. jupyter execute notebooks/diagnostics/regime_detector_v0_analysis.ipynb 2>&1 | tail -20
```

Or if running the .py source directly:

```bash
PYTHONPATH=. python notebooks/diagnostics/regime_detector_v0_analysis.py 2>&1 | tee /tmp/phase4_analysis.log | tail -50
```

Expected:
- All 6 sections complete without exception.
- Each section prints summary stats; the final "Summary for Phase 5" block has concrete numbers.
- Six PNG figures written to `diagnostics/regime/v0/figures/`.

If any analysis throws an exception, fix it. If the output looks "too clean" (e.g., one hypothesis explains 100%), suspect a bug per the decision gate.

- [ ] **Step 5.3: Save the Phase 5 input data**

Take a screenshot or save the final summary block's stdout to `diagnostics/regime/v0/phase4_summary.txt`:

```bash
PYTHONPATH=. python notebooks/diagnostics/regime_detector_v0_analysis.py 2>&1 | grep -A 100 "Summary for Phase 5" > diagnostics/regime/v0/phase4_summary.txt
cat diagnostics/regime/v0/phase4_summary.txt
```

- [ ] **Step 5.4: Commit Phase 4**

```bash
git add -f notebooks/diagnostics/regime_detector_v0_analysis.ipynb notebooks/diagnostics/regime_detector_v0_analysis.py
git add -f diagnostics/regime/v0/figures/
git add -f diagnostics/regime/v0/phase4_summary.txt
git commit -m "diagnostic(regime): Phase 4 analysis notebook + figures

Six analyses (A-F) testing H1-H5:
- A: regime distribution by year (H1)
- B: run-length distribution per regime (H4)
- C: empirical transition matrix (H4, H1)
- D: lag-to-event for G4 drawdowns (H5)
- E: input ablation on G1_BEAR mismatch days (H1, H2; most actionable)
- F: VIX lookback sensitivity 63/126/252/504d (H3)

Outputs:
- notebooks/diagnostics/regime_detector_v0_analysis.ipynb
- diagnostics/regime/v0/figures/*.png (6 figures)
- diagnostics/regime/v0/phase4_summary.txt (Phase 5 input)"
```

**Decision gate**: Before Phase 5, sanity-review the summary. If one hypothesis explains everything with no residual, suspect a bug. Real diagnostics produce mixed evidence.

---

## Task 6 (Phase 5): Synthesis report

**Files:**
- Modify: `docs/reports/ramp/20260523_regime_detector_diagnostic.md` (extend with Phase 5 sections)

**Goal**: Append hypothesis verdicts (H1-H5) + remediation ranking (A-E) + next-step recommendation to the existing report (which already has the Phase 0 writeup at the top).

This task is human-judgment-heavy. The plan provides the template; the implementer fills in verdicts based on the Phase 4 summary.

- [ ] **Step 6.1: Read the Phase 4 summary**

```bash
cat diagnostics/regime/v0/phase4_summary.txt
```

- [ ] **Step 6.2: Open the existing report and append sections**

Append to `docs/reports/ramp/20260523_regime_detector_diagnostic.md` after the Phase 0 section:

```markdown
## Phase 5: Synthesis

### Hypothesis verdicts

Each verdict is one of {SUPPORTED, REFUTED, INCONCLUSIVE} backed by Phase 4 quantitative evidence. Differences in Sharpe-like statistics smaller than 0.2 are flagged as not statistically meaningful (per the May 2026 root-cause investigation's SE estimate of ~0.17 on EXT-OOS windows).

#### H1: BEAR conjunction structurally too restrictive

**Verdict**: [SUPPORTED / REFUTED / INCONCLUSIVE]

**Evidence**:
- BEAR % overall: [from summary]
- BEAR % by year (max): [from summary]
- H1 falsification check: "BEAR < 5% in any year" -> [SUPPORTED if all years < 5% -> H1 supported; REFUTED if any year > 5% -> H1 refuted]
- Median onset lag vs G4 drawdown events: [from summary] days

**Reframing**: Per Phase 0, the detector is a score-based argmax, not a hard 5-AND conjunction. H1 as stated is partially wrong. The corrected H1 is "BEAR is not the argmax winner often enough." [Add evidence on this corrected form.]

#### H2: UNPREDICTABLE dead zones in uptrends

**Verdict**: [SUPPORTED / REFUTED / INCONCLUSIVE]

**Evidence**:
- Days with VIX > 25 AND SPY > 50-SMA: [count from joined data]
- Of those, % labeled UNPREDICTABLE: [%]
- Of those, % labeled WEAK_BULL or STRONG_BULL: [%]

[If WEAK_BULL/STRONG_BULL dominates: H2 supported. Otherwise refuted.]

#### H3: 252-day VIX percentile compresses adaptively

**Verdict**: [SUPPORTED / REFUTED / INCONCLUSIVE]

**Evidence (from Analysis F)**:
- Days passing VIX pct >= 70 at lookback=63d: [count and %]
- Days passing at lookback=252d (production): [count and %]
- Days passing at lookback=504d: [count and %]

[If 63d and 504d differ materially from 252d on bear-ish windows -> H3 supported.]

#### H4: No hysteresis -> label flicker

**Verdict**: [SUPPORTED / REFUTED / INCONCLUSIVE]

**Evidence (from Analyses B + C)**:
- Median run-length per regime: [from summary]
- Regimes with median < 3 days: [count]
- Transition matrix diagonal mass: [average of diagonal entries from C]

[If 2+ regimes have median run < 3 days AND diagonal mass < 0.7 -> H4 supported.]

#### H5: SMA-based inputs lag regime onset

**Verdict**: [SUPPORTED / REFUTED / INCONCLUSIVE]

**Evidence (from Analysis D)**:
- Median lag to first BEAR label (drawdown events): [from summary] days
- Events with NO BEAR label fired in window: [from summary]

[If median lag > 15 trading days OR > 1/3 of events have no BEAR fire -> H5 supported.]

### Remediation option ranking

| Rank | Option | Rationale | Hypothesis support |
|---:|---|---|---|
| 1 | [option] | [why this option is most supported] | [list H1-H5 supporting this option] |
| 2 | [option] | ... | ... |
| 3 | [option] | ... | ... |
| 4 | [option] | ... | ... |
| 5 | [option] | ... | ... |

### Next-step recommendation

[One of:]
- (a) Regime detector v1 design (specify which remediation option to start with).
- (b) RAMP BEAR-day cash logic (per May 2026 root-cause; the detector may not be the bottleneck).
- (c) Both in parallel (separate spec docs).

**The critical reminder, repeated**: Any regime-detector improvement must be validated against the V1 baseline (vanilla momentum, no regime overlay) on EXT-OOS at 1.5x costs. If improved-detector + RAMP does not beat V1, the detector was not the bottleneck and Phase D paper validation of V11 (already in flight on the ramp-phase4-turnover-regime-research branch) remains the higher-priority work stream.

### Decision: stop or proceed

[Choose:]
- "Proceed to v1 detector design (separate brainstorm spec)."
- "Pause detector work. Prioritize RAMP BEAR-day cash logic."
- "Both proceed in parallel."
```

- [ ] **Step 6.3: Fill in the verdicts from Phase 4 outputs**

Read the Phase 4 figures and `phase4_summary.txt`, then fill in each `[SUPPORTED / REFUTED / INCONCLUSIVE]` placeholder and the evidence bullet points with concrete numbers. Do NOT leave any placeholder in the final report.

For the remediation ranking, weigh which hypotheses are most strongly supported and which options address them. Example mapping:
- If H4 strongly supported -> Option B (hysteresis) high.
- If H3 strongly supported -> Option C (lookback adjustment) high.
- If E (ablation) shows VIX percentile is the dominant failure mode -> Options A + C high; B + D + E low.
- If E shows failures spread across criteria -> Option E (score-based) high.

- [ ] **Step 6.4: Self-review the synthesis**

Read the report top to bottom. Verify:
- Every `[bracketed placeholder]` has been replaced with a concrete value or judgment.
- Each verdict cites specific numbers (not "many" or "few").
- The ranking is justified by at least 2 sentences of rationale per option.
- The next-step recommendation is one of the three options, not freeform.
- The Sharpe-SE-0.17 caveat is mentioned at least once in the verdicts where applicable.

If any of these fail, fix inline.

- [ ] **Step 6.5: Commit Phase 5**

```bash
git add docs/reports/ramp/20260523_regime_detector_diagnostic.md
git commit -m "diagnostic(regime): Phase 5 synthesis with H1-H5 verdicts and option ranking

Hypothesis verdicts:
- H1: [verdict]
- H2: [verdict]
- H3: [verdict]
- H4: [verdict]
- H5: [verdict]

Top remediation option: [option] (rationale)
Next-step recommendation: [(a)/(b)/(c)]

Statistical caveat carried throughout: Sharpe SE ~0.17 on EXT-OOS, so
differences < 0.2 are flagged as not meaningful. The 'detector != bottleneck'
caveat remains: regardless of option ranking, any detector revision must beat
V1 (no-regime baseline) at 1.5x costs to be worth deploying."
```

---

## Verification

End-to-end checks after all 6 tasks:

1. `git log --oneline regime-detector-diagnostic` shows 6 diagnostic commits past `d60686e`.
2. `python -m pytest tests/diagnostics/ -v` -- all tests pass.
3. `diagnostics/data/spy_vix_2016_2026.parquet` exists locally (untracked).
4. `diagnostics/regime/v0/labels.parquet/year=*` exists and is committed.
5. `diagnostics/regime/ground_truth.parquet` exists and is committed.
6. `diagnostics/regime/v0/figures/` contains 6 PNGs and is committed.
7. `docs/reports/ramp/20260523_regime_detector_diagnostic.md` contains both Phase 0 writeup AND Phase 5 synthesis with no `[bracketed placeholders]`.
8. Production-parity check from Step 3.6 passes (no MISMATCH lines).
9. Replay regime distribution sanity: STRONG_BULL + WEAK_BULL + SIDEWAYS dominate (typical 70-90%); BEAR + UNPREDICTABLE are minority (typical 5-20% combined).
10. Phase 5 verdicts include specific numbers from Phase 4, not vague language.

## What this plan does NOT do

- Build a v1 detector. The synthesis recommends WHAT to build, not the build itself.
- Modify any production code. Read-only against `src/strategies/advanced/market_regime_detector.py`.
- Re-run RAMP backtests with a hypothetical revised detector.
- Touch the V11 paper-validation work on the `ramp-phase4-turnover-regime-research` branch.
- Merge to main. Branch stays open through Phase 5; merge after the synthesis is reviewed by the user.

## Self-review checklist (run before declaring the plan complete)

Quick pass-through to confirm before writing-plans hands off to executing-plans / subagent-driven-development:

**Spec coverage** -- spec sections vs plan tasks:
- Problem statement -> Phase 0 writeup (Task 1)
- 5 hypotheses H1-H5 -> Analyses A-F (Task 5), verdicts (Task 6)
- 5 remediation options A-E -> Ranking in Task 6
- 3 goals (per-day record, ground-truth labels, synthesis) -> Tasks 3, 4, 6
- Phase 0-5 -> Tasks 1-6
- Risk table -> not explicitly tasked; mitigations live in decision gates after each task
- Success criteria -> Task 6 + Verification

**Placeholder scan**: only the synthesis report (Task 6) has placeholders, and they're explicitly handled by Steps 6.3-6.4 which require filling them in.

**Type consistency**: file paths consistent across tasks (`diagnostics/regime/v0/labels.parquet`, `diagnostics/regime/ground_truth.parquet`). Function names `replay_one_day`, `replay_range`, `compute_alternative_vix_percentiles`, `label_g1_drawdown_bear`, `label_g2_forward_window_bear`, `label_g3_vol_spike`, `label_g4_hand_curated`, `build_ground_truth` defined in Tasks 3-4 and tested in their respective tests.

Plan ready for execution.
