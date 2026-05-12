# Homeguard Backtesting Methodology

**Status**: Authoritative
**Owners**: Shuyang
**Last reviewed**: 2026-05-12
**Location**: `docs/methodology/backtesting.md`
**Read by**: `strategy-lead`, `strategy-architect`, `strategy-implementer`, `code-reviewer`, `backtest-driver`, `backtest-optimizer`, `portfolio-integrator`

This file is the single source of truth for backtest integrity, statistical methodology, cost modeling, optimization stopping conditions, portfolio integration thresholds, and reproducibility requirements. Every agent in the strategy pipeline reads it before doing any quantitative work.

When this file and an agent's own prompt disagree, **this file wins**. When this file and a paper disagree, the paper wins -- file an issue.

---

## Purpose and usage

### For agents

At the start of any task that involves backtesting, optimization, validation, or portfolio integration, read the section(s) of this file relevant to that task. Section headers are stable and unique -- grep for them. Do not paraphrase rules from memory; reference them from this file.

When dispatching subagents, pass the line *"Consult `docs/methodology/backtesting.md` Section N before proceeding"* rather than copying the rules. Subagents must read this file directly.

### For Shuyang

When you want to change a rule (e.g., adjust a threshold, switch cost models, add a new asset class), edit this file and bump the changelog. Do not edit individual agent prompts to encode methodology changes -- they will drift.

When you find a disagreement between this file and an agent's behavior in practice, the agent has a bug. Either the agent isn't reading this file, or it's reading and ignoring it. Either way, fix the agent, not the methodology.

### Section index

1. Bias prevention -- what must never be done in a backtest
2. Statistical framework -- Sharpe, PSR, DSR, PBO with correct formulas
3. Walk-forward methodology -- purging, embargo, window construction
4. Cost models -- per asset class with explicit formulas
5. Stopping conditions -- when "best of their abilities" is reached
6. Portfolio integration -- correlation, marginal Sharpe, capacity
7. Data quality and point-in-time conventions
8. Reproducibility requirements
9. Experiment registry schema
10. Homeguard-specific reference (paths, regimes, brokers, data)

---

## Section 1: Bias prevention

These rules apply to every backtest and every optimization run. Violations invalidate results. The code-reviewer's job is to verify these on every strategy implementation.

### 1.1 Lookahead bias

**Rule**: At each decision timestamp `t`, a signal may only use information knowable strictly before `t`.

**Required patterns**:

```python
# Signal computation
signal_t = (sma_50.shift(1) > sma_200.shift(1))   # use shifted features
trade_decision_t = signal_t.iloc[t]                # decision at t uses t-1 features

# Future return for backtest evaluation (separate from decision)
forward_return_t = close.pct_change().shift(-1)    # tomorrow's return, evaluated only after t
```

**Forbidden patterns**:

| Pattern | Why it fails |
|---|---|
| `df['close'].pct_change()` used as same-bar signal | Today's return is unknown at today's open |
| Full-sample `.mean()`, `.std()`, `.quantile()` for thresholds | Uses future observations |
| Full-sample z-score normalization | Same problem; use expanding or rolling |
| `.fillna(method='bfill')` on price data | Backfills future values into past |
| Joining data without timestamp checks | Fundamentals dated 2024-03-31 may not be public until 2024-06-15 |
| Same-day news -> same-day signal | News timestamps must be respected at intraday resolution |
| Index membership "as of today" for historical backtests | Use point-in-time membership |

**Code-reviewer responsibilities**: For every signal computation in `src/strategies/`, verify:
- All features end with `.shift(k)` for `k >= 1` where appropriate
- No statistic computed across the full sample is used in a decision
- No `bfill` / `interpolate(method='time')` that could backfill future values
- Joins of multi-source data (prices + fundamentals + news) preserve timestamp order

### 1.2 Survivorship bias

**Rule**: The backtest universe at time `t` must include every symbol that *was* tradeable at `t`, not just those tradeable today.

**Mitigations**:

- For S&P 500 and similar index-based universes: use point-in-time membership lists. If unavailable, apply an empirical haircut to CAGR (see Section 7.2).
- For ETFs: the Homeguard ETF universes (OMR uses 20 leveraged ETFs) are largely surviving issues; survivorship is a smaller concern but still nonzero. Document which delisted ETFs you may have missed.
- For futures: continuous contracts (`.c.0`) handle roll automatically; ensure roll methodology is documented (calendar roll vs. open-interest roll).
- For crypto: include delisted pairs. Several major exchanges have delisted tokens that were tradeable historically.

**Sensitivity check**: Re-run the top configuration excluding the top 5 contributing symbols. If Sharpe degrades by more than 30%, the result is fragile to a few names -- flag as concentration-dependent.

### 1.3 Selection bias

**Rule**: The set of strategies, parameters, and universes evaluated must be specified before evaluation, not chosen after seeing results.

**Common violations**:

- "We tried 47 parameter combinations but only report the top 5" -- the comparison is over 47, not 5; multiple-testing correction must use 47 (see Section 2.4).
- "We tried 12 universes and report the best" -- same issue; the trial count is 12.
- "We selected the period 2020-2024 because it's most relevant" -- the comparison universe must include the periods you rejected. Use full data coverage instead (Section 1.6).

**Practical rule**: Every grid search must record `combinations_tested` in the experiment registry. DSR and PBO computations use the *project-wide* cumulative trial count, not the per-run count.

### 1.4 Normalization leakage

**Rule**: Any statistical transformation of features (z-score, robust scaling, ranking, winsorization) must use only past data at each decision point.

```python
# WRONG: uses full sample
z = (df['feature'] - df['feature'].mean()) / df['feature'].std()

# CORRECT: expanding window
z = (df['feature'] - df['feature'].expanding(min_periods=252).mean()) / df['feature'].expanding(min_periods=252).std()

# CORRECT: rolling window (preferred for non-stationary features)
z = (df['feature'] - df['feature'].rolling(252).mean()) / df['feature'].rolling(252).std()
```

For cross-sectional ranks (e.g., momentum across 500 symbols at time `t`), the rank is computed across symbols *at* `t` using only data available at `t`. This is acceptable because the rank uses no future data, only cross-sectional contemporary data.

### 1.5 Burn-in contamination

**Rule**: When a feature requires `N` bars of history (e.g., SMA(200) requires 200 prior bars), the first `N-1` bars of the OOS test period have features that include in-sample data. This is acceptable but must be documented.

**Mitigation**: For walk-forward, the test window does not include trades during the first `lookback` bars. Either skip those bars entirely or document that early-test signals are not independent of training data.

### 1.6 Full data coverage

**Rule**: Backtests must use the maximum available history for each symbol, not a cherry-picked subset.

**Procedure**:
1. For each symbol in the universe, query the earliest available date in `equities_1day/`, `futures_1min/`, etc.
2. Backtest start = max(earliest available, strategy lookback requirement)
3. If a symbol's history is shorter than the others, it enters the universe when its data becomes available -- the engine must handle a time-varying universe.
4. Reserve the most recent 12 months as a frozen out-of-sample period that is never touched during strategy development or optimization.

**Why this matters**: Earlier in this project, a strategy showed Sharpe 0.698 over 2022-2024 and Sharpe -0.767 over 2018-2024. The 3-year window was a regime artifact. Short-window backtests produce false confidence.

**Enforcement**: The orchestrator (strategy-lead) verifies the backtest's `window_start` is within 30 days of the earliest available data, otherwise flags the run as window-restricted.

### 1.7 Vol-target leakage

**Rule**: Volatility used to size a position at time `t` must use only data strictly before `t`.

```python
# WRONG: same-bar realized vol
vol_t = returns.rolling(20).std().iloc[t]
position_t = target_vol / vol_t

# CORRECT
vol_t = returns.rolling(20).std().shift(1).iloc[t]
position_t = target_vol / vol_t
```

---

## Section 2: Statistical framework

This section defines Sharpe, PSR, DSR, and PBO with explicit formulas. The previous version of the agent prompts confused these -- particularly DSR -- so every formula below is stated in full.

### 2.1 Sharpe ratio

The Sharpe ratio of a return series `r_t` over `n` observations:

$$\hat{SR} = \frac{\bar{r} - r_f}{\hat{\sigma}}$$

where:
- $\bar{r}$ is the sample mean return
- $r_f$ is the period risk-free rate (set to 0 for relative comparison, or use 3-month T-bill for absolute)
- $\hat{\sigma}$ is the sample standard deviation of returns

**Annualization**: $SR_{annual} = SR_{period} \times \sqrt{P}$ where $P$ is the number of periods per year (252 for daily, 12 for monthly, etc.). Do not annualize by simply multiplying -- use the square root.

**Convention in Homeguard reports**: Daily returns, $r_f = 0$, annualized by $\sqrt{252}$. State this explicitly in the report header. When $r_f > 0$ matters (post-2022 environment), recompute and state both.

### 2.2 Probabilistic Sharpe Ratio (PSR)

PSR estimates the probability that the true Sharpe ratio exceeds a benchmark $SR^*$, given the observed sample. From Bailey & Lopez de Prado (2012):

$$PSR(SR^*) = \Phi\left(\frac{(\hat{SR} - SR^*)\sqrt{n - 1}}{\sqrt{1 - \hat{\gamma}_3 \hat{SR} + \frac{\hat{\gamma}_4 - 1}{4}\hat{SR}^2}}\right)$$

where:
- $\hat{SR}$ is the observed Sharpe (same units as $SR^*$ -- both per-period or both annualized)
- $n$ is the number of return observations
- $\hat{\gamma}_3$ is the sample skewness of returns
- $\hat{\gamma}_4$ is the sample kurtosis (Pearson's, normal = 3, not excess)
- $\Phi$ is the standard normal CDF

**Interpretation**: PSR returns a probability in $[0, 1]$. A PSR of 0.95 means "95% confident the true Sharpe exceeds the benchmark."

**Standard threshold**: Reject the null of $SR \leq SR^*$ when $PSR > 0.95$. For most uses set $SR^* = 0$.

**Why the moment terms matter**: For Gaussian returns, the denominator simplifies to $\sqrt{1 + \frac{1}{2}\hat{SR}^2}$. For real strategies -- especially mean-reversion and short-vol -- return distributions have negative skew and high kurtosis, which *increase* the denominator and *decrease* PSR. Strategies that look good on Sharpe alone often fail PSR because their tail behavior is bad. This is exactly the property we want to catch.

### 2.3 Deflated Sharpe Ratio (DSR)

DSR adjusts PSR for the multiple-testing bias introduced by trying $N$ configurations and reporting only the best. From Bailey & Lopez de Prado (2014):

$$DSR = PSR(SR_0^*)$$

where the benchmark $SR_0^*$ is the expected maximum Sharpe under the null after $N$ independent trials:

$$SR_0^* = \sqrt{V[\{\hat{SR}_n\}]} \cdot \left[(1 - \gamma_{EM}) \cdot \Phi^{-1}\left(1 - \frac{1}{N}\right) + \gamma_{EM} \cdot \Phi^{-1}\left(1 - \frac{1}{N} \cdot e^{-1}\right)\right]$$

where:
- $V[\{\hat{SR}_n\}]$ is the variance of the Sharpe ratios across the $N$ trials
- $\gamma_{EM} \approx 0.5772$ is the Euler-Mascheroni constant
- $\Phi^{-1}$ is the inverse normal CDF
- $N$ is the cumulative project-wide trial count (from the experiment registry)

**Critical**: $N$ is project-wide, not run-specific. If you've optimized 10 strategies with 500 configs each, $N = 5000$ for any new optimization, not 500.

**Interpretation**: DSR is a probability. Pass threshold is the same as PSR: $DSR > 0.95$.

**What the old "Sharpe x (1 - ln(N)/(2T))" formula was**: A simple Bonferroni-style haircut. It has the right qualitative direction (lower Sharpe with more trials) but misses two things DSR captures: (1) the *variance* of trial Sharpes matters -- if all your trials cluster around similar Sharpes, the max isn't very inflated, but if they spread widely, it is; (2) the moment correction from PSR. The old formula would let through fat-tailed strategies that DSR catches.

**Reference implementation**:

```python
import numpy as np
from scipy.stats import norm

def psr(sr_hat, sr_benchmark, n, skew, kurt):
    """Probabilistic Sharpe Ratio. sr in same units as sr_benchmark."""
    denom = np.sqrt(1 - skew * sr_hat + ((kurt - 1) / 4) * sr_hat**2)
    z = (sr_hat - sr_benchmark) * np.sqrt(n - 1) / denom
    return norm.cdf(z)

def expected_max_sharpe(trial_sharpes, n_trials):
    """E[max SR] across N trials under the null. Bailey & Lopez de Prado 2014."""
    em = 0.5772156649  # Euler-Mascheroni
    v = np.var(trial_sharpes, ddof=1)
    return np.sqrt(v) * ((1 - em) * norm.ppf(1 - 1/n_trials)
                         + em * norm.ppf(1 - 1/(n_trials * np.e)))

def dsr(sr_hat, trial_sharpes, n, skew, kurt, n_trials_project):
    """Deflated Sharpe Ratio. Use project-wide cumulative trial count for n_trials_project."""
    sr_zero = expected_max_sharpe(trial_sharpes, n_trials_project)
    return psr(sr_hat, sr_zero, n, skew, kurt)
```

### 2.4 Probability of Backtest Overfitting (PBO)

PBO from Bailey, Borwein, Lopez de Prado, Zhu (2017) uses Combinatorially Symmetric Cross-Validation (CSCV) to estimate the probability that the best in-sample strategy underperforms the median out-of-sample.

**Procedure**:

1. Construct matrix $M$ of size $T \times N$ where $T$ is the number of return observations and $N$ is the number of strategy configurations evaluated. Cell $M_{t,n}$ is the return of config $n$ at time $t$.
2. Partition $M$ by rows into $S$ submatrices of equal size (typically $S = 16$).
3. Enumerate all $\binom{S}{S/2}$ combinations of $S/2$ submatrices to use as in-sample.
4. For each combination $c$:
   - Concatenate the $S/2$ in-sample submatrices into matrix $J_c$; the rest become $\bar{J}_c$.
   - Find $n^* = \arg\max_n \text{Sharpe}(J_c[:, n])$ -- the best config in-sample.
   - Compute the relative rank of $n^*$ in $\bar{J}_c$ (rank by Sharpe; 1.0 = best, 0.0 = worst).
   - Compute logit: $\lambda_c = \log\frac{w_c}{1 - w_c}$ where $w_c$ is the relative rank.
5. PBO is the fraction of $c$ where $\lambda_c < 0$ -- i.e., the in-sample best performed below median out-of-sample.

**Thresholds**:

| PBO | Interpretation |
|---|---|
| < 0.25 | Acceptable -- in-sample selection generalizes |
| 0.25 - 0.50 | Concerning -- selection bias is meaningful |
| > 0.50 | Strong overfitting -- your "best" performs worse than random selection out-of-sample |

**When to compute**: PBO is required whenever an optimization run evaluated more than 20 configurations and you intend to act on the top-N. The backtest-optimizer must produce and report PBO at the end of every parameter sweep.

### 2.5 Combined statistical gate

A strategy passes the statistical gate for live consideration when *all* of:

- $PSR(0) > 0.95$ on the OOS window -- true Sharpe likely positive
- $DSR > 0.95$ using project-wide trial count -- survives multiple-testing correction
- $PBO < 0.25$ -- best-in-sample generalizes
- Trade count $\geq 30$ on OOS -- sufficient sample for statistical inference
- OOS / IS Sharpe ratio $\geq 0.7$ -- performance degrades less than 30%

These five together are far more selective than any single Sharpe threshold. They are the operational definition of "statistically significant edge" for Homeguard.

### 2.6 Why we don't use magic-number thresholds

Earlier methodology used thresholds like "Sharpe > 3.0 REJECT" and "CAGR > 20% INVESTIGATE." These are removed. Reasons:

- Sharpe > 3.0 is achievable in some genuine strategies (high-frequency, certain stat arb). The right response is *investigate intensely*, not *reject*.
- CAGR depends on leverage, asset class, and risk budget. A 20% CAGR for an unlevered equity strategy is suspicious; for a 3x leveraged ETF strategy with appropriate drawdown, it's normal.
- The combined statistical gate (Section 2.5) does the work these thresholds were trying to do, but in a principled way that scales with strategy characteristics.

When a strategy looks too good, run the gate. Don't trust the gut feel of a threshold.

---

## Section 3: Walk-forward methodology

This section defines purging, embargo, and window construction. Earlier methodology defined embargo as "= feature lookback" -- that is wrong and is corrected here.

### 3.1 Walk-forward structure

Use **anchored expanding windows** by default; use **rolling windows** when the strategy must adapt to regime changes and stationarity of features over long horizons is suspect.

For an anchored walk-forward with $K$ folds:

```
Fold 1: train [t_0, t_0 + T_min],  test [t_0 + T_min, t_0 + T_min + T_test]
Fold 2: train [t_0, t_0 + T_min + T_test], test [t_0 + T_min + T_test, t_0 + T_min + 2*T_test]
...
Fold K: train [t_0, t_K - T_test], test [t_K - T_test, t_K]
```

Minimum folds: **5**. Fewer folds than 5 do not give enough degrees of freedom for stable aggregate statistics.

### 3.2 Purging

**Definition**: Remove from the training set any observation whose label-determination window overlaps with the test set.

**Why**: If the strategy's label is "20-day forward return," then an observation at time `t_i` "knows about" returns through `t_i + 20`. If the test set starts at `t_k` and `t_i + 20 > t_k`, then the training observation at `t_i` has information that overlaps the test set. Including it in training leaks future into past.

**Formula**: For a label horizon $h$, purge training observations $i$ where $t_i \in [t_k - h, t_k + T_{test}]$.

**Practical**: For daily-rebalance strategies, $h = 1$ and purging removes just the last training day. For strategies with longer holds (OMR: ~17 hours; momentum: 21 days), purge accordingly.

### 3.3 Embargo

**Definition**: A small additional gap, immediately after the test set, that is excluded from the *next* training window. This defends against serial correlation in features that aren't strictly point-in-time.

**Formula**: $\text{embargo} = \epsilon \cdot T$ where $\epsilon \in [0.01, 0.05]$ (1% to 5% of total observations). For daily data with 10 years of history ($T \approx 2520$), embargo is 25-125 trading days.

**Embargo is NOT the feature lookback**. Setting embargo = lookback (e.g., 200 days for SMA-200) would either consume the test set or, worse, be applied to the wrong side. The feature lookback controls when features start producing valid values; it has nothing to do with embargo.

### 3.4 Aggregation

Aggregate fold results by **pooling returns**, not by averaging Sharpes.

```python
# WRONG
mean_sharpe = np.mean([fold.sharpe for fold in folds])

# CORRECT
all_oos_returns = pd.concat([fold.oos_returns for fold in folds])
pooled_sharpe = sharpe(all_oos_returns)
```

Reason: averaging Sharpes is biased; pooling returns gives the correct asymptotic estimator.

Per-fold Sharpes are still reported alongside the pooled estimate, to show stability across folds.

### 3.5 IS/OOS ratio by parameter count

The training period must be longer than the test period in proportion to the number of free parameters:

| Free parameters | IS:OOS ratio |
|---|---|
| 1 | 2 : 1 |
| 2 | 3 : 1 |
| 3 | 4 : 1 |
| >= 4 | flag -- too many parameters; see Section 5.4 |

These are minimums, not targets. More training data is generally better, subject to non-stationarity concerns.

### 3.6 Implementation

Use `WalkForwardValidator` at `src/backtesting/chunking/walk_forward.py`. It exposes:

```python
WalkForwardValidator(
    engine=engine,
    train_months=12,        # minimum, scale with parameter count
    test_months=3,
    step_months=3,
    purge_days=...,         # = label horizon
    embargo_pct=0.02,       # 2% default
)
```

If you need behavior the validator doesn't support, extend it rather than reimplementing. Do not write a one-off walk-forward in a backtest script.

---

## Section 4: Cost models by asset class

Costs are mandatory in every backtest. A strategy that looks good before costs is not a strategy.

### 4.1 Equities

**Round-trip cost** = spread cost + commission + market impact.

| Liquidity tier | Round-trip total | Examples |
|---|---|---|
| Large-cap liquid | 5-15 bps | SPY, QQQ, AAPL, MSFT |
| Mid-cap | 15-30 bps | Russell 2000 names, sector ETFs |
| Leveraged ETFs | 15-30 bps | TQQQ, SOXL, SPXU |
| Small-cap / illiquid | 50-200+ bps | Micro-caps, low ADV |

**Commission**: For IBKR retail tiered: ~$0.0035/share, $0.35 minimum. For Alpaca: zero on tier; account for SEC/TAF fees on sells (~$0.000166 + $0.000145 per share, small but nonzero).

**Market impact** (for orders > 1% of ADV): use the square-root model:

$$\text{impact} = \sigma_{daily} \cdot \eta \cdot \sqrt{X / V_{daily}}$$

where $X$ is order size in shares, $V_{daily}$ is average daily volume, $\sigma_{daily}$ is daily return volatility, and $\eta \approx 0.5$ is an empirical impact coefficient.

**Position limits**: Single order $\leq 5\%$ of ADV. Participation rate $\leq 25\%$ of contemporaneous minute volume.

### 4.2 Futures (GLBX continuous .c.0)

**Per-contract costs**:

- IBKR commission: ~$0.85-$1.50 per side depending on contract
- Exchange + regulatory fees: ~$1.00-$2.50 per side
- Half-tick slippage typical for ES, NQ during regular hours; one-tick for off-hours

**Roll costs** (continuous contracts): The roll adjustment in the `.c.0` series should be calendar-based (not back-adjusted on price). When the strategy holds across a roll date, model the roll spread explicitly as an additional one-half-tick adjustment.

**Available in Homeguard**: ES, NQ, YM, RTY (equity indices), CL (oil), GC (gold), ZN (10Y notes), 6E (EUR/USD futures), ZC (corn).

### 4.3 FX

**Spread costs** in pips: typical retail tiers around 0.5-1.5 pips for major pairs during liquid sessions, widening to 2-5 pips during low-liquidity overnight hours.

**Commission**: ECN tiers ~$2-7 per $1M notional; retail spread-only tiers absorb this in the spread.

**Session sensitivity**: London/NY overlap (8:00-12:00 ET) has the tightest spreads. Asian hours and weekend opens are widest. Cost model should be session-aware for intraday strategies.

**Available in Homeguard**: 50 pairs including G10 majors, crosses, EM pairs across LatAm/Asia-Pacific/CEE-EMEA/Africa, plus XAU/USD and XAG/USD.

### 4.4 Crypto

**Maker/taker fees** (Coinbase Advanced Trade, retail base tier): 0.6% taker / 0.4% maker. Scales down with 30-day volume.

**Spread**: Major pairs (BTC, ETH) on Coinbase: 1-5 bps in normal conditions. Altcoins: highly variable, often 10-50 bps.

**Funding rates** (perps only): Separate from execution cost but affects holding cost. Not currently traded by Homeguard (CSCM uses Coinbase spot).

**Round-trip cost convention**: $60$ bps taker / $60$ bps taker = $120$ bps round-trip for naive retail. Use this as default for CSCM unless specific tier is verified.

**Available in Homeguard**: 33 pairs.

### 4.5 Options

Options costs are structurally different from underlyings.

**Per-contract commission** (IBKR retail): $0.50-$0.65 per contract.

**Slippage model**: Fill price relative to NBBO midpoint:

$$\text{fill}_{buy} = \text{mid} + \alpha \cdot \frac{1}{2}(\text{ask} - \text{bid})$$
$$\text{fill}_{sell} = \text{mid} - \alpha \cdot \frac{1}{2}(\text{ask} - \text{bid})$$

with $\alpha$ depending on liquidity and aggressiveness:

| Liquidity | $\alpha$ | Example |
|---|---|---|
| Very liquid (SPY ATM) | 0.3-0.5 | SPY weekly ATM |
| Liquid index ETF | 0.5-0.7 | QQQ, IWM ATM |
| Single-stock ATM | 0.7-1.0 | AAPL, MSFT |
| Wings / illiquid | 1.0+ (cross full spread) | OTM, far-dated |

This is the *fraction of the half-spread crossed*. $\alpha = 1.0$ means you cross the half-spread fully -- you pay/receive the ask/bid. $\alpha > 1$ for very illiquid options where even the NBBO doesn't fully represent fillable price.

**Do not** use "50-75% of the bid-ask" as slippage -- that's ambiguous and most readings render most strategies untradeable.

**Other options-specific**:
- Use quotes from 14 minutes before close (not EOD prints) for end-of-day strategies; EOD options prints are often stale.
- For American options on dividend-paying stocks, model early exercise risk.
- Theta and vega exposures are separate from execution cost but must be in the P&L.

### 4.6 Cost sensitivity testing

For every final validation (Phase 8 / backtest-driver after optimization), additionally run the top configuration at **1.5x the base cost tier**.

**Pass criterion**: Sharpe at 1.5x costs $\geq 0.5$ AND $PSR(0)$ at 1.5x costs $> 0.90$.

If the strategy fails cost sensitivity, its edge is too thin to survive execution variability and it should not graduate to paper trading regardless of base-cost metrics.

---

## Section 5: Stopping conditions for optimization

This section defines when "tweak to the best of their abilities" is reached. Without explicit stopping conditions, optimization converges to noise.

### 5.1 Statistical floor

Stop when $PSR(0) \geq 0.95$ AND $DSR \geq 0.95$ on the OOS window. The strategy has reached statistical significance accounting for multiple testing. Further optimization can only inflate apparent Sharpe at the cost of overfitting.

### 5.2 Diminishing returns

Stop when the most recent optimization round improved DSR-adjusted OOS Sharpe by less than 5% over the previous round.

Use DSR-adjusted, not raw, OOS Sharpe -- raw OOS Sharpe can drift up while DSR-adjusted (which accounts for cumulative trial count) drifts flat or down.

### 5.3 Overfitting trip

Stop immediately and report as overfit if any of:

- IS/OOS Sharpe ratio < 0.7 (OOS underperforms IS by more than 30%)
- $PBO > 0.50$ -- best in-sample is below-median out-of-sample
- Parameter sensitivity shows cliff-edge behavior on any parameter (Section 5.5)

### 5.4 Parameter count

Strategies with more than 3 free tunable parameters are flagged. Each additional parameter increases degrees of freedom and reduces the statistical significance of any apparent edge.

Hard rule: $\geq 5$ parameters triggers a methodology review -- the strategy must be simplified or have a strong economic rationale for each parameter before optimization proceeds.

### 5.5 Parameter sensitivity

For each parameter `p` with selected value `p*`, evaluate the configuration at `p* x (1 - d)` and `p* x (1 + d)` for $\delta \in \{0.10, 0.20\}$.

Classification:

| Behavior | Label |
|---|---|
| Both neighbors achieve $\geq 0.9 \times$ best Sharpe | STABLE |
| Neighbors achieve $0.5-0.9 \times$ best Sharpe | MODERATE |
| Either neighbor achieves $< 0.5 \times$ best Sharpe | BRITTLE |

BRITTLE on any parameter triggers a stop. The result is cliff-edged and will not survive small differences between backtest and live execution.

### 5.6 Hard caps

- Maximum 3 optimization rounds per strategy
- Maximum 6 hours of optimizer compute per strategy
- Maximum 5000 cumulative configurations per strategy

Whichever first.

### 5.7 Cost sensitivity (final gate)

Section 4.6 applies as a gate: 1.5x cost Sharpe < 0.5 or 1.5x cost $PSR(0) < 0.90$ means the strategy does not graduate, regardless of any other metric.

---

## Section 6: Portfolio integration

Once a strategy passes individual validation, it must be evaluated against the existing live book before consuming risk capital. The portfolio-integrator owns this.

### 6.1 Inputs

- Candidate strategy's OOS daily returns from final validation
- Incumbent strategies' daily returns from the experiment registry (last 24 months minimum)
- Current allocation weights for incumbents

For Homeguard as of 2026-05, incumbents are OMR, RAMP, CSCM. The integrator queries `experiments.duckdb` for their most recent validated return streams.

### 6.2 Correlation analysis

Compute pairwise correlations $\rho_{new, i}$ between the candidate and each incumbent, over the longest overlapping window.

| Max pairwise correlation | Action |
|---|---|
| < 0.4 | Strong diversifier -- proceed |
| 0.4 - 0.7 | Acceptable -- proceed with reduced initial allocation |
| > 0.7 | Reject unless candidate's DSR-adjusted Sharpe exceeds $1.2 \times$ the best incumbent it correlates with |

The exception clause exists because a strictly better version of an existing strategy is still worth running, even at high correlation -- but the bar is high.

### 6.3 Marginal portfolio Sharpe

Compute the portfolio Sharpe with and without the candidate, using inverse-volatility weighting as the baseline:

$$\Delta SR_{portfolio} = SR_{portfolio + candidate} - SR_{portfolio}$$

If $\Delta SR_{portfolio} \leq 0$, the candidate does not improve the portfolio on a risk-adjusted basis and should be rejected even if standalone metrics are strong.

### 6.4 Capacity check

Each strategy has a capacity ceiling -- the AUM beyond which market impact degrades returns. The integrator estimates capacity from:

- Average position size relative to ADV (equity)
- Open interest at typical strike/expiry (options)
- Order book depth at typical sizes (crypto)
- Position limits per exchange (futures)

**Rule**: Proposed allocation $\leq$ capacity / 2 (safety margin of 2x). If the strategy's capacity is below 2x the smallest meaningful allocation in the portfolio, it doesn't graduate.

### 6.5 Allocation recommendation

The integrator does not set allocations. It recommends, with rationale:

- Equal-weight baseline
- Inverse-volatility weighted (recommended default for 2-10 strategies)
- Risk-parity (only when 5+ strategies with 1+ year of concurrent history exist; before that, the covariance estimation noise exceeds the benefit)

Final allocation decisions are made by Shuyang based on the integrator's report and external constraints (broker margin, regulatory limits, conviction).

### 6.6 Real-time vs. batch

The integrator runs after **every** strategy that passes Phase 8, not only at the end of a batch. A high-correlation strategy is a no even before its standalone metrics matter -- running the integrator early avoids wasted Phase 8 compute on candidates that wouldn't graduate anyway.

The strategy-lead invokes the integrator immediately after a Phase 8 PASS verdict. If the integrator returns REJECT-FOR-CORRELATION, the strategy moves to `[-]` in TODO.md with the correlation rationale.

---

## Section 7: Data quality and point-in-time

### 7.1 Fundamental data lag

Earnings, revenue, balance sheet items: assume 45-90 day lag from the period end date. For 10-Q filings, 45 days is the SEC requirement for large accelerated filers. Use 60 days as a conservative default; document the actual lag if known per data source.

### 7.2 Index membership

Use point-in-time index constituent lists. For S&P 500, the membership at 2018-06-15 was not the same as today's membership. If point-in-time lists are unavailable, apply an empirical survivorship haircut:

| Universe | CAGR haircut | Rationale |
|---|---|---|
| Current S&P 500 over 5+ years | 1-2% | Empirical from Brown et al. |
| Current Russell 2000 over 5+ years | 2-4% | Higher turnover, larger haircut |
| Current Nasdaq 100 over 5+ years | 1-3% | Mid-range |

Document the haircut applied. This is itself a magic number that depends on rebalancing methodology; treat the haircut as a sensitivity test, not a final answer.

### 7.3 Corporate actions

Splits and dividends must be adjusted at the **announcement date** for prospective backtests, not retroactively. If using pre-adjusted data, verify the data source's adjustment methodology.

### 7.4 News timestamps

News in `news/symbol={SYM}/` has publication timestamps. Joining news to prices at daily resolution: news published before market open on day `d` is information available for day `d`'s decisions; news published intraday or after-hours on day `d` is information for day `d+1`'s decisions. Do not naively join on date.

For intraday strategies, respect the news publication timestamp at minute resolution.

### 7.5 Data snapshot conventions

Every backtest records its `data_snapshot_date` in the experiment registry -- the date the parquet store was last refreshed. This matters because revisions to historical data (split adjustments, exchange corrections, vendor reprocessing) change backtest results. Two backtests with identical code and parameters but different snapshot dates can produce different Sharpes; the snapshot date is part of the reproducibility identity.

---

## Section 8: Reproducibility requirements

Every backtest and optimization run records the following in the experiment registry. Runs missing any of these are rejected by the strategy-lead and re-run.

### 8.1 Required identity fields

| Field | Source | Why |
|---|---|---|
| `git_sha` | `git rev-parse HEAD` at run start | Code version |
| `config_sha` | SHA-256 of config YAML | Config edits change results |
| `data_snapshot_date` | mtime of relevant parquet partitions | Data revisions change results |
| `python_env_hash` | SHA of `pip freeze` output | Library version differences |
| `random_seeds` | JSON: `{numpy: N, python: P, optimizer: O, ...}` | Stochastic components |
| `wall_clock_start` / `wall_clock_end` | ISO-8601 UTC | Cost tracking |
| `host` | hostname | Hardware drift |

### 8.2 Forbidden patterns

- `random.random()` without a fixed seed
- `np.random.rand()` without a seeded `default_rng`
- Use of `time.time()` as a seed (irreproducible)
- Multiprocessing with workers that don't propagate the parent seed

### 8.3 Verification

The strategy-lead spot-checks reproducibility by re-running every 10th completed strategy at the recorded git SHA + config SHA. Differences in Sharpe > 0.05 trigger investigation.

---

## Section 9: Experiment registry schema

Single DuckDB file at `output/experiments.duckdb`. Appended to by every backtest and optimization run. Queried by the portfolio-integrator and the strategy-lead.

### 9.1 Table: `runs`

```sql
CREATE TABLE runs (
    run_id              VARCHAR PRIMARY KEY,        -- UUID4
    timestamp_utc       TIMESTAMP,
    strategy_name       VARCHAR,
    agent_name          VARCHAR,                    -- 'backtest-driver' | 'backtest-optimizer'
    phase               VARCHAR,                    -- 'initial' | 'optimization_round_N' | 'final'
    parent_run_id       VARCHAR,                    -- chain: optimization round -> final validation

    -- Configuration
    params              JSON,
    universe_name       VARCHAR,
    asset_class         VARCHAR,                    -- 'equities' | 'futures' | 'fx' | 'crypto' | 'options'
    data_frequency      VARCHAR,                    -- '1min' | '1hour' | '1day'

    -- Windows
    window_start        DATE,
    window_end          DATE,
    is_start            DATE,
    is_end              DATE,
    oos_start           DATE,
    oos_end             DATE,
    n_folds             INTEGER,

    -- Metrics (computed on OOS / pooled)
    metrics             JSON,                       -- sharpe, psr, dsr, pbo, cagr, max_dd, calmar, win_rate, trade_count, etc.
    regime_breakdown    JSON,                       -- per-regime metrics from MarketRegimeDetector
    fold_metrics        JSON,                       -- per-fold metrics for stability check

    -- Cost
    cost_tier_used      VARCHAR,                    -- 'large_cap' | 'mid_cap' | 'leveraged_etf' | ...
    cost_bps            DECIMAL,                    -- the actual bps applied
    cost_sensitivity    JSON,                       -- {1x: {...}, 1.5x: {...}, 2x: {...}}

    -- Multiple-testing
    combinations_in_run     INTEGER,                -- this run's configurations
    combinations_project    INTEGER,                -- project-wide cumulative at run time

    -- Reproducibility (Section 8.1)
    git_sha             VARCHAR,
    config_sha          VARCHAR,
    data_snapshot_date  DATE,
    python_env_hash     VARCHAR,
    random_seeds        JSON,
    wall_clock_start    TIMESTAMP,
    wall_clock_end      TIMESTAMP,
    host                VARCHAR,

    -- Outcomes
    verdict             VARCHAR,                    -- 'PASS' | 'FAIL' | 'PENDING'
    verdict_reasons     JSON,                       -- which gates passed/failed
    notes               VARCHAR
);
```

### 9.2 Table: `return_streams`

```sql
CREATE TABLE return_streams (
    run_id              VARCHAR,
    date                DATE,
    return_pct          DECIMAL,
    position_count      INTEGER,
    PRIMARY KEY (run_id, date)
);
```

This stores the daily OOS return stream of every passing strategy, so the portfolio-integrator can pull incumbents and compute correlations without re-running anything.

### 9.3 Append protocol

Every backtest script ends with an append call:

```python
from src.experiments.registry import append_run

append_run(
    run_id=run_id,
    strategy_name=strategy_name,
    agent_name='backtest-driver',
    phase='final',
    parent_run_id=optimizer_run_id,
    params=best_params,
    metrics=metrics_dict,
    return_stream=daily_oos_returns,
    ...
)
```

If the append fails, the run fails. No silent success.

### 9.4 Querying for DSR project-wide trial count

```python
import duckdb
con = duckdb.connect('output/experiments.duckdb')

n_trials_project = con.execute("""
    SELECT SUM(combinations_in_run)
    FROM runs
    WHERE agent_name = 'backtest-optimizer'
""").fetchone()[0]
```

This is the $N$ to use in DSR computation per Section 2.3.

---

## Section 10: Homeguard-specific reference

### 10.1 Repository paths

| What | Path |
|---|---|
| Strategy specifications | `docs/strategies/production/<NAME>_STRATEGY.md` |
| Strategy implementations | `src/strategies/advanced/<name>.py` |
| Strategy configs | `config/strategies/<name>.yaml` |
| Backtest configs | `config/backtesting/<name>.yaml` |
| Backtest engine | `src/backtesting/engine/` |
| Walk-forward | `src/backtesting/chunking/walk_forward.py` |
| Optimization framework | `src/backtesting/optimization/` |
| Regime detection (5-regime) | `src/strategies/advanced/market_regime_detector.py` |
| Regime detection (Bull/Bear/Sideways) | `src/backtesting/regimes/detector.py` |
| Reporting | `src/backtesting/reporting/` |
| Backtest scripts | `backtest_scripts/` (NOT `scripts/backtest_scripts/`) |
| Test suite | `tests/` (mirrors `src/`) |
| Backtest reports | `docs/reports/<strategy>/` |
| Optimization output | `output/optimization/<strategy>/` |
| Experiment registry | `output/experiments.duckdb` |
| Phase analysis output | `docs/agent-learnings/<strategy>/` |
| Methodology (this file) | `docs/methodology/backtesting.md` |
| Infra patterns | `docs/architecture/infra_patterns.md` |

### 10.2 Regime detectors -- disambiguation

Two regime systems coexist in Homeguard. Use the right one.

**`MarketRegimeDetector`** at `src/strategies/advanced/market_regime_detector.py`:
- Five named regimes: STRONG_BULL, WEAK_BULL, SIDEWAYS, UNPREDICTABLE, BEAR
- Used by OMR, RAMP, CSCM in production
- Inputs: SPY vs. SMA(20/50/200), VIX percentile, momentum slope
- This is the regime detector to use for trading strategies and regime-conditioned reporting

**`RegimeDetector`** at `src/backtesting/regimes/detector.py`:
- Separate detectors: TrendDetector (Bull/Bear/Sideways), VolatilityDetector (High/Low), DrawdownDetector (Drawdown/Recovery/Calm)
- Used by the backtesting engine for retrospective performance analysis
- This is for analyzing strategy behavior across decoupled regime dimensions

When in doubt: trading and signal generation use `MarketRegimeDetector`; performance analysis can use either depending on the question.

### 10.3 Brokers by strategy

| Strategy | Execution broker | Data source |
|---|---|---|
| OMR | IBKR (`ib_async`) | Alpaca SIP + ThetaData |
| RAMP | IBKR (`ib_async`) | Alpaca SIP |
| CSCM | Coinbase Advanced Trade | Coinbase + Binance.US REST |
| (research) RAMP-CSP | IBKR (options) | ThetaData |

The trade-log-analyzer must check IBKR (`ib_async`) error patterns for OMR/RAMP and Coinbase patterns for CSCM. Alpaca is data-only in current production.

### 10.4 Systemd services

| Service | Purpose |
|---|---|
| `homeguard-omr.service` | OMR live trading |
| `homeguard-ramp.service` | RAMP live trading |
| `homeguard-cscm.service` | CSCM live trading |
| `homeguard-cscm-paper.service` | CSCM paper |
| `homeguard-cscm-demo.service` | CSCM demo |
| `homeguard-trading.target` | Grouping target -- checking this alone does not surface individual service failures |

Always check individual services via `systemctl status homeguard-*.service`, not only the target.

### 10.5 Available data on disk

(Resolve storage root via `from src.settings import get_local_storage_dir`.)

| Asset class | Path | Coverage |
|---|---|---|
| Equities (1min/1hour/1day) | `equities_1min/`, `equities_1hour/`, `equities_1day/` | 3492 symbols, 2017+ |
| Futures (GLBX continuous .c.0) | `futures_1min/` | 9 contracts (ES, NQ, YM, RTY, CL, GC, ZN, 6E, ZC), 2010-10 to current |
| FX | `fx_1min/` | 50 pairs, varies by pair |
| Crypto | `crypto_1min/`, `crypto_1hour/`, `crypto_1day/` | 33 pairs, varies |
| Options | `options/{chains,gex_daily,options_combined}/` | EOD, varies |
| News | `news/symbol={SYM}/` | 502 symbols, per-event |

### 10.6 Environment

| Setting | Value |
|---|---|
| Production host | EC2 t4g.medium (ARM64, 4GB RAM, Amazon Linux 2023) |
| Memory threshold for ops alerts | > 3GB used (75% of 4GB) -- NOT 900MB |
| Python environment | conda env `fintech` |
| Python invocation (EC2) | `~/Homeguard/venv/bin/python` or `conda run -n fintech python` |
| Python invocation (local Windows) | `conda run -n fintech python` (do not hardcode user-specific paths) |

---

## Changelog

| Date | Change | Author |
|---|---|---|
| 2026-05-12 | Initial consolidated methodology. Replaces inline rules in `backtest-optimizer`, `backtest-driver`, `trading-lead`, and `trade-log-analyzer` agents. Fixes DSR formula, embargo definition, options slippage, regime detector path, systemd service references, and EC2 memory threshold. | Shuyang |

---

## Appendix: Reading priority for agents

Each agent reads only the sections it needs. A pointer table to avoid wasting context:

| Agent | Must read | Should read |
|---|---|---|
| strategy-lead | 1, 5, 6, 10 | 2 (for verdicts), 9 |
| strategy-architect | 1, 10 | 4 (for cost-aware design) |
| strategy-implementer | 1, 10 | 7 (point-in-time) |
| code-reviewer | 1, 7 | 10 (paths) |
| backtest-driver | 1, 2, 3, 4, 8, 9, 10 | 5 (for sanity check) |
| backtest-optimizer | 1, 2, 3, 5, 8, 9 | 4, 10 |
| portfolio-integrator | 6, 9 | 2 (DSR) |
| trade-log-analyzer | 10 (services, brokers, env) | -- |

This is the file. When in doubt, read it.
