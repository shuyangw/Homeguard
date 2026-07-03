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
| `homeguard-multi.service` | Active stocks trading unit (currently runs `--strategy ramp`; supersedes the per-strategy units below) |
| `homeguard-omr.service` | Legacy OMR unit (file exists, **disabled**) |
| `homeguard-ramp.service` | Legacy RAMP unit (file exists, **disabled**) |
| `homeguard-cscm.service` | CSCM live trading |
| `homeguard-cscm-paper.service` | CSCM paper |
| `homeguard-cscm-demo.service` | CSCM demo |
| `homeguard-trading.target` | Grouping target -- checking this alone does not surface individual service failures |

Always check individual services via `systemctl status homeguard-*.service`, not only the target. See `CLAUDE.md` "Production Strategies" for the canonical current-state description.

### 10.5 Available data on disk

(Resolve storage root via `from src.settings import get_local_storage_dir`.)

The **authoritative inventory** of what is on disk -- with row counts, file counts, byte sizes, date ranges, and dtype quirks -- lives in [`docs/reference/DATA_INVENTORY.md`](../reference/DATA_INVENTORY.md), regenerated regularly. Consult that file before any backtest depending on a specific asset class. Quick orientation:

| Asset class | Path |
|---|---|
| Equities (1min) | `equities_1min/`, plus the by-date mirror `equities_1min_by_date/` |
| Futures | `futures_1min/`, `futures_per_contract_1min/`, `futures_per_contract_daily/`, `futures_1min_oi_roll/`, `futures_status/` |
| FX | `fx_1min/`, `fx_quotes_raw/`, `fx_quotes_minute_aggregated/` |
| Crypto | `crypto_1min/` (Alpaca-sourced) plus `crypto_1min_alpaca_archive/` snapshot |
| Options | `options/options_combined/` (with stubs at `options/{chains,gex_daily}/`) |
| News & sentiment | `news/`, `sentiment/` |
| Alt-data | `alt_data/fred/`, `alt_data/cot/` |

### 10.6 Environment

| Setting | Value |
|---|---|
| Production host | EC2 t4g.medium (ARM64, 4GB RAM, Amazon Linux 2023) |
| Memory threshold for ops alerts | > 3GB used (75% of 4GB) -- NOT 900MB |
| Python environment | conda env `fintech` |
| Python invocation (EC2) | `~/Homeguard/venv/bin/python` or `conda run -n fintech python` |
| Python invocation (local Windows) | `conda run -n fintech python` (do not hardcode user-specific paths) |

---


## Section 11: Exit Logic and Profit-Taking Methodology

Exit logic -- stops, targets, trailing rules, time-based exits -- is the single most overfitting-prone aspect of strategy design. There is almost always a stop level that would have caught the worst training drawdown; the optimizer will find it; it will fail OOS. This section defines what's required to use exit logic correctly and to validate it without falling into the stop-optimization trap.

This section applies to every strategy with a non-time-based exit. Time-only-exit strategies (OMR's "exit at 9:31 AM next day", RAMP's "exit on next rebalance") are exempt from 11.2-11.6 but must still document the exit type per 11.1.

### 11.1 Exit-logic taxonomy

Every strategy blueprint declares its exit type(s). Implementer follows. Reviewer verifies.

| Type | Definition | Example |
|---|---|---|
| `time_fixed` | Exit after exactly N bars or at a fixed wall-clock time | OMR exits at 9:31 AM next day |
| `signal_reversal` | Exit when an opposite entry signal fires | MA crossover: long exits on bearish cross |
| `fixed_pct_stop` | Exit at fixed % below entry | -5% stop loss |
| `fixed_pct_target` | Exit at fixed % above entry | +10% profit target |
| `vol_scaled_stop` | Exit at k x ATR below entry | -1.5 ATR stop |
| `vol_scaled_target` | Exit at k x ATR above entry | +2.0 ATR target |
| `trailing_stop` | Stop ratchets on new high (longs) / low (shorts) | Trail by 0.8 ATR after +1.0 ATR profit |
| `time_stop` | Exit if not profitable after N bars | Close if down after 5 days |
| `conditional` | Exit on regime change, vol spike, correlation breakdown | Exit on regime -> BEAR |
| `scale_out` | Partial exits at multiple price levels | 50% at +5%, 50% at +10% |
| `greek_limit` | Options: exit when delta/gamma/vega crosses threshold | Close short put when delta < -0.5 |
| `premium_capture` | Options: close when X% of max premium captured | Close CSP at 50% premium decay |

Strategies typically combine 2-4 of these (e.g., signal_reversal + time_stop + trailing_stop). The taxonomy makes the combination explicit so reviewers and optimizers can verify each component independently.

### 11.2 Bar-resolution requirements

A stop or target validated on data coarser than the trigger condition is invalid. The backtest cannot know whether the stop hit before or after the close.

| Exit type | Minimum data frequency |
|---|---|
| `time_fixed` at daily granularity | Daily bars |
| `signal_reversal` at daily granularity | Daily bars |
| Any `fixed_pct_*` or `vol_scaled_*` with same-day trigger | 1-minute bars |
| Any `trailing_stop` with intraday trigger | 1-minute bars |
| `time_stop` based on holding period (no intraday trigger) | Daily bars |
| Same-bar entry + stop (e.g., ORB) | 1-minute bars, with chronological fill order documented |
| Options: `greek_limit` | EOD chains if checked at EOD; intraday chains if checked intraday |
| Options: `premium_capture` | EOD chains usually sufficient |

**Code-reviewer rule**: If a strategy declares an intraday stop on daily bars, flag as CRITICAL. The backtest can give either the lucky `H=stop_level - 0.01` answer or the unlucky `L=stop_level - 5%` answer; neither is real.

### 11.3 Same-bar fill-order convention

When both a stop and a target could trigger within the same bar (intraday, daily, whatever), the conservative convention is:

**Stops fill first.**

This is a backtest-pessimism choice: the backtest reports the worst plausible interpretation, so live results can only surprise upward. For strategies where the optimizer would otherwise exploit favorable fill-ordering, this is the only safe assumption.

For ORB-style strategies where entry and stop can hit the same bar, the implementer must use minute data and the actual chronological order from the data -- not the same-bar convention.

For mean-reversion strategies that frequently revisit entry-region prices intraday, the same-bar fill-order convention will mark some trades as stopped-out that in reality might have hit the target. This is correct conservatism -- backtest those trades as losses.

Implementer must document the fill-order assumption in the strategy's blueprint.

### 11.4 Gap modeling

When a stop or target is set overnight (or across any market closure) and the next session opens past it, the fill price is the open, not the stop level.

**Required model:**

For a long position with stop at price `S`:

```
if next_open <= S:
    fill_price = next_open  # gap fill, possibly far below S
else:
    # intraday -- check during the bar
    if next_low <= S:
        fill_price = S      # stop triggered intrabar at S
    else:
        # no trigger this bar
```

Symmetric for shorts and for targets (mirror condition with `>=` and `next_high`).

**Asset-class adjustments:**

| Asset class | Gap risk | Notes |
|---|---|---|
| US equities | High Mon-Fri overnight, very high Mon morning | Earnings, news |
| Futures (continuous) | Moderate at session boundaries; high at quarterly rolls | Roll modeled separately |
| FX | Moderate weekend gaps (Friday NY close -> Sunday Asia open) | |
| Crypto | Low (24/7), but flash crashes substitute | May 2021, Oct 2021 |
| Options | High -- both gap and IV crush at open | Avoid same-day options stops near close |

Strategies that hold across known gap risks (e.g., overnight equity positions through earnings) must either filter out names with imminent earnings or accept gap risk in cost modeling.

### 11.5 Stop-specific slippage

> **WIRED (as of 2026-05-14)**: the multipliers in the table below are applied by `simulate_portfolio_numba` at stop-loss fill sites only (exit_reason == EXIT_STOP_LOSS). Configure via `costs.stop_slippage_multiplier` in strategy YAML; default 1.5. The previous "WIRING IN FLIGHT" hard gate in `strategy-lead.md` Phase 9 has been lifted.

Stops execute as market orders at trigger. Slippage on stops is structurally worse than slippage on limit entries.

**Multiplier on standard cost tier (per Section 4):**

| Condition | Slippage multiplier |
|---|---|
| Stop fills during normal liquid hours | 1.5x standard cost tier |
| Stop fills during news event / volatility spike | 2.0x standard cost tier |
| Stop fills on gap-down open (equity) | 3.0x standard or actual gap, whichever larger |
| Stop fills during off-session (FX Asian, futures off-hours) | 2.0x standard cost tier |
| Stop fills during weekend (crypto thin liquidity) | 2.0x standard cost tier |
| Stop fills during flash crash event | Model as worst quartile of trade-by-trade slippage on training data |

**Cost-sensitivity test (extending Section 4.6):**

For strategies with stops, the 1.5x cost sensitivity test also runs at 2.0x **stop slippage** (entry slippage unchanged). This stresses the strategy on its actual failure mode rather than uniformly inflating costs.

Pass criterion (extending Section 4.6): Sharpe at 2.0x stop slippage >= 0.4 AND PSR(0) at 2.0x stop slippage > 0.85. Lower thresholds than the 1.5x general cost test because stop-stressed Sharpe is expected to be worse.

### 11.6 MAE/MFE methodology

Maximum Adverse Excursion and Maximum Favorable Excursion are the principled way to size stops and targets without overfitting. Computed from a backtest's trade log, they characterize the actual distribution of intra-trade price movement that the strategy experiences.

**Per-trade fields required in the trade log** (Code-reviewer enforces in Section 11.9):

```
entry_time            # ISO timestamp
entry_price           # Fill price including entry slippage
exit_time
exit_price            # Fill price including exit slippage
exit_reason           # signal_reversal | stop_hit | target_hit | time_expired | regime_change | etc.
mae_pct               # Worst paper loss during the trade, signed (negative for longs that went down)
mfe_pct               # Best paper profit during the trade, signed
mae_time              # When MAE occurred (timestamp)
mfe_time              # When MFE occurred (timestamp)
bars_held             # Integer count
hit_stop              # bool: did the trade ever touch the stop level?
hit_target            # bool: did the trade ever touch the target level?
```

Trades without these fields cannot be analyzed by Section 11.6 and Section 12. Code-reviewer flags any backtest engine or trade log producer missing these.

**Required outputs in every backtest report for strategies with non-time-based exits** (extending Section 12 driver requirements):

1. **Winners' MAE distribution**: mean, p25, p50, p75, p95. "How deep did winning trades go underwater before recovering?"
2. **Losers' MAE distribution**: same five quantiles. "How deep did losing trades go before exit?"
3. **Target attainment rate**: fraction of winning trades that hit the declared profit target (if applicable). Numbers near 1.0 suggest targets are too tight; numbers near 0 suggest no targets are being hit.
4. **Stop-touched-but-recovered rate**: fraction of *winning* trades whose intra-trade MAE breached the stop level. Indicates stops are too tight -- these trades would have stopped out in live trading with realistic execution.
5. **Target-exceeded-but-reverted rate**: fraction of *losing* trades whose intra-trade MFE exceeded the target level. Indicates targets are too tight or trailing rules too loose -- winners turned to losers.

**Stop-sizing from MAE/MFE (training-data-only procedure):**

1. Compute MAE distribution for winning trades on the training period.
2. Compute MAE distribution for losing trades on the training period.
3. Set the stop at the level where: (a) losers' 25th percentile MAE is exceeded, but (b) winners' 75th percentile MAE is not breached. If no such gap exists, the trade-by-trade signal isn't distinguishable on MAE alone -- use vol-scaled stops instead.
4. Validate on OOS: report points 1-5 above on OOS data. If winners' MAE distribution shifts up materially OOS, the stop is too tight.

This is the only stop-sizing procedure that doesn't fit to noise. **Optimizer-discovered stop levels without MAE/MFE backing are rejected** by the lead in Phase 9 validation.

### 11.7 Profit-taking by asset class

Profit targets, trailing stops, and scale-out rules have asset-class-specific interpretations. This subsection codifies the per-asset rules.

#### 11.7.1 Equities (cash and leveraged ETFs)

- Both stops and targets work in standard form: fixed-pct, vol-scaled, MAE/MFE-derived.
- Slippage per 11.5.
- Gap risk per 11.4. Stop slippage on equity gap-downs can be severe -- model explicitly for any strategy holding overnight.
- **Profit-taking analysis required**: target attainment rate, MFE-beyond-target rate, R:R distribution by exit type.
- For leveraged ETFs (TQQQ, SOXL, etc.): MFE distributions are wider; standard pct-based targets are usually too tight. Vol-scaled targets recommended.

#### 11.7.2 Futures (continuous .c.0 contracts)

- Stops in ticks, not pct. Tick value varies by contract (ES = $12.50, NQ = $5.00, CL = $10.00, etc.).
- Same-bar fill-order matters more than equities (faster markets).
- Slippage on news events: 1.5x to 3x normal during scheduled releases (FOMC, NFP, EIA inventory). Strategy specs should declare whether they trade through these or filter them out.
- Profit-taking: scaling-out is common due to contract granularity (you can exit 1 of 3 contracts at +1 ATR, etc.). Strategy spec must declare whether engine supports partials before scale-out is in the parameter space.
- **Required diagnostic for futures strategies**: target attainment rate broken down by intraday session (RTH vs ETH if applicable).

#### 11.7.3 FX

- Stops in pips. Pip value depends on pair and position size.
- Spread is part of stop slippage (you don't get filled on the bid as a long stopping out).
- Session-dependent: stops triggered in thin Asian hours can slip 2-5x tighter-session slippage.
- 24/5 trading with weekend gap risk: Sunday open can blow through Friday's stop. Required filter: strategy specs must declare whether they hold across weekends.
- Profit-taking on mean-reversion FX strategies (e.g., the Darwinex-inspired strategy): **tight targets relative to ATR are correct**. Letting winners run is the wrong instinct in mean-reversion contexts; the edge is the reversion, not trend continuation. Typical target ~ 0.5 to 1.0 x ATR.
- **Required diagnostic for FX strategies**: target attainment rate by session (Asian / London / NY overlap / NY).

#### 11.7.4 Crypto

- 24/7 mostly eliminates gap risk, but flash-crash risk substitutes. Document major events (May 19 2021 BTC, May 2022 LUNA, Oct 2021 various). Strategies with stops below recent significant lows are vulnerable.
- Weekend liquidity is thin on most pairs. Stops set during weekday liquidity may slip 2-3x worse Saturday/Sunday.
- Exchange outages create false stops (price prints on illiquid books) -- for CSCM on Coinbase, model assumes Coinbase doesn't have an outage. If a different exchange is added, that assumption changes.
- **Profit-taking trap**: crypto's positive skew is extreme; capping upside via fixed targets can destroy strategies that profit from rare large moves. **Default recommendation for momentum/trend crypto strategies: trailing stops, not fixed targets.** Mean-reversion strategies follow the FX rule (tight targets correct).
- **Required diagnostic for crypto strategies**: MFE distribution with explicit reporting of trades whose MFE > 2x target (suggests target is too tight).

#### 11.7.5 Options

This is the most distinct of the asset classes and needs the most care.

- **Stops on the option price are unreliable.** Option value moves with Greeks (delta, gamma, theta, vega), not just underlying price. A 50% stop on the option premium can trigger from a vol crush even with the underlying flat -- that's not a stop-out signal, it's a market-condition change.
- **Better stop framing**: stop on the *underlying* price ("if underlying hits X, close position regardless of option P&L") or stop on *Greeks* ("close if short put delta exceeds -0.5").
- **Profit-taking for premium sellers (CSPs, covered calls, credit spreads, iron condors)**: industry-standard practice (tastytrade research):
  - **Manage at 50% of max profit.** Close the position when 50% of the premium received has decayed.
  - **Manage at 21 DTE.** Close the position when 21 days to expiration remain, regardless of P&L. Gamma risk increases exponentially below 21 DTE.
  - **Roll on breach.** If the short strike is breached, roll to a further-OTM strike at the next monthly expiration.
- **Profit-taking for premium buyers (long calls, long puts, debit spreads)**: theta works against you. Pure signal-based exits typically outperform fixed targets. Fixed profit targets are acceptable for short-duration intraday options trades but generally fail for swing/positional buyers.
- **Required diagnostic for options strategies**: separate Greek-exposure tracking -- average delta, gamma, vega at exit; distribution of theta-decay-captured per trade.

For RAMP-CSP specifically (research strategy, blocked on options data gap): premium-capture exits at 50% with 21-DTE forced close is the right starting framework. Optimizer should *not* search around these parameters -- they're conventionally optimal per Sosnoff/tastytrade research and search would overfit. Search on entry parameters only (which strikes, which DTE entry, regime filters).

### 11.8 Stops and the parameter budget

Stops count toward the <=3 free-parameter budget from Section 5.4.

A strategy with `entry_signal_param + stop_loss_pct + profit_target_pct` is at 3 parameters. Adding a `trailing_threshold` makes it 4 -- over the budget. This is a hard limit; over-parameterization is the primary route to overfitting.

**Configurations that don't count against the budget:**

- Parameters fixed by methodology (50% premium capture for CSPs, 21-DTE for options management)
- Parameters fixed by asset class (tick value for futures)
- Parameters fixed by exchange (Coinbase fee tier for CSCM)

**Configurations that do count:**

- Anything searchable in `param_grid` or `param_space`
- Anything the strategy spec lists under `parameters_to_optimize`

If a strategy needs more than 3 tunable parameters including stops/targets, the strategy is over-parameterized. Methodology requires simplification -- either drop a parameter, fix it from first principles, or split into two strategies.

### 11.9 Code-reviewer responsibilities for exit logic

The code-reviewer, when reviewing changes under `src/strategies/`, `src/backtesting/engine/`, or any backtest script, must verify:

1. **Bar-resolution match**: If the strategy declares an intraday stop, the backtest config uses 1-minute bars. CRITICAL severity if mismatched.
2. **Same-bar fill-order documented**: Strategy blueprint includes the fill-order assumption (per 11.3) and the engine implements it. HIGH severity if undocumented.
3. **Gap modeling present**: For overnight-holding strategies, the engine implements the gap-fill model from 11.4. Verify by reading `src/backtesting/engine/backtest_engine.py` for the relevant fill logic. HIGH severity if missing.
4. **Trade log schema complete**: All fields from 11.6 are written by the engine. CRITICAL severity if `mae_pct`, `mfe_pct`, `hit_stop`, `hit_target` are missing -- those are required for downstream diagnostic outputs.
5. **Stop slippage applied**: If the engine has a slippage model, verify the stop multiplier (1.5x-3.0x per 11.5) is in effect on stop exits. HIGH severity if entries and stop exits use the same multiplier.
6. **Parameter budget**: Count exit-logic parameters against the <=3 budget per 11.8. MEDIUM severity if the strategy is at or over the budget.

These checks supplement Section 1 (bias prevention) and Section 7 (point-in-time) reviewer responsibilities. Reviewer prompts dispatched by the trading-lead must include the line "Consult docs/methodology/backtesting.md Section 11 for exit-logic checks."

### 11.10 Optimizer behavior with exit-level parameters

When the parameter space includes stop levels, profit-target levels, trailing thresholds, or any exit-logic parameter, the backtest-optimizer applies tightened sensitivity classification (Section 5.5).

**Standard sensitivity classification (Section 5.5):**

| Behavior | Label |
|---|---|
| Both neighbors achieve >= 0.9 x best Sharpe | STABLE |
| Neighbors achieve 0.5-0.9 x best Sharpe | MODERATE |
| Either neighbor achieves < 0.5 x best Sharpe | BRITTLE |

**Tightened classification for exit-logic parameters:**

| Behavior | Label |
|---|---|
| Both neighbors achieve >= 0.95 x best Sharpe | STABLE |
| Neighbors achieve 0.7-0.95 x best Sharpe | MODERATE |
| Either neighbor achieves < 0.7 x best Sharpe | BRITTLE |

BRITTLE on any exit-logic parameter triggers an immediate stop (Section 5.3). Stops and targets that aren't robust to small perturbations will fail in live trading -- execution variability alone will move the fill price more than the perturbation.

Additionally, when the optimizer's recommended stop/target levels are not backed by MAE/MFE analysis on training data (per 11.6), the lead's Phase 9 validation rejects the configuration regardless of statistical gate. MAE/MFE-derived stops are the only defensible source for these parameters.

### 11.11 Exit logic and the experiment registry

The registry schema (Section 9) gains two additional fields for backtests of strategies with non-time-based exits:

```sql
ALTER TABLE runs ADD COLUMN exit_logic_summary JSON;
-- {
--   "exit_types": ["signal_reversal", "fixed_pct_stop", "trailing_stop"],
--   "winners_mae_p75": -0.018,
--   "losers_mae_p25": -0.043,
--   "target_attainment_rate": 0.62,
--   "stop_touched_but_recovered_rate": 0.08,
--   "target_exceeded_but_reverted_rate": 0.14,
--   "stop_slippage_multiplier_applied": 1.5
-- }

ALTER TABLE runs ADD COLUMN mae_mfe_validated BOOLEAN;
-- TRUE if MAE/MFE distributions were computed and stops are MAE/MFE-derived per 11.6
```

The portfolio-integrator (Section 6) reads `exit_logic_summary` to verify that any strategy graduating to live trading has sane exit-logic diagnostics. A strategy with `target_exceeded_but_reverted_rate > 0.30` should not graduate even if its Sharpe is good -- too many winners are turning into losers due to target placement.

```

---

## Section 12: Required Diagnostic Outputs

This section consolidates the cross-cutting diagnostics required of every backtest and every optimization. Where Section 2 defines *what statistical tests must pass*, this section defines *what diagnostic information must be produced* -- even when the statistics pass.

The five Tier-1 diagnostics below are mandatory. The backtest-driver produces them in every report. The backtest-optimizer produces them for the top configuration of every optimization run. The trading-lead gates on them in Phase 6 and Phase 9 validation.

### 12.0 Trade log persistence (every backtest, every asset class)

Every backtest engine MUST persist its simulated-trade log -- the per-fill / per-position-change records -- for EVERY asset class (equity, crypto, futures, options, and any new asset-class path). Persisting only aggregate metrics and the equity curve, and discarding the fills, is a violation: the trade log is the input to 12.1, 12.2, and Section 11.6, so a run without it cannot be diagnosed or gated. **The code-reviewer flags any backtest engine or runner that produces results without persisting a trade log** (in addition to the per-trade field checks in 11.9). When a new engine or asset-class path is added, wiring trade-log persistence is part of the definition of done, not a follow-up. (Reference wirings: equity/crypto via `backtest_runner` -> `TradeLogger`, gated on `output.save_trades` default True; futures via `run_futures_backtest(..., log_trades=True)`.)

### 12.1 Trade-level metrics alongside portfolio metrics

Every backtest report includes both views:

**Portfolio-level** (from existing Section 2):
- Pooled-returns Sharpe ratio, PSR, DSR
- CAGR, max drawdown, Calmar ratio
- PBO from optimizer runs (Section 2.4)

**Trade-level** (new requirement):
- Win rate (fraction of round-trips with positive P&L after costs)
- Profit factor (gross wins / gross losses)
- Expectancy (mean P&L per trade in $)
- Average winner vs. average loser (R:R distribution)
- Longest losing streak (count of consecutive losing trades)
- Largest single trade win / largest single trade loss
- Win rate by holding period (bucketed: < 1 day, 1-5 days, 5-20 days, > 20 days)

**Why required**: portfolio Sharpe smooths across many trades and can mask a strategy whose individual trades are marginal. A strategy with portfolio Sharpe 1.5 but trade-level expectancy near zero after costs is one execution-quality regression away from breakeven. The reverse -- a strategy with low Sharpe but extreme R:R and high expectancy -- has different deployment characteristics (smaller AUM, more selective).

**Gate (trading-lead Phase 6 and 9)**: reject if `portfolio_sharpe > 1.0` AND `trade_expectancy_after_costs <= 0`. This is a P&L attribution mismatch -- the engine is computing portfolio returns differently than trade returns, or costs are being applied inconsistently. Dispatch code-reviewer to investigate before proceeding.

### 12.2 Capacity curve

Every backtest of a strategy aiming for live deployment runs the same backtest at multiple capital levels and reports the Sharpe / CAGR / max DD curve.

**Standard scale points**: $50K, $250K, $1M, $5M, $25M.

Method: re-evaluate the existing trade log at each capital level, applying the square-root market impact model from Section 4.1:

```
impact_pct = sigma_daily x eta x sqrt(position_size_dollars / daily_dollar_volume)
```

For positions exceeding ~5% of ADV, impact is non-trivial and Sharpe degrades. The curve tells you where the strategy's edge breaks down.

**Output format** (in metrics JSON, registry column `capacity_curve`):

```json
"capacity_curve": [
    {"capital_usd": 50000, "sharpe": 1.52, "cagr": 0.18, "max_dd": -0.07, "avg_impact_bps": 2.1},
    {"capital_usd": 250000, "sharpe": 1.48, "cagr": 0.17, "max_dd": -0.08, "avg_impact_bps": 4.7},
    {"capital_usd": 1000000, "sharpe": 1.41, "cagr": 0.15, "max_dd": -0.09, "avg_impact_bps": 9.4},
    {"capital_usd": 5000000, "sharpe": 1.18, "cagr": 0.11, "max_dd": -0.11, "avg_impact_bps": 21.0},
    {"capital_usd": 25000000, "sharpe": 0.62, "cagr": 0.05, "max_dd": -0.14, "avg_impact_bps": 47.0}
]
```

**Gate (portfolio-integrator)**: identify the capacity ceiling -- the capital level beyond which Sharpe drops below 80% of the $50K baseline. Allocation recommendation never exceeds capacity_ceiling / 2.

**Cost**: ~5-10 additional minutes per backtest. Cheap relative to the run itself.

### 12.3 Regime transition analysis

Every backtest of a strategy with 5+ years of data reports performance broken down by:

- Stable-regime periods (consecutive days in the same regime)
- Transition periods (+/-10 trading days around each regime boundary detected by `MarketRegimeDetector`)

**Output format**:

```json
"regime_transitions": {
    "n_transitions": 14,
    "transition_window_days": 10,
    "transition_sharpe": 0.31,
    "non_transition_sharpe": 1.42,
    "transition_max_dd": -0.18,
    "non_transition_max_dd": -0.07,
    "transition_pct_of_total_dd": 0.62,
    "transition_pct_of_total_pnl": 0.08
}
```

**Why required**: strategies often look fine when averaged across regimes but lose disproportionately at transitions, where regime detectors lag in real time. A strategy that earns Sharpe 1.5 in stable regimes but Sharpe -0.3 in transitions is structurally vulnerable to whichever delay the live regime detector has -- and OMR/RAMP both have live detectors with non-zero lag.

**Gate (trading-lead Phase 6 and 9)**: reject if `transition_pct_of_total_dd > 0.5` (more than half of drawdown happens at transitions) AND `transition_sharpe < 0` (transitions are net-negative). Strategies failing this gate should not graduate or should be paper-trading-only.

### 12.4 Hyperparameter temporal stability

When an optimization sweep has parameter count >= 2, the backtest-optimizer runs the optimization separately on each of the K walk-forward windows and reports parameter stability.

**Output format** (in optimization chronicle and registry):

```json
"parameter_stability": {
    "n_windows": 5,
    "windows": [
        {"train_end": "2020-06-30", "best": {"long_period": 21, "penalty_weight": 4.0, "top_n": 8}},
        {"train_end": "2021-06-30", "best": {"long_period": 21, "penalty_weight": 5.0, "top_n": 10}},
        {"train_end": "2022-06-30", "best": {"long_period": 21, "penalty_weight": 4.0, "top_n": 9}},
        {"train_end": "2023-06-30", "best": {"long_period": 21, "penalty_weight": 4.5, "top_n": 12}},
        {"train_end": "2024-06-30", "best": {"long_period": 21, "penalty_weight": 5.0, "top_n": 11}}
    ],
    "stability_by_parameter": {
        "long_period": {"mean": 21.0, "std": 0.0, "cv": 0.0, "classification": "STABLE"},
        "penalty_weight": {"mean": 4.5, "std": 0.5, "cv": 0.11, "classification": "STABLE"},
        "top_n": {"mean": 10.0, "std": 1.6, "cv": 0.16, "classification": "STABLE"}
    },
    "overall_classification": "STABLE"
}
```

**Classification by coefficient of variation**:

| CV | Label |
|---|---|
| < 0.20 | STABLE |
| 0.20 - 0.50 | MODERATE |
| > 0.50 | UNSTABLE |

Overall classification is the worst across all parameters.

**Gate (trading-lead Phase 9)**: reject if `overall_classification == UNSTABLE`. The parameters fit noise, not signal -- different windows recommend materially different configurations and there's no defensible reason to pick any single one.

**Cost**: real. K-window optimization multiplies optimizer wall-clock by K. For a 6-hour optimization with K=5, this is 30 hours. The methodology accepts this cost for strategies graduating to live deployment, and the optimizer's hard cap (Section 5.6) accommodates it: 5000 cumulative configurations per strategy includes the K-window decomposition.

Strategies in early research phase (not yet candidates for live deployment) may skip 12.4. The optimizer reports `"parameter_stability": "not_assessed_research_phase"` in that case. Lead's gate doesn't trigger for research-phase strategies.

### 12.5 Benchmark comparison and information ratio

Every backtest of an equity, ETF, or futures strategy reports benchmark-relative metrics:

- **Beta** to the benchmark (regression slope of strategy returns on benchmark returns)
- **Alpha** (regression intercept, annualized)
- **Tracking error** (std of strategy returns minus benchmark returns, annualized)
- **Information ratio** (alpha / tracking error)

**Benchmark mapping**:

| Strategy class | Benchmark |
|---|---|
| Long-only equity | SPY |
| Long-short equity | Equal-weighted SPY + cash |
| Sector-rotation equity | XLK + XLF + XLV equal-weighted |
| Long-only crypto | BTC |
| Long-only futures (equity index) | ES front-month |
| Long-only futures (commodity) | GSCI Commodity Index |
| FX (G10 carry trade) | DBV ETF |
| FX (mean reversion) | None -- report Sharpe only |
| Options (premium-selling) | None -- report Sharpe only |

**Output format**:

```json
"benchmark_comparison": {
    "benchmark_symbol": "SPY",
    "beta": 0.45,
    "alpha_annualized": 0.082,
    "tracking_error_annualized": 0.087,
    "information_ratio": 0.94,
    "r_squared": 0.31
}
```

**Why required**: a long-only equity strategy with Sharpe 1.0 in a bull market with Sharpe 0.8 is generating much of its Sharpe from market beta, not skill. Information ratio strips that out. Strategies with high Sharpe and low IR are paying complexity for market exposure that could be bought directly.

**Gate (trading-lead Phase 6 and 9)**:

| Asset class | Min information ratio | Action if below |
|---|---|---|
| Long-only equity | 0.30 | Reject -- strategy is not adding skill above market |
| Long-short equity | 0.50 | Reject |
| Sector-rotation equity | 0.40 | Reject |
| Long-only crypto | 0.40 | Reject |
| Long-only futures | 0.30 | Reject |
| Strategies with no defined benchmark | N/A | Report Sharpe only, no IR gate |

### 12.6 Required diagnostic outputs -- consolidated table

For convenience, the complete diagnostic checklist for each strategy class:

| Diagnostic | When required | Produced by | Gated by | Severity if missing |
|---|---|---|---|---|
| Portfolio metrics (Sharpe, PSR, DSR, max DD) | Every backtest | backtest-driver | trading-lead | CRITICAL |
| Trade-level metrics (12.1) | Every backtest | backtest-driver | trading-lead | HIGH |
| Capacity curve (12.2) | Every backtest aiming for live | backtest-driver | portfolio-integrator | HIGH |
| Regime transitions (12.3) | 5+ year backtests | backtest-driver | trading-lead | HIGH |
| Parameter stability (12.4) | Optimization with >= 2 params, live-bound | backtest-optimizer | trading-lead | HIGH |
| Benchmark / IR (12.5) | Strategies with defined benchmark | backtest-driver | trading-lead | MEDIUM |
| Exit logic summary (11.11) | Strategies with non-time exits | backtest-driver | portfolio-integrator | HIGH |
| MAE/MFE validation (11.6) | Strategies with stops/targets | backtest-driver | trading-lead | HIGH |
| PBO (Section 2.4) | Optimization with > 20 configs | backtest-optimizer | trading-lead | CRITICAL |
| Reproducibility identity (Section 8.1) | Every run | All run-producing agents | trading-lead | CRITICAL |

Severity defines the lead's response:
- **CRITICAL** missing: reject the phase, mark `[!]` in TODO.md, dispatch fix
- **HIGH** missing: reject the phase unless explicitly waived by user
- **MEDIUM** missing: warn in the report, proceed but flag

---

## Changelog

| Date | Change | Author |
|---|---|---|
| 2026-05-12 | v2: Section 11 (Exit Logic and Profit-Taking Methodology, 11 subsections) and Section 12 (Required Diagnostic Outputs, 6 subsections) appended. Registry schema extended (`exit_logic_summary`, `mae_mfe_validated`). Appendix table updated to reflect actual on-disk agents and to add a "future agents" note for `portfolio-integrator`, `strategy-architect`, `strategy-implementer` (decision B). Gates added: trade-expectancy consistency, capacity, regime transitions, parameter temporal stability, information ratio. Stop-loss governance: MAE/MFE-derived stops required for live deployment. | Shuyang |
| 2026-05-12 | v1: Initial consolidated methodology. Replaces inline rules in `backtest-optimizer`, `backtest-driver`, `trading-lead`, and `trade-log-analyzer` agents. Fixes DSR formula, embargo definition, options slippage, regime detector path, systemd service references, and EC2 memory threshold. | Shuyang |

---

## Appendix: Reading priority for agents

Each agent reads only the sections it needs. A pointer table to avoid wasting context:

Agents currently in `.claude/agents/`:

| Agent | Must read | Should read |
|---|---|---|
| strategy-lead | 1, 5, 6, 10, 11, 12 | 2 (for verdicts), 9 |
| code-architect (when used for strategy work) | 1, 10, 11 | 4 (cost-aware design) |
| code-explorer | 10 | -- |
| code-reviewer | 1, 7, 11 (for strategies with exits) | 10 (paths) |
| backtest-driver | 1, 2, 3, 4, 8, 9, 10, 11, 12 | 5 (sanity check) |
| backtest-optimizer | 1, 2, 3, 5, 8, 9, 11, 12 | 4, 10 |
| trade-log-analyzer | 10 (services, brokers, env) | -- |
| live-ops | 10 | -- |
| codebase-analyzer | -- | -- |

**Future agents** (decision B -- defer until trigger):

- `portfolio-integrator`: trigger = first portfolio-integration question requiring multi-file return-stream analysis the orchestrator can't fit in its head. Methodology Section 6 (the rules) is in effect; lead handles inline until then. Must read 6, 9, 11.11, 12.
- `strategy-architect` and `strategy-implementer`: trigger = first strategy where the blueprint phase needs its own context budget. Currently `code-architect` and the general-purpose agent handle these.

When a future agent is created, move its row into the main table above.

This is the file. When in doubt, read it.
