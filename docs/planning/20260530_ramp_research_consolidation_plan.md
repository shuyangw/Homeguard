# RAMP Research Harness: Consolidation, Rename, and Variant Registry (refined)

**Status**: Approved design -- ready for implementation planning
**Owner**: Shuyang
**Author**: Claude (refined from the 2026-05-30 draft via a brainstorming/verification pass)
**Created**: 2026-05-30
**Location**: `docs/planning/20260530_ramp_research_consolidation_plan.md`
**Supersedes**: the external draft (`~/Downloads/20260530_ramp_research_consolidation_plan.md`)
**Depends on**: Phase 4 harness (landed -- `src/research/ramp_phase4/`, 41 passing tests); Phase 3A/3B variant exploration (findings in `docs/reports/ramp/20260505_*.md`)

---

## Corrections from the draft (read first)

This spec is the draft plan re-verified against `origin/main` and amended where the
draft's "current state" claims were stale. Two load-bearing claims changed; both
reshape scope. The verification was done after fast-forwarding a local checkout that
was 5 commits behind `origin/main` -- the draft had been authored against an older tree.

| Draft claim | Verdict on `origin/main` | Effect on plan |
|---|---|---|
| Engine ignores `rebalance_frequency` (the field is declared but the loop may not branch on it) | **CONFIRMED** -- `engine.py` has zero weekday/weekly branching; it rebalances every day unconditionally | Weekly rebalance is **unbuilt code**, not a CLI passthrough. The weekly comparison is split into a **gated PR 6**; PR 5 ships **daily-only** first. |
| `config/trading/strategy_toggle.yaml` carries a **dead/deletable** `ramp.variant: v11` field | **RETRACTED** -- the field is **present and live-shaped** (`ramp: v11`, `cscm/mp/omr: v01`), added by the deploy-sync commit `00ed0cb (#7)`. It is **present-but-inert**: round-tripped in the raw toggle dict, read by no accessor | PR 4 drops all variant work. The field is acknowledged in a **deferred-work appendix**, not edited. |
| Toggle read path is the `ToggleConfig` dataclass (which "drops" `variant` at parse time) | **REFINED** -- accessors read the **raw dict** (`config.get('enabled')`); `ToggleConfig` exists but is not the read path. A future variant accessor is `config.get('variant')` mirroring `is_enabled` | Captured in the deferred appendix so the eventual wiring targets the right code path. |
| *(new, from the data-reorg commits)* | A canonical-paths module `src/settings/data_paths.py` (+ `get_data_dir(subdir)`) landed in the 5 catch-up commits | Implementation **must branch from `origin/main`**; PR 1.2 may route the cache through `get_data_dir(...)`. |

**Net structural change:** six PRs instead of five (PR 6 = gated weekly), PR 2's
`--rebalance-frequency` flag is fail-loud on `weekly_*`, PR 4 loses its variant-toggle
part and gains a deferred appendix, and PR 5 becomes PR 5a (daily verdict). Everything
else in the draft stood up to verification and is preserved.

---

## How to use this document

This is an execution plan for a coding agent. Ground rules:

1. **Branch from `origin/main`.** `git fetch && git checkout -b feature/ramp-research-consolidation origin/main`. Do not start from a stale local `main`; the data-reorg (`src/settings/data_paths.py`) and toggle-sync (`ramp.variant`) commits must be present, or your "current state" will diverge from this spec exactly as the draft's did.
2. **Inspect before editing.** Every "current state" claim was verified at authoring time, but re-read each file immediately before editing. Line numbers are *locators*, not guarantees -- grep for the quoted code, don't trust the integer.
3. **One PR per section, in order.** PRs are ordered by leverage and risk. Each has a "Definition of done" and a "Validation" block -- run validation before opening the next PR. Do not batch.
4. **No behavior changes disguised as refactors.** PR 1 (module rename) and PR 2 (CLI rename) change zero numeric outputs. PR 3 is the only PR that adds computation paths, and even there the existing two variants produce byte-identical results before and after.
5. **Decisions are already made.** Where this plan resolves a choice, it says so. If you discover a fact that invalidates a decision (as happened with the draft), STOP and surface it rather than improvising.
6. **The point is the turnover measurement in PR 5a.** PRs 1-4 are the instrument; PR 5a is the experiment that decides whether RAMP is salvageable as a daily strategy. PR 6 (weekly) exists only if PR 5a says daily fails its cost floor. Keep that framing -- do not let the refactor become the goal.

---

## Executive summary

The RAMP research code has three compounding problems:

1. **A useless module name.** `src/research/ramp_phase4/` is a high-quality, stateful, cost-aware, walk-forward backtest harness with 41 passing tests -- buried inside a name that encodes project chronology ("phase 4") rather than function. The name makes a reusable instrument look like throwaway scaffolding.

2. **A variant-naming collision with no decoder.** The only *registered* variants are `V01` and `V03` (both run the production regime overlay; they differ only on whether they honor the 0.5x crash multiplier). The *decision-relevant* variants -- `plain` (vanilla momentum, no overlay) and `bear_cash` (overlay but BEAR->cash) -- exist only as inline `if variant == 'V1':` branches inside four dated investigation scripts, each of which reimplements the entire backtest loop. The script-world `V1` (no overlay) and the library-world `V01` (overlay) are near-identical strings meaning *opposite* things, with no legend anywhere across seven reports.

3. **A present-but-inert toggle field.** `config/trading/strategy_toggle.yaml` carries `ramp.variant: v11` (and `v01` on the other three strategies). It is round-tripped in the raw toggle dict but read by no accessor -- it does nothing while *looking* like live RAMP behavior is versioned and selectable. (Note: `v11` is not a defined registry id; see PR 4's deferred appendix.)

**Six PRs, ordered by leverage and risk:**

| PR | Content | Risk | Effort | Blocks |
|---|---|---|---|---|
| **PR 1** | Rename module `ramp_phase4` -> `regime_momentum_lab`; decouple data-cache path from module name | Low (mechanical) | 1-2 hr | PR 2, PR 3 |
| **PR 2** | Rename CLI `ramp_phase4_backtest.py` -> `run_momentum_variant.py`; add `--rebalance-frequency` (fail-loud on `weekly_*`) + turnover summary line | Low | 1 hr | PR 5a |
| **PR 3** | Variant registry: rename `V01`/`V03` -> `prod_no_crash`/`prod`; add `plain` + `bear_cash`; add `aliases` + `resolve()` | Medium | 1 day | PR 5a |
| **PR 4** | Archive the four dated investigation scripts under `_archived/`; record the inert `ramp.variant` field in a deferred-work appendix (no edit) | Low | 1-2 hr | None |
| **PR 5a** | Measure realized **daily** turnover; daily cost sensitivity; write findings report with the verdict | None (research) | 1 day | -- |
| **PR 6** (gated) | **Only if PR 5a shows daily fails its cost floor:** build a tested weekly-rebalance feature in the engine, then run the weekly cost-sensitivity comparison + report addendum | Medium | 1-1.5 days | -- |

**Total estimated effort**: ~3 working days for PRs 1-5a; +1-1.5 days for PR 6 if triggered.

**Outcome.** After PR 4 the active tree loses four ~500-line near-duplicate scripts, all variants run through one tested engine, the module name describes what the code is, the `v{n}` collision is gone (old report labels still resolve via aliases), and the inert toggle field is documented rather than misleading. After PR 5a you have a measured (not assumed) daily turnover number and a cost verdict on whether RAMP clears its transaction-cost floor at daily cadence in any of {`plain`, `prod`, `bear_cash`}. PR 6 answers the same at weekly cadence only if daily fails.

---

## Goals and non-goals

### Goals

- Rename `src/research/ramp_phase4/` -> `src/research/regime_momentum_lab/` (module + tests + the two importing scripts), with zero change to any numeric output.
- Stop the code-module name leaking into on-disk data paths (`cache/ramp_phase4/...` -> a stable `cache/regime_momentum/...`, optionally via `get_data_dir()`).
- Make the variant registry in `variants.py` the single source of truth for all RAMP backtest variants.
- Replace `V0n` numbering with descriptive names (`plain`, `prod`, `prod_no_crash`, `bear_cash`), preserving traceability to dated reports via an `aliases` field and a `resolve()` lookup.
- Add the two decision-relevant variants that currently live only in dated scripts: `plain` (no overlay) and `bear_cash` (overlay + BEAR->cash; one flag, since `compute_plan` already takes `bear_to_cash`).
- Archive the four dated investigation scripts under `_archived/` so their audit trail survives but their dead reimplemented-loop code leaves the active tree.
- Measure realized **daily** turnover and produce a findings report answering: what is realized daily turnover, and does any variant clear realistic transaction costs at daily cadence?
- **(Gated)** If daily fails, build a tested weekly-rebalance feature in the engine and extend the findings report with the weekly comparison.

### Non-goals

- **Detector rewrite.** The dead `volatility_regime` criterion, missing hysteresis, and unused `confidence`-for-sizing are real defects but out of scope. Worth fixing only *if* PR 5a/6 shows the overlay is worth keeping. Do not touch `market_regime_detector.py`.
- **Variant toggle wiring.** The `ramp.variant` field is acknowledged in a deferred appendix and left as-is. Wiring it to switch live behavior is a production change that belongs to the post-PR-5 deploy decision, not this plan. Do not edit `strategy_toggle.yaml` (it is gitignored + force-tracked + per-host runtime state; a committed edit won't reach EC2).
- **Production deployment / A7 gate changes.** This plan deploys no variant and does not touch `run_live_paper_trading.py`'s execution path or the A7 paper-validation timer/collector/comparator.
- **Renaming history.** Dated progress docs (`docs/progress/20260515_RAMP_PHASE4*.md`), ops scripts (`scripts/ops/ramp_phase4_close_progress_doc.sh`, `check_ramp_paper_session.sh`), and dated reports keep their names. Only the *importable module* and its *live CLI* get the new name.
- **Universe expansion.** S&P 500 u NASDAQ-100 expansion is deferred -- it is the wrong lever for a transaction-cost-floor problem. Revisit only after PR 5a/6.
- **New cost models.** `costs.py` keeps flat-bps only. PR 5a/6 use flat-bps tiers.
- **FX harness reuse.** Pointing `regime_momentum_lab` at the FX strategy is the long-term payoff but not part of this plan; just stop the name from being RAMP-phase-specific.

---

## Background: current state (verified against `origin/main`)

### The module under rename

```
src/research/ramp_phase4/
  __init__.py     (empty)
  config.py       HarnessConfig (frozen dataclass): start_date, end_date, universe_csv,
                  initial_capital, cost_bps_per_side, timing_mode, rebalance_frequency
                  ('daily' | 'weekly_friday' | 'weekly_wednesday'), rounding_mode,
                  min_trade_value_usd. PURE DATA, no logic.
                  NOTE: rebalance_frequency is DECLARED here but the engine does NOT
                  branch on it (see engine.py note). PR 2 makes that gap loud; PR 6 closes it.
  costs.py        flat_bps_cost(trades, bps) -> float.
  data.py         load_universe_panel(...); _read_closes_from_parquet(...).
                  SIP_DAILY_CACHE_REL = 'cache/ramp_phase4/equities_daily_from_sip.parquet'
                  (locator ~line 45), plus LEGACY_DAILY_CACHE_REL fallback and
                  case-insensitive symbol-column handling. Resolves under the H: storage root.
                  THIS is the only real-world side effect of the rename.
  engine.py       run_variant(cfg, variant_spec) -> List[DailyRecord]. The stateful loop:
                  HarnessState, DailyRecord, compute_trades (whole-share rounding,
                  min_trade_value drop), apply_trades (sells-before-buys), NaN forced-exit,
                  overleverage guard. *** Does NOT read cfg.rebalance_frequency anywhere. ***
  metrics.py      sharpe_ratio, cagr, max_drawdown, avg_daily_turnover, cost_drag_pct,
                  regime_attribution. avg_daily_turnover ALREADY EXISTS -- the turnover
                  instrument is already built.
  reports.py      build_variant_report(...); build_parity_report(...).
  variants.py     REGISTRY: {'V01': VariantSpec, 'V03': VariantSpec}. VariantSpec(id,
                  description, plan_fn). No aliases, no resolve(). V01 ignores exposure_pct
                  (gross always 1.0); V03 honors it (0.5 in crash).
```

### Reference graph for `ramp_phase4` (re-verify before editing)

**Code imports (change with the rename):**
- `scripts/backtest_scripts/ramp_phase4_backtest.py` -- 4 imports. Also renamed in PR 2.
- `scripts/backtest_scripts/_make_parity_report.py` -- 4 imports. Keeps its name.
- Internal sibling imports inside `engine.py`, `reports.py`, `data.py`.

**Test imports (directory moves + monkeypatch string targets):**
- `tests/research/ramp_phase4/` -> 8 files.
- `test_data.py` / `test_engine.py` contain string-literal monkeypatch targets like
  `'src.research.ramp_phase4.data._read_closes_from_parquet'` and
  `'src.research.ramp_phase4.engine.load_universe_panel'`. A find-and-replace MISS here
  breaks the mock *silently*. **Highest-risk part of the rename.**
- `test_variants.py` asserts literal keys `'V01'`/`'V03'` -- these change in PR 3, not PR 1.

**On-disk data path (one-time migration tail):**
- `data.py` `SIP_DAILY_CACHE_REL = 'cache/ramp_phase4/equities_daily_from_sip.parquet'`
  (+ docstring refs). Resolves under the H: storage root.

**Docs / ops scripts (DO NOT rename -- historical):**
- `docs/progress/20260515_RAMP_PHASE4*.md`, `docs/progress/20260519_RAMP_PHASE4_*.md`
- `scripts/ops/ramp_phase4_close_progress_doc.sh`, `scripts/ops/check_ramp_paper_session.sh`

### The planner API the new variants depend on

`compute_plan` in `src/strategies/advanced/ramp_target_planner.py` (keyword-only) takes,
among others, `bear_to_cash: bool = False` -- when `True` AND `regime == "BEAR"`, returns a
plan with `targets={}` and all current positions in `exits`. So `bear_cash` is a one-flag
change. `RampPlan` fields the engine reads: `plan.targets` (keys = symbols), `plan.regime`,
`plan.exposure_pct`.

### The toggle field (for PR 4's deferred appendix)

- `config/trading/strategy_toggle.yaml` (on `origin/main`, last touched by `00ed0cb (#7)`,
  2026-05-26) has, under each strategy, `enabled`, `shutdown_requested`, and `variant`:
  `ramp: v11`; `cscm`/`mp`/`omr`: `v01`.
- `StrategyStateManager` reads the toggle as a **raw dict**: `is_enabled` does
  `self._toggle.get('strategies',{}).get(strategy,{}).get('enabled', False)`. The
  `ToggleConfig` dataclass (`enabled`, `shutdown_requested`) exists but is **not** the
  read path. `variant` is therefore present in the dict but read by nothing.
- `create_ramp_adapter(broker, data_provider=None, initial_capital=None, metrics_registry=None, *, broker_name: str)` -- no `variant` parameter; hardcodes `use_target_planner=True`.

---

## PR 1: Rename module `ramp_phase4` -> `regime_momentum_lab`

**Goal**: Move `src/research/ramp_phase4/` to `src/research/regime_momentum_lab/`, update all imports and test monkeypatch strings, and decouple the on-disk cache path from the module name. Zero numeric-output change.

**Why first**: lowest-risk change; everything else imports the new path. Landing it alone proves the rename is purely mechanical -- if the full suite is green before and after with identical numbers, the move is confirmed.

**Decision (resolved)**: New name is `regime_momentum_lab`. The harness's defining feature is regime-gated variant comparison; "regime_momentum" names the strategy family precisely, "lab" signals "where variants get tested." Strategy-agnostic enough to fit a future non-RAMP regime-momentum variant, but not so generic it implies arbitrary-strategy support (it assumes the RAMP panel shape with SPY/VIX columns).

### Files touched

```
src/research/ramp_phase4/                  -> src/research/regime_momentum_lab/   (git mv the dir)
tests/research/ramp_phase4/                -> tests/research/regime_momentum_lab/  (git mv the dir)
scripts/backtest_scripts/ramp_phase4_backtest.py   (import lines only; file rename is PR 2)
scripts/backtest_scripts/_make_parity_report.py    (import lines only)
```

### Specific changes

**1.1 Move the source directory.** `git mv src/research/ramp_phase4 src/research/regime_momentum_lab` (history follows). Then update internal sibling imports. Grep: `grep -rln 'ramp_phase4' src/research/regime_momentum_lab/`. Replace `from src.research.ramp_phase4.X` -> `from src.research.regime_momentum_lab.X`. Do NOT change any function names, signatures, or logic.

**1.2 Decouple the cache path from the module name.** In `data.py`:
- `SIP_DAILY_CACHE_REL = 'cache/ramp_phase4/...'` -> `'cache/regime_momentum/equities_daily_from_sip.parquet'`.
- Update the docstring references to match.
- **Optional refinement:** route the resolved path through the new `get_data_dir('cache/regime_momentum/...')` helper from `src/settings/data_paths.py` rather than manually joining the storage root, for consistency with the post-reorg canonical-path convention. If you do this, confirm `get_data_dir` resolves under the same H: storage root the loader currently uses (`get_local_storage_dir`).
- **One-time data migration (operational, not code):** the cached Parquet lives at `<H: root>/cache/ramp_phase4/equities_daily_from_sip.parquet`. Pick ONE and note it in the PR description:
  - (a) Physically move it: `mv <root>/cache/ramp_phase4/ <root>/cache/regime_momentum/`. Preferred -- preserves the cached SIP pull.
  - (b) Let the loader re-create it on next run (the SIP-build path exists). Slower first run.
- If the agent cannot reach the H: drive, it MUST NOT attempt the move -- leave a `MIGRATION REQUIRED` note in the PR description and ensure the loader **fails loud** (not silently empty) if the cache is absent.

**1.3 Move the test directory and fix monkeypatch strings.** `git mv tests/research/ramp_phase4 tests/research/regime_momentum_lab`. Then:
- Update all `from src.research.ramp_phase4.X` imports across the 8 test files.
- **Critical:** grep specifically for the string form (not caught by an import-only search):
  `grep -rn "src.research.ramp_phase4" tests/research/regime_momentum_lab/`. Replace
  `ramp_phase4` -> `regime_momentum_lab` in every monkeypatch/patch target.
- Do NOT change `test_variants.py` assertions (`'V01'`, `'V03'`) -- those change in PR 3.

**1.4 Update the two importing scripts' import lines** (file rename of the CLI is PR 2).

### Definition of done

- `grep -rn 'ramp_phase4' src/ tests/` returns zero hits in any `.py` import or patch string (docs/ops-script hits are expected, out of scope).
- `src/research/ramp_phase4/` no longer exists; `src/research/regime_momentum_lab/` exists with all 8 source files.

### Validation

```
pytest tests/research/regime_momentum_lab/ -v        # expect 41/41 passed (same count)
```
Then a numeric-identity check (capture OLD-tree output first, then NEW):
```
python scripts/backtest_scripts/ramp_phase4_backtest.py --variant V03 \
  --start 2025-01-01 --end 2026-04-30 --cost-bps 0,5 --output /tmp/v03_after.md
diff /tmp/v03_before.md /tmp/v03_after.md   # expect: only the git-sha line differs
```
Anything other than the git-sha line differing means the rename changed behavior -- STOP.

---

## PR 2: Rename CLI -> `run_momentum_variant.py` + rebalance/turnover surface

**Goal**: Rename the live backtest CLI to a function-descriptive name, add a `--rebalance-frequency` argument that is **fail-loud** on values the engine doesn't yet support, and add a one-line realized-turnover summary to stdout.

**Estimated effort**: 1 hour.

### Files touched

```
scripts/backtest_scripts/ramp_phase4_backtest.py  -> scripts/backtest_scripts/run_momentum_variant.py
```

### Specific changes

**2.1 Rename the file.** `git mv scripts/backtest_scripts/ramp_phase4_backtest.py scripts/backtest_scripts/run_momentum_variant.py`. Update the module docstring to "Run a regime-momentum backtest variant against SIP data and emit a Markdown report."

**2.2 Add `--rebalance-frequency` (fail-loud).** The engine does **not** branch on `cfg.rebalance_frequency` (verified). So the flag must not silently no-op:
```python
p.add_argument('--rebalance-frequency',
               choices=['daily', 'weekly_friday', 'weekly_wednesday'],
               default='daily',
               help='Rebalance cadence. Only "daily" is implemented today; '
                    'weekly_* is built in PR 6 (gated on the daily cost verdict).')
```
Pass it into `HarnessConfig(rebalance_frequency=args.rebalance_frequency)`. Then, **before running**, guard:
```python
if args.rebalance_frequency != 'daily':
    raise NotImplementedError(
        f'rebalance_frequency={args.rebalance_frequency!r} is not implemented in the '
        f'engine yet (see PR 6). Only "daily" is supported. The HarnessConfig field '
        f'exists but engine.run_variant does not branch on it.')
```
This converts a silent trap into a loud, honest failure. `daily` works today.

**2.3 Add a realized-turnover summary line to stdout.** After the per-tier runs:
```python
from src.research.regime_momentum_lab.metrics import avg_daily_turnover
for bps, records in records_by_tier.items():
    t = avg_daily_turnover(records)
    print(f'[turnover] {args.variant} @ {bps}bps: avg_daily_turnover={t:.4f} '
          f'({t * 100:.2f}% of portfolio/day)')
```

### Definition of done

- `ramp_phase4_backtest.py` no longer exists; `run_momentum_variant.py` exists.
- `--help` shows `--rebalance-frequency` with the "only daily implemented" note.
- A daily run prints a `[turnover]` line per cost tier.
- Passing `--rebalance-frequency weekly_friday` raises `NotImplementedError` (does not silently run daily).

### Validation

```
python scripts/backtest_scripts/run_momentum_variant.py --variant V03 \
  --start 2025-01-01 --end 2026-04-30 --cost-bps 5 \
  --rebalance-frequency daily --output /tmp/v03_daily.md
# expect a [turnover] line; /tmp/v03_daily.md identical to PR-1's v03 output at 5bps
python scripts/backtest_scripts/run_momentum_variant.py --variant V03 \
  --rebalance-frequency weekly_friday --start 2025-01-01 --end 2026-04-30 --cost-bps 5
# expect: NotImplementedError, non-zero exit
```
(This PR still uses the old id `V03` -- renamed in PR 3.)

---

## PR 3: Variant registry -- descriptive names, aliases, `plain` + `bear_cash`

**Goal**: Make `variants.py` the single source of truth. Rename `V01`/`V03` to descriptive ids, add an `aliases` field + `resolve()` so old report labels still map, and add `plain` + `bear_cash`.

**Why medium-risk**: it adds new computation paths and renames registry keys that tests and the CLI's variant `choices` depend on. The existing two variants MUST produce byte-identical output before and after (same plan_fns under new ids).

**Estimated effort**: 1 day.

### The variant model (target)

| New id | Semantics | Overlay? | Exposure | Aliases |
|---|---|---|---|---|
| `plain` | Vanilla momentum, fixed params, no regime-based switching | No (records regime for forensics, does not act on it) | Always full (1.0) | `V1` |
| `prod` | Production: regime overlay + per-regime params + 0.5x crash multiplier honored | Yes | exposure_pct (0.5 in crash) | `V03`, `V0` |
| `prod_no_crash` | Overlay + per-regime params, crash multiplier ignored (gross always 1.0). Parity-test baseline | Yes | Always full (1.0) | `V01` |
| `bear_cash` | `prod` but BEAR regime -> 100% cash | Yes | exposure_pct, except BEAR=0.0 | `V8` |

**Decision (resolved) -- `plain` records the regime but does not act on it.** Call the detector for the `__regime__` label, but pin params to a fixed set (`V1_PARAMS = {'long_p': 21, 'short_p': 5, 'long_w': 0.3, 'pen_w': 5.0, 'top_n': 10}`, origin: `ramp_root_cause_20260505.py`) and full exposure. The regime label has zero effect on selection, sizing, or exposure -- metadata only -- so PR 5a forensics can see what regime *would* have been called on days `plain` beats `prod`.

### Specific changes

**3.1 Extend `VariantSpec` with aliases + add `resolve()`.**
```python
@dataclass(frozen=True)
class VariantSpec:
    id: str
    description: str
    plan_fn: PlanFn
    aliases: tuple[str, ...] = ()

def resolve(name: str) -> VariantSpec:
    """Look up a variant by canonical id or any registered alias.
    Lets historical report labels (V0, V01, V03, V1, V8) keep resolving.
    Raises KeyError with the full id list on miss."""
    if name in REGISTRY:
        return REGISTRY[name]
    for spec in REGISTRY.values():
        if name in spec.aliases:
            return spec
    raise KeyError(
        f'Unknown variant {name!r}. Known ids: {sorted(REGISTRY)}; '
        f'aliases: {{a for s in REGISTRY.values() for a in s.aliases}}')
```

**3.2 Rebuild REGISTRY with the four variants.** Keep `_compute_plan_from_panel` (the shared spine). The existing `_variant_v01`/`_variant_v03` bodies become `_variant_prod_no_crash`/`_variant_prod` (rename functions, do not change bodies). Add:
- `_variant_bear_cash`: identical to `_variant_prod` except `compute_plan` receives `bear_to_cash=True`. Cleanest: factor `_compute_plan_from_panel` to accept `bear_to_cash: bool = False` and thread it into the `compute_plan(...)` call. When the resulting plan has empty `targets` (BEAR cashed out), the engine must sell existing positions to zero -- **verify against `engine.compute_trades` that empty target_weights + existing positions produces sells-to-zero.**
- `_variant_plain`: call the detector for the regime label only; compute momentum with `V1_PARAMS`; return equal weights `1/top_n` across the top 10 at full exposure; do NOT route through `compute_plan`'s regime-param path.
```python
REGISTRY: Dict[str, VariantSpec] = {
    'plain': VariantSpec(id='plain',
        description='Vanilla momentum, fixed params (pen_w=5.0, top_n=10), no regime overlay; '
                    'regime recorded for forensics only',
        plan_fn=_variant_plain, aliases=('V1',)),
    'prod': VariantSpec(id='prod',
        description='Production RAMP: regime overlay + per-regime params + 0.5x crash multiplier',
        plan_fn=_variant_prod, aliases=('V03', 'V0')),
    'prod_no_crash': VariantSpec(id='prod_no_crash',
        description='Overlay + per-regime params, crash multiplier ignored (parity-test baseline)',
        plan_fn=_variant_prod_no_crash, aliases=('V01',)),
    'bear_cash': VariantSpec(id='bear_cash',
        description='Production overlay but BEAR regime -> 100% cash (bear_to_cash=True)',
        plan_fn=_variant_bear_cash, aliases=('V8',)),
}
```

**3.3 Update `test_variants.py`.**
- Replace the `V01`/`V03`-contains test with one asserting the four canonical ids are present `VariantSpec` instances.
- `test_resolve_aliases`: `resolve('V03').id == 'prod'`, `resolve('V01').id == 'prod_no_crash'`, `resolve('V1').id == 'plain'`, `resolve('V8').id == 'bear_cash'`, `resolve('nonsense')` raises `KeyError`.
- Rename the calm/crash tests to the new ids; numeric assertions unchanged (same plan_fns).
- `test_plain_ignores_regime`: in a crash panel, `plain` gross stays ~1.0 while `prod` drops to <=0.6.
- `test_bear_cash_goes_to_cash_in_bear`: construct a BEAR-classifying panel; assert body weights sum to 0.0. If deterministic BEAR from a synthetic panel is hard, `@pytest.mark.skip(reason="needs deterministic BEAR fixture")` with a TODO rather than asserting something false.

**3.4 Update `_make_parity_report.py` ids.** Change its `REGISTRY['V01']`/`REGISTRY['V03']` to `resolve('V01')`/`resolve('V03')` (immune to future renames). Leave the `v01_records`/`v03_records` parameter names in `build_parity_report` as-is (internal) unless you also update `reports.py` + `test_reports.py` together.

**3.5 CLI picks up new ids via `resolve()`.** `run_momentum_variant.py` should call `spec = resolve(args.variant)` and drop the hard `choices=` list (print known ids in help text instead), so historical invocations with `V03` still work. **Decision (resolved):** use `resolve()`, remove the hard `choices=`.

### Definition of done

- `REGISTRY` has exactly four keys: `plain`, `prod`, `prod_no_crash`, `bear_cash`.
- `resolve()` maps all five legacy labels (`V0`, `V01`, `V03`, `V1`, `V8`) to the right specs.
- No bare `V0n` string appears as a registry KEY in `src/`.
- `prod`/`prod_no_crash` produce byte-identical reports to PR-2's `V03`/`V01` runs (modulo the variant-id header string).

### Validation

```
pytest tests/research/regime_momentum_lab/test_variants.py -v
python scripts/backtest_scripts/run_momentum_variant.py --variant prod \
  --start 2025-01-01 --end 2026-04-30 --cost-bps 0,5 --output /tmp/prod.md
# metric tables in /tmp/prod.md must match /tmp/v03_daily.md from PR 2
python scripts/backtest_scripts/run_momentum_variant.py --variant plain   ... --output /tmp/plain.md
python scripts/backtest_scripts/run_momentum_variant.py --variant bear_cash ... --output /tmp/bear_cash.md
python scripts/backtest_scripts/run_momentum_variant.py --variant V03      ... --output /tmp/alias_check.md
```
**Sanity cross-check (not a hard gate):** Phase 3A recorded, at 0% cost on 2025-01-01..2026-04-30, EXT-OOS Sharpe ~0.070 (production-equivalent), ~0.314 (vanilla), ~0.571 (BEAR-to-cash). Exact matches are NOT expected (different data source, stateful harness), but the **ordering** should hold: `bear_cash > plain > prod` at 0% cost. If it inverts, a plan_fn is wrong -- investigate before PR 5a.

---

## PR 4: Archive dated scripts; record the inert toggle field

**Goal**: Remove the four dated investigation scripts from the active tree (preserving them under `_archived/`), and document the present-but-inert `ramp.variant` field in a deferred-work appendix. **No variant code or config is changed in this plan.**

**Estimated effort**: 1-2 hours.

### Part A: Archive the dated scripts

The repo's `pytest.ini` has `norecursedirs = _archived ...`. There is currently NO `_archived` directory -- this PR creates the first one. Use `_archived` (matching pytest.ini), NOT `archive/`.

**4.1 Create the archive and move the scripts.** (Re-verify exact filenames with `ls scripts/backtest_scripts/ramp_*` first.)
```
mkdir -p scripts/backtest_scripts/_archived
git mv scripts/backtest_scripts/ramp_root_cause_20260505.py            scripts/backtest_scripts/_archived/
git mv scripts/backtest_scripts/ramp_phase3a_variants_20260505.py      scripts/backtest_scripts/_archived/
git mv scripts/backtest_scripts/ramp_phase3b_bear_optimizer_20260505.py scripts/backtest_scripts/_archived/
git mv scripts/backtest_scripts/ramp_re_eval_20260504.py               scripts/backtest_scripts/_archived/
```
**Before moving, confirm nothing imports them:** `grep -rn "ramp_root_cause\|ramp_phase3a\|ramp_phase3b\|ramp_re_eval" --include=*.py src/ scripts/ tests/`. Expect zero import hits. If any are imported, STOP -- that changes archival from "move" to "extract-then-move."

**4.2 Add `scripts/backtest_scripts/_archived/README.md`:**
```markdown
# Archived RAMP investigation scripts

These dated scripts performed the RAMP alpha-decay root-cause investigation (2026-05).
Their FINDINGS are preserved in docs/reports/ramp/20260505_*.md. Their CODE is archived
(not deleted) for audit. Each reimplemented its own backtest loop, metrics, and data
loader; that functionality now lives in the tested harness at
src/research/regime_momentum_lab/. Variants V0/V01/V03/V1/V8 referenced here map to the
registry ids prod/prod_no_crash/plain/bear_cash via variants.resolve(). Excluded from
pytest discovery via norecursedirs in pytest.ini.

Do not re-activate by importing. To reproduce a finding, run the equivalent registry
variant through scripts/backtest_scripts/run_momentum_variant.py.
```

### Part B: Record the inert toggle field (documentation only -- no edit)

The `ramp.variant: v11` field is **present and inert**, not dead-deletable. Wiring it to
switch live behavior is a production change out of scope here (post-PR-5 deploy decision).
This PR does **not** edit `strategy_toggle.yaml` or `strategy_state_manager.py`. It records
the three facts below in Appendix B so they are never rediscovered. (No code or config
change ships in Part B.)

### Definition of done

- `scripts/backtest_scripts/_archived/` contains the four scripts + README; `pytest --collect-only 2>&1 | grep -c _archived` returns 0.
- Appendix B (deferred toggle work) is present in this spec / the eventual plan.
- `strategy_toggle.yaml` is **unchanged**.

### Validation

```
pytest --collect-only -q 2>&1 | grep _archived   # expect: no output (not collected)
pytest tests/ -q                                  # full suite still green
```

---

## PR 5a: The experiment -- realized DAILY turnover + cost verdict

**Goal**: Answer the question the whole effort exists to enable, at daily cadence: *what is RAMP's realized daily turnover, and does any variant clear realistic transaction costs daily?* -- and write it up.

**Why this is the payoff**: every cost figure in the Phase 3A report assumed turnover = 1.0 (full daily rotation), which the report flags as conservative. A top-10 momentum book does not fully rotate daily. If realized turnover is ~0.15-0.25, cost drag drops materially and the strategy may clear 5 bps. This PR produces the measured number.

**This PR writes NO production code** -- it runs the harness built in PRs 1-3 and writes a report. Weekly cadence is PR 6, gated on this PR's verdict.

**Estimated effort**: 1 day.

### Steps

**5a.1 Measure realized daily turnover** (the headline number `T`). Run each variant at 0 bps over the full IS+OOS+EXT-OOS span and the EXT-OOS span separately, capturing the `[turnover]` line:
```
for V in plain prod bear_cash; do
  python scripts/backtest_scripts/run_momentum_variant.py --variant $V \
    --start 2017-01-01 --end 2026-04-30 --cost-bps 0 --rebalance-frequency daily \
    --output docs/reports/ramp/_scratch/${V}_full_daily.md
  python scripts/backtest_scripts/run_momentum_variant.py --variant $V \
    --start 2025-01-01 --end 2026-04-30 --cost-bps 0 --rebalance-frequency daily \
    --output docs/reports/ramp/_scratch/${V}_ext_daily.md
done
```

**5a.2 Daily cost sensitivity at realistic tiers.** The CLI iterates cost tiers and the engine applies them via `flat_bps_cost`, so drag is modeled from realized trades, not back-of-envelope:
```
for V in plain prod bear_cash; do
  python scripts/backtest_scripts/run_momentum_variant.py --variant $V \
    --start 2025-01-01 --end 2026-04-30 --cost-bps 0,2.5,5,7.5 --rebalance-frequency daily \
    --output docs/reports/ramp/_scratch/${V}_ext_costsweep_daily.md
done
```

**5a.3 Write the findings report** at `docs/reports/ramp/YYYYMMDD_turnover_cost_sensitivity.md` (house style: Context / Methodology / results tables / Conclusion / Implications). It must answer explicitly:
1. **Realized daily turnover** `T` for each variant, vs the assumed 1.0; implied cost drag at 5 bps (`2 * 5bps * T` per day) against each variant's gross daily return.
2. **Net Sharpe/CAGR/MaxDD at 0/2.5/5/7.5 bps, daily** for `plain`, `prod`, `bear_cash` -- harness-measured.
3. **The daily verdict**, in one of:
   - "RAMP clears realistic costs at daily realized turnover" -> recommend a variant; deploy decision moves to a separate plan. **PR 6 not triggered.**
   - "RAMP does not clear costs at daily cadence" -> **PR 6 triggered**: test the weekly structural lever (note CSCM clears costs weekly).
4. **Statistical honesty**: Sharpe SE on ~331 EXT-OOS days is ~0.17 -- differences below ~0.2 are noise. Do not tune any parameter on EXT-OOS.
5. **Forensics**: on days `plain` beats `prod`, what regime was recorded? (Uses `plain`'s metadata-only regime label.)

**5a.4 Update the RAMP report index** with a pointer to the findings report and the turnover number `T`, so it is not re-derived later.

### Definition of done

- A findings report exists with answers 1-5; `T` is stated per variant; a daily verdict is recommended; the report explicitly states whether PR 6 is triggered.

### Validation (internal consistency)

The harness-measured turnover at 0 bps must match the turnover implied by cost drag at higher tiers (same trades). Spot-check one variant: cost_drag at 5 bps should equal ~`2 * 0.0005 * T * days * mean_portfolio_value`. If not, the cost model and turnover metric disagree -- one is wrong.
Also smoke-check the daily SIP loader works post-data-reorg on the synced tree (the reorg didn't touch `src/research/`, but `data.py`'s `src/settings` imports should resolve).

---

## PR 6 (gated): Weekly rebalance -- engine feature + comparison

**Trigger**: Build this PR **only if** PR 5a's verdict is "RAMP does not clear costs at daily cadence." If daily clears costs, skip PR 6 and move to the deploy decision.

**Goal**: Implement weekly rebalancing in the engine (the `rebalance_frequency` field is declared but unhonored), then re-run the cost sweep weekly and extend the findings report.

**Why it is its own PR**: weekly rebalancing is a genuine **behavior-adding** engine change with a correctness surface of its own. The draft mis-scoped it as a CLI passthrough; it is not.

**Estimated effort**: 1-1.5 days.

### Part A: Engine weekly-rebalance feature

**6.1** In `engine.run_variant`, branch on `cfg.rebalance_frequency`:
- `daily` (default): unchanged behavior.
- `weekly_friday` / `weekly_wednesday`: only **recompute and apply trades** on the target weekday (or the last/first trading day of the week if the target day is a market holiday); on non-rebalance days, **carry positions** and mark-to-market (still record a `DailyRecord` with zero `turnover_usd`/`cost_usd`, updated `portfolio_value`/`daily_return`).
- Resolve the holiday/short-week rule explicitly (e.g., "rebalance on the last trading day on-or-before the target weekday each week"). Document the chosen rule in the engine docstring.
- Ensure `avg_daily_turnover` and `flat_bps_cost` remain correct across skipped days (turnover is concentrated on rebalance days; the average is over all days).

**6.2** Remove the `NotImplementedError` guard from `run_momentum_variant.py` (PR 2.2) now that weekly is real.

**6.3 Tests** (`test_engine.py`): assert `weekly_friday` produces trades only on Fridays (or the resolved last-trading-day-of-week), zero turnover on other days, and that a full-week carry preserves positions; assert turnover over a multi-week window is ~1/5 of the daily run on the same window (order-of-magnitude, not exact).

### Part B: Weekly comparison + report addendum

**6.4** Re-run 5a.2's cost sweep with `--rebalance-frequency weekly_friday`:
```
for V in plain prod bear_cash; do
  python scripts/backtest_scripts/run_momentum_variant.py --variant $V \
    --start 2025-01-01 --end 2026-04-30 --cost-bps 0,2.5,5,7.5 --rebalance-frequency weekly_friday \
    --output docs/reports/ramp/_scratch/${V}_ext_costsweep_weekly.md
done
```
**6.5** Extend the findings report with: weekly net Sharpe/CAGR/MaxDD at each tier; the turnover reduction weekly achieves vs daily; and the final verdict, in one of:
- "Clears costs only at weekly cadence" -> recommend weekly rebalance (the CSCM-style structural fix).
- "Does not clear costs at any tested cadence" -> recommend retiring RAMP as a standalone daily strategy, fold the momentum signal into a multi-factor sleeve, and redeploy `regime_momentum_lab` to the FX work.

### Definition of done

- `engine.run_variant` honors `rebalance_frequency`; tests prove weekly fires only on the target day.
- The `NotImplementedError` guard is gone.
- The findings report has the weekly comparison and a final cross-cadence verdict.

---

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Starting from a stale local `main` (as the draft did) | Medium | High (re-introduces the corrected findings) | Ground rule 1: branch from `origin/main`; confirm `ramp.variant` present and `src/settings/data_paths.py` exists before starting |
| Monkeypatch string-literal miss in test rename (PR 1.3) | Medium | High (silent broken mock) | Dedicated grep for the string form; confirm 41/41 AND spot-check one patched test fails correctly when the real function is broken |
| Cache-path change orphans the SIP Parquet on H: (PR 1.2) | High (agent likely can't reach H:) | Medium | `MIGRATION REQUIRED` note; loader fails loud on missing cache, never silently empty |
| `bear_cash` empty-targets case mishandled by `compute_trades` (PR 3.2) | Low | Medium | `test_bear_cash_goes_to_cash_in_bear` asserts body weights sum to 0; verify against `engine.compute_trades` |
| Variant-ordering inverts vs Phase 3A (PR 3 sanity check) | Low | High (a plan_fn is wrong) | Cross-check `bear_cash > plain > prod` at 0% cost before PR 5a |
| Weekly rebalance turnover/holiday accounting wrong (PR 6.1) | Medium | Medium | Explicit holiday rule in docstring; tests assert trades only on target day and ~1/5 turnover |
| Rename changes a numeric output (PR 1) | Low | High (silent research corruption) | Byte-diff a `prod`/`V03` report before vs after; only the git-sha line may differ |
| Someone wires the inert toggle field expecting `v11` to resolve | Low | Medium | Appendix B documents that `v11` is NOT a registry id/alias and must be normalized to `prod` (or aliased) first |

## Rollback

Every PR is a clean git revert. PR 1's data-cache migration is the only step with a non-git
side effect: if rolled back, move `cache/regime_momentum/` back to `cache/ramp_phase4/` on the
data drive (or let it re-pull). No PR writes to production state, broker config, the live
decision log, or `strategy_toggle.yaml`. PR 6 is gated and independently revertable.

## Appendix A: Legacy-label -> canonical-id map (for report readers)

| Legacy label | Where it appeared | Canonical id | `resolve()` returns |
|---|---|---|---|
| `V0` | root_cause, phase3a (production reference) | `prod` | `prod` |
| `V03` | library variants.py, phase4 reports | `prod` | `prod` |
| `V01` | library variants.py, phase4 reports | `prod_no_crash` | `prod_no_crash` |
| `V1` | root_cause, phase3a (vanilla momentum) | `plain` | `plain` |
| `V8` | phase3a (BEAR-to-cash) | `bear_cash` | `bear_cash` |
| `V2` | root_cause (inverse-vol) | -- (not ported) | KeyError |
| `V4` | root_cause (SPY-vol overlay) | -- (not ported) | KeyError |
| `V5a/b/c` | phase3a (vol-adj momentum) | -- (not ported) | KeyError |
| `v11` | `strategy_toggle.yaml` (inert) | -- (NOT a variant id; see Appendix B) | KeyError |

`V2`, `V4`, `V5a/b/c` are deliberately NOT ported (Phase 3A/3B classified them RESEARCH ONLY). `v11` is a toggle value, never a defined backtest variant.

## Appendix B: Deferred work -- the `ramp.variant` toggle field

The `strategy_toggle.yaml` `variant` field is deliberately left untouched by this plan. Whether
to make it real is a **post-PR-5 deploy decision** (it only matters once a variant is worth
deploying). Three facts to carry forward so they are not rediscovered:

1. **The field is present, not deleted.** On `origin/main` (since `00ed0cb (#7)`), all four
   strategies carry `variant`: `ramp: v11`, `cscm`/`mp`/`omr`: `v01`. It is round-tripped in
   the toggle dict and read by no accessor -- present-but-inert.
2. **The read path is the raw dict, not `ToggleConfig`.** A future accessor should be
   `get_variant(strategy)` doing `self._toggle.get('strategies',{}).get(strategy,{}).get('variant')`,
   mirroring `is_enabled` -- NOT a new `ToggleConfig` field (the dataclass is not the read path).
3. **`v11` is not a valid registry id and not in the alias set** (aliases cover `V0/V01/V03/V1/V8`).
   Any future `resolve()`-based startup guard MUST first normalize `v11 -> prod` (the behavior the
   live adapter runs today) or add a `v11` alias to the `prod` spec -- otherwise RAMP fails to start.
4. **Do not hand-edit `strategy_toggle.yaml`.** It is gitignored + force-tracked + per-host
   runtime state; a committed edit will not reach EC2, and the live value also lives in per-host
   runtime state. Any change goes through the toggle's write path (`StrategyStateManager`), not a
   manual edit.

## Appendix C: Final target tree (after PR 4; PR 6 additions in brackets)

```
src/research/regime_momentum_lab/
  __init__.py  config.py  costs.py  data.py  engine.py  metrics.py  reports.py  variants.py
  [engine.py honors rebalance_frequency after PR 6]

tests/research/regime_momentum_lab/
  __init__.py  test_config.py  test_costs.py  test_data.py  test_engine.py
  test_metrics.py  test_reports.py  test_variants.py

scripts/backtest_scripts/
  run_momentum_variant.py        (renamed CLI; --rebalance-frequency fail-loud; turnover line; resolve())
  _make_parity_report.py         (ids -> prod/prod_no_crash via resolve())
  _archived/
    README.md
    ramp_root_cause_20260505.py
    ramp_phase3a_variants_20260505.py
    ramp_phase3b_bear_optimizer_20260505.py
    ramp_re_eval_20260504.py

config/trading/strategy_toggle.yaml   (UNCHANGED -- variant field inert; see Appendix B)
docs/reports/ramp/
  YYYYMMDD_turnover_cost_sensitivity.md   (PR 5a output; PR 6 addendum if triggered)
  (existing dated reports unchanged)
```
