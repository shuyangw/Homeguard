# Methodology Changelog

Tracks versioned changes to `docs/methodology/backtesting.md`. Each entry summarises the change and links to relevant commits or PRs in `docs/planning/`.

## v2 (2026-05-12)

- Section 11 added: Exit Logic and Profit-Taking Methodology (11 subsections covering taxonomy, bar-resolution, same-bar fill order, gap modeling, stop slippage, MAE/MFE methodology, profit-taking by asset class, parameter budget for exits, code-reviewer responsibilities, optimizer behavior on exit parameters, registry integration).
- Section 12 added: Required Diagnostic Outputs (6 subsections covering trade-level metrics, capacity curve, regime transitions, hyperparameter temporal stability, benchmark / information ratio, consolidated checklist).
- Registry schema extended (Section 11.11): `exit_logic_summary JSON`, `mae_mfe_validated BOOLEAN` columns. Append wiring follows in a separate PR.
- Appendix "Reading priority for agents" table updated to reflect agents actually on disk. `portfolio-integrator`, `strategy-architect`, `strategy-implementer` moved to a "Future agents" block per decision B (defer until concrete trigger).
- New gates added to operationalize Section 12:
  - Trade-expectancy consistency: reject if portfolio Sharpe > 1.0 AND trade-level expectancy after costs <= 0 (12.1).
  - Regime-transition concentration: reject if > 50% of total drawdown happens at regime transitions AND transition Sharpe < 0 (12.3).
  - Parameter temporal stability: reject if any tunable parameter is UNSTABLE across walk-forward windows (12.4).
  - Information ratio floors per asset class (12.5).
- Stop-loss governance: MAE/MFE-derived stops are now the only defensible source for stop levels in live-bound strategies. Optimizer-discovered stops without MAE/MFE backing are rejected at Phase 9 (11.6, 11.10).

Related: `docs/planning/20260512_methodology_rollout_v3_plan.md` PR 2.

## v1 (2026-05-12)

- Initial consolidated methodology (Sections 1-10).
- Bias prevention, statistical framework (PSR / DSR / PBO with correct formulas), walk-forward (purge + embargo as distinct concepts), cost models per asset class, stopping conditions, portfolio integration rules, point-in-time data conventions, reproducibility identity fields, registry schema, Homeguard-specific reference.
- Replaces inline rules previously scattered across `.claude/agents/backtest-driver.md`, `backtest-optimizer.md`, `trading-lead.md`, `trade-log-analyzer.md`.
- Fixes: DSR formula (replaces old `Sharpe * (1 - ln(N)/(2T))` approximation with Bailey & Lopez de Prado 2014), embargo definition (NOT equal to feature lookback per 3.3), options slippage (alpha-of-half-spread instead of "50-75% of bid-ask"), regime detector path (`MarketRegimeDetector` at `src/strategies/advanced/market_regime_detector.py`), systemd service references, EC2 memory threshold (3 GB on t4g.medium, not 900 MB).
