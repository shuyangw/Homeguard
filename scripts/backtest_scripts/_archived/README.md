# Archived RAMP investigation scripts

These dated scripts performed the RAMP alpha-decay root-cause investigation
(2026-05). Their FINDINGS are preserved in docs/reports/ramp/20260505_*.md.
Their CODE is archived (not deleted) for audit. Each reimplemented its own
backtest loop, metrics, and data loader; that functionality now lives in the
tested harness at src/research/regime_momentum_lab/. Variants V0/V01/V03/V1/V8
referenced here map to the registry ids prod/prod_no_crash/plain/bear_cash via
variants.resolve(). Excluded from pytest discovery via norecursedirs in pytest.ini.

Do not re-activate by importing. To reproduce a finding, run the equivalent
registry variant through scripts/backtest_scripts/run_momentum_variant.py.
