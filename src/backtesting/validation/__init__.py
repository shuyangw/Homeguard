"""Statistical validation framework for backtested strategies.

Provides reusable tools for assessing whether a strategy's observed
performance is statistically significant or likely due to overfitting.

Modules:
    deflated_sharpe - Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)
    bootstrap       - Bootstrap confidence intervals for strategy metrics
    cpcv            - Combinatorial Purged Cross-Validation
    permutation     - Permutation test framework for signal significance
    combined_gate   - Combined CPCV + DSR + PBO pass/fail statistical gate
"""

from src.backtesting.validation.deflated_sharpe import (
    DSRResult,
    compute_deflated_sharpe,
)
from src.backtesting.validation.bootstrap import (
    BootstrapCIResult,
    BootstrapSuiteResult,
    bootstrap_metric,
    bootstrap_strategy_suite,
)
from src.backtesting.validation.cpcv import (
    CPCVResult,
    cpcv_splits,
    generate_cpcv_splits,
    run_cpcv,
)
from src.backtesting.validation.permutation import (
    PermutationResult,
    run_permutation_test,
)
from src.backtesting.validation.combined_gate import combined_gate

__all__ = [
    "DSRResult",
    "compute_deflated_sharpe",
    "BootstrapCIResult",
    "BootstrapSuiteResult",
    "bootstrap_metric",
    "bootstrap_strategy_suite",
    "CPCVResult",
    "cpcv_splits",
    "generate_cpcv_splits",
    "run_cpcv",
    "PermutationResult",
    "run_permutation_test",
    "combined_gate",
]
