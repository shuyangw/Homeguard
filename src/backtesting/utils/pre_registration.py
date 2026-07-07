"""Pre-registration gate for futures strategy backtests.

Enforces that every futures backtest config declares its trial up front
(construction, expected sign, hypothesis) so the DSR trial count stays honest
and post-hoc sign flips are visible. The block is stored verbatim in the
experiment registry via append_run(params=config)."""
from __future__ import annotations

from typing import Any, Mapping

_REQUIRED = ("construction", "expected_sign", "hypothesis")
_VALID_SIGNS = {"long", "short", "long_short", "neutral"}


def validate_pre_registration(config: Mapping[str, Any]) -> None:
    block = config.get("pre_registration")
    if not block or not isinstance(block, Mapping):
        raise ValueError(
            "config is missing a non-empty 'pre_registration' block "
            "(construction, expected_sign, hypothesis)"
        )
    for key in _REQUIRED:
        val = block.get(key)
        if val is None or (isinstance(val, str) and not val.strip()):
            raise ValueError(f"pre_registration.{key} is missing or empty")
    if block["expected_sign"] not in _VALID_SIGNS:
        raise ValueError(
            f"pre_registration.expected_sign must be one of {sorted(_VALID_SIGNS)}, "
            f"got {block['expected_sign']!r}"
        )
