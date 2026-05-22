"""Pure target-weight filter functions for Phase C Wave 1 variants.

Each filter takes a proposed-target dict and returns an adjusted dict.
Filters compose by chaining. The output of each filter sums to 1.0
(equal-weight renormalization) so consecutive filters maintain the
same invariant.
"""

from __future__ import annotations

from datetime import datetime
from math import ceil
from typing import Dict

import pandas as pd

from src.research.ramp_phase4.engine import HarnessState


def rank_buffer(
    proposed_targets: Dict[str, float],
    state: HarnessState,
    buffer_size: int,
    universe_ranking: "pd.Series",
    top_n: int,
) -> Dict[str, float]:
    """V04: keep currently-held names that rank within top_n + buffer.

    Returns target weights that include the proposed names PLUS any
    currently-held symbol whose rank in universe_ranking is within
    [1, top_n + buffer_size]. Final dict is equal-weighted and sums to 1.0.

    Args:
        proposed_targets: dict[symbol -> proposed weight] from V01 base.
        state: harness state (for state.positions).
        buffer_size: how many ranks past top_n to tolerate before dropping a held name.
        universe_ranking: pd.Series mapping symbol -> momentum rank (1 = best).
        top_n: the target position count for the current regime.

    Returns:
        dict[symbol -> equal-weight] summing to 1.0.
    """
    held_symbols = set(state.positions.keys())
    proposed_symbols = set(proposed_targets.keys())

    # Held names within the buffer zone (rank <= top_n + buffer_size) are retained
    # even if not in the proposed set.
    buffer_limit = top_n + buffer_size
    retained = {
        sym for sym in held_symbols - proposed_symbols
        if sym in universe_ranking.index and universe_ranking[sym] <= buffer_limit
    }
    final_symbols = proposed_symbols | retained

    if not final_symbols:
        return {}
    weight = 1.0 / len(final_symbols)
    return {sym: weight for sym in final_symbols}
