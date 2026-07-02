"""SPAN-style approximate futures margin.

Scan-range per-contract margin from contract_specs, plus an optional
inter-commodity offset matrix (credit % applied to the smaller leg when two
roots are held opposite-signed). Replaceable: a true SPAN engine can implement
the same requirement()/check_and_scale() interface later without touching the
simulator.
"""
from __future__ import annotations

import math

from src.data.futures.contract_specs import get_spec

# Inter-commodity offset credits are opt-in: pass `offset_matrix` explicitly.
# With no matrix, requirement() is pure gross scan-range margin (no netting).
# Example credits (fraction, applied only when the two roots are held
# OPPOSITE-signed) for a caller who wants to enable them:
#   {("ES", "NQ"): 0.75, ("ZN", "ZB"): 0.70}
DEFAULT_OFFSETS: dict[tuple[str, str], float] = {}


class MarginModel:
    def __init__(self, offset_matrix: dict[tuple[str, str], float] | None = None):
        raw = DEFAULT_OFFSETS if offset_matrix is None else offset_matrix
        # store symmetrically for easy lookup
        self._offsets: dict[frozenset[str], float] = {
            frozenset(k): v for k, v in raw.items()
        }

    def _gross(self, positions: dict[str, int]) -> float:
        return sum(abs(n) * get_spec(root).initial_margin for root, n in positions.items())

    def requirement(self, positions: dict[str, int]) -> float:
        total = self._gross(positions)
        # subtract offset credits for opposite-signed pairs present in the book
        roots = list(positions)
        for i in range(len(roots)):
            for j in range(i + 1, len(roots)):
                a, b = roots[i], roots[j]
                credit = self._offsets.get(frozenset((a, b)))
                if credit is None:
                    continue
                na, nb = positions[a], positions[b]
                if na == 0 or nb == 0 or (na > 0) == (nb > 0):
                    continue  # same direction or empty -> no offset
                leg_a = abs(na) * get_spec(a).initial_margin
                leg_b = abs(nb) * get_spec(b).initial_margin
                total -= credit * min(leg_a, leg_b)
        return max(total, 0.0)

    def utilization(self, positions: dict[str, int], equity: float) -> float:
        if equity <= 0:
            return float("inf")
        return self.requirement(positions) / equity

    def check_and_scale(self, targets: dict[str, int], equity: float, cap: float = 0.5) -> dict[str, int]:
        req = self.requirement(targets)
        budget = cap * equity
        if req <= budget or req <= 0:
            return dict(targets)
        factor = budget / req
        return {root: int(math.floor(abs(n) * factor)) * (1 if n >= 0 else -1)
                for root, n in targets.items()}
