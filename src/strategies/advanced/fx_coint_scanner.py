"""CointScanner (#37): rolling Engle-Granger cointegration scanner.

Monthly, causal. On each scan date a trailing ``scan_window``-day (capped at the
data available so far) Engle-Granger test (``test_pair`` on ``ln`` levels, data
up to that date only) is run over every candidate pair -- all instrument pairs
sharing at most one 3-letter currency code and not forming a mechanical triangle.
A pair is tradeable when its ADF p-value clears ``adf_max``, its OU half-life
lies in ``half_life_range``, and its spread vol is large enough that a 1.5-sigma
move clears twice the nominal round-trip cost. Tradeable pairs are ranked by an
edge/cost proxy (spread vol / cost) and the top ``top_n`` form the watchlist. On
each monthly scan the watchlist is rebuilt from that scan's fresh top-N (meta --
hedge, half-life, adf baseline -- recomputed so re-qualifying pairs enter on a
current hedge and stale flat pairs age out), except a pair with an OPEN position
is retained regardless of the fresh ranking, keeping its entry-time hedge fixed
for the life of the trade so its structural exit stays manageable.

Watchlist pairs run a per-date mean-reversion state machine on the current-date
residual z-score: enter when ``|z| > entry_z`` (signed against the divergence,
``-sign(z)``); exit on ``|z| < target_z`` (reverted), ``|z| > stop_z``
(blown out), ``2 * half_life`` days held, or the STRUCTURAL exit -- the pair's
rolling ADF p-value degrades by more than 0.2 versus its selection baseline for
10 consecutive days, which also removes it from the watchlist. Active pairs emit
a single ``Spread(a, b, hedge_ratio, strength)`` with the trailing spread vol
from the base class. All computation uses only rows up to the current date.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from src.backtesting.costs.fx import fx_round_trip_pips
from src.backtesting.engine.spread_sizing import Spread
from src.data.artifacts.cointegration import test_pair
from src.strategies.advanced.fx_spread_base import SpreadStrategy

_STRENGTH_CAP = 20.0
_STRUCTURAL_ADF_DEGRADE = 0.2
_STRUCTURAL_SUSTAIN_DAYS = 10
_MIN_SCAN_DAYS = 30


def _is_mechanical_triangle(x: str, y: str) -> bool:
    # A shared currency that is the base of one leg and the quote of the other
    # makes the two legs a triangle (x = A/S, y = S/B -> A/B), which is a
    # mechanical relationship. Same-numeraire pairs (shared currency in the same
    # slot, e.g. both quote CAD) are legitimate relative-value candidates.
    xb, xq, yb, yq = x[:3], x[3:], y[:3], y[3:]
    return xq == yb or xb == yq


def _candidate_pairs(pairs) -> list[tuple]:
    """Candidate legs for the cointegration scan.

    Scope note: this filter is NOT a general "reducible to a third pair"
    exclusion. It excludes only cross-slot mechanical-triangle chains -- where
    the shared currency is the BASE of one leg and the QUOTE of the other
    (e.g. EURGBP + GBPUSD, which chains to EURUSD). It KEEPS same-slot
    common-numeraire pairs, where the shared currency sits in the same slot of
    both legs (both quote or both base it), e.g. EURCAD/AUDCAD (AUD-vs-CAD-vs-USD
    relative value). Keeping same-numeraire crosses is the standard FX
    relative-value convention, so it is intentional, not an oversight.
    """
    out = []
    for x, y in combinations(pairs, 2):
        shared = {x[:3], x[3:]} & {y[:3], y[3:]}
        if len(shared) <= 1 and not _is_mechanical_triangle(x, y):
            out.append((x, y))
    return out


class CointScanner(SpreadStrategy):
    def __init__(self, universe, scan_window=250, half_life_range=(5, 25),
                 adf_max=0.05, top_n=5, entry_z=2.0, target_z=0.5, stop_z=3.5):
        self.universe = list(universe)
        self.scan_window = int(scan_window)
        self.hl_lo, self.hl_hi = float(half_life_range[0]), float(half_life_range[1])
        self.adf_max = float(adf_max)
        self.top_n = int(top_n)
        self.entry_z = float(entry_z)
        self.target_z = float(target_z)
        self.stop_z = float(stop_z)
        self.candidates = _candidate_pairs(self.universe)
        # fx_round_trip_pips already returns the ROUND-TRIP cost (it doubles the
        # per-side spread internally), so this is the round-trip cost in price
        # terms. The tradeable gate below multiplies by 2 to require a 1.5-sigma
        # move to clear TWICE round-trip; do not double it a second time here.
        self._cost = fx_round_trip_pips("major") * 0.0001

    @staticmethod
    def _is_scan(d, prev_d) -> bool:
        if prev_d is None:
            return True
        return (d.year, d.month) != (prev_d.year, prev_d.month)

    def _window(self, series, i):
        lo = max(0, i - self.scan_window + 1)
        w = series[lo:i + 1]
        return w if np.all(np.isfinite(w)) else None

    def _adf_pvalue(self, la, lb, i):
        a, b = self._window(la, i), self._window(lb, i)
        if a is None or b is None:
            return None
        try:
            return test_pair(pd.Series(a), pd.Series(b))["adf_pvalue"]
        except Exception:
            return None

    def _resid_z(self, la, lb, hedge, i):
        a, b = self._window(la, i), self._window(lb, i)
        if a is None or b is None:
            return None
        spread = a - hedge * b
        sd = spread.std()
        if not np.isfinite(sd) or sd <= 0:
            return None
        z = (spread[-1] - spread.mean()) / sd
        return float(z) if np.isfinite(z) else None

    def _strength(self, z: float) -> float:
        scaled = -z * (10.0 / self.entry_z)
        return float(np.clip(scaled, -_STRENGTH_CAP, _STRENGTH_CAP))

    def _scan(self, close_panel, log_panel, i):
        scored = []
        for a, b in self.candidates:
            la, lb = self._window(log_panel[a], i), self._window(log_panel[b], i)
            if la is None or lb is None or len(la) < _MIN_SCAN_DAYS:
                continue
            try:
                res = test_pair(pd.Series(la), pd.Series(lb))
            except Exception:
                continue
            if res["adf_pvalue"] >= self.adf_max:
                continue
            if not (self.hl_lo <= res["half_life"] <= self.hl_hi):
                continue
            sig = self._spread_sigma(close_panel, a, b, res["hedge_ratio"], i)
            if sig is None:
                continue
            if not (1.5 * sig > 2.0 * self._cost):
                continue
            scored.append((sig / self._cost, (a, b), res))
        scored.sort(key=lambda t: t[0], reverse=True)
        return {
            pair: {"hedge": res["hedge_ratio"], "half_life": res["half_life"],
                   "adf_base": res["adf_pvalue"]}
            for _, pair, res in scored[:self.top_n]
        }

    def _exit_reason(self, st, z, log_panel, pair, i):
        # Structural degradation is only evaluated while a position is open.
        if abs(z) < self.target_z or abs(z) > self.stop_z:
            return "target_stop"
        if (i - st["entry_idx"]) >= 2.0 * st["half_life"]:
            return "time"
        a, b = pair
        cur = self._adf_pvalue(log_panel[a], log_panel[b], i)
        if cur is not None and (cur - st["adf_base"]) > _STRUCTURAL_ADF_DEGRADE:
            st["degrade_streak"] += 1
        else:
            st["degrade_streak"] = 0
        if st["degrade_streak"] >= _STRUCTURAL_SUSTAIN_DAYS:
            return "structural"
        return None

    def spread_book(self, close_panel):
        close_panel = close_panel.sort_index()
        dates = list(close_panel.index)
        log_panel = {c: np.log(close_panel[c].astype(float).values) for c in self.universe}

        book, sigma = {}, {}
        watch = {}  # pair -> {hedge, half_life, adf_base, dir, entry_idx, degrade_streak}
        prev_d = None

        for i, d in enumerate(dates):
            if self._is_scan(d, prev_d):
                # Rebuild the watchlist from this scan's fresh top-N qualifying
                # pairs with REFRESHED meta (hedge/half_life/adf_base recomputed
                # now), so a re-qualifying pair enters on a current hedge and
                # stale flat pairs age out (top-N enforced in steady state). Any
                # pair with an OPEN position is retained regardless of the fresh
                # ranking, keeping its position state and entry-time hedge so its
                # structural exit stays manageable -- the hedge is held fixed for
                # the life of an open trade (no mid-trade re-hedge).
                fresh = self._scan(close_panel, log_panel, i)
                new_watch = {pair: st for pair, st in watch.items()
                             if st["dir"] != 0}
                for pair, meta in fresh.items():
                    if pair not in new_watch:
                        new_watch[pair] = {**meta, "dir": 0, "entry_idx": None,
                                           "degrade_streak": 0}
                watch = new_watch
            prev_d = d

            day_spreads, day_sigma = [], {}
            for pair, st in list(watch.items()):
                a, b = pair
                z = self._resid_z(log_panel[a], log_panel[b], st["hedge"], i)
                if z is None:
                    # Emitting nothing for this pair makes the simulator flatten
                    # it and charge a round trip; keep the state machine in sync
                    # or it phantom-re-enters later (silent-skip defect).
                    st["dir"], st["entry_idx"], st["degrade_streak"] = 0, None, 0
                    continue

                if st["dir"] != 0:
                    reason = self._exit_reason(st, z, log_panel, pair, i)
                    if reason == "structural":
                        del watch[pair]
                        continue
                    if reason is not None:
                        st["dir"], st["entry_idx"], st["degrade_streak"] = 0, None, 0
                        continue
                elif abs(z) > self.entry_z:
                    st["dir"], st["entry_idx"] = (-1 if z > 0 else 1), i
                else:
                    continue

                sig = self._spread_sigma(close_panel, a, b, st["hedge"], i)
                if sig is None:
                    st["dir"], st["entry_idx"], st["degrade_streak"] = 0, None, 0
                    continue
                day_spreads.append(Spread(a, b, st["hedge"], self._strength(z)))
                day_sigma[(a, b)] = sig

            if day_spreads:
                book[d] = day_spreads
                sigma[d] = day_sigma

        return book, sigma
