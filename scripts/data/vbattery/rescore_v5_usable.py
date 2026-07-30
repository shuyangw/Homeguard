"""Re-score the V5 gate on USABLE-VALUE rate (Phase-1 Gap 3).

REGISTERED GATE -- restated verbatim, threshold unchanged
--------------------------------------------------------
"Null rate for `implied_vol`, `delta`, `theta`, `vega` per root x year. Sanity:
`IV in (0.01, 5.0)`, `|delta| <= 1`, delta sign correct per `right`. **>=90%**
non-null per root-year = PASS. Below => that root-year is flagged **untrusted**,
routed to later recomputation. Do **not** recompute now."

What changes here is the METRIC the 90% threshold is applied to, per CORRECTION
ADDENDUM C1 of `20260727_options_vbattery_report.md`: the shipped gate bound on
the *finite/non-null* rate alone and never applied the registered sanity bounds.
The corrected metric is the USABLE-VALUE rate --

    usable_rate = min(
        rate(implied_vol finite AND 0.01 < iv < 5.0),
        rate(delta finite AND |delta| <= 1),
        rate(delta finite AND sign correct per `right`),
        rate(theta finite),
        rate(vega finite),
    )

DATA SOURCE AND THE NaN-vs-NULL TRAP
------------------------------------
This does NOT rescan 24bn rows. It reuses two existing measurements:

1. `output/vbattery/sweep/v5_gate_by_root_year.csv` -- the shipped V5 sweep.
   VERIFIED AT CODE LEVEL (`sweep_v1_v2_v3_v5_v9_v11.py`, the `---- V5 ----`
   block): every V5 accumulator is built with `np.isfinite()` on the VALUES.
   The columns named `nonnull_*` are therefore FINITE-VALUE rates, not
   `null_count` rates -- they are already NaN-aware and the ALL_NAN root-years
   correctly score 0.0. See the divergence note in the Phase-1 readiness doc.
2. `docs/strategies/research/options-slate/20260728_options_greek_coverage_census.csv`
   -- the per-partition OK / ALL_NAN / NO_COLUMN / PARTIAL_NAN classification,
   used as an INDEPENDENT cross-check that (1) is consistent per root-year.

Writes `output/vbattery/sweep/v5_gate_usable_by_root_year.csv`.

DATA-PROPERTY MEASUREMENT ONLY. No positions, no P&L.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

os.environ.setdefault("POLARS_MAX_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

#: The registered threshold. NOT tunable -- restated, never changed.
V5_THRESHOLD = 0.90

V5_SWEEP_CSV = PROJECT_ROOT / "output" / "vbattery" / "sweep" / "v5_gate_by_root_year.csv"
CENSUS_CSV = (
    PROJECT_ROOT / "docs" / "strategies" / "research" / "options-slate"
    / "20260728_options_greek_coverage_census.csv"
)
OUT_CSV = (
    PROJECT_ROOT / "output" / "vbattery" / "sweep" / "v5_gate_usable_by_root_year.csv"
)

#: Each is a rate of (finite value AND its registered sanity bound, if any).
_USABLE_COMPONENTS = [
    "plausible_iv_frac",
    "abs_delta_le1_frac",
    "delta_sign_ok_frac",
    "nonnull_theta",
    "nonnull_vega",
]


def rescore_v5(sweep: pd.DataFrame) -> pd.DataFrame:
    """Apply the verbatim 90% threshold to the corrected usable-value rate."""
    out = sweep.copy()
    out["usable_rate"] = out[_USABLE_COMPONENTS].min(axis=1)
    out["usable_binding_column"] = out[_USABLE_COMPONENTS].idxmin(axis=1)
    out["gate_usable"] = [
        "PASS" if r >= V5_THRESHOLD else "UNTRUSTED" for r in out["usable_rate"]
    ]
    out["changed"] = out["gate_usable"] != out["gate"]
    return out


def census_usability_by_root_year(census: pd.DataFrame) -> pd.DataFrame:
    """Collapse the per-partition greek census to a per-root-year summary."""
    g = census.groupby(["root", "year"])["greek_status"]
    out = pd.DataFrame({
        "n_partitions": g.size(),
        "n_ok": g.apply(lambda s: int((s == "OK").sum())),
        "n_all_nan": g.apply(lambda s: int((s == "ALL_NAN").sum())),
        "n_no_column": g.apply(lambda s: int((s == "NO_COLUMN").sum())),
        "n_partial_nan": g.apply(lambda s: int((s == "PARTIAL_NAN").sum())),
    }).reset_index()
    out["census_ok_frac"] = out["n_ok"] / out["n_partitions"]
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Re-score V5 on usable-value rate.")
    ap.add_argument("--sweep", default=str(V5_SWEEP_CSV))
    ap.add_argument("--census", default=str(CENSUS_CSV))
    ap.add_argument("--out", default=str(OUT_CSV))
    args = ap.parse_args(argv)

    sweep = pd.read_csv(args.sweep)
    census = census_usability_by_root_year(pd.read_csv(args.census))
    scored = rescore_v5(sweep).merge(census, on=["root", "year"], how="left")

    # Independent cross-check: a root-year with zero OK partitions must not pass.
    contradiction = scored[
        (scored["census_ok_frac"] == 0.0) & (scored["gate_usable"] == "PASS")
    ]
    if len(contradiction):
        logger.error(
            f"[-] {len(contradiction)} root-years pass the usable gate but have "
            f"ZERO OK partitions in the census: "
            f"{contradiction[['root', 'year']].to_dict('records')}"
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.out, index=False)

    n_pass = int((scored["gate_usable"] == "PASS").sum())
    n_untrusted = int((scored["gate_usable"] == "UNTRUSTED").sum())
    n_changed = int(scored["changed"].sum())
    logger.info(
        f"[+] V5 re-scored on usable-value rate at the registered "
        f"{V5_THRESHOLD:.0%} threshold: {n_pass} PASS / {n_untrusted} UNTRUSTED "
        f"of {len(scored)} root-years; {n_changed} changed vs the shipped gate"
    )
    if n_changed:
        for _, r in scored[scored["changed"]].iterrows():
            logger.warning(
                f"[!] {r['root']} {int(r['year'])}: {r['gate']} -> "
                f"{r['gate_usable']} (usable {r['usable_rate']:.4f}, "
                f"binding {r['usable_binding_column']})"
            )
    logger.info(f"[+] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
