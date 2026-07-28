"""V13 -- point-in-time universe coverage measurement (MEASUREMENT ONLY).

Registered gate:
  "V13 PIT universe coverage. For every point-in-time-defined universe
   (rank/weight-based membership), fraction of sessions where all required
   members exist on disk; report the earliest date from which coverage is
   complete. Gate: <95% of sessions fully covered => that universe is NOT
   usable as registered; report the first-full-coverage date and the implied
   usable window."

No backtest, no P&L. Reads parquet FOOTER statistics only -- never row data.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.settings import get_local_storage_dir  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

OUT_DIR = REPO_ROOT / "output" / "vbattery" / "v13"

INDEX_ETF_ROOTS = {"SPY", "QQQ", "IWM", "DIA", "EEM", "FXI", "GLD", "SLV",
                   "SMH", "TLT", "XLE", "XLF", "XLI", "XLK", "XLV", "SPX", "VIX", "IBIT"}


def scan_partitions() -> pd.DataFrame:
    base = Path(get_local_storage_dir()) / "options" / "options_combined"
    rows = []
    for f in sorted(base.glob("root=*/year=*/month=*/data.parquet")):
        root = f.parts[-4].split("=", 1)[1]
        md = pq.ParquetFile(f).metadata
        ts_idx = md.schema.names.index("timestamp")
        lo, hi = None, None
        for rg in range(md.num_row_groups):
            st = md.row_group(rg).column(ts_idx).statistics
            if st is None or not st.has_min_max:
                continue
            lo = st.min if lo is None else min(lo, st.min)
            hi = st.max if hi is None else max(hi, st.max)
        if lo is None:
            logger.warning(f"[!] no timestamp statistics: {f}")
            continue
        rows.append({
            "root": root,
            "partition": f"{f.parts[-3]}/{f.parts[-2]}",
            "first_date": pd.Timestamp(lo[:10]),
            "last_date": pd.Timestamp(hi[:10]),
            "num_rows": md.num_rows,
        })
    return pd.DataFrame(rows)


def build_root_session_matrix(parts: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    roots = sorted(parts["root"].unique())
    mat = pd.DataFrame(False, index=sessions, columns=roots)
    for root, grp in parts.groupby("root"):
        covered = pd.Series(False, index=sessions)
        for _, r in grp.iterrows():
            covered |= (sessions >= r["first_date"]) & (sessions <= r["last_date"])
        mat[root] = covered.values
    return mat


def coverage_for_fixed_members(mat: pd.DataFrame, members: list[str]) -> pd.Series:
    missing = [m for m in members if m not in mat.columns]
    if missing:
        return pd.Series(False, index=mat.index)
    return mat[members].all(axis=1)


def first_full_coverage_date(covered: pd.Series):
    """Earliest date from which coverage is CONTINUOUSLY complete through the end."""
    if not covered.any():
        return None
    rev_all = covered[::-1].cummin()[::-1]
    if not rev_all.any():
        return None
    return rev_all.idxmax()


def gap_ranges(sessions: pd.DatetimeIndex, flags: pd.Series) -> list[tuple]:
    bad = pd.Series(flags[~flags].index)
    if bad.empty:
        return []
    grp = (bad.diff().dt.days.fillna(9999) > 5).cumsum()
    return [(v.iloc[0].date(), v.iloc[-1].date(), len(v)) for _, v in bad.groupby(grp)]


def longest_covered_span(covered: pd.Series):
    if not covered.any():
        return None, 0
    grp = (~covered).cumsum()
    best = covered[covered].groupby(grp[covered]).apply(lambda s: (s.index[0], s.index[-1], len(s)))
    lo, hi, n = max(best.tolist(), key=lambda t: t[2])
    return f"{lo.date()} -> {hi.date()}", n


def summarize(name: str, definition_src: str, construction: str, pit_source: bool,
              members: list[str], covered: pd.Series, measurable: bool, note: str) -> dict:
    ffc = first_full_coverage_date(covered)
    frac = float(covered.mean())
    span, span_n = longest_covered_span(covered)
    return {
        "universe": name,
        "definition_source": definition_src,
        "construction_used": construction,
        "genuine_pit_source": "yes" if pit_source else "no",
        "measurable_as_registered": "yes" if measurable else "NO",
        "members_used": ",".join(members),
        "n_sessions": int(len(covered)),
        "n_sessions_covered": int(covered.sum()),
        "frac_sessions_covered": round(frac, 6),
        "first_full_coverage_date": None if ffc is None else str(ffc.date()),
        "usable_window": "none" if ffc is None else f"{ffc.date()} -> {covered.index[-1].date()}",
        "longest_continuous_covered_span": span,
        "longest_continuous_covered_sessions": span_n,
        "gate_95pct": "PASS" if frac >= 0.95 else "FAIL",
        "note": note,
    }


def best_subset_upper_bound(mat: pd.DataFrame, candidates: list[str], k: int):
    """Unconstrained upper bound: best k-subset of candidate roots by covered fraction."""
    if len(candidates) < k:
        return None, pd.Series(False, index=mat.index)
    best, best_cov = None, None
    for combo in itertools.combinations(candidates, k):
        cov = mat[list(combo)].all(axis=1)
        if best_cov is None or cov.sum() > best_cov.sum():
            best, best_cov = list(combo), cov
    return best, best_cov


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("[*] scanning parquet footers")
    parts = scan_partitions()
    logger.info(f"[+] {len(parts)} partitions across {parts['root'].nunique()} roots")

    root_ranges = (parts.groupby("root")
                        .agg(first_session=("first_date", "min"),
                             last_session=("last_date", "max"),
                             n_month_partitions=("partition", "count"),
                             total_rows=("num_rows", "sum"))
                        .reset_index())
    # missing-month detection
    gaps = []
    for _, r in root_ranges.iterrows():
        want = pd.period_range(r["first_session"].to_period("M"),
                               r["last_session"].to_period("M"), freq="M")
        have = set(parts.loc[parts["root"] == r["root"], "partition"])
        miss = [str(p) for p in want
                if f"year={p.year}/month={p.month:02d}" not in have]
        gaps.append(",".join(miss))
    root_ranges["missing_months_within_range"] = gaps

    start = parts["first_date"].min()
    end = parts["last_date"].max()
    sessions = pd.DatetimeIndex(
        mcal.get_calendar("XNYS").valid_days(start_date=start, end_date=end).tz_localize(None))
    logger.info(f"[+] session calendar {sessions[0].date()} -> {sessions[-1].date()} "
                f"({len(sessions)} sessions)")

    mat = build_root_session_matrix(parts, sessions)
    on_disk_singles = sorted(c for c in mat.columns if c not in INDEX_ETF_ROOTS)
    logger.info(f"[+] on-disk single names ({len(on_disk_singles)}): {on_disk_singles}")

    spec = "docs/strategies/research/options-slate/2026-07-25_options_slate_cc_handoff_spec_v2.md#1.4"
    rows, per_session = [], {}

    # --- U_INDEX: fixed membership, fully measurable ---
    u_index = ["SPY", "QQQ", "IWM"]
    cov = coverage_for_fixed_members(mat, u_index)
    per_session["U_INDEX"] = cov
    rows.append(summarize("U_INDEX", spec, "fixed literal membership {SPY,QQQ,IWM}",
                          True, u_index, cov, True,
                          "membership is fixed, not rank-based -> no PIT source needed"))

    # --- rank/weight-based universes: NOT MEASURABLE AS REGISTERED ---
    no_pit = ("no point-in-time membership/weight source exists in repo; "
              "config/universes/*-2025.csv are TODAY snapshots (survivorship-biased) "
              "and src/strategies/universe/equity_universe.py pulls today's Wikipedia list")

    # biased today-snapshot proxies (explicitly labelled)
    sp500_today = pd.read_csv(REPO_ROOT / "config" / "universes" / "sp500-2025.csv")
    top_by_mcap_today = [s for s in sp500_today.sort_values("Ranking")["Symbol"].tolist()]
    top6_today = top_by_mcap_today[:6]
    cov = coverage_for_fixed_members(mat, top6_today)
    per_session["U_TOP6_SPY__today_snapshot_upper_bound"] = cov
    rows.append(summarize("U_TOP6_SPY", spec,
                          "NOT CONSTRUCTIBLE (needs SPY index weights at date t)",
                          False, [], pd.Series(False, index=sessions), False, no_pit))
    rows.append(summarize("U_TOP6_SPY__today_snapshot_upper_bound", spec,
                          "BIASED: top 6 of config/universes/sp500-2025.csv by today's Ranking",
                          False, top6_today, cov, False,
                          "explicitly survivorship-biased upper bound, NOT the registered measure"))

    best6, cov6 = best_subset_upper_bound(mat, on_disk_singles, 6)
    if best6:
        per_session["U_TOP6_SPY__best_6_subset_upper_bound"] = cov6
        rows.append(summarize("U_TOP6_SPY__best_6_subset_upper_bound", spec,
                              "UNCONSTRAINED BOUND: best-covered 6-subset of on-disk singles",
                              False, best6, cov6, False,
                              "ignores index weights entirely; a ceiling no PIT membership can beat"))

    for name, k in (("U_MEGA10", 10), ("U_MEGA20", 20), ("U_TIER1_100", 100)):
        rows.append(summarize(name, spec,
                              "NOT CONSTRUCTIBLE (needs cross-market PIT option-volume ranking; "
                              f"only {mat.shape[1]} roots on disk)",
                              False, [], pd.Series(False, index=sessions), False, no_pit))
        best, covk = best_subset_upper_bound(mat, on_disk_singles, k)
        if best is None:
            rows.append(summarize(f"{name}__best_{k}_subset_upper_bound", spec,
                                  f"UNCONSTRAINED BOUND: impossible -- only "
                                  f"{len(on_disk_singles)} single names on disk, {k} required",
                                  False, [], pd.Series(False, index=sessions), False,
                                  "upper bound is structurally 0% of sessions"))
        else:
            per_session[f"{name}__best_{k}_subset_upper_bound"] = covk
            rows.append(summarize(f"{name}__best_{k}_subset_upper_bound", spec,
                                  f"UNCONSTRAINED BOUND: best-covered {k}-subset of on-disk singles",
                                  False, best, covk, False,
                                  "ignores option-volume ranking; a ceiling no PIT ranking can beat"))

    summary = pd.DataFrame(rows)
    ps = pd.DataFrame(per_session, index=sessions)
    ps.index.name = "session"

    gap_rows = []
    for root in mat.columns:
        rng = root_ranges.set_index("root").loc[root]
        in_life = (sessions >= rng["first_session"]) & (sessions <= rng["last_session"])
        flags = pd.Series(mat[root].values, index=sessions)[in_life]
        for lo, hi, n in gap_ranges(sessions, flags):
            gap_rows.append({"root": root, "gap_start": str(lo), "gap_end": str(hi),
                             "n_sessions": n})
    gaps_df = pd.DataFrame(gap_rows)
    gaps_df.to_csv(OUT_DIR / "v13_root_session_gaps.csv", index=False)

    parts.to_parquet(OUT_DIR / "v13_partition_inventory.parquet", index=False)
    root_ranges.to_csv(OUT_DIR / "v13_root_coverage.csv", index=False)
    root_ranges.to_parquet(OUT_DIR / "v13_root_coverage.parquet", index=False)
    ps.to_parquet(OUT_DIR / "v13_per_session_coverage.parquet")
    summary.to_csv(OUT_DIR / "v13_universe_summary.csv", index=False)
    summary.to_parquet(OUT_DIR / "v13_universe_summary.parquet", index=False)

    for _, r in summary.iterrows():
        tag = "[+]" if r["gate_95pct"] == "PASS" else "[-]"
        logger.info(f"{tag} {r['universe']}: covered={r['frac_sessions_covered']:.4f} "
                    f"first_full={r['first_full_coverage_date']} gate={r['gate_95pct']}")
    logger.info(f"[+] wrote shards to {OUT_DIR}")


if __name__ == "__main__":
    main()
