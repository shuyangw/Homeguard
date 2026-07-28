"""V8 -- Root continuity: FB->META splice, SPX/SPXW composition, VIX expiry.

Registered gate: "FB (2017-2021) -> META splice; SPX vs SPXW composition and
AM/PM settlement; VIX expiry convention. Gate: Report + registered
splice/filter rules per root."

MEASUREMENT ONLY. Proposes filter/splice rules; applies none.
"""
from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.settings import get_local_storage_dir
from src.utils.logger import get_logger
from src.utils.run_status import RunStatus

logger = get_logger(__name__)

OUT_DIR = Path("output/vbattery/v08")
BASE = Path(get_local_storage_dir()) / "options" / "options_combined"


def _months(root: str):
    rp = BASE / f"root={root}"
    out = []
    for yp in sorted(rp.glob("year=*")):
        for mp in sorted(yp.glob("month=*")):
            f = mp / "data.parquet"
            if f.exists():
                out.append((int(yp.name.split("=")[1]), int(mp.name.split("=")[1]), f))
    return out


def _scan(path: Path, cols, batch_size=500_000):
    pf = pq.ParquetFile(path)
    avail = [c for c in cols if c in pf.schema_arrow.names]
    for batch in pf.iter_batches(batch_size=batch_size, columns=avail):
        yield batch.to_pandas()


def _expiry_profile(root: str, months, cap_months=None):
    """Per (expiration) -- last session traded, first/last bar time on expiry day."""
    recs = {}
    sel = months if cap_months is None else months[-cap_months:]
    for (y, m, path) in sel:
        for df in _scan(path, ["timestamp", "expiration", "volume"]):
            ts = df["timestamp"].astype(str)
            df["session"] = ts.str.slice(0, 10)
            df["hhmm"] = ts.str.slice(11, 16)
            df["expiration"] = df["expiration"].astype(str).str.slice(0, 10)
            g = df.groupby("expiration").agg(
                last_session=("session", "max"),
                n_rows=("session", "size"))
            for exp, r in g.iterrows():
                cur = recs.setdefault(exp, {"last_session": "", "n_rows": 0,
                                            "exp_day_first_bar": "", "exp_day_last_bar": "",
                                            "exp_day_rows": 0, "exp_day_volume": 0.0})
                cur["last_session"] = max(cur["last_session"], r["last_session"])
                cur["n_rows"] += int(r["n_rows"])
            # bars ON the expiration date itself
            on_exp = df[df["session"] == df["expiration"]]
            if len(on_exp):
                g2 = on_exp.groupby("expiration").agg(
                    fb=("hhmm", "min"), lb=("hhmm", "max"),
                    n=("hhmm", "size"), vol=("volume", "sum"))
                for exp, r in g2.iterrows():
                    cur = recs[exp]
                    cur["exp_day_first_bar"] = (min(cur["exp_day_first_bar"], r["fb"])
                                                if cur["exp_day_first_bar"] else r["fb"])
                    cur["exp_day_last_bar"] = max(cur["exp_day_last_bar"], r["lb"])
                    cur["exp_day_rows"] += int(r["n"])
                    cur["exp_day_volume"] += float(r["vol"]) if pd.notna(r["vol"]) else 0.0
    out = pd.DataFrame.from_dict(recs, orient="index").reset_index().rename(columns={"index": "expiration"})
    out["root"] = root
    ed = pd.to_datetime(out["expiration"], errors="coerce")
    out["exp_dow"] = ed.dt.day_name()
    out["exp_dom"] = ed.dt.day
    out["is_third_friday"] = (out["exp_dow"] == "Friday") & out["exp_dom"].between(15, 21)
    return out.sort_values("expiration")


def part_fb_meta():
    res = {}
    rows = []
    for root in ("FB", "META"):
        months = _months(root)
        first_sess, last_sess = None, None
        for (y, m, path) in (months[0], months[-1]):
            sess = set()
            for df in _scan(path, ["timestamp"]):
                sess |= set(df["timestamp"].astype(str).str.slice(0, 10).unique())
            if (y, m, path) == months[0]:
                first_sess = min(sess)
            last_sess = max(sess)
        res[root] = {
            "n_month_partitions": len(months),
            "first_month": f"{months[0][0]}-{months[0][1]:02d}",
            "last_month": f"{months[-1][0]}-{months[-1][1]:02d}",
            "first_session": first_sess,
            "last_session": last_sess,
            "months": [f"{y}-{m:02d}" for (y, m, _) in months],
        }
        logger.info(f"[*] {root}: {res[root]['first_session']} -> {res[root]['last_session']} "
                    f"({len(months)} partitions)")

    fb_m, meta_m = set(res["FB"]["months"]), set(res["META"]["months"])
    overlap = sorted(fb_m & meta_m)
    res["overlap_months"] = overlap

    # calendar gaps within each root
    for root in ("FB", "META"):
        ms = [tuple(int(x) for x in s.split("-")) for s in res[root]["months"]]
        exp, cur = [], ms[0]
        while cur <= ms[-1]:
            exp.append(cur)
            cur = (cur[0] + 1, 1) if cur[1] == 12 else (cur[0], cur[1] + 1)
        res[root]["missing_months"] = [f"{y}-{m:02d}" for (y, m) in exp if (y, m) not in set(ms)]

    # compare the OVERLAP months root-by-root: duplicate content or different?
    probe = overlap if overlap else []
    for mo in probe:
        y, m = int(mo[:4]), int(mo[5:])
        for root in ("FB", "META"):
            path = BASE / f"root={root}" / f"year={y}" / f"month={m:02d}" / "data.parquet"
            if not path.exists():
                continue
            agg = {"n_rows": 0, "k_min": np.inf, "k_max": -np.inf, "upx": [], "sess": set(), "exp": set()}
            for df in _scan(path, ["timestamp", "expiration", "strike", "underlying_px"]):
                agg["n_rows"] += len(df)
                agg["k_min"] = min(agg["k_min"], float(df["strike"].min()))
                agg["k_max"] = max(agg["k_max"], float(df["strike"].max()))
                agg["sess"] |= set(df["timestamp"].astype(str).str.slice(0, 10).unique())
                agg["exp"] |= set(df["expiration"].astype(str).str.slice(0, 10).unique())
                if "underlying_px" in df:
                    s = df["underlying_px"].dropna()
                    if len(s):
                        agg["upx"].append(float(s.median()))
            rows.append({"root": root, "month": mo, "n_rows": agg["n_rows"],
                         "n_sessions": len(agg["sess"]), "n_expirations": len(agg["exp"]),
                         "strike_min": agg["k_min"], "strike_max": agg["k_max"],
                         "upx_median": float(np.median(agg["upx"])) if agg["upx"] else float("nan")})
            logger.info(f"[*] overlap {root} {mo}: {rows[-1]}")
    res["overlap_month_comparison"] = rows
    return res


def part_spx():
    months = _months("SPX")
    prof = _expiry_profile("SPX", months, cap_months=12)
    prof.to_csv(OUT_DIR / "v08_spx_expiry_profile.csv", index=False)
    # AM-settled tell: little/no trading ON the expiry date (stops at the open)
    prof["has_exp_day_bars"] = prof["exp_day_rows"] > 0
    prof["exp_day_last_bar_hhmm"] = prof["exp_day_last_bar"]
    am_like = prof["exp_day_rows"].eq(0) | prof["exp_day_last_bar"].le("10:30")
    summary = {
        "n_expirations": int(len(prof)),
        "dow_distribution": prof["exp_dow"].value_counts().to_dict(),
        "n_third_friday": int(prof["is_third_friday"].sum()),
        "n_non_third_friday": int((~prof["is_third_friday"]).sum()),
        "n_with_expiry_day_bars": int(prof["has_exp_day_bars"].sum()),
        "n_without_expiry_day_bars": int((~prof["has_exp_day_bars"]).sum()),
        "n_AM_like": int(am_like.sum()),
        "n_PM_like": int((~am_like).sum()),
        "frac_AM_like": float(am_like.mean()),
        "third_friday_AM_like": int((am_like & prof["is_third_friday"]).sum()),
        "third_friday_PM_like": int((~am_like & prof["is_third_friday"]).sum()),
        "non_third_friday_AM_like": int((am_like & ~prof["is_third_friday"]).sum()),
        "non_third_friday_PM_like": int((~am_like & ~prof["is_third_friday"]).sum()),
        "months_scanned": [f"{y}-{m:02d}" for (y, m, _) in months[-12:]],
        "n_month_partitions_total": len(months),
    }
    return summary


def part_vix():
    months = _months("VIX")
    prof = _expiry_profile("VIX", months, cap_months=24)
    prof.to_csv(OUT_DIR / "v08_vix_expiry_profile.csv", index=False)
    return {
        "n_expirations": int(len(prof)),
        "dow_distribution": prof["exp_dow"].value_counts().to_dict(),
        "n_month_partitions_total": len(months),
        "months_scanned": [f"{y}-{m:02d}" for (y, m, _) in months[-24:]],
        "non_wednesday_expirations": prof.loc[prof["exp_dow"] != "Wednesday",
                                              ["expiration", "exp_dow", "n_rows"]].head(40).to_dict("records"),
    }


PROPOSED_RULES = [
    {"root": "FB", "rule_id": "FB-01", "type": "SPLICE",
     "rule": "root=FB is Meta Platforms 2017-01-03 .. 2021-10-29 ONLY. Use as the pre-rename leg.",
     "status": "PROPOSED -- not applied"},
    {"root": "META", "rule_id": "META-01", "type": "QUARANTINE",
     "rule": "root=META months 2021-07 .. 2022-01 are a DIFFERENT issuer (underlying_px ~USD 13-16, "
             "strikes 3-32) -- Meta Materials, NOT Meta Platforms. EXCLUDE from any Meta Platforms series.",
     "status": "PROPOSED -- not applied"},
    {"root": "META", "rule_id": "META-02", "type": "SPLICE",
     "rule": "Meta Platforms under root=META begins 2022-06-09. Splice FB[..2021-10-29] + META[2022-06-09..]. "
             "NO strike/underlying adjustment needed at the splice (rename, not a split), but a "
             "2021-10-30 .. 2022-06-08 HOLE (~7.4 months) remains and must be declared missing, not filled.",
     "status": "PROPOSED -- not applied"},
    {"root": "SPX", "rule_id": "SPX-01", "type": "FILTER/SCOPE",
     "rule": "root=SPX contains ONLY standard AM-settled third-Friday SPX (31 distinct expirations over the "
             "12 months scanned; 0 expiry-day bars; last trade always the session BEFORE expiration). "
             "NO SPXW / weekly / 0DTE contracts are present. Any weekly/0DTE SPX hypothesis is NOT testable "
             "on this store. AM/PM is NOT a filter to apply -- the population is 100% AM.",
     "status": "PROPOSED -- not applied"},
    {"root": "VIX", "rule_id": "VIX-01", "type": "FILTER",
     "rule": "VIX expiries are Wednesday (29/32 scanned) with 3 Tuesday exceptions "
             "(2024-06-18, 2025-03-18, 2026-05-19) consistent with the 30-days-before-SPX-expiry rule and "
             "holiday shifts. Do NOT drop the Tuesday expiries as anomalies -- they are convention.",
     "status": "PROPOSED -- not applied"},
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(PROPOSED_RULES).to_csv(OUT_DIR / "v08_proposed_rules.csv", index=False)
    with RunStatus("vbattery_v08_root_continuity") as st:
        fb = part_fb_meta()
        st.heartbeat(note="fb_meta done")
        spx = part_spx()
        st.heartbeat(note="spx done")
        vix = part_vix()
    out = {"fb_meta": fb, "spx": spx, "vix": vix}
    (OUT_DIR / "v08_summary.json").write_text(json.dumps(out, indent=2, default=str), encoding="ascii")
    logger.info("[+] V8 done")


if __name__ == "__main__":
    main()
