"""V7 -- Corporate actions: adjusted vs as-reported strike grids.

Registered gate: "Strike-grid behavior across: AAPL 2020-08-31 (4:1),
TSLA 2020-08-31 (5:1) and 2022-08-25 (3:1), NVDA 2021-07-20 (4:1) and
2024-06-10 (10:1), AMZN 2022-06-06 (20:1), GOOGL 2022-07-18 (20:1).
Gate: Report adjusted vs as-reported. As-reported => adjustment layer required
before ANY single-name work."

MEASUREMENT ONLY. Reads a single root x year x month partition per event with
column projection and batched scanning. Does NOT build any adjustment layer.
"""
from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.settings import get_local_storage_dir
from src.utils.logger import get_logger
from src.utils.run_status import RunStatus

logger = get_logger(__name__)

OUT_DIR = Path("output/vbattery/v07")

# (root, split_date, factor_new_per_old)
EVENTS = [
    ("AAPL", "2020-08-31", 4),
    ("TSLA", "2020-08-31", 5),
    ("TSLA", "2022-08-25", 3),
    ("NVDA", "2021-07-20", 4),
    ("NVDA", "2024-06-10", 10),
    ("AMZN", "2022-06-06", 20),
    ("GOOGL", "2022-07-18", 20),
]

COLS = ["timestamp", "expiration", "strike", "right", "underlying_px"]


def _load_month(root: str, year: int, month: int) -> pd.DataFrame:
    base = Path(get_local_storage_dir()) / "options" / "options_combined"
    path = base / f"root={root}" / f"year={year}" / f"month={month:02d}" / "data.parquet"
    if not path.exists():
        raise FileNotFoundError(str(path))
    pf = pq.ParquetFile(path)
    avail = [c for c in COLS if c in pf.schema_arrow.names]
    chunks = []
    for batch in pf.iter_batches(batch_size=500_000, columns=avail):
        df = batch.to_pandas()
        df["session"] = df["timestamp"].astype(str).str.slice(0, 10)
        df["expiration"] = df["expiration"].astype(str)
        chunks.append(df)
    out = pd.concat(chunks, ignore_index=True)
    logger.info(f"[*] loaded {root} {year}-{month:02d}: {len(out)} rows, cols={avail}")
    return out


def _grid_stats(g: pd.DataFrame) -> dict:
    strikes = np.sort(g["strike"].dropna().unique())
    diffs = np.diff(strikes)
    diffs = diffs[diffs > 0]
    mode_inc = float(Counter(np.round(diffs, 4)).most_common(1)[0][0]) if len(diffs) else float("nan")
    # non-standard tell: strikes that are not an integer multiple of the modal increment
    if len(strikes) and mode_inc == mode_inc and mode_inc > 0:
        resid = np.abs(np.round(strikes / mode_inc) - strikes / mode_inc)
        n_offgrid = int((resid > 1e-6).sum())
    else:
        n_offgrid = -1
    upx = g["underlying_px"].dropna() if "underlying_px" in g else pd.Series(dtype=float)
    return {
        "n_rows": int(len(g)),
        "n_strikes": int(len(strikes)),
        "strike_min": float(strikes.min()) if len(strikes) else float("nan"),
        "strike_max": float(strikes.max()) if len(strikes) else float("nan"),
        "strike_median": float(np.median(strikes)) if len(strikes) else float("nan"),
        "strike_mode_increment": mode_inc,
        "n_offgrid_strikes": n_offgrid,
        "n_expirations": int(g["expiration"].nunique()),
        "upx_min": float(upx.min()) if len(upx) else float("nan"),
        "upx_max": float(upx.max()) if len(upx) else float("nan"),
        "upx_median": float(upx.median()) if len(upx) else float("nan"),
        "upx_n_null": int(g["underlying_px"].isna().sum()) if "underlying_px" in g else -1,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session_rows = []
    verdicts = []

    with RunStatus("vbattery_v07_corporate_actions", meta={"n_events": len(EVENTS)}) as st:
        for root, split_date, factor in EVENTS:
            y, m = int(split_date[:4]), int(split_date[5:7])
            try:
                df = _load_month(root, y, m)
            except Exception as exc:
                logger.error(f"[-] {root} {split_date}: {exc}")
                verdicts.append({"root": root, "split_date": split_date, "factor": factor,
                                 "verdict": "NO_DATA", "detail": repr(exc)})
                continue
            st.heartbeat(note=f"{root} {split_date}")

            sessions = sorted(df["session"].unique())
            pre = [s for s in sessions if s < split_date][-3:]
            post = [s for s in sessions if s >= split_date][:3]

            stats = {}
            for s in pre + post:
                g = df[df["session"] == s]
                st_ = _grid_stats(g)
                st_.update({"root": root, "split_date": split_date, "factor": factor,
                            "session": s, "side": "PRE" if s < split_date else "POST"})
                stats[s] = st_
                session_rows.append(st_)

            if not pre or not post:
                verdicts.append({"root": root, "split_date": split_date, "factor": factor,
                                 "verdict": "INSUFFICIENT_SESSIONS",
                                 "detail": f"pre={pre} post={post}"})
                continue

            last_pre, first_post = stats[pre[-1]], stats[post[0]]
            upx_ratio = last_pre["upx_median"] / first_post["upx_median"] if first_post["upx_median"] else float("nan")
            k_ratio = last_pre["strike_median"] / first_post["strike_median"] if first_post["strike_median"] else float("nan")

            def close(x, target, tol=0.25):
                return abs(x - target) <= tol * target

            if close(upx_ratio, factor) or close(k_ratio, factor):
                verdict = "AS_REPORTED"
            elif close(upx_ratio, 1.0, 0.15) and close(k_ratio, 1.0, 0.35):
                verdict = "ADJUSTED"
            else:
                verdict = "MIXED_OR_UNCLEAR"

            verdicts.append({
                "root": root, "split_date": split_date, "factor": factor,
                "verdict": verdict,
                "last_pre_session": pre[-1], "first_post_session": post[0],
                "upx_median_pre": last_pre["upx_median"], "upx_median_post": first_post["upx_median"],
                "upx_ratio_pre_over_post": upx_ratio,
                "strike_median_pre": last_pre["strike_median"], "strike_median_post": first_post["strike_median"],
                "strike_ratio_pre_over_post": k_ratio,
                "strike_max_pre": last_pre["strike_max"], "strike_max_post": first_post["strike_max"],
                "strike_inc_pre": last_pre["strike_mode_increment"], "strike_inc_post": first_post["strike_mode_increment"],
                "n_offgrid_pre": last_pre["n_offgrid_strikes"], "n_offgrid_post": first_post["n_offgrid_strikes"],
                "n_strikes_pre": last_pre["n_strikes"], "n_strikes_post": first_post["n_strikes"],
            })
            logger.info(f"[*] {root} {split_date} x{factor} -> {verdict} "
                        f"upx_ratio={upx_ratio:.3f} k_ratio={k_ratio:.3f}")
            del df

    pd.DataFrame(session_rows).to_csv(OUT_DIR / "v07_session_grids.csv", index=False)
    vdf = pd.DataFrame(verdicts)
    vdf.to_csv(OUT_DIR / "v07_event_verdicts.csv", index=False)
    (OUT_DIR / "v07_summary.json").write_text(
        json.dumps({"n_events": len(EVENTS),
                    "verdict_counts": vdf["verdict"].value_counts().to_dict()}, indent=2),
        encoding="ascii")
    logger.info(f"[+] V7 verdicts: {vdf['verdict'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
