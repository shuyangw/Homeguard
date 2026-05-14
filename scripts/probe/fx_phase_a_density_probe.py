"""Phase A density probe: which Layer 1/2/3 candidate pairs have real density on Polygon S3?

Probes each candidate at:
  - 2014-09-08 (historical week, Mon)
  - 2026-04-07 (recent week, Tue)
Threshold: >= 50% of 1440 minute bars in BOTH windows to qualify.

Output: docs/planning/20260514_fx_phase_a_results.json
"""
from __future__ import annotations

import gzip
import io
import json
import sys
from datetime import date
from pathlib import Path

import boto3
from botocore.client import Config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.acquisition.plugins.massive_fx_flat import make_s3_client, key_for

CANDIDATES = [
    # Layer 1: G10 cross matrix
    ("NZDCAD", "C:NZD-CAD"), ("NZDNOK", "C:NZD-NOK"), ("NZDSEK", "C:NZD-SEK"),
    ("CADNOK", "C:CAD-NOK"), ("CADSEK", "C:CAD-SEK"),
    ("CHFNOK", "C:CHF-NOK"), ("CHFSEK", "C:CHF-SEK"),
    ("NOKJPY", "C:NOK-JPY"), ("SEKJPY", "C:SEK-JPY"),
    ("AUDNOK", "C:AUD-NOK"), ("AUDSEK", "C:AUD-SEK"),
    # Layer 2: Metals crosses
    ("XAUEUR", "C:XAU-EUR"), ("XAUGBP", "C:XAU-GBP"), ("XAUJPY", "C:XAU-JPY"),
    ("XAUAUD", "C:XAU-AUD"), ("XAUCHF", "C:XAU-CHF"),
    ("XAGEUR", "C:XAG-EUR"), ("XAGGBP", "C:XAG-GBP"), ("XAGJPY", "C:XAG-JPY"),
    # Layer 3: EUR/AUD-EM crosses
    ("EURMXN", "C:EUR-MXN"), ("EURZAR", "C:EUR-ZAR"),
    ("EURCNH", "C:EUR-CNH"), ("EURPLN", "C:EUR-PLN"),
    ("GBPMXN", "C:GBP-MXN"), ("GBPZAR", "C:GBP-ZAR"), ("GBPCNH", "C:GBP-CNH"),
    ("AUDCNH", "C:AUD-CNH"), ("AUDMXN", "C:AUD-MXN"),
]

PROBE_DAYS = [date(2014, 9, 8), date(2026, 4, 7)]
THRESHOLD = 0.50  # 50% density required in BOTH windows


def count_pair_rows_in_day(s3, day: date, target_tickers: set[str]) -> dict:
    """Download one daily file; return rows-per-ticker for our targets."""
    bucket = "flatfiles"
    key = key_for(day)
    buf = io.BytesIO()
    s3.download_fileobj(bucket, key, buf)
    text = gzip.decompress(buf.getvalue()).decode("utf-8")
    counts: dict[str, int] = {}
    for line in text.splitlines()[1:]:
        comma = line.find(",")
        if comma < 0:
            continue
        t = line[:comma]
        if t in target_tickers:
            counts[t] = counts.get(t, 0) + 1
    return counts


def main() -> int:
    s3 = make_s3_client()
    target_to_hg = {t: hg for hg, t in CANDIDATES}
    target_set = set(target_to_hg)

    results = []
    per_day_counts = {}
    for d in PROBE_DAYS:
        per_day_counts[d.isoformat()] = count_pair_rows_in_day(s3, d, target_set)

    for hg, t in CANDIDATES:
        densities = []
        for d in PROBE_DAYS:
            n = per_day_counts[d.isoformat()].get(t, 0)
            densities.append(n / 1440.0)
        verdict = "include" if min(densities) >= THRESHOLD else "skip"
        results.append({
            "hg_symbol": hg,
            "massive_ticker": t,
            "historical_density": round(densities[0], 3),
            "recent_density": round(densities[1], 3),
            "verdict": verdict,
        })

    out = Path("docs/planning/20260514_fx_phase_a_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "generated_at": "2026-05-14",
        "probe_days": [d.isoformat() for d in PROBE_DAYS],
        "threshold": THRESHOLD,
        "results": results,
    }, indent=2))
    print(f"wrote {out}")

    print(f"\n{'pair':<10} {'ticker':<14} {'hist':<8} {'recent':<8} verdict")
    for r in results:
        print(f"{r['hg_symbol']:<10} {r['massive_ticker']:<14} "
              f"{r['historical_density']:<8} {r['recent_density']:<8} {r['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
