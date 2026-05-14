"""Phase 0 Probe 1: Inventory the Massive S3 flat-files bucket.

Lists prefixes at known and candidate paths to confirm what data is
accessible under the current MASSIVE_S3_* credentials. Output is
appended to docs/planning/20260514_fx_phase_0_results.md.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger
logger = get_logger(__name__)


def _env(name: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    for line in (PROJECT_ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith(name + "="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(f"{name} not found")


def make_client():
    return boto3.client(
        "s3",
        endpoint_url=_env("MASSIVE_S3_ENDPOINT"),
        aws_access_key_id=_env("MASSIVE_S3_ACCESS_KEY"),
        aws_secret_access_key=_env("MASSIVE_S3_SECRET_KEY"),
        config=Config(signature_version="s3v4"),
    )


PROBES = [
    "global_forex/minute_aggs_v1/",
    "global_forex/quotes_v1/",
    "global_forex/trades_v1/",
    "global_forex/day_aggs_v1/",
    "us_indices/",
    "us_treasuries/",
    "us_stocks_sip/",
    "us_options_opra/",
    "us_futures_cme/",
    "us_futures_cbot/",
    "us_futures_comex/",
    "us_futures_nymex/",
]


def probe_prefix(s3, bucket: str, prefix: str) -> dict:
    try:
        resp = s3.list_objects_v2(
            Bucket=bucket, Prefix=prefix, Delimiter="/", MaxKeys=20,
        )
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "?")
        return {"prefix": prefix, "accessible": False, "error": code,
                "children": [], "sample_size_bytes": None}
    cps = [cp["Prefix"] for cp in (resp.get("CommonPrefixes") or [])]
    contents = resp.get("Contents") or []
    sample_size = contents[0]["Size"] if contents else None
    return {
        "prefix": prefix, "accessible": True, "error": None,
        "children": cps, "sample_size_bytes": sample_size,
    }


def main() -> int:
    bucket = _env("MASSIVE_S3_BUCKET")
    s3 = make_client()
    results = [probe_prefix(s3, bucket, p) for p in PROBES]

    print(f"{'prefix':<40} {'accessible':<12} {'children':<5} {'sample bytes':<14} {'error'}")
    print("-" * 90)
    for r in results:
        n_child = len(r["children"])
        size = r["sample_size_bytes"] or 0
        err = r["error"] or ""
        print(f"{r['prefix']:<40} {str(r['accessible']):<12} {n_child:<5} {size:<14} {err}")
        for c in r["children"][:5]:
            print(f"    {c}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
