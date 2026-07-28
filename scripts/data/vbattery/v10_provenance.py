"""V10 -- Provenance mix (ThetaData vs IBKR chains) in options_combined.

Registered gate: "DATA_INVENTORY.md lists ThetaData + IBKR chains; the download
scripts are ThetaData. Which rows, if any, are IBKR-sourced; are populations
distinguishable. Gate: If mixed and indistinguishable, V1/V3 statistics must be
computed conservatively over the whole."

MEASUREMENT ONLY. Inspects schemas (no provenance column?), the writer scripts,
and the download logs under <options>/_logs.
"""
from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.settings import get_local_storage_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

OUT_DIR = Path("output/vbattery/v10")
REPO = Path(".")

PROVENANCE_TOKENS = ["source", "provenance", "vendor", "feed", "origin", "venue", "exchange", "symbol"]
IBKR_PAT = re.compile(r"ibkr|ib_async|ib_insync|interactive\s*brokers|tws|gateway|port\s*=?\s*(4001|4002|7496|7497)", re.I)
THETA_PAT = re.compile(r"thetadata|theta[_ ]terminal|25503|localhost:25510", re.I)


def scan_schema_columns() -> dict:
    """Does ANY partition carry a provenance-bearing column?"""
    schemas = pd.read_parquet("output/vbattery/v04/v04_partition_schemas.parquet")
    allcols = set()
    for c in schemas["columns"]:
        allcols |= set(c.split(","))
    hits = {t: sorted(c for c in allcols if t in c.lower()) for t in PROVENANCE_TOKENS}
    return {"distinct_columns_across_store": sorted(allcols),
            "provenance_token_hits": {k: v for k, v in hits.items() if v}}


def sample_symbol_column_values() -> dict:
    """The 21-col variants carry `symbol` -- does it encode a vendor/source?"""
    schemas = pd.read_parquet("output/vbattery/v04/v04_partition_schemas.parquet")
    with_sym = schemas[schemas["has_symbol_col"]]
    out = []
    for _, r in with_sym.groupby("schema_hash").head(1).iterrows():
        pf = pq.ParquetFile(r["path"])
        batch = next(pf.iter_batches(batch_size=20_000, columns=["symbol"]))
        vals = batch.to_pandas()["symbol"].astype(str)
        out.append({"schema_hash": r["schema_hash"], "path": r["path"],
                    "n_distinct_in_sample": int(vals.nunique()),
                    "sample_values": vals.drop_duplicates().head(8).tolist()})
    return {"n_partitions_with_symbol": int(len(with_sym)), "samples": out}


def scan_writer_scripts() -> dict:
    files = sorted(Path("scripts/data").glob("*options*.py")) + \
            sorted(Path("src/data/options").glob("*.py"))
    rows = []
    for f in files:
        txt = f.read_text(encoding="utf-8", errors="replace")
        rows.append({
            "file": str(f.resolve()),
            "n_lines": txt.count("\n") + 1,
            "mentions_thetadata": bool(THETA_PAT.search(txt)),
            "mentions_ibkr": bool(IBKR_PAT.search(txt)),
            "writes_options_combined": "options_combined" in txt,
        })
    return {"writers": rows}


def scan_logs() -> dict:
    log_dir = Path(get_local_storage_dir()) / "options" / "_logs"
    rows = []
    if not log_dir.exists():
        return {"log_dir": str(log_dir), "exists": False, "logs": []}
    for f in sorted(log_dir.iterdir()):
        if not f.is_file():
            continue
        n_theta = n_ibkr = 0
        try:
            with f.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if THETA_PAT.search(line):
                        n_theta += 1
                    if IBKR_PAT.search(line):
                        n_ibkr += 1
        except Exception as exc:
            logger.error(f"[-] log read failed {f}: {exc}")
        rows.append({"log": f.name, "bytes": f.stat().st_size,
                     "n_thetadata_lines": n_theta, "n_ibkr_lines": n_ibkr})
        logger.info(f"[*] {f.name}: theta={n_theta} ibkr={n_ibkr}")
    return {"log_dir": str(log_dir), "exists": True, "logs": rows}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_columns": scan_schema_columns(),
        "symbol_column_samples": sample_symbol_column_values(),
        "writer_scripts": scan_writer_scripts(),
        "logs": scan_logs(),
    }
    logs = result["logs"]["logs"]
    result["verdict_inputs"] = {
        "any_provenance_column": bool(
            set(result["schema_columns"]["provenance_token_hits"]) - {"symbol"}),
        "n_logs": len(logs),
        "total_ibkr_log_lines": sum(r["n_ibkr_lines"] for r in logs),
        "total_thetadata_log_lines": sum(r["n_thetadata_lines"] for r in logs),
        "n_writers_mentioning_ibkr": sum(1 for r in result["writer_scripts"]["writers"] if r["mentions_ibkr"]),
        "n_writers_mentioning_thetadata": sum(1 for r in result["writer_scripts"]["writers"] if r["mentions_thetadata"]),
    }
    (OUT_DIR / "v10_provenance.json").write_text(json.dumps(result, indent=2, default=str), encoding="ascii")
    pd.DataFrame(logs).to_csv(OUT_DIR / "v10_log_scan.csv", index=False)
    pd.DataFrame(result["writer_scripts"]["writers"]).to_csv(OUT_DIR / "v10_writer_scripts.csv", index=False)
    logger.info(f"[+] V10 verdict inputs: {json.dumps(result['verdict_inputs'])}")


if __name__ == "__main__":
    main()
