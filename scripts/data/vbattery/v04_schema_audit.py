"""V4 -- Schema reconciliation + dtype audit over the equity-options store.

Registered gate: "20 vs 21 columns (leading `symbol`); full dtype audit against
the Datetime[us, UTC] standard. Gate: Report only -- output is the
canonicalization mapping."

Reads PARQUET FOOTER METADATA ONLY (schema_arrow, num_rows, num_row_groups).
No row data is read. Emits one row per root x year x month partition.
"""
from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.settings import get_local_storage_dir
from src.utils.logger import get_logger
from src.utils.run_status import RunStatus

logger = get_logger(__name__)

OUT_DIR = Path("output/vbattery/v04")


def _partitions(root_dir: Path):
    for root_p in sorted(root_dir.glob("root=*")):
        for year_p in sorted(root_p.glob("year=*")):
            for month_p in sorted(year_p.glob("month=*")):
                f = month_p / "data.parquet"
                if f.exists():
                    yield root_p.name.split("=", 1)[1], year_p.name.split("=", 1)[1], month_p.name.split("=", 1)[1], f


CANONICAL_20 = [
    "timestamp", "expiration", "strike", "right",
    "open", "high", "low", "close", "volume", "trade_count", "vwap",
    "bid_close", "ask_close", "implied_vol", "delta", "theta", "vega",
    "underlying_px", "gamma_eod", "open_interest_eod",
]
CANONICAL_DTYPES = {
    "timestamp": "timestamp[us, tz=UTC]", "expiration": "date32[day]", "strike": "double",
    "right": "string", "open": "double", "high": "double", "low": "double", "close": "double",
    "volume": "int64", "trade_count": "int64", "vwap": "double", "bid_close": "double",
    "ask_close": "double", "implied_vol": "double", "delta": "double", "theta": "double",
    "vega": "double", "underlying_px": "double", "gamma_eod": "double",
    "open_interest_eod": "int64",
}


def emit_canonicalization_mapping(df: pd.DataFrame) -> None:
    """Per schema variant x canonical column: the required read-side transform."""
    rows = []
    for h, g in df.groupby("schema_hash"):
        r = g.iloc[0]
        cols = r["columns"].split("|")
        dts = dict(zip(cols, r["dtypes"].split("|")))
        for cc in CANONICAL_20:
            present = cc in dts
            src = dts.get(cc, "")
            tgt = CANONICAL_DTYPES[cc]
            if not present:
                action = "MISSING -- column absent; quarantine partition for any use requiring this field"
            elif src == "null":
                action = "ALL_NULL -- typed null column, carries no data; treat as MISSING"
            elif src == tgt:
                action = "OK"
            elif cc == "timestamp":
                action = f"PARSE_STRING_TO_DATETIME ({src} -> {tgt}); tz-naive input, localization rule must be registered"
            elif cc == "expiration":
                action = (f"CAST {src} -> {tgt}" if src == "date32[day]"
                          else f"PARSE_STRING_TO_DATE ({src} -> {tgt})")
            elif src.startswith("dictionary"):
                action = f"DECODE_DICTIONARY -> {tgt}"
            elif src in ("float", "float32"):
                action = f"UPCAST float32 -> {tgt} (PRECISION ALREADY LOST at write time)"
            elif src in ("int32", "int64", "double", "string", "large_string"):
                action = f"CAST {src} -> {tgt}"
            else:
                action = f"CAST {src} -> {tgt}"
            rows.append({
                "schema_hash": h, "n_columns": int(r["n_columns"]),
                "has_symbol_col": bool(r["has_symbol_col"]),
                "n_partitions": int(len(g)), "n_rows": int(g["num_rows"].sum()),
                "roots": ",".join(sorted(g["root"].unique())),
                "canonical_column": cc,
                "source_position": cols.index(cc) if present else -1,
                "source_dtype": src or "ABSENT",
                "target_dtype": tgt,
                "action": action,
            })
        for extra in cols:
            if extra not in CANONICAL_20:
                rows.append({
                    "schema_hash": h, "n_columns": int(r["n_columns"]),
                    "has_symbol_col": bool(r["has_symbol_col"]),
                    "n_partitions": int(len(g)), "n_rows": int(g["num_rows"].sum()),
                    "roots": ",".join(sorted(g["root"].unique())),
                    "canonical_column": f"<extra:{extra}>",
                    "source_position": cols.index(extra),
                    "source_dtype": dts[extra], "target_dtype": "DROP",
                    "action": "DROP -- not in canonical 20; redundant with partition key" if extra == "symbol"
                              else "DROP -- not in canonical 20",
                })
    pd.DataFrame(rows).to_csv(OUT_DIR / "v04_canonicalization_mapping.csv", index=False)
    logger.info(f"[+] canonicalization mapping rows: {len(rows)}")


def main() -> None:
    base = Path(get_local_storage_dir()) / "options" / "options_combined"
    if not base.exists():
        raise FileNotFoundError(f"options_combined not found: {base}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cached = OUT_DIR / "v04_partition_schemas.parquet"
    if os.environ.get("V04_MAPPING_ONLY") == "1" and cached.exists():
        emit_canonicalization_mapping(pd.read_parquet(cached))
        return
    parts = list(_partitions(base))
    logger.info(f"[*] V4 partitions discovered: {len(parts)}")

    rows = []
    errors = []
    with RunStatus("vbattery_v04_schema_audit", meta={"n_partitions": len(parts)}) as st:
        for i, (root, year, month, path) in enumerate(parts):
            try:
                pf = pq.ParquetFile(path)
                sch = pf.schema_arrow
                names = list(sch.names)
                dtypes = [str(sch.field(n).type) for n in names]
                sig = "|".join(f"{n}:{d}" for n, d in zip(names, dtypes))
                rows.append({
                    "root": root,
                    "year": int(year),
                    "month": int(month),
                    "n_columns": len(names),
                    "schema_hash": hashlib.sha1(sig.encode("ascii", "replace")).hexdigest()[:12],
                    "columns": "|".join(names),
                    "dtypes": "|".join(dtypes),
                    "has_symbol_col": "symbol" in names,
                    "timestamp_dtype": str(sch.field("timestamp").type) if "timestamp" in names else "MISSING",
                    "expiration_dtype": str(sch.field("expiration").type) if "expiration" in names else "MISSING",
                    "num_rows": pf.metadata.num_rows,
                    "num_row_groups": pf.metadata.num_row_groups,
                    "file_bytes": path.stat().st_size,
                    "path": str(path),
                })
            except Exception as exc:
                logger.error(f"[-] failed metadata read {path}: {exc}")
                errors.append({"path": str(path), "error": repr(exc)})
            if i % 500 == 0:
                st.heartbeat(note=f"{i}/{len(parts)}")
                logger.info(f"[*] {i}/{len(parts)}")

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_DIR / "v04_partition_schemas.parquet", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(OUT_DIR / "v04_metadata_errors.csv", index=False)

    variants = (df.groupby(["schema_hash", "n_columns", "has_symbol_col", "timestamp_dtype",
                            "expiration_dtype", "columns", "dtypes"], dropna=False)
                  .agg(n_partitions=("root", "size"),
                       n_roots=("root", "nunique"),
                       total_rows=("num_rows", "sum"),
                       total_bytes=("file_bytes", "sum"),
                       roots=("root", lambda s: ",".join(sorted(set(s)))),
                       min_ym=("year", "min"),
                       max_ym=("year", "max"))
                  .reset_index())
    variants.to_csv(OUT_DIR / "v04_schema_variants.csv", index=False)

    summary = {
        "n_partitions": int(len(df)),
        "n_roots": int(df["root"].nunique()),
        "total_rows": int(df["num_rows"].sum()),
        "total_bytes": int(df["file_bytes"].sum()),
        "n_schema_variants": int(variants.shape[0]),
        "n_partitions_21_col": int((df["n_columns"] == 21).sum()),
        "n_partitions_with_symbol": int(df["has_symbol_col"].sum()),
        "column_count_distribution": {str(k): int(v) for k, v in df["n_columns"].value_counts().items()},
        "timestamp_dtype_distribution": {str(k): int(v) for k, v in df["timestamp_dtype"].value_counts().items()},
        "expiration_dtype_distribution": {str(k): int(v) for k, v in df["expiration_dtype"].value_counts().items()},
        "n_metadata_errors": len(errors),
    }
    emit_canonicalization_mapping(df)
    (OUT_DIR / "v04_summary.json").write_text(json.dumps(summary, indent=2), encoding="ascii")
    logger.info(f"[+] V4 summary: {json.dumps(summary)[:2000]}")


if __name__ == "__main__":
    main()
