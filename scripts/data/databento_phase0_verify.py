"""Phase 0 verification per DATABENTO_BULK_PULL_PLAN.md Section 12.

Runs metadata-only checks (zero cost) before committing to any paid batch jobs:
  0.1 Confirm GLBX.MDP3 dataset range
  0.2 Verify .v.0 roll resolves cleanly on GC and ZC across full history
  0.3 Cost preview for Section A (continuous, ~50 symbols) and MBP-1 sliver

Run before subscribing to Databento Standard or submitting any batch jobs.
"""

import os
from datetime import datetime
from pathlib import Path

import databento as db
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DATASET = "GLBX.MDP3"
START_FLOOR = "2010-06-06"
END = "2026-02-22"

ALL_CONTINUOUS = [
    "ES.v.0", "NQ.v.0", "YM.v.0", "RTY.v.0",
    "MES.v.0", "MNQ.v.0", "M2K.v.0", "MYM.v.0",
    "CL.v.0", "NG.v.0", "HO.v.0", "RB.v.0", "BZ.v.0", "MCL.v.0", "MNG.v.0",
    "GC.v.0", "GC.n.0", "SI.v.0", "HG.v.0", "PL.v.0", "MGC.v.0", "SIL.v.0",
    "ZT.v.0", "ZF.v.0", "ZN.v.0", "TN.v.0", "ZB.v.0", "UB.v.0",
    "SR3.v.0", "SR1.v.0",
    "10Y.v.0", "30Y.v.0", "5YY.v.0", "2YY.v.0",
    "6E.v.0", "6J.v.0", "6B.v.0", "6A.v.0", "6C.v.0", "6S.v.0", "6N.v.0", "6M.v.0",
    "ZC.v.0", "ZS.v.0", "ZW.v.0", "KE.v.0", "ZL.v.0", "ZM.v.0", "LE.v.0", "HE.v.0",
    "BTC.v.0", "MBT.v.0", "ETH.v.0", "MET.v.0",
]

# CME month codes; last char of raw symbol after root
EXPECTED_GC_MONTHS = {"G", "J", "M", "Q", "V", "Z"}  # Feb Apr Jun Aug Oct Dec
EXPECTED_ZC_MONTHS = {"H", "K", "N", "U", "Z"}        # Mar May Jul Sep Dec


def _root_month_year(raw_symbol: str, root: str) -> tuple[str, str]:
    """Extract month-code letter and year-digits from a raw CME symbol like 'GCG4'."""
    tail = raw_symbol[len(root):]
    if not tail:
        return ("", "")
    return tail[0], tail[1:]


def step_01_dataset_range(client: db.Historical) -> None:
    print("=" * 70)
    print("0.1 Dataset range check")
    print("=" * 70)
    rng = client.metadata.get_dataset_range(dataset=DATASET)
    print(f"  Reported range: {rng}")
    start = rng.get("start_date") or rng.get("start")
    if isinstance(start, str) and start.startswith(START_FLOOR):
        print(f"  [OK] Start floor confirmed: {start}")
    else:
        print(f"  [WARN] Expected floor {START_FLOOR}, got {start}")


def step_02_roll_resolution(client: db.Historical) -> None:
    print()
    print("=" * 70)
    print("0.2 Roll-rule resolution: GC.v.0 and ZC.v.0")
    print("=" * 70)

    for sym, root, expected_months in [
        ("GC.v.0", "GC", EXPECTED_GC_MONTHS),
        ("ZC.v.0", "ZC", EXPECTED_ZC_MONTHS),
    ]:
        print(f"\nResolving {sym} {START_FLOOR} -> {END}...")
        try:
            res = client.symbology.resolve(
                dataset=DATASET,
                symbols=[sym],
                stype_in="continuous",
                stype_out="raw_symbol",
                start_date=START_FLOOR,
                end_date=END,
            )
        except Exception as e:
            print(f"  [FAIL] {sym} resolve raised: {type(e).__name__}: {e}")
            continue

        result = res.get("result") if isinstance(res, dict) else getattr(res, "result", None)
        if not result:
            print(f"  [FAIL] {sym} no result mappings returned")
            continue

        mappings = result.get(sym, [])
        if not mappings:
            print(f"  [FAIL] {sym} mapping list is empty (broken resolution)")
            continue

        unique_raws = sorted({m.get("s") for m in mappings if m.get("s")})
        print(f"  [OK] {sym} has {len(mappings)} mapping intervals, "
              f"{len(unique_raws)} unique raw contracts")
        # Check that month codes are in the expected cycle
        bad_months = []
        for raw in unique_raws:
            month_code, year_digits = _root_month_year(raw, root)
            if month_code and month_code not in expected_months:
                bad_months.append(raw)
        if bad_months:
            print(f"  [WARN] {sym} has off-cycle contracts: {bad_months[:5]}")
        else:
            print(f"  [OK] {sym} all contracts in expected cycle "
                  f"{sorted(expected_months)}")
        print(f"  First 5 contracts: {unique_raws[:5]}")
        print(f"  Last 5 contracts:  {unique_raws[-5:]}")


def step_03_cost_preview(client: db.Historical) -> None:
    print()
    print("=" * 70)
    print("0.3 Cost preview")
    print("=" * 70)

    # Section A: full continuous OHLCV-1m
    try:
        cost_a = client.metadata.get_cost(
            dataset=DATASET,
            symbols=ALL_CONTINUOUS,
            stype_in="continuous",
            schema="ohlcv-1m",
            start=START_FLOOR,
            end=END,
        )
        print(f"  Section A (continuous OHLCV-1m, {len(ALL_CONTINUOUS)} symbols): "
              f"${cost_a:.4f}")
    except Exception as e:
        print(f"  [FAIL] Section A cost preview: {type(e).__name__}: {e}")

    # Section F: MBP-1 first 12 months (should be free under L1 12mo inclusion)
    mbp_syms = ["ES.v.0", "MES.v.0", "NQ.v.0", "MNQ.v.0"]
    try:
        cost_mbp1_free = client.metadata.get_cost(
            dataset=DATASET,
            symbols=mbp_syms,
            stype_in="continuous",
            schema="mbp-1",
            start="2025-02-22",
            end="2026-02-22",
        )
        print(f"  Section F MBP-1 most-recent 12mo (should be free): "
              f"${cost_mbp1_free:.4f}")
    except Exception as e:
        print(f"  [FAIL] MBP-1 12mo cost preview: {type(e).__name__}: {e}")

    try:
        cost_mbp1_paid = client.metadata.get_cost(
            dataset=DATASET,
            symbols=mbp_syms,
            stype_in="continuous",
            schema="mbp-1",
            start="2024-02-22",
            end="2025-02-22",
        )
        print(f"  Section F MBP-1 second 12mo PAYG: ${cost_mbp1_paid:.4f}")
    except Exception as e:
        print(f"  [FAIL] MBP-1 PAYG cost preview: {type(e).__name__}: {e}")


def main() -> int:
    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        print("[FAIL] DATABENTO_API_KEY not set")
        return 1

    client = db.Historical(api_key)

    step_01_dataset_range(client)
    step_02_roll_resolution(client)
    step_03_cost_preview(client)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
