"""Parallel driver for the M7 smoothed IV surface build.

The work is embarrassingly parallel across root-months. Each worker pins
POLARS_MAX_THREADS/OMP_NUM_THREADS to 1 so `--jobs N` really means N total
threads (per the repo's parallel-thread cap note), and the whole run is wrapped
in `RunStatus` so a killed run leaves a stale RUNNING sentinel with its last
heartbeat rather than a silent gap.

Sharding exists because backgrounded jobs are reaped at ~60 minutes. Use
`--shard i/n` to split a long build across several sequential invocations.

    python scripts/data/build_m7_surface.py --jobs 8 --shard 0/2
    python scripts/data/build_m7_surface.py --jobs 8 --shard 1/2
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("POLARS_MAX_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import multiprocessing as mp
from typing import List, Optional, Sequence, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)

Job = Tuple[str, int, int]


def _init_worker() -> None:
    os.environ["POLARS_MAX_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"


def _run_one(job: Job) -> Tuple[Job, bool, str]:
    from src.data.options.iv_surface_build import build_month

    root, year, month = job
    try:
        build_month(root, year, month, overwrite=False)
        return job, True, ""
    except Exception as exc:  # a failed month must not kill the wave
        logger.error(f"[-] {root} {year}-{month:02d} failed: {exc}")
        return job, False, str(exc)


def _collect_jobs(roots: Sequence[str]) -> List[Job]:
    from src.data.options.iv_surface_build import _available_months

    jobs: List[Job] = []
    for root in roots:
        for year, month in _available_months(root):
            jobs.append((root, year, month))
    return jobs


def main(argv: Optional[Sequence[str]] = None) -> int:
    from src.utils.run_status import RunStatus

    ap = argparse.ArgumentParser(description="Build the M7 IV surface in parallel.")
    ap.add_argument("--roots", default="SPY,QQQ")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--shard", default="0/1", help="i/n round-robin shard")
    args = ap.parse_args(argv)

    roots = [r.strip() for r in args.roots.split(",") if r.strip()]
    jobs = _collect_jobs(roots)
    i, n = (int(x) for x in args.shard.split("/"))
    # Round-robin so months of very different sizes spread evenly.
    jobs = [j for idx, j in enumerate(jobs) if idx % n == i]
    if not jobs:
        logger.error("[-] no root-months matched")
        return 1

    logger.info(f"[+] {len(jobs)} root-months, shard {i}/{n}, jobs={args.jobs}")
    failures: List[Tuple[Job, str]] = []
    with RunStatus(
        f"m7_surface_shard{i}of{n}",
        meta={"roots": roots, "months": len(jobs), "jobs": args.jobs},
    ) as status:
        with mp.Pool(args.jobs, initializer=_init_worker) as pool:
            for done, (job, ok, err) in enumerate(
                pool.imap_unordered(_run_one, jobs), start=1
            ):
                if not ok:
                    failures.append((job, err))
                status.heartbeat(
                    note=f"{done}/{len(jobs)} done, {len(failures)} failed"
                )

    if failures:
        logger.error(f"[-] {len(failures)} root-months failed:")
        for job, err in failures:
            logger.error(f"    {job}: {err}")
        return 1
    logger.info(f"[+] shard {i}/{n} complete: {len(jobs)} root-months")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
