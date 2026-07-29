#!/usr/bin/env bash
# Parallel driver for options_chain_eod materialization.
# Usage: build_eod_chain_parallel.sh <jobs_file> <n_jobs>
# jobs_file lines: "<ROOT> <YEAR> <MONTH>"
set -u
JOBS_FILE="$1"
NJOBS="${2:-8}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="C:/Users/qwqw1/anaconda3/envs/fintech/python.exe"
export POLARS_MAX_THREADS=1
export OMP_NUM_THREADS=1

run_one() {
    root="$1"; year="$2"; month="$3"
    cd "$REPO" || exit 1
    log="output/wave0/logs/eod_${root}_${year}_${month}.log"
    if [ -f "H:/Stock_Data/options/options_chain_eod/root=${root}/year=${year}/month=$(printf '%02d' "$month")/data.parquet" ]; then
        echo "SKIP ${root} ${year}-${month}" >> output/wave0/logs/eod_driver.log
        return 0
    fi
    "$PY" scripts/data/build_options_chain_eod.py --root "$root" --year "$year" --month "$month" > "$log" 2>&1
    echo "EXIT=$? ${root} ${year}-${month}" >> output/wave0/logs/eod_driver.log
}
export -f run_one
export REPO PY

xargs -a "$JOBS_FILE" -n 3 -P "$NJOBS" bash -c 'run_one "$@"' _
echo "DRIVER_DONE"
