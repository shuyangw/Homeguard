#!/usr/bin/env bash
# Batch driver for the V1/V2/V3/V5/V9/V11 sweep.
# Usage: run_sweep_batch.sh <mode> <jobs> <spec> [<spec> ...]
#   mode: prepass | sweep
#   spec: ROOT  or  ROOT:YFROM:YTO
set -u
MODE="$1"; shift
JOBS="$1"; shift
PY="C:/Users/qwqw1/anaconda3/envs/fintech/python.exe"
SCRIPT="scripts/data/vbattery/sweep_v1_v2_v3_v5_v9_v11.py"
export PYTHONPATH=. POLARS_MAX_THREADS=1 OMP_NUM_THREADS=1
mkdir -p output/vbattery/sweep/logs

run_one() {
  spec="$1"
  root="${spec%%:*}"
  rest="${spec#*:}"
  args=(--root "$root")
  tag="$root"
  if [ "$rest" != "$spec" ]; then
    yf="${rest%%:*}"; yt="${rest##*:}"
    args+=(--year-from "$yf" --year-to "$yt")
    tag="${root}_${yf}_${yt}"
  fi
  [ "$MODE" = "prepass" ] && args+=(--prepass)
  "$PY" "$SCRIPT" "${args[@]}" > "output/vbattery/sweep/logs/${MODE}_${tag}.log" 2>&1
  echo "[done] $MODE $tag rc=$?"
}

i=0
for spec in "$@"; do
  run_one "$spec" &
  i=$((i+1))
  if [ $((i % JOBS)) -eq 0 ]; then wait; fi
done
wait
echo "[batch complete] $MODE"
