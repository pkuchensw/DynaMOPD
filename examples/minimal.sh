#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
if [[ -z ${RUN_DIR:-} ]]; then
  mkdir -p runs
  RUN_DIR=$(mktemp -d runs/minimal-XXXXXXXX)
fi
echo "DynaMOPD output: $RUN_DIR"
exec bash "$repo/launch.sh" --config "$repo/configs/minimal-4gpu.json" \
  --data "$repo/data/native-smoke.jsonl" --out "$RUN_DIR" "$@"
