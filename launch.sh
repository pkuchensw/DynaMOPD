#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
base=$(dirname "$here")
export PYTHONPATH="$base${FA2_DIR:+:$FA2_DIR}${PY_CACHE:+:$PY_CACHE}${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_HUB_DISABLE_PROGRESS_BARS=${HF_HUB_DISABLE_PROGRESS_BARS:-1}
export PYTHONUNBUFFERED=1
exec "${PYTHON:-python}" -m DynaMOPD.run "$@"
