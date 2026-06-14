#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/data}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${1:-${SCRIPT_DIR}/results/final_${STAMP}_final_autodl}"

mkdir -p "${DATA_ROOT}" "${OUT_DIR}"
cd "${REPO_ROOT}"

python -m pip install -r "${SCRIPT_DIR}/requirements-autodl.txt"

python - <<'PY'
import torch
import torchvision

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

python experiments/hefl_runtime/calibrate_he.py \
  --backend tenseal \
  --out "${OUT_DIR}/he_calibration.json"

python experiments/hefl_runtime/run_suite.py \
  --suite final_autodl \
  --out "${OUT_DIR}" \
  --data-root "${DATA_ROOT}" \
  --calibration-path "${OUT_DIR}/he_calibration.json"

python experiments/hefl_runtime/aggregate_results.py --suite-dir "${OUT_DIR}"

echo "Final suite complete: ${OUT_DIR}"
