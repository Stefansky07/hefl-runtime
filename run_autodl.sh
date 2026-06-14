#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${SCRIPT_DIR}/autodl_logs"

mkdir -p "${LOG_DIR}"
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

for name in smoke cifar_tiny leakage resnet18_profile; do
  config="experiments/hefl_runtime/configs/${name}.json"
  log="${LOG_DIR}/${name}.log"
  echo "Running ${config}"
  python experiments/hefl_runtime/run_experiment.py --config "${config}" 2>&1 | tee "${log}"
done

echo "Done. Results are under experiments/hefl_runtime/results/"
