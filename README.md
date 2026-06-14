# HE-FL Runtime Experiments

This directory contains experiment code for federated averaging with CKKS-like secure aggregation profiling.

## Structure

```text
hefl_runtime/
├── hefl/                  # core package
├── configs/               # JSON experiment configs
├── run_experiment.py      # run one config
├── run_suite.py           # run a multi-config suite
├── aggregate_results.py   # aggregate suite outputs
├── calibrate_he.py        # optional TenSEAL timing calibration
├── summarize_results.py   # regenerate summaries for one run
└── requirements-autodl.txt
```

## Install

Install PyTorch and torchvision for your environment first, then install the remaining dependencies:

```bash
python -m pip install -r requirements-autodl.txt
```

## Run One Experiment

```bash
python run_experiment.py --config configs/smoke.json
python run_experiment.py --config configs/cifar10_tinycnn_iid.json
```

Each run writes outputs under:

```text
results/<timestamp>_<run_name>/
```

## Run A Suite

Run a short suite:

```bash
python run_suite.py --suite dev_autodl --out results/dev --data-root data --allow-analytic
python aggregate_results.py --suite-dir results/dev
```

Run the full suite with calibration:

```bash
python calibrate_he.py --backend tenseal --out results/final/he_calibration.json
python run_suite.py --suite paper_autodl --out results/final --data-root data --calibration-path results/final/he_calibration.json
python aggregate_results.py --suite-dir results/final
```

## Data

Real dataset configs use torchvision datasets. Set `dataset.root` or pass `--data-root` to choose the cache directory.

Final configs set:

```json
"allow_fake_fallback": false
```

so missing real data fails explicitly instead of silently falling back to FakeData. `configs/smoke.json` keeps FakeData for quick checks.

## Outputs

Single runs produce:

```text
config.json
config_resolved.json
hardware.json
round_trace.csv
client_metrics.csv
summary.csv
layout_plan.json
error_decomposition.csv
metadata_features.csv
leakage_report.json
run_summary.json
manifest.json
report.md
summary.md
*.png
```

Suite runs additionally produce:

```text
manifest.json
seed_summary.csv
final_summary.csv
final_summary.md
aggregate_manifest.json
```

## Main Configs

```text
configs/smoke.json
configs/cifar10_tinycnn_iid.json
configs/cifar10_tinycnn_dirichlet05.json
configs/cifar10_tinycnn_dirichlet01.json
configs/fashionmnist_tinymlp_dirichlet05.json
configs/resnet18_cifar10_profile.json
```

## Optional HE Calibration

`calibrate_he.py` uses TenSEAL to measure representative CKKS vector operations and writes a JSON calibration file. `run_suite.py` can pass that file to the SimHE backend with `--calibration-path`.

