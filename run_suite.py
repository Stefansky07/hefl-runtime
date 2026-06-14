from __future__ import annotations

import argparse
import copy
import datetime as dt
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aggregate_results import aggregate_suite
from hefl.config import load_config
from hefl.utils import ensure_dir, write_json
from run_experiment import run


PAPER_AUTODL_CONFIGS = [
    "cifar10_tinycnn_iid",
    "cifar10_tinycnn_dirichlet05",
    "cifar10_tinycnn_dirichlet01",
    "fashionmnist_tinymlp_dirichlet05",
    "resnet18_cifar10_profile",
]

DEV_AUTODL_CONFIGS = [
    "cifar10_tinycnn_iid",
    "cifar10_tinycnn_dirichlet05",
    "fashionmnist_tinymlp_dirichlet05",
    "resnet18_cifar10_profile",
]

SUITES = {
    "dev_autodl": {
        "configs": DEV_AUTODL_CONFIGS,
        "seeds": [2026],
        "rounds": 3,
        "subset_train": 2048,
        "subset_test": 512,
        "purpose": "development validation on real datasets with small subsets",
    },
    "paper_autodl": {
        "configs": PAPER_AUTODL_CONFIGS,
        "seeds": None,
        "rounds": None,
        "subset_train": None,
        "subset_test": None,
        "purpose": "ordinary paper-level final evaluation on real datasets",
    },
    "final_autodl": {
        "configs": PAPER_AUTODL_CONFIGS,
        "seeds": None,
        "rounds": None,
        "subset_train": None,
        "subset_test": None,
        "purpose": "backward-compatible alias for paper_autodl",
    },
}

RESOURCE_PRESETS = {
    "dev_autodl": {
        "recommended": "CPU 8 cores / 32GB RAM, or any low-cost GPU instance",
        "gpu": "optional",
        "notes": "TenSEAL calibration is CPU-bound; small PyTorch workloads do not require a high-end GPU.",
    },
    "paper_autodl": {
        "recommended": "RTX 3090/4090 24GB or A10 24GB; CPU 16 cores; RAM 64GB",
        "gpu": "recommended but not mandatory",
        "notes": "GPU speeds up PyTorch training; HE calibration remains CPU-bound.",
    },
    "final_autodl": {
        "recommended": "RTX 3090/4090 24GB or A10 24GB; CPU 16 cores; RAM 64GB",
        "gpu": "recommended but not mandatory",
        "notes": "Alias for paper_autodl.",
    },
}


def _suite_output_dir(out: str | Path | None, suite: str) -> Path:
    if out:
        return ensure_dir(out).resolve()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_dir(ROOT / "results" / f"suite_{stamp}_{suite}").resolve()


def _resolved_config(
    base: Dict[str, Any],
    workload: str,
    seed: int,
    suite_dir: Path,
    data_root: str,
    calibration_path: Path,
    allow_analytic: bool,
    suite_spec: Dict[str, Any],
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg["seed"] = int(seed)
    cfg["run_name"] = f"{workload}_seed{seed}"
    cfg["result_base_dir"] = str(suite_dir / "runs")
    cfg.setdefault("dataset", {})
    cfg["dataset"]["root"] = data_root
    cfg["dataset"]["allow_fake_fallback"] = False
    if suite_spec.get("subset_train") is not None:
        cfg["dataset"]["subset_train"] = suite_spec["subset_train"]
    if suite_spec.get("subset_test") is not None:
        cfg["dataset"]["subset_test"] = suite_spec["subset_test"]
    if suite_spec.get("rounds") is not None:
        cfg.setdefault("fed", {})
        cfg["fed"]["rounds"] = suite_spec["rounds"]
    cfg.setdefault("he_backend", {})
    if calibration_path.exists():
        cfg["he_backend"]["mode"] = "calibrated_simhe"
        cfg["he_backend"]["calibration_path"] = str(calibration_path)
    elif allow_analytic:
        cfg["he_backend"]["mode"] = "simhe"
        cfg["he_backend"]["calibration_path"] = None
    else:
        raise FileNotFoundError(
            f"Missing calibration file: {calibration_path}. "
            "Run calibrate_he.py first or pass --allow-analytic for development only."
        )
    return cfg


def run_suite(
    suite: str,
    out: str | Path | None,
    data_root: str,
    calibration_path: str | Path | None,
    allow_analytic: bool,
) -> Path:
    if suite not in SUITES:
        raise ValueError(f"Unknown suite: {suite}")
    suite_spec = SUITES[suite]
    suite_dir = _suite_output_dir(out, suite)
    config_dir = ensure_dir(suite_dir / "resolved_configs")
    ensure_dir(suite_dir / "runs")
    if calibration_path:
        calibration = Path(calibration_path).expanduser()
        if not calibration.is_absolute():
            calibration = (Path.cwd() / calibration).resolve()
    else:
        calibration = suite_dir / "he_calibration.json"

    runs: List[Dict[str, Any]] = []
    for workload in suite_spec["configs"]:
        base_path = ROOT / "configs" / f"{workload}.json"
        base = load_config(base_path)
        if suite_spec.get("seeds") is not None:
            seeds = [int(s) for s in suite_spec["seeds"]]
        else:
            seeds = [int(s) for s in base.get("experiment", {}).get("seeds", [base.get("seed", 2026)])]
        for seed in seeds:
            cfg = _resolved_config(
                base=base,
                workload=workload,
                seed=seed,
                suite_dir=suite_dir,
                data_root=data_root,
                calibration_path=calibration,
                allow_analytic=allow_analytic,
                suite_spec=suite_spec,
            )
            resolved_path = config_dir / f"{workload}_seed{seed}.json"
            write_json(resolved_path, cfg)
            result_dir = run(resolved_path)
            run_manifest = {
                "suite": suite,
                "workload": workload,
                "seed": seed,
                "run_id": result_dir.name,
                "result_dir": str(result_dir),
                "config_path": str(resolved_path),
            }
            runs.append(run_manifest)

    manifest = {
        "kind": "suite",
        "suite": suite,
        "suite_dir": str(suite_dir),
        "purpose": suite_spec["purpose"],
        "resource_preset": RESOURCE_PRESETS[suite],
        "data_root": data_root,
        "calibration_path": str(calibration),
        "runs": runs,
        "workloads": suite_spec["configs"],
    }
    write_json(suite_dir / "manifest.json", manifest)
    aggregate_suite(suite_dir)
    print(suite_dir)
    return suite_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a HE-FL experiment suite.")
    parser.add_argument("--suite", default="paper_autodl", choices=sorted(SUITES))
    parser.add_argument("--out", default=None, help="Suite output directory. Defaults to results/suite_<timestamp>.")
    parser.add_argument("--data-root", default="/root/autodl-tmp/data")
    parser.add_argument("--calibration-path", default=None)
    parser.add_argument("--allow-analytic", action="store_true", help="Run without TenSEAL calibration.")
    args = parser.parse_args()
    run_suite(
        suite=args.suite,
        out=args.out,
        data_root=args.data_root,
        calibration_path=args.calibration_path,
        allow_analytic=args.allow_analytic,
    )


if __name__ == "__main__":
    main()
