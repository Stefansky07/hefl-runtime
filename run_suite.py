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


FINAL_AUTODL_CONFIGS = [
    "cifar10_tinycnn_iid",
    "cifar10_tinycnn_dirichlet05",
    "cifar10_tinycnn_dirichlet01",
    "fashionmnist_tinymlp_dirichlet05",
    "resnet18_cifar10_profile",
]


def _suite_output_dir(out: str | Path | None, suite: str) -> Path:
    if out:
        return ensure_dir(out).resolve()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_dir(ROOT / "results" / f"final_{stamp}_{suite}").resolve()


def _resolved_config(
    base: Dict[str, Any],
    workload: str,
    seed: int,
    suite_dir: Path,
    data_root: str,
    calibration_path: Path,
    allow_analytic: bool,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg["seed"] = int(seed)
    cfg["run_name"] = f"{workload}_seed{seed}"
    cfg["result_base_dir"] = str(suite_dir / "runs")
    cfg.setdefault("dataset", {})
    cfg["dataset"]["root"] = data_root
    cfg["dataset"]["allow_fake_fallback"] = False
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
    if suite != "final_autodl":
        raise ValueError(f"Unknown suite: {suite}")
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
    for workload in FINAL_AUTODL_CONFIGS:
        base_path = ROOT / "configs" / f"{workload}.json"
        base = load_config(base_path)
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
        "data_root": data_root,
        "calibration_path": str(calibration),
        "runs": runs,
        "workloads": FINAL_AUTODL_CONFIGS,
    }
    write_json(suite_dir / "manifest.json", manifest)
    aggregate_suite(suite_dir)
    print(suite_dir)
    return suite_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the final HE-FL experiment suite.")
    parser.add_argument("--suite", default="final_autodl", choices=["final_autodl"])
    parser.add_argument("--out", default=None, help="Suite output directory. Defaults to results/final_<timestamp>.")
    parser.add_argument("--data-root", default="/root/autodl-tmp/data")
    parser.add_argument("--calibration-path", default=None)
    parser.add_argument("--allow-analytic", action="store_true", help="Development only: run without TenSEAL calibration.")
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
