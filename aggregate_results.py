from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from hefl.reporting import write_rows_csv
from hefl.utils import read_json, write_json


def _metadata_stability(run_dir: Path, baseline: str) -> float:
    path = run_dir / "leakage_report.json"
    if not path.exists():
        return 0.0
    report = read_json(path)
    mode_by_baseline = {
        "simhe_fixed_template_runtime": "fixed",
        "simhe_layer_order": "fixed_reference_simhe_layer_order",
        "simhe_manual_packed": "fixed_reference_simhe_manual_packed",
        "dynamic_template_or_selective": "dynamic",
    }
    mode = mode_by_baseline.get(baseline)
    if not mode:
        return 0.0
    return float(report.get("by_template_mode", {}).get(mode, {}).get("template_stability_rate", 0.0))


def aggregate_suite(suite_dir: str | Path) -> Dict[str, Any]:
    suite_dir = Path(suite_dir)
    manifest_path = suite_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = read_json(manifest_path)
    seed_rows: List[Dict[str, Any]] = []

    for run in manifest.get("runs", []):
        run_dir = Path(run["result_dir"])
        summary_path = run_dir / "summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        summary = pd.read_csv(summary_path).to_dict(orient="records")
        for row in summary:
            baseline = str(row.get("baseline", ""))
            seed_rows.append(
                {
                    "suite": manifest.get("suite"),
                    "workload": run.get("workload"),
                    "seed": run.get("seed"),
                    "run_id": run.get("run_id"),
                    "result_dir": str(run_dir),
                    "baseline": baseline,
                    "rounds": row.get("rounds", 0),
                    "avg_round_ms": row.get("avg_round_ms", 0.0),
                    "avg_train_ms": row.get("avg_train_ms", 0.0),
                    "avg_serialized_mb_per_client": row.get("avg_serialized_mb_per_client", 0.0),
                    "avg_raw_mb_per_client": row.get("avg_raw_mb_per_client", 0.0),
                    "avg_ciphertext_count": row.get("avg_ciphertext_count", 0.0),
                    "avg_slot_utilization": row.get("avg_slot_utilization", 0.0),
                    "avg_total_l2": row.get("avg_total_l2", 0.0),
                    "initial_accuracy": row.get("initial_accuracy", 0.0),
                    "final_accuracy": row.get("final_accuracy", 0.0),
                    "metadata_stability": _metadata_stability(run_dir, baseline),
                }
            )

    write_rows_csv(suite_dir / "seed_summary.csv", seed_rows)
    df = pd.DataFrame(seed_rows)
    final_rows: List[Dict[str, Any]] = []
    numeric_cols = [
        "rounds",
        "avg_round_ms",
        "avg_train_ms",
        "avg_serialized_mb_per_client",
        "avg_raw_mb_per_client",
        "avg_ciphertext_count",
        "avg_slot_utilization",
        "avg_total_l2",
        "initial_accuracy",
        "final_accuracy",
        "metadata_stability",
    ]
    if not df.empty:
        for (workload, baseline), sub in df.groupby(["workload", "baseline"]):
            item: Dict[str, Any] = {
                "workload": workload,
                "baseline": baseline,
                "seeds": int(sub["seed"].nunique()),
            }
            for col in numeric_cols:
                vals = pd.to_numeric(sub[col], errors="coerce")
                mean = float(vals.mean())
                std = float(vals.std(ddof=1)) if len(vals.dropna()) > 1 else 0.0
                ci95 = 1.96 * std / math.sqrt(max(int(vals.count()), 1)) if vals.count() else 0.0
                item[f"{col}_mean"] = mean
                item[f"{col}_std"] = std
                item[f"{col}_ci95"] = ci95
            final_rows.append(item)

    write_rows_csv(suite_dir / "final_summary.csv", final_rows)
    lines = ["# Final HE-FL Suite Summary", ""]
    if final_rows:
        cols = [
            "workload",
            "baseline",
            "seeds",
            "final_accuracy_mean",
            "avg_round_ms_mean",
            "avg_serialized_mb_per_client_mean",
            "avg_ciphertext_count_mean",
            "avg_slot_utilization_mean",
            "avg_total_l2_mean",
            "metadata_stability_mean",
        ]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for row in final_rows:
            values = []
            for col in cols:
                val = row.get(col, "")
                values.append(f"{val:.6g}" if isinstance(val, float) else str(val))
            lines.append("| " + " | ".join(values) + " |")
    (suite_dir / "final_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    aggregate_manifest = {
        "suite_dir": str(suite_dir),
        "seed_summary": "seed_summary.csv",
        "final_summary": "final_summary.csv",
        "rows": len(final_rows),
    }
    write_json(suite_dir / "aggregate_manifest.json", aggregate_manifest)
    return aggregate_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate a final HE-FL experiment suite across seeds.")
    parser.add_argument("--suite-dir", required=True, help="Suite output directory containing manifest.json.")
    args = parser.parse_args()
    result = aggregate_suite(args.suite_dir)
    print(result["suite_dir"])


if __name__ == "__main__":
    main()
