from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from hefl.reporting import plot_results, summarize_rows, write_markdown_report, write_rows_csv
from hefl.utils import read_json, write_json


def summarize(result_dir: str | Path) -> None:
    result_dir = Path(result_dir)
    round_path = result_dir / "round_trace.csv"
    if not round_path.exists():
        raise FileNotFoundError(round_path)
    round_rows = pd.read_csv(round_path).to_dict(orient="records")
    summary = summarize_rows(round_rows)
    write_rows_csv(result_dir / "summary.csv", summary)
    leakage_report = read_json(result_dir / "leakage_report.json") if (result_dir / "leakage_report.json").exists() else {}
    config = read_json(result_dir / "config_resolved.json") if (result_dir / "config_resolved.json").exists() else {}
    plot_paths = plot_results(result_dir)
    artifacts = [
        "round_trace.csv",
        "summary.csv",
        "error_decomposition.csv",
        "metadata_features.csv",
        "leakage_report.json",
        *[Path(p).name for p in plot_paths],
    ]
    write_markdown_report(result_dir / "summary.md", config, summary, leakage_report, artifacts)
    write_json(result_dir / "summarize_status.json", {"status": "ok", "result_dir": str(result_dir)})
    print(result_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a SimHE runtime result directory.")
    parser.add_argument("--results", required=True, help="Result directory to summarize.")
    args = parser.parse_args()
    summarize(args.results)


if __name__ == "__main__":
    main()
