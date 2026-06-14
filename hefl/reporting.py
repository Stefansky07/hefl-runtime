from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from .utils import ensure_dir, write_json


def create_run_dir(base_dir: str | Path, run_name: str) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_dir(Path(base_dir) / f"{stamp}_{run_name}")


def write_rows_csv(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown_report(
    path: str | Path,
    config: Dict[str, Any],
    summary_rows: List[Dict[str, Any]],
    leakage_report: Dict[str, Any],
    artifacts: List[str],
) -> None:
    lines = [
        f"# SimHE Lightweight HE-FL Runtime Report",
        "",
        f"- Run name: `{config.get('run_name')}`",
        f"- Dataset: `{config.get('dataset', {}).get('name')}`",
        f"- Model: `{config.get('model', {}).get('name')}`",
        f"- Important: these HE numbers are `SimHE` profiling estimates, not real CKKS measurements.",
        "",
        "## Summary",
        "",
    ]
    if summary_rows:
        cols = [
            "baseline",
            "rounds",
            "avg_round_ms",
            "avg_serialized_mb_per_client",
            "avg_ciphertext_count",
            "avg_slot_utilization",
            "avg_total_l2",
            "final_accuracy",
        ]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for row in summary_rows:
            vals = []
            for col in cols:
                val = row.get(col, "")
                vals.append(f"{val:.6g}" if isinstance(val, float) else str(val))
            lines.append("| " + " | ".join(vals) + " |")
    lines.extend(["", "## Leakage Proxy", ""])
    for mode, row in leakage_report.get("by_template_mode", {}).items():
        lines.append(
            f"- `{mode}`: bytes variance={row.get('serialized_bytes_variance', 0):.6g}, "
            f"occupancy variance={row.get('slot_utilization_variance', 0):.6g}, "
            f"template stability={row.get('template_stability_rate', 0):.3f}, "
            f"toy attack accuracy={row.get('toy_attack_accuracy', 0):.3f}"
        )
    lines.extend(["", "## Artifacts", ""])
    for artifact in artifacts:
        lines.append(f"- `{artifact}`")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_results(result_dir: str | Path) -> List[str]:
    result_dir = Path(result_dir)
    outputs: List[str] = []
    try:
        import matplotlib.pyplot as plt

        round_path = result_dir / "round_trace.csv"
        error_path = result_dir / "error_decomposition.csv"
        if round_path.exists() and round_path.stat().st_size > 0:
            df = pd.read_csv(round_path)
            if not df.empty:
                latency_cols = [c for c in ["train_ms", "pack_ms", "encode_sim_ms", "encrypt_sim_ms", "tx_sim_ms", "agg_ms", "decrypt_sim_ms", "rebuild_ms"] if c in df]
                if latency_cols:
                    grouped = df.groupby("baseline")[latency_cols].mean()
                    ax = grouped.plot(kind="bar", stacked=True, figsize=(9, 4))
                    ax.set_ylabel("mean ms / round")
                    ax.set_title("SimHE runtime breakdown")
                    plt.tight_layout()
                    out = result_dir / "latency_breakdown.png"
                    plt.savefig(out, dpi=160)
                    plt.close()
                    outputs.append(str(out))
                if {"baseline", "slot_utilization", "serialized_ciphertext_bytes_per_client"}.issubset(df.columns):
                    grouped = df.groupby("baseline")[["slot_utilization", "serialized_ciphertext_bytes_per_client"]].mean()
                    ax = grouped["slot_utilization"].plot(kind="bar", figsize=(8, 4))
                    ax.set_ylim(0, 1.05)
                    ax.set_ylabel("slot utilization")
                    ax.set_title("Slot utilization by baseline")
                    plt.tight_layout()
                    out = result_dir / "slot_utilization.png"
                    plt.savefig(out, dpi=160)
                    plt.close()
                    outputs.append(str(out))
        if error_path.exists() and error_path.stat().st_size > 0:
            df = pd.read_csv(error_path)
            if not df.empty and {"baseline", "total_l2", "pack_l2", "he_l2"}.issubset(df.columns):
                grouped = df.groupby("baseline")[["pack_l2", "he_l2", "total_l2"]].mean()
                ax = grouped.plot(kind="bar", figsize=(9, 4))
                ax.set_ylabel("L2 error")
                ax.set_title("Aggregation error decomposition")
                plt.tight_layout()
                out = result_dir / "error_decomposition.png"
                plt.savefig(out, dpi=160)
                plt.close()
                outputs.append(str(out))
    except Exception as exc:
        write_json(result_dir / "plot_error.json", {"error": str(exc)})
    return outputs


def summarize_rows(round_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not round_rows:
        return []
    df = pd.DataFrame(round_rows)
    out: List[Dict[str, Any]] = []
    for baseline, sub in df.groupby("baseline"):
        row = {
            "baseline": baseline,
            "rounds": int(sub["round"].nunique()) if "round" in sub else len(sub),
            "avg_round_ms": float(sub.get("round_ms", pd.Series([0.0])).mean()),
            "avg_train_ms": float(sub.get("train_ms", pd.Series([0.0])).mean()),
            "avg_serialized_mb_per_client": float(sub.get("serialized_ciphertext_bytes_per_client", pd.Series([0.0])).mean() / 1024**2),
            "avg_raw_mb_per_client": float(sub.get("raw_ciphertext_bytes_per_client", pd.Series([0.0])).mean() / 1024**2),
            "avg_ciphertext_count": float(sub.get("ciphertext_count", sub.get("bundle_count", pd.Series([0.0]))).mean()),
            "avg_slot_utilization": float(sub.get("slot_utilization", pd.Series([0.0])).mean()),
            "avg_total_l2": float(sub.get("total_l2", pd.Series([0.0])).mean()),
        }
        out.append(row)
    return out
