from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hefl.config import crypto_params_from_config, environment_info, load_config, resolve_device
from hefl.data import make_client_loaders, make_dataset, make_eval_loader, make_partitions
from hefl.fedavg import apply_update, clone_state_dict, evaluate, fedavg_plain, run_client_round
from hefl.layout import HeuristicLayoutPlanner, baseline_layout_config
from hefl.leakage import LeakageAuditor, MetadataRecord
from hefl.metrics import error_decomposition
from hefl.models import get_model, parameter_count, state_parameter_profile
from hefl.reporting import create_run_dir, plot_results, summarize_rows, write_markdown_report, write_rows_csv
from hefl.sim_he import SimHEBackend
from hefl.types import dataclass_to_dict
from hefl.utils import scale_state_dict, set_seed, write_json


def _selected_clients(config: Dict[str, Any]) -> List[int]:
    fed = config.get("fed", {})
    num_clients = int(fed.get("num_clients", 4))
    clients_per_round = int(fed.get("clients_per_round", num_clients))
    return list(range(min(num_clients, clients_per_round)))


def _sim_baselines(config: Dict[str, Any]) -> List[str]:
    baselines = list(config.get("baselines", []))
    if config.get("run_dynamic_template_ablation") and "dynamic_template_or_selective" not in baselines:
        baselines.append("dynamic_template_or_selective")
    return baselines


def _attack_labels(update_norms: List[float]) -> List[str]:
    if not update_norms:
        return []
    median = statistics.median(update_norms)
    return ["high_norm" if v >= median else "low_norm" for v in update_norms]


def _metadata_records(
    run_id: str,
    round_id: int,
    baseline: str,
    layout: Any,
    client_updates: Any,
    labels: List[str],
    crypto_serialized_ct_bytes: int,
) -> List[MetadataRecord]:
    rows: List[MetadataRecord] = []
    norms = [u.update_norm for u in client_updates]
    min_norm = min(norms) if norms else 0.0
    max_norm = max(norms) if norms else 1.0
    span = max(max_norm - min_norm, 1e-12)
    dynamic = baseline == "dynamic_template_or_selective"
    for idx, update in enumerate(client_updates):
        if dynamic:
            fraction = 0.45 + 0.55 * ((update.update_norm - min_norm) / span)
            visible_used = max(1, int(layout.num_params * fraction))
            visible_bundles = max(1, (visible_used + layout.slot_capacity - 1) // layout.slot_capacity)
            visible_padding = visible_bundles * layout.slot_capacity - visible_used
            visible_util = visible_used / max(visible_bundles * layout.slot_capacity, 1)
            template_id = f"dyn_{visible_bundles}"
            mode = "dynamic"
        else:
            visible_bundles = layout.bundle_count
            visible_padding = layout.padding_slots
            visible_util = layout.slot_utilization
            template_id = layout.template_id
            mode = "fixed" if baseline == "simhe_fixed_template_runtime" else f"fixed_reference_{baseline}"
        rows.append(
            MetadataRecord(
                run_id=run_id,
                round=round_id,
                baseline=baseline,
                client_id=update.client_id,
                template_mode=mode,
                template_id=template_id,
                bundle_count=int(visible_bundles),
                serialized_bytes=int(visible_bundles * crypto_serialized_ct_bytes),
                slot_utilization=float(visible_util),
                padding_slots=int(visible_padding),
                update_norm=float(update.update_norm),
                attack_label=labels[idx] if idx < len(labels) else "unknown",
            )
        )
    return rows


def run(config_path: str | Path) -> Path:
    config = load_config(config_path)
    set_seed(int(config.get("seed", 2026)))
    device = resolve_device(config)
    result_base = Path(config.get("result_base_dir", ROOT / "results"))
    if not result_base.is_absolute():
        result_base = ROOT / result_base
    result_dir = create_run_dir(result_base, str(config.get("run_name", "run")))
    run_id = result_dir.name

    train_dataset, test_dataset, dataset_info = make_dataset(config)
    num_classes = int(dataset_info["num_classes"])
    image_size = tuple(int(x) for x in dataset_info["image_size"])
    partitions = make_partitions(config, train_dataset, num_classes=num_classes)
    batch_size = int(config.get("dataset", {}).get("batch_size", 64))
    client_loaders = make_client_loaders(train_dataset, partitions, batch_size=batch_size, seed=int(config.get("seed", 2026)))
    eval_loader = make_eval_loader(test_dataset, batch_size=batch_size)

    model = get_model(config.get("model", {}).get("name", "tiny_cnn"), num_classes=num_classes, image_size=image_size).to(device)
    crypto = crypto_params_from_config(config)
    planner = HeuristicLayoutPlanner(crypto, bundle_buckets=config.get("layout", {}).get("bundle_buckets"))

    config_out = dict(config)
    config_out["dataset_resolved"] = dataset_info
    config_out["environment"] = environment_info(device)
    config_out["crypto_resolved"] = crypto.to_dict()
    config_out["model_profile"] = {
        "parameter_count": parameter_count(model),
        "state_tensors": state_parameter_profile(model),
    }
    config_out["partition"] = {
        "sample_counts": partitions.sample_counts,
        "label_histograms": partitions.label_histograms,
    }
    write_json(result_dir / "config.json", config)
    write_json(result_dir / "config_resolved.json", config_out)
    write_json(result_dir / "hardware.json", config_out["environment"])

    fed = config.get("fed", {})
    baselines = _sim_baselines(config)
    sim_he_cfg = config.get("sim_he", {})
    he_backend_cfg = config.get("he_backend", {})
    he_backend_mode = str(he_backend_cfg.get("mode", "simhe"))
    calibration_path = he_backend_cfg.get("calibration_path")
    if he_backend_mode == "calibrated_simhe" and not calibration_path:
        raise ValueError("he_backend.mode=calibrated_simhe requires he_backend.calibration_path")
    selected = _selected_clients(config)
    rounds = int(fed.get("rounds", 2))
    lr = float(fed.get("lr", 0.05))
    local_epochs = int(fed.get("local_epochs", 1))

    round_rows: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []
    client_rows: List[Dict[str, Any]] = []
    layout_rows: List[Dict[str, Any]] = []
    auditor = LeakageAuditor()

    initial_eval = evaluate(model, eval_loader, device)
    global_state = clone_state_dict(model)

    for round_id in range(1, rounds + 1):
        train_start = time.perf_counter()
        client_updates = run_client_round(
            global_model=model,
            global_state=global_state,
            client_loaders=client_loaders,
            selected_clients=selected,
            device=device,
            lr=lr,
            local_epochs=local_epochs,
            seed=int(config.get("seed", 2026)) + round_id * 100,
        )
        train_ms = (time.perf_counter() - train_start) * 1000.0
        plain_aggregate = fedavg_plain(client_updates)
        total_samples = sum(u.sample_count for u in client_updates)
        weights = [u.sample_count / max(total_samples, 1) for u in client_updates]
        weighted_updates = [scale_state_dict(u.update, w) for u, w in zip(client_updates, weights)]
        norm_labels = _attack_labels([u.update_norm for u in client_updates])

        for idx, update in enumerate(client_updates):
            client_rows.append(
                {
                    "run_id": run_id,
                    "round": round_id,
                    "client_id": update.client_id,
                    "sample_count": update.sample_count,
                    "train_loss": update.train_loss,
                    "train_ms": update.train_ms,
                    "update_norm": update.update_norm,
                    "attack_label": norm_labels[idx] if idx < len(norm_labels) else "unknown",
                }
            )

        if "plain_fedavg" in baselines:
            row = {
                "run_id": run_id,
                "round": round_id,
                "baseline": "plain_fedavg",
                "scheme": "plaintext",
                "train_ms": train_ms,
                "round_ms": train_ms,
                "num_clients": len(client_updates),
                "slot_utilization": 0.0,
                "bundle_count": 0,
                "serialized_ciphertext_bytes_per_client": 0,
                "raw_ciphertext_bytes_per_client": 0,
                "plain_float32_bytes_per_client": sum(t.numel() for t in plain_aggregate.values()) * 4,
                "pack_l2": 0.0,
                "pack_linf": 0.0,
                "pack_mse": 0.0,
                "he_l2": 0.0,
                "he_linf": 0.0,
                "he_mse": 0.0,
                "total_l2": 0.0,
                "total_linf": 0.0,
                "total_mse": 0.0,
            }
            round_rows.append(row)
            error_rows.append({"run_id": run_id, "round": round_id, "baseline": "plain_fedavg", **{k: row[k] for k in row if k.endswith(("_l2", "_linf", "_mse"))}})

        for baseline in baselines:
            if baseline == "plain_fedavg":
                continue
            layout_cfg = baseline_layout_config(baseline, config.get("layout", {}))
            layout = planner.plan(
                global_state,
                strategy=layout_cfg.get("strategy", "layer_order"),
                template_policy=layout_cfg.get("template_policy", "bucketed_fixed"),
                target_bundle_count=layout_cfg.get("target_bundle_count"),
            )
            backend = SimHEBackend(
                crypto=crypto,
                quant_bits=int(sim_he_cfg.get("quant_bits", 0) or 0),
                clip_norm=sim_he_cfg.get("clip_norm"),
                add_noise=bool(sim_he_cfg.get("add_noise", True)),
                seed=int(config.get("seed", 2026)) + round_id,
                calibration_path=calibration_path,
            )
            sim_result = backend.aggregate_updates(weighted_updates, layout, reference=plain_aggregate)
            errors = error_decomposition(plain_aggregate, sim_result.pack_plain_update, sim_result.aggregate_update)
            round_ms = train_ms + sum(sim_result.timings.values())
            row = {
                "run_id": run_id,
                "round": round_id,
                "baseline": baseline,
                "scheme": "SimHE",
                "train_ms": train_ms,
                "round_ms": round_ms,
                "num_clients": len(client_updates),
                "bundle_count": layout.bundle_count,
                "ciphertext_count": layout.bundle_count,
                "slot_utilization": layout.slot_utilization,
                "padding_slots": layout.padding_slots,
                "template_id": layout.template_id,
                **sim_result.timings,
                **sim_result.bytes,
                **errors,
                "he_backend_mode": he_backend_mode,
                "calibration_source": sim_result.metadata.get("calibration_source"),
            }
            round_rows.append(row)
            error_rows.append({"run_id": run_id, "round": round_id, "baseline": baseline, **errors})
            if round_id == 1:
                layout_rows.append({"baseline": baseline, **dataclass_to_dict(layout)})
            metadata = _metadata_records(
                run_id=run_id,
                round_id=round_id,
                baseline=baseline,
                layout=layout,
                client_updates=client_updates,
                labels=norm_labels,
                crypto_serialized_ct_bytes=crypto.serialized_ct_bytes,
            )
            auditor.extend(metadata)

        apply_update(model, plain_aggregate, scale=1.0)
        global_state = clone_state_dict(model)

    final_eval = evaluate(model, eval_loader, device)
    leakage_report = auditor.summarize()
    metadata_rows = [r.to_dict() for r in auditor.records]
    summary = summarize_rows(round_rows)
    for row in summary:
        row["initial_accuracy"] = initial_eval["accuracy"]
        row["final_accuracy"] = final_eval["accuracy"]
        row["initial_loss"] = initial_eval["loss"]
        row["final_loss"] = final_eval["loss"]

    write_rows_csv(result_dir / "round_trace.csv", round_rows)
    write_rows_csv(result_dir / "client_metrics.csv", client_rows)
    write_rows_csv(result_dir / "summary.csv", summary)
    write_rows_csv(result_dir / "error_decomposition.csv", error_rows)
    write_rows_csv(result_dir / "metadata_features.csv", metadata_rows)
    write_json(result_dir / "layout_plan.json", layout_rows)
    write_json(result_dir / "leakage_report.json", leakage_report)
    write_json(result_dir / "run_summary.json", {"initial_eval": initial_eval, "final_eval": final_eval, "summary": summary})
    write_json(
        result_dir / "manifest.json",
        {
            "kind": "single_run",
            "run_id": run_id,
            "run_name": config.get("run_name", "run"),
            "seed": config.get("seed"),
            "result_dir": str(result_dir),
            "config_path": str(config_path),
            "dataset": dataset_info,
            "model": config.get("model", {}),
            "fed": config.get("fed", {}),
            "he_backend": he_backend_cfg,
            "artifacts": [
                "config.json",
                "config_resolved.json",
                "hardware.json",
                "round_trace.csv",
                "client_metrics.csv",
                "summary.csv",
                "layout_plan.json",
                "run_summary.json",
                "report.md",
                "error_decomposition.csv",
                "metadata_features.csv",
                "leakage_report.json",
            ],
        },
    )
    plot_paths = plot_results(result_dir)
    artifacts = [
        "config_resolved.json",
        "round_trace.csv",
        "client_metrics.csv",
        "summary.csv",
        "layout_plan.json",
        "error_decomposition.csv",
        "metadata_features.csv",
        "leakage_report.json",
        *[Path(p).name for p in plot_paths],
    ]
    write_markdown_report(result_dir / "report.md", config_out, summary, leakage_report, artifacts)
    print(result_dir)
    return result_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight SimHE HE-FL runtime experiment.")
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
