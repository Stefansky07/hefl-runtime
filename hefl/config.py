from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any, Dict

import torch

from .types import CryptoParams
from .utils import read_json


DEFAULT_CONFIG: Dict[str, Any] = {
    "run_name": "smoke",
    "seed": 2026,
    "device": "auto",
    "dataset": {
        "name": "fake_cifar",
        "root": "data",
        "download": False,
        "allow_fake_fallback": True,
        "subset_train": None,
        "subset_test": None,
        "num_samples": 512,
        "num_classes": 10,
        "image_size": [3, 32, 32],
        "batch_size": 64,
    },
    "model": {"name": "tiny_cnn"},
    "fed": {
        "num_clients": 4,
        "rounds": 2,
        "local_epochs": 1,
        "partition": "iid",
        "dirichlet_alpha": 0.5,
        "clients_per_round": 4,
        "lr": 0.05,
        "dropout_rate": 0.0,
        "quorum": 1,
    },
    "layout": {
        "strategy": "layer_order",
        "template_policy": "bucketed_fixed",
        "target_bundle_count": None,
        "bundle_buckets": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
    },
    "sim_he": {
        "enabled": True,
        "quant_bits": 0,
        "clip_norm": None,
        "add_noise": True,
        "network_profile": "lan",
    },
    "he_backend": {
        "mode": "simhe",
        "calibration_path": None,
    },
    "experiment": {
        "seeds": [2026],
    },
    "baselines": ["plain_fedavg", "simhe_layer_order", "simhe_manual_packed", "simhe_fixed_template_runtime"],
}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> Dict[str, Any]:
    return deep_merge(DEFAULT_CONFIG, read_json(path))


def resolve_device(config: Dict[str, Any]) -> torch.device:
    requested = config.get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def crypto_params_from_config(config: Dict[str, Any]) -> CryptoParams:
    crypto = config.get("crypto", {})
    return CryptoParams(**crypto)


def environment_info(device: torch.device) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "gpu_name": props.name,
                "gpu_total_memory_gb": round(props.total_memory / 1024**3, 3),
                "torch_cuda": torch.version.cuda,
            }
        )
    return info
