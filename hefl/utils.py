from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_hash(data: Any, length: int = 12) -> str:
    blob = json.dumps(data, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:length]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def flatten_state_dict(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    parts = []
    for _, tensor in state_dict.items():
        parts.append(tensor.detach().reshape(-1).cpu().float())
    if not parts:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(parts)


def unflatten_to_state_dict(flat: torch.Tensor, reference: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    pos = 0
    flat_cpu = flat.detach().cpu()
    for name, tensor in reference.items():
        n = tensor.numel()
        out[name] = flat_cpu[pos : pos + n].reshape(tensor.shape).to(dtype=tensor.dtype)
        pos += n
    if pos != flat_cpu.numel():
        raise ValueError(f"Unflatten consumed {pos} elements but flat has {flat_cpu.numel()}")
    return out


def add_state_dict(base: Dict[str, torch.Tensor], update: Dict[str, torch.Tensor], scale: float = 1.0) -> Dict[str, torch.Tensor]:
    return {k: base[k].detach().cpu() + update[k].detach().cpu() * scale for k in base}


def scale_state_dict(update: Dict[str, torch.Tensor], scale: float) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().float() * float(scale) for k, v in update.items()}


def sub_state_dict(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: a[k].detach().cpu() - b[k].detach().cpu() for k in a}


def state_dict_numel(state_dict: Dict[str, torch.Tensor]) -> int:
    return int(sum(t.numel() for t in state_dict.values()))


def weighted_average_updates(updates: list[Dict[str, torch.Tensor]], weights: list[float]) -> Dict[str, torch.Tensor]:
    if not updates:
        raise ValueError("No updates to average")
    out = {k: torch.zeros_like(v.detach().cpu(), dtype=torch.float32) for k, v in updates[0].items()}
    for update, weight in zip(updates, weights):
        for name, tensor in update.items():
            out[name] += tensor.detach().cpu().float() * float(weight)
    return out
