from __future__ import annotations

from typing import Any, Dict

import torch

from .utils import flatten_state_dict


def vector_error(reference: torch.Tensor, candidate: torch.Tensor, prefix: str = "error") -> Dict[str, float]:
    ref = reference.detach().cpu().float().reshape(-1)
    cand = candidate.detach().cpu().float().reshape(-1)
    if ref.numel() != cand.numel():
        raise ValueError(f"Shape mismatch in vector_error: {ref.numel()} vs {cand.numel()}")
    diff = cand - ref
    return {
        f"{prefix}_l2": float(torch.linalg.vector_norm(diff).item()),
        f"{prefix}_linf": float(torch.max(torch.abs(diff)).item()) if diff.numel() else 0.0,
        f"{prefix}_mse": float(torch.mean(diff * diff).item()) if diff.numel() else 0.0,
    }


def state_error(
    reference: Dict[str, torch.Tensor],
    candidate: Dict[str, torch.Tensor],
    prefix: str = "error",
) -> Dict[str, float]:
    return vector_error(flatten_state_dict(reference), flatten_state_dict(candidate), prefix=prefix)


def error_decomposition(
    plain: Dict[str, torch.Tensor],
    packed_plain: Dict[str, torch.Tensor],
    simhe: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    row: Dict[str, float] = {}
    row.update({"clip_l2": 0.0, "clip_linf": 0.0, "clip_mse": 0.0})
    row.update({"quant_l2": 0.0, "quant_linf": 0.0, "quant_mse": 0.0})
    row.update(state_error(plain, packed_plain, prefix="pack"))
    row.update(state_error(packed_plain, simhe, prefix="he"))
    row.update(state_error(plain, simhe, prefix="total"))
    return row


def summarize_update(update: Dict[str, torch.Tensor], prefix: str = "update") -> Dict[str, float]:
    flat = flatten_state_dict(update)
    if flat.numel() == 0:
        return {f"{prefix}_l2": 0.0, f"{prefix}_mean": 0.0, f"{prefix}_max_abs": 0.0}
    return {
        f"{prefix}_l2": float(torch.linalg.vector_norm(flat.float()).item()),
        f"{prefix}_mean": float(torch.mean(flat.float()).item()),
        f"{prefix}_max_abs": float(torch.max(torch.abs(flat.float())).item()),
    }
