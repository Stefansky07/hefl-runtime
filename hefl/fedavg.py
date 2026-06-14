from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from .utils import add_state_dict, sub_state_dict, weighted_average_updates


@dataclass
class ClientUpdate:
    client_id: int
    update: Dict[str, torch.Tensor]
    sample_count: int
    train_loss: float
    train_ms: float
    update_norm: float


def clone_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_state_dict_cpu(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    model.load_state_dict({k: v.detach().cpu() for k, v in state.items()})


def apply_update(model: nn.Module, update: Dict[str, torch.Tensor], scale: float = 1.0) -> None:
    current = clone_state_dict(model)
    model.load_state_dict(add_state_dict(current, update, scale=scale))


def update_l2_norm(update: Dict[str, torch.Tensor]) -> float:
    total = 0.0
    for tensor in update.values():
        t = tensor.detach().cpu().float()
        total += float(torch.sum(t * t).item())
    return total**0.5


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss_sum += float(criterion(logits, y).item())
            correct += int((logits.argmax(dim=1) == y).sum().item())
            total += int(y.numel())
    return {
        "loss": loss_sum / max(total, 1),
        "accuracy": correct / max(total, 1),
        "num_examples": total,
    }


def _synthetic_update(
    state: Dict[str, torch.Tensor],
    client_id: int,
    seed: int,
    scale: float = 1e-4,
) -> Dict[str, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + client_id * 7919)
    out: Dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        if tensor.is_floating_point():
            out[name] = torch.randn(tensor.shape, generator=gen, dtype=torch.float32) * scale
        else:
            out[name] = torch.zeros_like(tensor.detach().cpu(), dtype=torch.float32)
    return out


def local_train(
    global_model: nn.Module,
    global_state: Dict[str, torch.Tensor],
    loader: DataLoader,
    client_id: int,
    device: torch.device,
    lr: float,
    local_epochs: int,
    seed: int,
) -> ClientUpdate:
    sample_count = len(loader.dataset)
    if local_epochs <= 0:
        update = _synthetic_update(global_state, client_id=client_id, seed=seed)
        return ClientUpdate(
            client_id=client_id,
            update=update,
            sample_count=sample_count,
            train_loss=0.0,
            train_ms=0.0,
            update_norm=update_l2_norm(update),
        )

    model = copy.deepcopy(global_model).to(device)
    model.load_state_dict(global_state)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    loss_sum = 0.0
    seen = 0
    start = time.perf_counter()
    for _ in range(local_epochs):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * int(y.numel())
            seen += int(y.numel())
    train_ms = (time.perf_counter() - start) * 1000.0
    local_state = clone_state_dict(model)
    update = sub_state_dict(local_state, global_state)
    return ClientUpdate(
        client_id=client_id,
        update=update,
        sample_count=sample_count,
        train_loss=loss_sum / max(seen, 1),
        train_ms=train_ms,
        update_norm=update_l2_norm(update),
    )


def run_client_round(
    global_model: nn.Module,
    global_state: Dict[str, torch.Tensor],
    client_loaders: Sequence[DataLoader],
    selected_clients: Sequence[int],
    device: torch.device,
    lr: float,
    local_epochs: int,
    seed: int,
) -> List[ClientUpdate]:
    updates = []
    for client_id in selected_clients:
        updates.append(
            local_train(
                global_model=global_model,
                global_state=global_state,
                loader=client_loaders[client_id],
                client_id=client_id,
                device=device,
                lr=lr,
                local_epochs=local_epochs,
                seed=seed,
            )
        )
    return updates


def fedavg_plain(client_updates: Sequence[ClientUpdate]) -> Dict[str, torch.Tensor]:
    total = sum(u.sample_count for u in client_updates)
    weights = [u.sample_count / max(total, 1) for u in client_updates]
    return weighted_average_updates([u.update for u in client_updates], weights)
