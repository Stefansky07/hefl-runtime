from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


@dataclass
class PartitionResult:
    indices: List[List[int]]
    label_histograms: List[Dict[int, int]]
    sample_counts: List[int]


def _dataset_targets(dataset: Dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        return np.asarray(getattr(dataset, "targets"), dtype=np.int64)
    labels = []
    for i in range(len(dataset)):
        _, target = dataset[i]
        labels.append(int(target))
    return np.asarray(labels, dtype=np.int64)


def _maybe_subset(dataset: Dataset, max_samples: int | None, seed: int) -> Dataset:
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    return Subset(dataset, indices[: int(max_samples)].astype(int).tolist())


def _load_real_dataset(
    name: str,
    root: Path,
    download: bool,
    transform: Any,
) -> Tuple[Dataset, Dataset, Dict[str, Any]]:
    if name in {"cifar10", "cifar-10"}:
        train = datasets.CIFAR10(root=str(root), train=True, download=download, transform=transform)
        test = datasets.CIFAR10(root=str(root), train=False, download=download, transform=transform)
        return train, test, {"source": "CIFAR10", "image_size": [3, 32, 32], "num_classes": 10}
    if name in {"fashionmnist", "fashion-mnist", "fashion_mnist"}:
        train = datasets.FashionMNIST(root=str(root), train=True, download=download, transform=transform)
        test = datasets.FashionMNIST(root=str(root), train=False, download=download, transform=transform)
        return train, test, {"source": "FashionMNIST", "image_size": [1, 28, 28], "num_classes": 10}
    if name in {"mnist"}:
        train = datasets.MNIST(root=str(root), train=True, download=download, transform=transform)
        test = datasets.MNIST(root=str(root), train=False, download=download, transform=transform)
        return train, test, {"source": "MNIST", "image_size": [1, 28, 28], "num_classes": 10}
    raise ValueError(f"Unknown real dataset: {name}")


def make_dataset(config: Dict[str, Any]) -> Tuple[Dataset, Dataset, Dict[str, Any]]:
    dcfg = config.get("dataset", {})
    name = str(dcfg.get("name", "fake_cifar")).lower()
    num_samples = int(dcfg.get("num_samples", 512))
    num_classes = int(dcfg.get("num_classes", 10))
    image_size = tuple(int(x) for x in dcfg.get("image_size", [3, 32, 32]))
    seed = int(config.get("seed", 2026))
    root = Path(dcfg.get("root", "data"))
    download = bool(dcfg.get("download", False))
    allow_fake_fallback = bool(dcfg.get("allow_fake_fallback", True))
    subset_train = dcfg.get("subset_train")
    subset_test = dcfg.get("subset_test")
    transform = transforms.Compose([transforms.ToTensor()])

    if name not in {"fake_cifar", "fake", "fakedata"}:
        try:
            train, test, info = _load_real_dataset(name, root=root, download=download, transform=transform)
            train = _maybe_subset(train, int(subset_train) if subset_train else None, seed=seed)
            test = _maybe_subset(test, int(subset_test) if subset_test else None, seed=seed + 100_000)
            info.update(
                {
                    "root": str(root),
                    "download": download,
                    "allow_fake_fallback": allow_fake_fallback,
                    "train_size": len(train),
                    "test_size": len(test),
                    "transform": "ToTensor",
                    "subset_train": subset_train,
                    "subset_test": subset_test,
                }
            )
            return train, test, info
        except Exception as exc:
            if not allow_fake_fallback:
                raise RuntimeError(
                    f"Failed to load dataset {name!r} from {root}. "
                    "Set dataset.download=true on AutoDL or pre-place the data cache."
                ) from exc
            if name in {"cifar10", "cifar-10"}:
                image_size = (3, 32, 32)
                num_classes = 10
            elif name in {"fashionmnist", "fashion-mnist", "fashion_mnist", "mnist"}:
                image_size = (1, 28, 28)
                num_classes = 10

    train = datasets.FakeData(
        size=num_samples,
        image_size=image_size,
        num_classes=num_classes,
        transform=transform,
        random_offset=seed,
    )
    test = datasets.FakeData(
        size=max(128, num_samples // 4),
        image_size=image_size,
        num_classes=num_classes,
        transform=transform,
        random_offset=seed + 100_000,
    )
    return train, test, {
        "source": "FakeData",
        "image_size": list(image_size),
        "num_classes": num_classes,
        "root": str(root),
        "download": download,
        "allow_fake_fallback": allow_fake_fallback,
        "train_size": len(train),
        "test_size": len(test),
        "transform": "ToTensor",
        "subset_train": subset_train,
        "subset_test": subset_test,
    }


def label_histogram(dataset: Dataset, indices: Sequence[int], num_classes: int) -> Dict[int, int]:
    targets = _dataset_targets(dataset)
    counts = Counter(int(targets[i]) for i in indices)
    return {label: int(counts.get(label, 0)) for label in range(num_classes)}


def partition_iid(dataset: Dataset, num_clients: int, seed: int, num_classes: int) -> PartitionResult:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    splits = [arr.astype(int).tolist() for arr in np.array_split(indices, num_clients)]
    return PartitionResult(
        indices=splits,
        label_histograms=[label_histogram(dataset, split, num_classes) for split in splits],
        sample_counts=[len(split) for split in splits],
    )


def partition_dirichlet(
    dataset: Dataset,
    num_clients: int,
    alpha: float,
    seed: int,
    num_classes: int,
    min_samples_per_client: int = 1,
) -> PartitionResult:
    targets = _dataset_targets(dataset)
    rng = np.random.default_rng(seed)
    min_samples = max(0, int(min_samples_per_client))
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    for _ in range(100):
        client_indices = [[] for _ in range(num_clients)]
        for label in range(num_classes):
            label_indices = np.where(targets == label)[0]
            rng.shuffle(label_indices)
            if label_indices.size == 0:
                continue
            proportions = rng.dirichlet(np.full(num_clients, max(alpha, 1e-3)))
            cuts = (np.cumsum(proportions)[:-1] * label_indices.size).astype(int)
            for client_id, part in enumerate(np.split(label_indices, cuts)):
                client_indices[client_id].extend(int(i) for i in part)
        if all(len(split) >= min_samples for split in client_indices):
            break
    while min_samples and any(len(split) < min_samples for split in client_indices):
        receiver = min(range(num_clients), key=lambda idx: len(client_indices[idx]))
        donor = max(range(num_clients), key=lambda idx: len(client_indices[idx]))
        if len(client_indices[donor]) <= min_samples:
            break
        moved_at = int(rng.integers(0, len(client_indices[donor])))
        client_indices[receiver].append(client_indices[donor].pop(moved_at))
    for split in client_indices:
        rng.shuffle(split)
    return PartitionResult(
        indices=client_indices,
        label_histograms=[label_histogram(dataset, split, num_classes) for split in client_indices],
        sample_counts=[len(split) for split in client_indices],
    )


def make_partitions(config: Dict[str, Any], dataset: Dataset, num_classes: int) -> PartitionResult:
    fed = config.get("fed", {})
    num_clients = int(fed.get("num_clients", 4))
    partition = str(fed.get("partition", "iid")).lower()
    seed = int(config.get("seed", 2026))
    if partition == "dirichlet":
        return partition_dirichlet(
            dataset=dataset,
            num_clients=num_clients,
            alpha=float(fed.get("dirichlet_alpha", 0.5)),
            seed=seed,
            num_classes=num_classes,
            min_samples_per_client=int(fed.get("min_samples_per_client", 1)),
        )
    return partition_iid(dataset=dataset, num_clients=num_clients, seed=seed, num_classes=num_classes)


def make_client_loaders(
    dataset: Dataset,
    partitions: PartitionResult,
    batch_size: int,
    seed: int,
) -> List[DataLoader]:
    loaders = []
    generator = torch.Generator()
    generator.manual_seed(seed)
    for split in partitions.indices:
        loaders.append(
            DataLoader(
                Subset(dataset, split),
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                generator=generator,
            )
        )
    return loaders


def make_eval_loader(dataset: Dataset, batch_size: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
