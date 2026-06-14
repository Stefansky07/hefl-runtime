from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List

import numpy as np


@dataclass
class MetadataRecord:
    run_id: str
    round: int
    baseline: str
    client_id: int
    template_mode: str
    template_id: str
    bundle_count: int
    serialized_bytes: int
    slot_utilization: float
    padding_slots: int
    update_norm: float
    attack_label: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LeakageAuditor:
    def __init__(self) -> None:
        self.records: List[MetadataRecord] = []

    def add(self, record: MetadataRecord) -> None:
        self.records.append(record)

    def extend(self, records: Iterable[MetadataRecord]) -> None:
        for record in records:
            self.add(record)

    @staticmethod
    def _nearest_centroid_accuracy(rows: List[MetadataRecord]) -> float:
        labels = sorted(set(r.attack_label for r in rows))
        if len(labels) < 2 or len(rows) < 3:
            return 0.0
        features = np.asarray(
            [[r.bundle_count, r.serialized_bytes / 1024.0, r.slot_utilization, r.padding_slots] for r in rows],
            dtype=np.float64,
        )
        y = np.asarray([labels.index(r.attack_label) for r in rows], dtype=np.int64)
        correct = 0
        total = 0
        for i in range(len(rows)):
            train_mask = np.arange(len(rows)) != i
            centroids = []
            for label_idx in range(len(labels)):
                subset = features[train_mask & (y == label_idx)]
                if subset.size == 0:
                    centroids.append(np.full(features.shape[1], np.inf))
                else:
                    centroids.append(subset.mean(axis=0))
            distances = [float(np.linalg.norm(features[i] - c)) for c in centroids]
            pred = int(np.argmin(distances))
            correct += int(pred == int(y[i]))
            total += 1
        return correct / max(total, 1)

    def summarize(self) -> Dict[str, Any]:
        rows = self.records
        if not rows:
            return {"num_records": 0}
        out: Dict[str, Any] = {"num_records": len(rows), "by_template_mode": {}, "by_baseline_template_mode": {}}

        def summarize_subset(subset: List[MetadataRecord]) -> Dict[str, Any]:
            bundles = np.asarray([r.bundle_count for r in subset], dtype=np.float64)
            bytes_ = np.asarray([r.serialized_bytes for r in subset], dtype=np.float64)
            occ = np.asarray([r.slot_utilization for r in subset], dtype=np.float64)
            templates = [r.template_id for r in subset]
            most_common = max(templates.count(t) for t in set(templates))
            return {
                "records": len(subset),
                "bundle_count_variance": float(np.var(bundles)),
                "serialized_bytes_variance": float(np.var(bytes_)),
                "slot_utilization_variance": float(np.var(occ)),
                "template_stability_rate": float(most_common / max(len(subset), 1)),
                "toy_attack_accuracy": float(self._nearest_centroid_accuracy(subset)),
                "leakage_surrogate_score": float(np.var(bytes_) / (float(np.mean(bytes_)) + 1e-9) + np.var(occ)),
            }

        for mode in sorted(set(r.template_mode for r in rows)):
            subset = [r for r in rows if r.template_mode == mode]
            out["by_template_mode"][mode] = summarize_subset(subset)
        for baseline in sorted(set(r.baseline for r in rows)):
            for mode in sorted(set(r.template_mode for r in rows if r.baseline == baseline)):
                subset = [r for r in rows if r.baseline == baseline and r.template_mode == mode]
                out["by_baseline_template_mode"][f"{baseline}:{mode}"] = summarize_subset(subset)
        return out
