from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional

import torch

from .types import CryptoParams, LayoutPlan, SegmentSpec, TensorSpec
from .utils import stable_hash


def tensor_specs_from_state(state_dict: Dict[str, torch.Tensor]) -> List[TensorSpec]:
    specs: List[TensorSpec] = []
    offset = 0
    for name, tensor in state_dict.items():
        numel = int(tensor.numel())
        specs.append(TensorSpec(name=name, shape=tuple(tensor.shape), numel=numel, offset=offset))
        offset += numel
    return specs


class HeuristicLayoutPlanner:
    def __init__(
        self,
        crypto: CryptoParams,
        bundle_buckets: Optional[Iterable[int]] = None,
    ) -> None:
        self.crypto = crypto
        self.bundle_buckets = list(bundle_buckets or [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])

    def _bundle_count(self, num_params: int, template_policy: str, target_bundle_count: Optional[int]) -> int:
        tight = max(1, math.ceil(num_params / self.crypto.slots))
        policy = template_policy.lower()
        if policy in {"tight", "layer_order", "manual_packed"}:
            return tight
        if policy in {"padded_fixed", "fixed"}:
            if target_bundle_count is None:
                return tight
            if target_bundle_count < tight:
                raise ValueError(f"target_bundle_count={target_bundle_count} is smaller than tight count={tight}")
            return int(target_bundle_count)
        if policy in {"bucketed_fixed", "fixed_template", "fixed_template_runtime"}:
            for bucket in self.bundle_buckets:
                if bucket >= tight:
                    return int(bucket)
            return tight
        if policy in {"dynamic", "dynamic_template", "dynamic_template_ablation"}:
            return tight
        raise ValueError(f"Unknown template policy: {template_policy}")

    def plan(
        self,
        state_dict: Dict[str, torch.Tensor],
        strategy: str = "layer_order",
        template_policy: str = "bucketed_fixed",
        target_bundle_count: Optional[int] = None,
        public_bundle_count: Optional[int] = None,
    ) -> LayoutPlan:
        specs = tensor_specs_from_state(state_dict)
        num_params = int(sum(s.numel for s in specs))
        bundle_count = self._bundle_count(num_params, template_policy, target_bundle_count)
        if public_bundle_count is not None:
            if public_bundle_count < bundle_count:
                raise ValueError("public_bundle_count cannot be smaller than actual bundle count")
            bundle_count = int(public_bundle_count)

        segments: List[SegmentSpec] = []
        for spec in specs:
            remaining = spec.numel
            tensor_pos = 0
            flat_pos = spec.offset
            while remaining > 0:
                bundle_id = flat_pos // self.crypto.slots
                slot_start = flat_pos % self.crypto.slots
                take = min(remaining, self.crypto.slots - slot_start)
                segments.append(
                    SegmentSpec(
                        tensor_name=spec.name,
                        tensor_start=tensor_pos,
                        flat_start=flat_pos,
                        length=take,
                        bundle_id=bundle_id,
                        slot_start=slot_start,
                    )
                )
                tensor_pos += take
                flat_pos += take
                remaining -= take

        used_slots = num_params
        total_slots = bundle_count * self.crypto.slots
        padding_slots = total_slots - used_slots
        if padding_slots < 0:
            raise ValueError("Layout has negative padding; bundle count is inconsistent")
        public_template = {
            "scheme": "SimHE-CKKS-like",
            "raw_ct_bytes": self.crypto.raw_ct_bytes,
            "serialized_ct_bytes": self.crypto.serialized_ct_bytes,
            "bundle_count": bundle_count,
            "serialized_bytes": bundle_count * self.crypto.serialized_ct_bytes,
            "occupancy_bucket": round(used_slots / max(total_slots, 1), 6),
            "template_policy": template_policy,
        }
        template_payload = {
            "strategy": strategy,
            "template_policy": template_policy,
            "slots": self.crypto.slots,
            "bundle_count": bundle_count,
            "specs": [(s.name, s.shape, s.numel, s.offset) for s in specs],
        }
        template_id = stable_hash(template_payload, length=12)
        return LayoutPlan(
            strategy=strategy,
            template_policy=template_policy,
            template_id=template_id,
            slot_capacity=self.crypto.slots,
            num_params=num_params,
            bundle_count=bundle_count,
            used_slots=used_slots,
            padding_slots=padding_slots,
            slot_utilization=used_slots / max(total_slots, 1),
            tensor_specs=specs,
            segments=segments,
            public_template=public_template,
        )


def baseline_layout_config(baseline: str, base_layout_config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(base_layout_config)
    name = baseline.lower()
    if name == "simhe_layer_order":
        cfg["strategy"] = "layer_order"
        cfg["template_policy"] = "tight"
        cfg["target_bundle_count"] = None
    elif name == "simhe_manual_packed":
        cfg["strategy"] = "manual_packed"
        cfg["template_policy"] = "tight"
        cfg["target_bundle_count"] = None
    elif name == "simhe_fixed_template_runtime":
        cfg["strategy"] = "fixed_template_runtime"
        cfg["template_policy"] = cfg.get("template_policy", "bucketed_fixed")
    elif name == "dynamic_template_or_selective":
        cfg["strategy"] = "dynamic_template_ablation"
        cfg["template_policy"] = "dynamic_template_ablation"
    return cfg
