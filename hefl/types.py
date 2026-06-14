from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class CryptoParams:
    scheme: str = "CKKS"
    poly_modulus_degree: int = 8192
    slots: int = 4096
    coeff_mod_bit_sizes: Tuple[int, ...] = (60, 40, 40, 60)
    ciphertext_size: int = 2
    scale_bits: int = 40
    security_level: str = "128-bit nominal"
    serialized_compression_ratio: float = 2.0
    encode_per_ct_ms: float = 0.20
    encrypt_per_ct_ms: float = 0.80
    decrypt_per_ct_ms: float = 0.50
    ct_add_per_ct_us: float = 20.0
    serializer_mbps: float = 500.0
    network_mbps: float = 100.0
    network_rtt_ms: float = 0.0
    noise_multiplier: float = 1024.0

    @property
    def num_primes(self) -> int:
        return len(self.coeff_mod_bit_sizes)

    @property
    def raw_ct_bytes(self) -> int:
        return 8 * self.poly_modulus_degree * self.num_primes * self.ciphertext_size

    @property
    def serialized_ct_bytes(self) -> int:
        return int(math.ceil(self.raw_ct_bytes / self.serialized_compression_ratio))

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["num_primes"] = self.num_primes
        data["raw_ct_bytes"] = self.raw_ct_bytes
        data["serialized_ct_bytes"] = self.serialized_ct_bytes
        return data


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: Tuple[int, ...]
    numel: int
    offset: int


@dataclass(frozen=True)
class SegmentSpec:
    tensor_name: str
    tensor_start: int
    flat_start: int
    length: int
    bundle_id: int
    slot_start: int


@dataclass
class LayoutPlan:
    strategy: str
    template_policy: str
    template_id: str
    slot_capacity: int
    num_params: int
    bundle_count: int
    used_slots: int
    padding_slots: int
    slot_utilization: float
    tensor_specs: List[TensorSpec] = field(default_factory=list)
    segments: List[SegmentSpec] = field(default_factory=list)
    slot_mapping: List[Dict[str, Any]] = field(default_factory=list)
    public_template: Dict[str, Any] = field(default_factory=dict)

    @property
    def raw_ciphertext_bytes(self) -> int:
        raw_per_ct = int(self.public_template.get("raw_ct_bytes", 0))
        return self.bundle_count * raw_per_ct

    @property
    def serialized_ciphertext_bytes(self) -> int:
        serialized_per_ct = int(self.public_template.get("serialized_ct_bytes", 0))
        return self.bundle_count * serialized_per_ct

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["raw_ciphertext_bytes"] = self.raw_ciphertext_bytes
        data["serialized_ciphertext_bytes"] = self.serialized_ciphertext_bytes
        return data


@dataclass
class SimCiphertext:
    bundle_id: int
    slots: Any
    used_slots: int
    padding_slots: int
    raw_bytes: int
    serialized_bytes: int
    scale_bits: int
    level: int
    noise_std: float
    timing: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    run_id: str
    result_dir: str
    summary: Dict[str, Any]


def dataclass_to_dict(obj: Any) -> Any:
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [dataclass_to_dict(v) for v in obj]
    return obj
