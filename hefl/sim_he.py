from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

from .types import CryptoParams, LayoutPlan, SimCiphertext
from .utils import flatten_state_dict, unflatten_to_state_dict


@dataclass
class SimHEResult:
    aggregate_update: Dict[str, torch.Tensor]
    pack_plain_update: Dict[str, torch.Tensor]
    ciphertexts: List[SimCiphertext]
    aggregate_ciphertexts: List[SimCiphertext]
    timings: Dict[str, float]
    bytes: Dict[str, int]
    metadata: Dict[str, Any]


class SimHEBackend:
    def __init__(
        self,
        crypto: CryptoParams,
        quant_bits: int = 0,
        clip_norm: float | None = None,
        add_noise: bool = True,
        seed: int = 2026,
        calibration_path: str | Path | None = None,
    ) -> None:
        self.crypto = crypto
        self.quant_bits = int(quant_bits or 0)
        self.clip_norm = clip_norm
        self.add_noise = bool(add_noise)
        self.seed = int(seed)
        self.calibration = self._load_calibration(calibration_path)

    def _load_calibration(self, calibration_path: str | Path | None) -> Dict[str, Any]:
        if not calibration_path:
            return {}
        path = Path(calibration_path)
        with path.open("r", encoding="utf-8") as f:
            import json

            data = json.load(f)
        data.setdefault("path", str(path))
        return data

    def _calibrated_per_ct(self) -> Dict[str, float]:
        per_ct = self.calibration.get("per_ct", {}) if self.calibration else {}
        encode_encrypt = float(per_ct.get("encode_encrypt_ms", 0.0) or 0.0)
        encode_ms = float(per_ct.get("encode_ms", encode_encrypt * 0.25 if encode_encrypt else self.crypto.encode_per_ct_ms))
        encrypt_ms = float(per_ct.get("encrypt_ms", encode_encrypt * 0.75 if encode_encrypt else self.crypto.encrypt_per_ct_ms))
        serialize_ms = float(per_ct.get("serialize_ms", self.crypto.serialized_ct_bytes / (self.crypto.serializer_mbps * 1024 * 1024) * 1000.0))
        upload_ms = self.serialized_ct_bytes / (self.crypto.network_mbps * 1024 * 1024) * 1000.0
        decrypt_ms = float(per_ct.get("decrypt_ms", self.crypto.decrypt_per_ct_ms))
        add_ms = float(per_ct.get("add_ms", self.crypto.ct_add_per_ct_us / 1000.0))
        return {
            "encode_ms": encode_ms,
            "encrypt_ms": encrypt_ms,
            "serialize_ms": serialize_ms,
            "upload_ms": upload_ms,
            "decrypt_ms": decrypt_ms,
            "add_ms": add_ms,
        }

    @property
    def serialized_ct_bytes(self) -> int:
        per_ct = self.calibration.get("per_ct", {}) if self.calibration else {}
        return int(per_ct.get("serialized_ct_bytes", self.crypto.serialized_ct_bytes))

    def _clip_flat(self, flat: torch.Tensor) -> torch.Tensor:
        if self.clip_norm is None:
            return flat
        norm = torch.linalg.vector_norm(flat.float())
        if float(norm) <= float(self.clip_norm):
            return flat
        return flat * (float(self.clip_norm) / (float(norm) + 1e-12))

    def _quantize_flat(self, flat: torch.Tensor) -> torch.Tensor:
        if self.quant_bits <= 0:
            return flat.float()
        qmax = (2 ** (self.quant_bits - 1)) - 1
        max_abs = float(torch.max(torch.abs(flat)).item())
        if max_abs == 0.0:
            return flat.float()
        scale = qmax / max_abs
        return torch.round(flat * scale).clamp(-qmax, qmax) / scale

    def preprocess(self, update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        flat = flatten_state_dict(update).float()
        flat = self._clip_flat(flat)
        flat = self._quantize_flat(flat)
        return unflatten_to_state_dict(flat, update)

    def pack(self, update: Dict[str, torch.Tensor], layout: LayoutPlan) -> List[torch.Tensor]:
        flat = flatten_state_dict(update).float()
        total_slots = layout.bundle_count * layout.slot_capacity
        if flat.numel() != layout.num_params:
            raise ValueError(f"Update has {flat.numel()} params but layout expects {layout.num_params}")
        padded = torch.zeros(total_slots, dtype=torch.float32)
        padded[: flat.numel()] = flat
        return [padded[i * layout.slot_capacity : (i + 1) * layout.slot_capacity].clone() for i in range(layout.bundle_count)]

    def encode_encrypt(self, bundles: Sequence[torch.Tensor], layout: LayoutPlan) -> List[SimCiphertext]:
        cts: List[SimCiphertext] = []
        per_ct = self._calibrated_per_ct()
        for bundle_id, slots in enumerate(bundles):
            start = bundle_id * layout.slot_capacity
            used = max(0, min(layout.num_params - start, layout.slot_capacity))
            padding = layout.slot_capacity - used
            timing = {
                "encode_ms": per_ct["encode_ms"],
                "encrypt_ms": per_ct["encrypt_ms"],
                "serialize_ms": per_ct["serialize_ms"],
                "upload_ms": per_ct["upload_ms"] + self.crypto.network_rtt_ms / max(layout.bundle_count, 1),
            }
            cts.append(
                SimCiphertext(
                    bundle_id=bundle_id,
                    slots=slots.clone(),
                    used_slots=used,
                    padding_slots=padding,
                    raw_bytes=self.crypto.raw_ct_bytes,
                    serialized_bytes=self.serialized_ct_bytes,
                    scale_bits=self.crypto.scale_bits,
                    level=self.crypto.num_primes - 1,
                    noise_std=0.0,
                    timing=timing,
                    metadata={"calibration_source": self.calibration.get("backend") if self.calibration else "analytic"},
                )
            )
        return cts

    def aggregate(self, client_ciphertexts: Sequence[Sequence[SimCiphertext]], layout: LayoutPlan) -> List[SimCiphertext]:
        if not client_ciphertexts:
            raise ValueError("No ciphertexts to aggregate")
        out: List[SimCiphertext] = []
        per_ct = self._calibrated_per_ct()
        for bundle_id in range(layout.bundle_count):
            summed = torch.zeros(layout.slot_capacity, dtype=torch.float32)
            used = 0
            padding = layout.slot_capacity
            for client_cts in client_ciphertexts:
                ct = client_cts[bundle_id]
                summed += ct.slots.float()
                used = ct.used_slots
                padding = ct.padding_slots
            out.append(
                SimCiphertext(
                    bundle_id=bundle_id,
                    slots=summed,
                    used_slots=used,
                    padding_slots=padding,
                    raw_bytes=self.crypto.raw_ct_bytes,
                    serialized_bytes=self.serialized_ct_bytes,
                    scale_bits=self.crypto.scale_bits,
                    level=self.crypto.num_primes - 1,
                    noise_std=0.0,
                    timing={"aggregate_ms": per_ct["add_ms"] * len(client_ciphertexts)},
                )
            )
        return out

    def decrypt(self, aggregate_ciphertexts: Sequence[SimCiphertext], layout: LayoutPlan) -> List[torch.Tensor]:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed + layout.bundle_count * 104729)
        noise_std = (2.0 ** (-self.crypto.scale_bits)) * self.crypto.noise_multiplier if self.add_noise else 0.0
        out: List[torch.Tensor] = []
        for ct in aggregate_ciphertexts:
            slots = ct.slots.clone().float()
            if noise_std > 0.0:
                slots += torch.randn(slots.shape, generator=gen, dtype=torch.float32) * noise_std
            out.append(slots)
        return out

    def rebuild(self, bundles: Sequence[torch.Tensor], layout: LayoutPlan, reference: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        flat = torch.cat([b.float().reshape(-1) for b in bundles], dim=0)[: layout.num_params]
        return unflatten_to_state_dict(flat, reference)

    def aggregate_updates(
        self,
        weighted_updates: Sequence[Dict[str, torch.Tensor]],
        layout: LayoutPlan,
        reference: Dict[str, torch.Tensor],
    ) -> SimHEResult:
        timings: Dict[str, float] = {
            "preprocess_ms": 0.0,
            "pack_ms": 0.0,
            "encode_sim_ms": 0.0,
            "encrypt_sim_ms": 0.0,
            "serialize_ms": 0.0,
            "tx_sim_ms": 0.0,
            "agg_ms": 0.0,
            "decrypt_sim_ms": 0.0,
            "rebuild_ms": 0.0,
        }
        client_ciphertexts: List[List[SimCiphertext]] = []
        pack_plain_weighted: List[Dict[str, torch.Tensor]] = []
        last_ciphertexts: List[SimCiphertext] = []

        for update in weighted_updates:
            t0 = time.perf_counter()
            processed = self.preprocess(update)
            timings["preprocess_ms"] += (time.perf_counter() - t0) * 1000.0
            t0 = time.perf_counter()
            bundles = self.pack(processed, layout)
            unpacked = self.rebuild(bundles, layout, processed)
            timings["pack_ms"] += (time.perf_counter() - t0) * 1000.0
            pack_plain_weighted.append(unpacked)
            cts = self.encode_encrypt(bundles, layout)
            timings["encode_sim_ms"] += sum(ct.timing.get("encode_ms", 0.0) for ct in cts)
            timings["encrypt_sim_ms"] += sum(ct.timing.get("encrypt_ms", 0.0) for ct in cts)
            timings["serialize_ms"] += sum(ct.timing.get("serialize_ms", 0.0) for ct in cts)
            timings["tx_sim_ms"] += sum(ct.timing.get("upload_ms", 0.0) for ct in cts)
            client_ciphertexts.append(cts)
            last_ciphertexts = cts

        t0 = time.perf_counter()
        aggregate_ciphertexts = self.aggregate(client_ciphertexts, layout)
        timings["agg_ms"] += (time.perf_counter() - t0) * 1000.0
        timings["agg_ms"] += sum(ct.timing.get("aggregate_ms", 0.0) for ct in aggregate_ciphertexts)

        per_ct = self._calibrated_per_ct()
        timings["decrypt_sim_ms"] += layout.bundle_count * per_ct["decrypt_ms"]
        t0 = time.perf_counter()
        decrypted = self.decrypt(aggregate_ciphertexts, layout)
        aggregate_update = self.rebuild(decrypted, layout, reference)
        timings["rebuild_ms"] += (time.perf_counter() - t0) * 1000.0

        pack_plain_slots = self.aggregate(client_ciphertexts, layout)
        pack_plain_update = self.rebuild([ct.slots for ct in pack_plain_slots], layout, reference)

        bytes_meta = {
            "plain_float32_bytes_per_client": layout.num_params * 4,
            "raw_ciphertext_bytes_per_client": layout.bundle_count * self.crypto.raw_ct_bytes,
            "serialized_ciphertext_bytes_per_client": layout.bundle_count * self.serialized_ct_bytes,
            "raw_ciphertext_bytes_total": layout.bundle_count * self.crypto.raw_ct_bytes * len(weighted_updates),
            "serialized_ciphertext_bytes_total": layout.bundle_count * self.serialized_ct_bytes * len(weighted_updates),
        }
        metadata = {
            "scheme": "SimHE",
            "calibration_source": self.calibration.get("backend") if self.calibration else "analytic",
            "calibration_path": self.calibration.get("path") if self.calibration else None,
            "ciphertext_count": layout.bundle_count,
            "bundle_count": layout.bundle_count,
            "slot_utilization": layout.slot_utilization,
            "padding_slots": layout.padding_slots,
            "template_id": layout.template_id,
            "client_count": len(weighted_updates),
        }
        return SimHEResult(
            aggregate_update=aggregate_update,
            pack_plain_update=pack_plain_update,
            ciphertexts=last_ciphertexts,
            aggregate_ciphertexts=aggregate_ciphertexts,
            timings=timings,
            bytes=bytes_meta,
            metadata=metadata,
        )
