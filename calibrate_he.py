from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hefl.utils import ensure_dir, write_json


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return float(statistics.mean(vals)) if vals else 0.0


def _stdev(values: Iterable[float]) -> float:
    vals = list(values)
    return float(statistics.stdev(vals)) if len(vals) > 1 else 0.0


def calibrate_tenseal(
    out: str | Path,
    poly_modulus_degree: int,
    coeff_mod_bit_sizes: List[int],
    scale_bits: int,
    vector_length: int,
    trials: int,
) -> Dict[str, Any]:
    try:
        import tenseal as ts
    except Exception as exc:
        raise RuntimeError("TenSEAL is not installed. Install it with `python -m pip install tenseal`.") from exc

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes,
    )
    context.global_scale = 2**scale_bits

    values_a = [((i % 17) - 8) / 17.0 for i in range(vector_length)]
    values_b = [((i % 11) - 5) / 11.0 for i in range(vector_length)]
    encode_encrypt_ms: List[float] = []
    serialize_ms: List[float] = []
    add_ms: List[float] = []
    decrypt_ms: List[float] = []
    serialized_bytes: List[int] = []

    for _ in range(max(1, int(trials))):
        start = time.perf_counter()
        enc_a = ts.ckks_vector(context, values_a)
        enc_b = ts.ckks_vector(context, values_b)
        encode_encrypt_ms.append((time.perf_counter() - start) * 1000.0 / 2.0)

        start = time.perf_counter()
        blob = enc_a.serialize()
        serialize_ms.append((time.perf_counter() - start) * 1000.0)
        serialized_bytes.append(len(blob))

        start = time.perf_counter()
        enc_sum = enc_a + enc_b
        add_ms.append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        _ = enc_sum.decrypt()
        decrypt_ms.append((time.perf_counter() - start) * 1000.0)

    result = {
        "backend": "tenseal",
        "status": "ok",
        "vector_length": vector_length,
        "trials": max(1, int(trials)),
        "crypto": {
            "scheme": "CKKS",
            "poly_modulus_degree": poly_modulus_degree,
            "coeff_mod_bit_sizes": coeff_mod_bit_sizes,
            "scale_bits": scale_bits,
        },
        "per_ct": {
            "encode_encrypt_ms": _mean(encode_encrypt_ms),
            "serialize_ms": _mean(serialize_ms),
            "add_ms": _mean(add_ms),
            "decrypt_ms": _mean(decrypt_ms),
            "serialized_ct_bytes": int(round(_mean(serialized_bytes))),
        },
        "std": {
            "encode_encrypt_ms": _stdev(encode_encrypt_ms),
            "serialize_ms": _stdev(serialize_ms),
            "add_ms": _stdev(add_ms),
            "decrypt_ms": _stdev(decrypt_ms),
            "serialized_ct_bytes": _stdev(serialized_bytes),
        },
    }
    out_path = Path(out)
    ensure_dir(out_path.parent)
    write_json(out_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate real CKKS microbenchmarks for calibrated SimHE.")
    parser.add_argument("--backend", default="tenseal", choices=["tenseal"])
    parser.add_argument("--out", required=True, help="Output he_calibration.json path.")
    parser.add_argument("--poly-modulus-degree", type=int, default=8192)
    parser.add_argument("--coeff-mod-bit-sizes", default="60,40,40,60")
    parser.add_argument("--scale-bits", type=int, default=40)
    parser.add_argument("--vector-length", type=int, default=4096)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    coeffs = [int(x.strip()) for x in args.coeff_mod_bit_sizes.split(",") if x.strip()]
    result = calibrate_tenseal(
        out=args.out,
        poly_modulus_degree=args.poly_modulus_degree,
        coeff_mod_bit_sizes=coeffs,
        scale_bits=args.scale_bits,
        vector_length=args.vector_length,
        trials=args.trials,
    )
    print(Path(args.out))
    print(result["per_ct"])


if __name__ == "__main__":
    main()
