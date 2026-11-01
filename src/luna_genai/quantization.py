"""Transparent toy uniform quantization; this is NOT NF4 or a model backend."""
from __future__ import annotations
import numpy as np


def absmax_quantize(values, bits: int = 4) -> dict:
    """Finite nonempty array -> symmetric signed integers, reconstruction/errors.

    bits: integer 2..8. qmax=2**(bits-1)-1, step=max(abs(x))/qmax.
    Rounding uses NumPy ties-to-even, then clipping. Zero array uses step=1.
    Returns new arrays; no mutation, model loading or fitted codebook.
    Example: absmax_quantize([-3.5, 0, 3.5], 4)['q'] -> [-7,0,7].
    Metadata/packing is not included: q is an int64 observation array, NOT an
    actually packed 4-bit storage file. step has the same units as values.
    """
    x = np.asarray(values, dtype=float)
    if x.size == 0 or not np.isfinite(x).all() or type(bits) is not int or not 2 <= bits <= 8:
        raise ValueError('유한한 비어 있지 않은 배열과 bits=2..8이 필요합니다.')
    qmax = 2 ** (bits - 1) - 1
    peak = float(np.abs(x).max())
    step = peak / qmax if peak else 1.
    q = np.clip(np.rint(x / step), -qmax, qmax).astype(np.int64)
    restored = q * step
    error = restored - x
    return {'bits': bits, 'qmax': qmax, 'step': step, 'input': x.copy(), 'q': q,
            'restored': restored, 'error': error, 'mae': float(np.abs(error).mean()),
            'max_error': float(np.abs(error).max()),
            'mode': '교육용 균등 정수 격자; NF4 아님; int64 관찰 배열'}


def weight_storage(parameters: int, bits: int) -> dict:
    """Parameter count, bits/weight -> theoretical WEIGHT-only bytes/GB/GiB.

    Does not measure process RAM or include scales, nonquantized layers,
    gradients, optimizer, activations, KV cache. No allocation or model load.
    Example: weight_storage(70_000_000_000,16)['GB'] == 140.
    """
    if type(parameters) is not int or parameters < 1 or bits not in (4, 8, 16, 32):
        raise ValueError('양의 정수 모수 수와 bits=4/8/16/32가 필요합니다.')
    n = parameters * bits / 8
    return {'parameters': parameters, 'bits': bits, 'bytes': n, 'GB': n / 1e9,
            'GiB': n / 2**30, 'scope': '가중치만의 이론적 하한; 실제 peak memory 아님'}


def lora_dimensions(d: int, k: int, rank: int) -> dict:
    """One dense layer W[d,k], A[r,k], B[d,r] -> shapes/counts, no fitting.

    Biases are excluded. rank must be 1..min(d,k). This compares one chosen
    layer only; whole-model reduction also depends on selected target_modules.
    """
    if any(type(v) is not int or v < 1 for v in (d, k, rank)) or rank > min(d, k):
        raise ValueError('양의 정수 d,k와 rank<=min(d,k)가 필요합니다.')
    return {'W': [d, k], 'A': [rank, k], 'B': [d, rank], 'full': d*k,
            'adapter': rank*(d+k), 'ratio': rank*(d+k)/(d*k)}
