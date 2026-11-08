"""W11A: visible, measured image-editing mechanisms, without training on import.

Small numerical functions are teaching decompositions, not replicas of Compel,
SEGA, ControlNet or IP-Adapter. tiny_edit uses the actual bundled pretrained
class denoiser. Full SD examples are separately hashed execution records.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from diffusers import DDIMScheduler
from .diffusion import make_scheduler

ASSETS = Path(__file__).parent / 'assets' / 'w11a'


def blend_mask(original: np.ndarray, edited: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Float arrays [H,W,C], mask [H,W] in [0,1]; white selects edited pixels.

    Returns a new float array. This explicit output overlay preserves black
    pixels exactly; a diffusion inpainting pipeline by itself need not do so.
    """
    a, b, m = np.asarray(original, float), np.asarray(edited, float), np.asarray(mask, float)
    if a.shape != b.shape or a.ndim != 3 or m.shape != a.shape[:2]:
        raise ValueError('Expected equal HWC images and an HW mask')
    if not all(np.isfinite(x).all() for x in (a, b, m)) or (m < 0).any() or (m > 1).any():
        raise ValueError('Finite values and mask in [0,1] required')
    return (1 - m[..., None]) * a + m[..., None] * b


def region_mse(original: np.ndarray, edited: np.ndarray, mask: np.ndarray) -> dict:
    """Mean per-channel squared error in binary inside/outside regions.

    Inputs HWC float images and HW binary mask; empty region returns None,
    never zero evidence for a missing region. Scale depends on input units.
    """
    blend_mask(original, edited, mask)  # validate shapes and finite inputs
    if not np.isin(mask, [0, 1]).all():
        raise ValueError('Metric requires a binary mask')
    errors = ((np.asarray(original, float) - edited) ** 2).mean(axis=-1)
    return {name: float(errors[region].mean()) if region.any() else None
            for name, region in [('inside', mask == 1), ('outside', mask == 0)]}


def weighted_embedding(embeddings: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Teaching linear combination [K,T,D], [K] -> [T,D], not a Compel parser.

    Weights are coefficients, not probabilities or object-area percentages.
    No normalization is applied, no tokens parsed, no model state changed.
    """
    if embeddings.ndim != 3 or weights.shape != (len(embeddings),):
        raise ValueError('Expected KTD embeddings and K coefficients')
    if not torch.isfinite(embeddings).all() or not torch.isfinite(weights).all():
        raise ValueError('Finite embeddings/coefficients required')
    return (embeddings * weights[:, None, None]).sum(dim=0)


def semantic_delta(base: torch.Tensor, concept: torch.Tensor, *, scale: float = 1.,
                   reverse: bool = False, quantile: float = .5) -> dict:
    """One-concept SEGA-inspired local calculation; NOT the full SEGA scheduler.

    Equal float tensors -> delta and guided prediction. Keep magnitudes >=
    quantile, optionally reverse, then add scale*delta. Omits warmup/momentum.
    """
    if base.shape != concept.shape or base.numel() == 0:
        raise ValueError('Nonempty equal prediction shapes required')
    if not 0 <= quantile <= 1 or not math.isfinite(scale):
        raise ValueError('Quantile in [0,1] and finite scale required')
    if not torch.isfinite(base).all() or not torch.isfinite(concept).all():
        raise ValueError('Finite predictions required')
    direction = concept - base
    threshold = torch.quantile(direction.abs().float(), quantile)
    selected = torch.where(direction.abs() >= threshold, direction, 0.)
    delta = (-1 if reverse else 1) * scale * selected
    return {'direction': direction, 'threshold': threshold, 'delta': delta, 'guided': base + delta}


def separate_attention(query: torch.Tensor, text_key: torch.Tensor, text_value: torch.Tensor,
                       image_key: torch.Tensor, image_value: torch.Tensor, scale: float = .5) -> dict:
    """IP-Adapter-inspired single-head algebra, not a trained IP-Adapter.

    Q[N,D], Kt[T,D], Vt[T,V], Ki[I,D], Vi[I,V] -> separate softmax maps and
    text + scale*image [N,V]. Concatenating tokens changes the denominator.
    """
    tensors = (query, text_key, text_value, image_key, image_value)
    if any(x.ndim != 2 or not torch.isfinite(x).all() for x in tensors):
        raise ValueError('Finite 2D tensors required')
    if not (query.shape[1] == text_key.shape[1] == image_key.shape[1]) or query.shape[1] < 1:
        raise ValueError('Query/key dimensions must match')
    if text_key.shape[0] != text_value.shape[0] or image_key.shape[0] != image_value.shape[0] or text_value.shape[1] != image_value.shape[1]:
        raise ValueError('Key/value token counts and value dimensions must match')
    if text_key.shape[0] == 0 or image_key.shape[0] == 0 or not math.isfinite(scale):
        raise ValueError('Nonempty tokens and finite scale required')
    factor = query.shape[1] ** -.5
    at = (query @ text_key.T * factor).softmax(-1)
    ai = (query @ image_key.T * factor).softmax(-1)
    text, image = at @ text_value, ai @ image_value
    return {'text_weights': at, 'image_weights': ai, 'text': text, 'image': image,
            'combined': text + scale * image}


def add_control(base: torch.Tensor, residual: torch.Tensor, scale: float = 1.) -> torch.Tensor:
    """ControlNet-inspired residual addition only; neither trains nor encodes an image."""
    if base.shape != residual.shape or not all(torch.isfinite(x).all() for x in (base, residual)) or not math.isfinite(scale):
        raise ValueError('Equal finite shapes and finite scale required')
    return base + scale * residual


@torch.inference_mode()
def tiny_edit(model, clean: torch.Tensor, *, strength: float = .5, steps: int = 20,
              label: int = 8, seed: int = 42, mask: torch.Tensor | None = None) -> dict:
    """Real CPU image2image with pretrained 28x28 class-conditioned denoiser.

    clean [B,1,28,28] in [-1,1], mask same shape with 0/1. strength picks a
    suffix of DDIM steps. With mask, black pixels are explicitly restored at
    each noise level and in final output. This is pixel-space teaching, not
    SDXL's nine-channel latent inpainting. Model weights/training mode preserved.
    """
    if clean.ndim != 4 or clean.shape[1:] != (1, 28, 28) or clean.device.type != 'cpu' or not torch.isfinite(clean).all() or clean.min() < -1 or clean.max() > 1:
        raise ValueError('CPU finite B1x28x28 clean images in [-1,1] required')
    if not 0 <= strength <= 1 or isinstance(steps, bool) or not isinstance(steps, int) or not 2 <= steps <= 100:
        raise ValueError('strength in [0,1], integer steps in 2..100 required')
    if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label <= 9:
        raise ValueError('Class label in 0..9 required')
    if mask is not None and (mask.shape != clean.shape or not torch.isin(mask, torch.tensor([0., 1.])).all()):
        raise ValueError('Binary mask must have the clean image shape')
    scheduler = DDIMScheduler.from_config(make_scheduler().config, timestep_spacing='trailing')
    scheduler.set_timesteps(steps)
    count = min(steps, max(1, int(steps * strength))) if strength > 0 else 0
    times = scheduler.timesteps[-count:] if count else scheduler.timesteps[:0]
    noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(seed))
    x = scheduler.add_noise(clean, noise, times[:1]) if count else clean.clone()
    initial = x.clone(); trace = []; mode = model.training; model.eval()
    try:
        for i, t in enumerate(times):
            predicted = model(x, torch.full((len(x),), int(t), dtype=torch.long), torch.full((len(x),), label, dtype=torch.long))
            x = scheduler.step(predicted, t, x, eta=0).prev_sample
            if mask is not None:
                kept = scheduler.add_noise(clean, noise, times[i + 1:i + 2]) if i + 1 < count else clean
                x = mask * x + (1 - mask) * kept
            trace.append({'t': int(t), 'epsilon_rms': float(predicted.square().mean().sqrt()), 'x_rms': float(x.square().mean().sqrt())})
    finally:
        model.train(mode)
    if not torch.isfinite(x).all():
        raise RuntimeError('Nonfinite denoising output; no fallback')
    return {'images': x, 'initial': initial, 'noise': noise, 'trace': trace,
            'executed_steps': count, 'strength': strength, 'seed': seed,
            'mode': 'current actual CPU tiny class denoising, not SD'}


def load_edit_records() -> dict:
    """Load checked actual full-SD execution records; never execute/download SD.

    Returns manifest and asset directory. Verify every recorded SHA before any
    image is shown. Missing/tampered records raise; no synthetic replacement.
    """
    manifest = json.loads((ASSETS / 'record-manifest.json').read_text(encoding='utf-8'))
    for name, digest in manifest['sha256'].items():
        path = ASSETS / name
        if path.parent.resolve() != ASSETS.resolve() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError('Editing record checksum/path mismatch: ' + name)
    if manifest['mode'] != 'actual SD CPU execution record':
        raise RuntimeError('Unexpected record provenance')
    return {'manifest': manifest, 'directory': ASSETS}
