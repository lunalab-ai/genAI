"""W10B: measured adaptation of the genuinely pretrained W06B CPU denoiser.

The small model predicts pixel-space noise conditioned on a Fashion-MNIST
class ID. It is not Stable Diffusion, a text encoder, or a complete DreamBooth
reproduction. No download, training, RNG change, or file write occurs on import.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from .conditional import load_conditional_checkpoint, sample_classes
from .diffusion import forward_noise

ASSETS = Path(__file__).parent / 'assets' / 'w10b'


def _integer(value: Any, name: str, low: int, high: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    if not low <= int(value) <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def _finite(value: float, name: str, low: float = 0., high: float = 10.) -> float:
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{name} must be finite in {low}..{high}')
    return value


def load_teaching_images() -> dict:
    """Verify eight bundled, procedurally drawn non-person tote-bag images.

    Returns manifest, RGB PIL images, and CPU float32 train/validation tensors
    [6,1,28,28]/[2,1,28,28] in [-1,1]. Alpha defines the silhouette against a
    black CPU background; this is distinct from the RGB input used by SD's VAE.
    Package installation prepares the data automatically. No private path or
    manual upload is needed. Missing/corrupt files raise; there is no fallback.
    """
    manifest = json.loads((ASSETS / 'image-manifest.json').read_text(encoding='utf-8'))
    records = manifest['images']
    if len(records) != 8 or len({r['id'] for r in records}) != 8:
        raise ValueError('Expected eight unique original view IDs')
    group_sets = {s: {r['group_id'] for r in records if r['split'] == s}
                  for s in ('train', 'validation')}
    if not all(group_sets.values()) or group_sets['train'] & group_sets['validation']:
        raise ValueError('Original view groups must be disjoint before augmentation')
    result: dict[str, Any] = {'manifest': manifest, 'rgb': [], 'ids': {}, 'groups': group_sets}
    arrays: dict[str, list] = {'train': [], 'validation': []}
    for record in records:
        filename = record['file']
        if Path(filename).name != filename or record['split'] not in arrays:
            raise ValueError('Invalid image manifest entry')
        path = ASSETS / filename
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Image checksum mismatch: ' + filename)
        with Image.open(path) as im:
            rgba = im.convert('RGBA')
            canvas = Image.new('RGBA', rgba.size, record['background'])
            canvas.alpha_composite(rgba)
            result['rgb'].append(canvas.convert('RGB'))
            small = rgba.resize((28, 28), Image.Resampling.LANCZOS)
            a = np.asarray(small, dtype=np.float32) / 255.
            alpha = a[..., 3]
            gray = a[..., :3].mean(axis=-1)
            # Fashion-MNIST's learned convention is a light object on black.
            pixels = alpha * (.45 + .55 * gray)
            arrays[record['split']].append(2 * pixels - 1)
    for split, items in arrays.items():
        result[split] = torch.from_numpy(np.stack(items)[:, None].copy()).float()
        result['ids'][split] = [r['id'] for r in records if r['split'] == split]
    return result


def parameter_fingerprint(model: nn.Module, *, adapters: bool = False) -> str:
    """SHA-256 of named CPU parameter bytes, with canonical base-layer names.

    adapters=False excludes lora_A/B. Canonical names allow comparing a plain
    Linear to its unchanged wrapped base. Reads one tensor at a time, not a
    second full model copy. Hash equality is evidence about weights, not quality.
    """
    digest = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        is_adapter = 'lora_A' in name or 'lora_B' in name
        if is_adapter != adapters:
            continue
        canonical = name.replace('.base_layer.', '.')
        digest.update(canonical.encode())
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class LowRankLinear(nn.Module):
    """Frozen Linear plus (alpha/r) B A x; A[r,in], B[out,r].

    B starts at zero and A at a seeded normal draw. Input [...,in] returns
    [...,out]. Only A/B require gradients. This tiny model adapts time-feature
    projections; SD adapts attention projections in a separate module.
    """
    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float, seed: int):
        super().__init__()
        self.rank = _integer(rank, 'rank', 1, min(base_layer.in_features, base_layer.out_features))
        self.alpha = _finite(alpha, 'alpha', .001, 1024.)
        self.scale = self.alpha / self.rank
        self.base_layer = base_layer
        self.base_layer.requires_grad_(False)
        g = torch.Generator().manual_seed(seed)
        self.lora_A = nn.Parameter(torch.randn((rank, base_layer.in_features), generator=g) * .02)
        self.lora_B = nn.Parameter(torch.zeros((base_layer.out_features, rank)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_layer(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale


def prior_loss_components(prediction: torch.Tensor, target: torch.Tensor,
                          instance_count: int, prior_weight: float = 1.) -> dict:
    """Split [instance rows, class-prior rows] and compute two mean MSEs.

    Returns differentiable instance/prior/total tensors. The means have their
    own denominators; total=instance+weight*prior, not the mean of all rows.
    With weight=0 the prior is observed but contributes no gradient.
    """
    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError('Prediction/target must share a batched shape')
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError('Loss inputs must be finite')
    n = _integer(instance_count, 'instance_count', 1, len(prediction) - 1)
    weight = _finite(prior_weight, 'prior_weight')
    inst = F.mse_loss(prediction[:n], target[:n])
    prior = F.mse_loss(prediction[n:], target[n:])
    return {'instance': inst, 'prior': prior, 'total': inst + weight * prior}


class TinyAdaptationLab:
    """An independent copy of the 6,000-step W06B checkpoint on CPU.

    mode=frozen/full/lora. Construction reads a checked package asset; it never
    trains or downloads. fit() explicitly changes this instance and optimizer.
    Inputs are CPU float32 [N,1,28,28]; class condition 8 means Bag, not a target
    label for a classifier and not a learned textual identifier.
    """
    def __init__(self, mode: str = 'lora', *, rank: int = 4,
                 alpha: float | None = None, seed: int = 42):
        if mode not in ('frozen', 'full', 'lora'):
            raise ValueError('Choose frozen, full, or lora')
        self.mode, self.seed = mode, _integer(seed, 'seed', 0, 2**31 - 1)
        self.rank = _integer(rank, 'rank', 1, 16)
        self.alpha = float(2 * rank if alpha is None else alpha)
        self.model, self.base_record = load_conditional_checkpoint()
        self.original_base_hash = parameter_fingerprint(self.model)
        self.targets: list[str] = []
        if mode in ('frozen', 'lora'):
            self.model.requires_grad_(False)
        if mode == 'lora':
            modules = [(n, m) for n, m in self.model.named_modules() if isinstance(m, nn.Linear)]
            for i, (name, layer) in enumerate(modules):
                parent_name, _, attr = name.rpartition('.')
                parent = self.model.get_submodule(parent_name) if parent_name else self.model
                setattr(parent, attr, LowRankLinear(layer, rank, self.alpha, seed + i))
                self.targets.append(name)
            if parameter_fingerprint(self.model) != self.original_base_hash:
                raise RuntimeError('Wrapping changed base weights')
        self.history: list[dict] = []
        self.training: dict | None = None

    def state(self) -> dict:
        """Counts, scale, base provenance, and actual trained-update count; no mutation."""
        named = list(self.model.named_parameters())
        return {'model': 'W06B TinyClassUNet / pixel-class-conditioned, not SD',
                'mode': self.mode, 'device': 'cpu', 'rank': self.rank if self.mode == 'lora' else None,
                'scale': self.alpha / self.rank if self.mode == 'lora' else None,
                'trainable_parameters': sum(p.numel() for _, p in named if p.requires_grad),
                'trainable_base': sum(p.numel() for n, p in named if p.requires_grad and 'lora_' not in n),
                'trainable_adapter': sum(p.numel() for n, p in named if p.requires_grad and 'lora_' in n),
                'target_modules': list(self.targets), 'base_checkpoint_sha256': self.base_record['sha256'],
                'base_pretraining_steps': self.base_record['steps'], 'updates': len(self.history)}

    @torch.no_grad()
    def evaluate(self, images: torch.Tensor, *, label: int = 8, seed: int = 733) -> dict:
        """Common held-out noise MSE at t=24,49,74; same seed across cases.

        No optimizer or model mutation. Returns measured noise MSE and the
        zero-predictor MSE, not image fidelity or subject-identity accuracy.
        """
        label = _integer(label, 'label', 0, 9)
        g = torch.Generator().manual_seed(seed)
        noise = torch.randn(images.shape, generator=g)
        y = torch.full((len(images),), label, dtype=torch.long)
        old = self.model.training
        self.model.eval()
        rows = []
        try:
            for level in (24, 49, 74):
                t = torch.full((len(images),), level, dtype=torch.long)
                pred = self.model(forward_noise(images, t, noise), t, y)
                rows.append(float(F.mse_loss(pred, noise)))
        finally:
            self.model.train(old)
        return {'noise_mse': float(np.mean(rows)), 'per_timestep_mse': rows,
                'zero_predictor_mse': float(noise.square().mean()), 'images': len(images), 'seed': seed}

    def fit(self, instances: torch.Tensor, prior_images: torch.Tensor, *, steps: int = 16,
            batch_size: int = 2, prior_weight: float = 1., learning_rate: float = .0002,
            accumulation_steps: int = 1) -> dict:
        """Actual MSE/backward/Adam updates with paired, controlled noise draws.

        Defaults:16 optimizer updates,2 instance+2 prior rows/microbatch.
        All variants consume identical draws, including when prior_weight=0.
        Accumulation averages microbatch losses before each optimizer update.
        Mutates only this lab's weights/optimizer/history. Never asserts quality
        improvement. Returns hashes, actual changes, first gradients and time.
        """
        if self.mode == 'frozen':
            raise ValueError('A frozen baseline cannot be fitted')
        if self.history:
            raise ValueError('For a controlled comparison create a fresh lab before fitting again')
        steps = _integer(steps, 'steps', 1, 256)
        batch_size = _integer(batch_size, 'batch_size', 1, 8)
        accumulation_steps = _integer(accumulation_steps, 'accumulation_steps', 1, 8)
        weight = _finite(prior_weight, 'prior_weight')
        lr = _finite(learning_rate, 'learning_rate', 1e-7, .01)
        if len(instances) < batch_size or len(prior_images) < batch_size:
            raise ValueError('Both datasets must contain a complete microbatch')
        before_base = parameter_fingerprint(self.model)
        before_adapters = parameter_fingerprint(self.model, adapters=True)
        original = {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}
        optimizer = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=lr)
        g = torch.Generator().manual_seed(self.seed + 100)
        self.model.train()
        first_gradients = {}
        started = time.perf_counter()
        for update in range(steps):
            optimizer.zero_grad(set_to_none=True)
            values = []
            for micro in range(accumulation_steps):
                offset = (update * accumulation_steps + micro) * batch_size
                ix = torch.arange(offset, offset + batch_size) % len(instances)
                px = torch.arange(offset, offset + batch_size) % len(prior_images)
                clean = torch.cat([instances[ix], prior_images[px]])
                t = torch.randint(0, 100, (len(clean),), generator=g)
                noise = torch.randn(clean.shape, generator=g)
                y = torch.full((len(clean),), 8, dtype=torch.long)
                prediction = self.model(forward_noise(clean, t, noise), t, y)
                losses = prior_loss_components(prediction, noise, batch_size, weight)
                (losses['total'] / accumulation_steps).backward()
                values.append({k: float(v.detach()) for k, v in losses.items()})
            grads = [p.grad for p in self.model.parameters() if p.requires_grad and p.grad is not None]
            norm = float(torch.sqrt(sum(v.float().square().sum() for v in grads)))
            if not math.isfinite(norm) or norm == 0:
                raise RuntimeError('Expected finite, nonzero trainable gradients')
            if update == 0 and self.mode == 'lora':
                first_gradients = {kind: float(torch.sqrt(sum(p.grad.square().sum() for n, p in self.model.named_parameters()
                                    if kind in n and p.grad is not None))) for kind in ('lora_A', 'lora_B')}
            optimizer.step()
            self.history.append({'update': update + 1,
                **{k + '_loss': float(np.mean([v[k] for v in values])) for k in ('instance', 'prior', 'total')},
                'gradient_norm': norm, 'seconds': time.perf_counter() - started})
        changes = {n: float((p.detach() - original[n]).norm()) for n, p in self.model.named_parameters() if n in original}
        self.training = {'mode': 'live_small_conditional_denoiser', 'state': self.state(),
            'steps': steps, 'instance_microbatch': batch_size, 'prior_microbatch': batch_size,
            'accumulation_steps': accumulation_steps, 'effective_instance_batch': batch_size * accumulation_steps,
            'prior_weight': weight, 'learning_rate': lr, 'seed': self.seed,
            'base_before': before_base, 'base_after': parameter_fingerprint(self.model),
            'base_unchanged': before_base == parameter_fingerprint(self.model),
            'adapter_before': before_adapters, 'adapter_after': parameter_fingerprint(self.model, adapters=True),
            'parameter_changes': changes, 'first_gradients': first_gradients,
            'history': copy.deepcopy(self.history), 'seconds': time.perf_counter() - started}
        if self.mode == 'lora' and not self.training['base_unchanged']:
            raise RuntimeError('LoRA training unexpectedly changed base parameters')
        if not any(v > 0 for v in changes.values()):
            raise RuntimeError('Optimizer did not change any trainable parameter')
        self.model.eval()
        return copy.deepcopy(self.training)

    def sample(self, *, seed: int = 42, count: int = 2, steps: int = 20) -> dict:
        """Actual DDIM samples from this lab, same initial noise for paired cases."""
        count = _integer(count, 'count', 1, 8)
        return sample_classes(self.model, [8] * count, seed=seed, steps=steps, matched_noise=False)

    def save_reload_check(self, directory: str | Path, probe_images: torch.Tensor) -> dict:
        """Save only A/B as NPZ + base metadata, load a fresh same-base lab, compare.

        Writes a new directory; refuses to overwrite an existing adapter file.
        Uses allow_pickle=False. Compares fixed-input predicted-noise tensors,
        not whether a generated picture looks similar. Does not change this lab.
        """
        if self.mode != 'lora' or not self.history:
            raise ValueError('Fit a LoRA lab first')
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        filename = path / 'adapter.npz'
        if filename.exists():
            raise FileExistsError('Use a fresh run directory')
        arrays = {n: p.detach().cpu().numpy() for n, p in self.model.named_parameters() if 'lora_' in n}
        np.savez_compressed(filename, **arrays)
        metadata = {'base_checkpoint_sha256': self.base_record['sha256'], 'base_parameter_sha256': self.original_base_hash,
                    'rank': self.rank, 'alpha': self.alpha, 'seed': self.seed,
                    'sha256': hashlib.sha256(filename.read_bytes()).hexdigest(), 'target_modules': self.targets}
        (path/'adapter.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        other = TinyAdaptationLab('lora', rank=self.rank, alpha=self.alpha, seed=self.seed)
        if other.original_base_hash != metadata['base_parameter_sha256']:
            raise RuntimeError('Wrong base for saved adapter')
        with np.load(filename, allow_pickle=False) as archive:
            if set(archive.files) != set(arrays):
                raise RuntimeError('Adapter keys differ')
            for name, value in other.model.named_parameters():
                if name in archive.files:
                    if tuple(value.shape) != archive[name].shape:
                        raise RuntimeError('Adapter shape differs')
                    value.data.copy_(torch.from_numpy(archive[name].copy()))
        g = torch.Generator().manual_seed(901)
        noise = torch.randn(probe_images.shape, generator=g)
        t = torch.full((len(probe_images),), 49, dtype=torch.long)
        noisy = forward_noise(probe_images, t, noise)
        y = torch.full((len(probe_images),), 8, dtype=torch.long)
        self.model.eval(); other.model.eval()
        with torch.no_grad():
            left, right = self.model(noisy, t, y), other.model(noisy, t, y)
        error = float((left - right).abs().max())
        if error > 1e-6:
            raise RuntimeError('Reload changed predictions')
        return {'max_abs_prediction_error': error, 'base_matches': True,
                'adapter_sha256': metadata['sha256'], 'bytes': filename.stat().st_size,
                'arrays': len(arrays), 'format': 'NPZ, A/B only, allow_pickle=False'}


def run_tiny_comparison(*, steps: int = 16, rank: int = 4, seed: int = 42) -> dict:
    """Run a frozen baseline and full/LoRA x prior0/1 from the SAME checkpoint.

    Returns data, independently fitted lab objects, and a common-evaluation
    DataFrame. Class priors are sampled from the frozen original model FIRST;
    held-out prior and unrelated-class samples are separate. No download.
    """
    data = load_teaching_images()
    baseline = TinyAdaptationLab('frozen', seed=seed)
    prior_all = sample_classes(baseline.model, [8] * 8, seed=810, steps=20, matched_noise=False)['images'].clamp(-1, 1)
    prior, prior_validation = prior_all[:4].clone(), prior_all[4:].clone()
    unrelated = sample_classes(baseline.model, [0, 0], seed=811, steps=20, matched_noise=False)['images'].clamp(-1, 1)
    cases: dict[str, TinyAdaptationLab] = {'base': baseline}
    for mode in ('full', 'lora'):
        for weight in (0., 1.):
            lab = TinyAdaptationLab(mode, rank=rank, alpha=2 * rank, seed=seed)
            lab.fit(data['train'], prior, steps=steps, prior_weight=weight)
            cases[f'{mode}_prior{int(weight)}'] = lab
    rows = []
    for name, lab in cases.items():
        rows.append({'case': name, 'updates': len(lab.history),
            'trainable': lab.state()['trainable_parameters'],
            'new_view_noise_mse': lab.evaluate(data['validation'])['noise_mse'],
            'class_prior_noise_mse': lab.evaluate(prior_validation)['noise_mse'],
            'unrelated_class_noise_mse': lab.evaluate(unrelated, label=0)['noise_mse'],
            'seconds': lab.training['seconds'] if lab.training else 0.})
    return {'data': data, 'cases': cases, 'table': pd.DataFrame(rows),
            'prior_train': prior, 'prior_validation': prior_validation, 'unrelated': unrelated}


def inpainting_channels(noisy_latent: torch.Tensor, mask: torch.Tensor,
                        masked_image_latent: torch.Tensor) -> torch.Tensor:
    """SD1 inpaint input: noisy[B,4,H,W]+white-edit mask[B,1,H,W]+masked VAE[B,4,H,W].

    Validates matching floating tensors and returns [B,9,H,W]. This operation
    assembles channels; it does not train or run an inpainting network. The
    masked-image latent must come from encoding the masked RGB image, not from
    asserting that VAE encoding commutes with multiplying a latent mask.
    """
    if noisy_latent.ndim != 4 or noisy_latent.shape[1] != 4:
        raise ValueError('Expected a batched four-channel noisy latent')
    n, _, h, w = noisy_latent.shape
    if mask.shape != (n, 1, h, w) or masked_image_latent.shape != noisy_latent.shape:
        raise ValueError('Mask/latent shapes do not align')
    for x in (noisy_latent, mask, masked_image_latent):
        if not x.is_floating_point() or x.dtype != noisy_latent.dtype or x.device != noisy_latent.device or not x.isfinite().all():
            raise ValueError('Inputs must share a finite floating dtype and device')
    if (mask < 0).any() or (mask > 1).any():
        raise ValueError('Mask values must be in [0,1]; white=edit')
    return torch.cat([noisy_latent, mask, masked_image_latent], dim=1)


def sdxl_condition_shapes(batch: int = 1, tokens: int = 77) -> dict:
    """Read-only standard SDXL-base configuration map; no SDXL weights/execution.

    768+1280 token features ->2048; second pooled features1280 plus six size/crop
    values, each embedded to256 ->2816. Other architectures may differ.
    """
    batch = _integer(batch, 'batch', 1, 8)
    tokens = _integer(tokens, 'tokens', 1, 77)
    return {'mode': 'SDXL configuration calculation; training/generation not executed',
            'token_hidden': [batch, tokens, 2048], 'pooled_text': [batch, 1280],
            'size_crop_values': [batch, 6], 'added_condition_width': 1280 + 6 * 256,
            'example_time_ids': [1024, 1024, 0, 0, 1024, 1024]}


def load_actual_sd_record() -> dict:
    """Load verified, actually executed SD1.5 records/images, never generate.

    Missing assets or a hash mismatch raise. Selecting a saved prompt/seed is
    record analysis and does not run SD now. SDXL execution has its own flag.
    """
    record = json.loads((ASSETS/'sd15-record.json').read_text(encoding='utf-8'))
    if record.get('model_id') != 'stable-diffusion-v1-5/stable-diffusion-v1-5' or not record.get('training', {}).get('base_unchanged'):
        raise ValueError('Expected an actual frozen-base SD1.5 LoRA training record')
    for item in record['files']:
        name = item['file']
        if Path(name).name != name or hashlib.sha256((ASSETS/name).read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Recorded image integrity failure: ' + name)
    return record
