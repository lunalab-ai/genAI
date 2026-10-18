"""W07A: inspectable SD1.5 DDIM inference; never training or silent replay."""
from __future__ import annotations
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any
import numpy as np
import torch
from PIL import Image
from .stable_diffusion import MODEL_ID, MODEL_REVISION, DEFAULT_PROMPT, cfg_combine

ASSETS = Path(__file__).parent / 'assets' / 'w07a'


def ddim_scalar(x_t: float, epsilon: float, alpha_t: float, alpha_previous: float) -> dict:
    """One illustrative deterministic DDIM step, with cumulative alpha values.

    Returns predicted clean x0 and the next less-noisy state. This scalar
    example has eta=0, no clipping; it neither calls SD nor measures quality.
    Example: ddim_scalar(.6,.2,.25,.64)['previous'] is about .802872.
    """
    if not all(math.isfinite(float(x)) for x in (x_t, epsilon, alpha_t, alpha_previous)):
        raise ValueError('All inputs must be finite')
    if not 0 < alpha_t <= alpha_previous <= 1:
        raise ValueError('Require 0 < alpha_t <= alpha_previous <= 1')
    x0 = (x_t - math.sqrt(1-alpha_t)*epsilon) / math.sqrt(alpha_t)
    previous = math.sqrt(alpha_previous)*x0 + math.sqrt(1-alpha_previous)*epsilon
    return {'predicted_x0': x0, 'previous': previous}


def _integer(value: Any, name: str, low: int, high: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(name+' must be an integer')
    value = float(value)
    if not math.isfinite(value) or not value.is_integer() or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in [{low}, {high}]')
    return int(value)


def make_initial_latents(pipe: Any, seed: int = 42) -> torch.Tensor:
    """New CPU generator on EACH call; return SD1.5 raw noise [1,4,64,64].

    Draw float32 noise on CPU, then convert to pipe device/U-Net dtype.
    Does not multiply scheduler.init_noise_sigma yet. No model execution.
    Identical calls in this environment return equal tensors. Cross-device
    generated images are not promised bitwise identical.
    """
    seed = _integer(seed, 'seed', 0, 2**32-1)
    noise = torch.randn((1,4,64,64), generator=torch.Generator(device='cpu').manual_seed(seed), dtype=torch.float32)
    return noise.to(device=pipe.device, dtype=pipe.unet.dtype)


@torch.inference_mode()
def decode_checked(pipe: Any, scaled_latent: torch.Tensor, *, preview: bool = False) -> Image.Image:
    """Decode one scaled latent using VAE config and the active safety checker.

    Returns PIL RGB. Raises on missing/final blocked assessment. For a blocked
    preview only, returns the checker's BLACK image, labelled in image.info;
    never returns the blocked pixels. Intermediate input
    is noisy z, not the scheduler's predicted x0; label previews accordingly.
    """
    if pipe.safety_checker is None:
        raise RuntimeError('An active safety checker is required')
    pixels = pipe.vae.decode(scaled_latent/pipe.vae.config.scaling_factor, return_dict=False)[0]
    pixels, flags = pipe.run_safety_checker(pixels, pipe.device, pipe.text_encoder.dtype)
    if flags is None or (any(flags) and not preview):
        raise RuntimeError('Output blocked or not assessed; choose a different prompt')
    image = pipe.image_processor.postprocess(pixels, output_type='pil', do_denormalize=[not bool(flags[0])])[0]
    image.info['safety_status'] = 'blocked preview (black image)' if any(flags) else 'passed'
    return image


@torch.inference_mode()
def manual_sample(pipe: Any, prompt: str = DEFAULT_PROMPT, *, negative_prompt: str = '',
                  seed: int = 42, guidance: float = 7.5, steps: int = 20,
                  initial_latents: torch.Tensor | None = None, previews: bool = True) -> dict:
    """Real single-image SD1.5 DDIM loop with visible intermediate tensors.

    Input: loaded SD1.5 epsilon-prediction pipeline, English prompt/negative
    strings, seed, CFG in [1,20], steps in [2,50]. Resolution fixed at 512.
    Optional raw initial_latents [1,4,64,64] is cloned, never mutated.
    Returns image, final_latent, trace (one row/step), frames and settings.
    No optimizer/backward/weight updates. Uses independent scheduler state.
    CFG=1 uses the conditional branch only, so negative prompt is inactive.
    Call load_sd15(device='cuda') once before repeated calls.
    """
    from diffusers import DDIMScheduler
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('Enter a nonempty prompt')
    if not isinstance(negative_prompt, str):
        raise ValueError('negative_prompt must be text')
    seed = _integer(seed, 'seed', 0, 2**32-1)
    steps = _integer(steps, 'steps', 2, 50)
    guidance = float(guidance)
    if not math.isfinite(guidance) or not 1 <= guidance <= 20:
        raise ValueError('guidance must be finite and in [1,20]')
    if not isinstance(pipe.scheduler, DDIMScheduler) or pipe.scheduler.config.prediction_type != 'epsilon':
        raise ValueError('This teaching loop supports DDIM epsilon prediction only')
    if pipe.unet.config.in_channels != 4 or pipe.vae_scale_factor != 8:
        raise ValueError('Expected SD1.5 latent layout')
    if pipe.safety_checker is None:
        raise RuntimeError('An active safety checker is required')
    if pipe.device.type == 'cuda': torch.cuda.synchronize()
    started = time.perf_counter()
    scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    scheduler.set_timesteps(steps, device=pipe.device)
    use_cfg = guidance > 1
    positive, negative = pipe.encode_prompt(prompt, pipe.device, 1, use_cfg, negative_prompt=negative_prompt)
    context = torch.cat([negative, positive]) if use_cfg else positive
    raw = make_initial_latents(pipe, seed) if initial_latents is None else initial_latents.detach().clone().to(pipe.device, pipe.unet.dtype)
    if raw.shape != (1,4,64,64) or not torch.isfinite(raw).all():
        raise ValueError('Initial noise must be finite [1,4,64,64]')
    initial_hash = hashlib.sha256(raw.float().cpu().numpy().tobytes()).hexdigest()
    z = raw * scheduler.init_noise_sigma
    frames, trace = [], []
    capture = {0, steps//4, steps//2, 3*steps//4, steps-1}
    for index, t in enumerate(scheduler.timesteps):
        # 1: same noisy state, two text contexts, SAME U-Net weights.
        model_input = torch.cat([z,z]) if use_cfg else z
        model_input = scheduler.scale_model_input(model_input, t)
        # 2: U-Net predicts epsilon; it does not directly return next z.
        prediction = pipe.unet(model_input, t, encoder_hidden_states=context, return_dict=False)[0]
        if use_cfg:
            eps_negative, eps_positive = prediction.chunk(2)
            epsilon = cfg_combine(eps_negative, eps_positive, guidance)
        else:
            eps_positive = prediction
            eps_negative = None
            epsilon = prediction
        before = float(z.float().square().mean().sqrt())
        # 3: DDIM changes the sample, NOT the learned parameters.
        z = scheduler.step(epsilon, t, z, eta=0.0, return_dict=False)[0]
        if not torch.isfinite(z).all(): raise RuntimeError('Nonfinite latent in sampling')
        trace.append({'step': index+1, 'input_t': int(t),
                      'next_t': int(scheduler.timesteps[index+1]) if index+1<steps else None,
                      'latent_rms_before': before, 'latent_rms_after': float(z.float().square().mean().sqrt()),
                      'negative_mean': float(eps_negative.float().mean()) if use_cfg else None,
                      'positive_mean': float(eps_positive.float().mean()),
                      'guided_rms': float(epsilon.float().square().mean().sqrt())})
        if index in capture:
            frames.append({'step': index+1, 'input_t': int(t), 'latent': z.float().cpu().clone()})
    image = decode_checked(pipe, z)
    if previews:
        for frame in frames:
            frame['image'] = image if frame['step']==steps else decode_checked(pipe, frame['latent'].to(pipe.device, pipe.vae.dtype), preview=True)
            frame['safety_status'] = frame['image'].info['safety_status']
    if pipe.device.type == 'cuda': torch.cuda.synchronize()
    settings = dict(model_id=MODEL_ID, revision=MODEL_REVISION, prompt=prompt,
                    negative_prompt=negative_prompt, seed=seed, initial_sha256=initial_hash,
                    guidance_scale=guidance, steps=steps, scheduler='DDIMScheduler', eta=0,
                    height=512, width=512, dtype=str(pipe.unet.dtype), device=str(pipe.device),
                    seconds=time.perf_counter()-started, mode='live manual SD generation',
                    safety_checked=True, negative_active=use_cfg,
                    preview_kind='direct decode of updated noisy latent; not predicted x0')
    return dict(image=image, final_latent=z.detach().clone(), trace=trace, frames=frames, settings=settings)


@torch.inference_mode()
def compare_sampling_paths(pipe: Any, *, prompt: str = DEFAULT_PROMPT, negative_prompt: str = '',
                           seed: int = 42, guidance: float = 7.5, steps: int = 20) -> dict:
    """Compare real manual vs pipeline final latents from the SAME raw noise.

    Fix DDIM, eta=0, resolution, prompt, negative prompt, dtype and steps.
    Pipeline output_type='latent' is for numerical comparison, not display;
    the manual final image is safety-checked. Reports measured errors and
    allclose(atol=.002,rtol=.002). A false result is returned, not concealed.
    """
    from diffusers import DDIMScheduler
    initial = make_initial_latents(pipe, seed)
    manual = manual_sample(pipe, prompt, negative_prompt=negative_prompt, seed=seed,
                           guidance=guidance, steps=steps, initial_latents=initial, previews=False)
    original_scheduler = pipe.scheduler
    try:
        pipe.scheduler = DDIMScheduler.from_config(original_scheduler.config)
        reference = pipe(prompt, negative_prompt=negative_prompt, latents=initial.clone(), height=512, width=512,
                         guidance_scale=float(guidance), num_inference_steps=steps, eta=0.0,
                         num_images_per_prompt=1, output_type='latent').images
    finally:
        pipe.scheduler = original_scheduler
    delta = (manual['final_latent'].float()-reference.float()).abs()
    return dict(max_abs=float(delta.max()), mean_abs=float(delta.mean()), atol=.002, rtol=.002,
                passed=bool(torch.allclose(manual['final_latent'], reference, atol=.002, rtol=.002)),
                initial_sha256=manual['settings']['initial_sha256'], settings=manual['settings'])


def load_sampling_records() -> dict:
    """Read hash-checked W07A actual T4 records. No model load or new inference.

    Returns metadata, runs, parity evidence and relative file names. Array
    archive and every picture are verified before use; errors stay visible.
    Use ASSETS/run['file'] for a final image. No network is performed here.
    """
    record = json.loads((ASSETS/'sampling-records.json').read_text(encoding='utf-8'))
    for name, digest in record['files'].items():
        if Path(name).name != name: raise ValueError('Invalid record path')
        if hashlib.sha256((ASSETS/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Sampling record checksum mismatch: '+name)
    return record
