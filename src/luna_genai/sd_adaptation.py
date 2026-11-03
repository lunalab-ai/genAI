"""W10B opt-in actual SD1.5 DreamBooth-LoRA, separate from CPU record analysis.

Uses the existing pinned base and active image assessment. No model download,
training, or network call on import. SDXL is a separate optional project recipe.
"""
from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .diffusion_adaptation import (_finite, _integer, load_teaching_images,
                                   parameter_fingerprint, prior_loss_components)
from .stable_diffusion import MODEL_ID, MODEL_REVISION, load_sd15


class BlockedImageError(RuntimeError):
    """An assessed blocked attempt, carrying settings but no original image."""
    def __init__(self, settings: dict):
        super().__init__('Image blocked by the active assessment; no evaluable image is available')
        self.settings = settings


def identifier_tokens(text: str = 'an illustration of lunabag tote bag') -> list[dict]:
    """Real pinned SD1.5 CLIP tokenizer IDs, not new vocabulary insertion.

    Downloads only tokenizer files (roughly a few MB) if missing and reuses the
    Hub cache. Returns position/ID/piece. Network/auth failures raise explicitly.
    The identifier may split into multiple existing pieces.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError('A nonempty prompt is required')
    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder='tokenizer', revision=MODEL_REVISION)
    ids = tokenizer(text, add_special_tokens=True)['input_ids']
    return [{'position': i, 'id': int(value), 'piece': tokenizer.convert_ids_to_tokens(int(value))}
            for i, value in enumerate(ids)]


class StableDiffusionAdapterLab:
    """Load an actual pinned SD1.5 pipeline; constructor does not train.

    device=cpu/cuda must be explicit; unavailable CUDA raises. CPU uses float32,
    CUDA base uses float16 and LoRA trainables float32. Initial model download
    is about2.6GB, cached subsequently. This is not the SDXL architecture.
    """
    def __init__(self, *, device: str = 'cpu', local_files_only: bool = False):
        self.pipe = load_sd15(device=device, local_files_only=local_files_only)
        self.device = device
        self.adapter_name: str | None = None
        self.training: dict | None = None
        self._probe: tuple | None = None
        self.pipe.vae.requires_grad_(False).eval()
        self.pipe.text_encoder.requires_grad_(False).eval()
        self.pipe.unet.requires_grad_(False).eval()
        self.pipe.set_progress_bar_config(disable=True)

    def state(self) -> dict:
        """Actual trainable counts/dtypes and immutable base identity; no mutation."""
        named = list(self.pipe.unet.named_parameters())
        return {'model_id': MODEL_ID, 'revision': MODEL_REVISION, 'device': self.device,
            'mode': 'actual SD1.5, not SDXL', 'base_dtype': str(self.pipe.unet.dtype),
            'trainable_base': sum(p.numel() for n, p in named if p.requires_grad and 'lora_' not in n),
            'trainable_adapter': sum(p.numel() for n, p in named if p.requires_grad and 'lora_' in n),
            'adapter_name': self.adapter_name,
            'adapter_dtypes': sorted({str(p.dtype) for n, p in named if 'lora_' in n}),
            'torch': torch.__version__, 'diffusers': version('diffusers'), 'peft': version('peft')}

    def _embedding(self, prompt: str) -> torch.Tensor:
        tokenizer = self.pipe.tokenizer
        encoded = tokenizer(prompt, padding='max_length', max_length=tokenizer.model_max_length,
                            truncation=True, return_tensors='pt')
        with torch.no_grad():
            return self.pipe.text_encoder(encoded.input_ids.to(self.pipe.device))[0].detach()

    def _latents(self, images: list[Image.Image], resolution: int) -> torch.Tensor:
        values = []
        for image in images:
            pixels = np.asarray(image.convert('RGB').resize((resolution, resolution), Image.Resampling.LANCZOS), dtype=np.float32)
            x = torch.from_numpy(pixels.copy()).permute(2, 0, 1)[None] / 127.5 - 1
            x = x.to(self.pipe.device, dtype=self.pipe.vae.dtype)
            with torch.no_grad():
                # Fixed posterior mode makes this small comparison reproducible.
                latent = self.pipe.vae.encode(x).latent_dist.mode() * self.pipe.vae.config.scaling_factor
            values.append(latent.detach().to(dtype=self.pipe.unet.dtype))
        return torch.cat(values)

    def generate(self, prompt: str, *, seed: int = 42, steps: int = 12,
                 resolution: int = 256, scale: float = 1., guidance: float = 7.5) -> dict:
        """Actually run this pipeline, preserving the active image assessment.

        Fresh CPU generator per call gives paired starting noise in this setup;
        cross-device bitwise-identical pictures are not promised. scale changes
        inference adapter strength and never performs training.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('Enter a nonempty prompt')
        seed = _integer(seed, 'seed', 0, 2**31 - 1)
        steps = _integer(steps, 'steps', 2, 40)
        resolution = _integer(resolution, 'resolution', 64, 512)
        if resolution % 8:
            raise ValueError('Resolution must be a multiple of8')
        scale = _finite(scale, 'scale', 0., 2.)
        guidance = _finite(guidance, 'guidance', 1., 12.)
        if self.adapter_name:
            self.pipe.set_adapters(self.adapter_name, adapter_weights=scale)
        self.pipe.unet.eval()
        started = time.perf_counter()
        result = self.pipe(prompt, height=resolution, width=resolution,
            num_inference_steps=steps, guidance_scale=guidance,
            generator=torch.Generator(device='cpu').manual_seed(seed))
        flags = getattr(result, 'nsfw_content_detected', None)
        settings = {'mode': 'live_sd15', 'model_id': MODEL_ID,
            'revision': MODEL_REVISION, 'prompt': prompt, 'seed': seed, 'steps': steps,
            'resolution': resolution, 'guidance': guidance,
            'adapter_scale': scale if self.adapter_name else 0., 'adapter_active': bool(self.adapter_name),
            'device': self.device, 'seconds': time.perf_counter() - started,
            'image_assessment': 'not_assessed' if flags is None else ('blocked' if any(flags) else 'passed')}
        if flags is None:
            raise RuntimeError('No image assessment result is available')
        if any(flags):
            raise BlockedImageError(settings)
        return {'image': result.images[0], 'settings': settings}

    def fit(self, prior_images: list[Image.Image], *, steps: int = 4, resolution: int = 256,
            rank: int = 4, learning_rate: float = .0001, prior_weight: float = 1., seed: int = 42) -> dict:
        """Actual instance+prior denoising loss/backward/AdamW for attention LoRA.

        Uses the six bundled instance views and caller-supplied class images
        generated by the ORIGINAL base before fitting. VAE/text encoder/base
        UNet stay frozen. Instance batch1+prior1; one optimizer update per step.
        Fixed VAE posterior modes are an explicit teaching simplification.
        Short runs demonstrate updating, not successful subject personalization.
        """
        if self.adapter_name is not None or self.training is not None:
            raise ValueError('Create a fresh base lab for another controlled training run')
        steps = _integer(steps, 'steps', 1, 256)
        resolution = _integer(resolution, 'resolution', 64, 512)
        if resolution % 8 or not prior_images:
            raise ValueError('Use a multiple-of8 resolution and nonempty original-base class images')
        rank = _integer(rank, 'rank', 1, 16)
        learning_rate = _finite(learning_rate, 'learning_rate', 1e-7, .001)
        prior_weight = _finite(prior_weight, 'prior_weight')
        seed = _integer(seed, 'seed', 0, 2**31 - 1)
        from diffusers import DDPMScheduler
        from peft import LoraConfig
        data = load_teaching_images()
        images = [image for image, record in zip(data['rgb'], data['manifest']['images']) if record['split'] == 'train']
        before_base = parameter_fingerprint(self.pipe.unet)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.pipe.unet.add_adapter(LoraConfig(r=rank, lora_alpha=rank, lora_dropout=0.,
                init_lora_weights='gaussian', target_modules=['to_q', 'to_k', 'to_v', 'to_out.0']))
        self.adapter_name = 'default'
        for name, value in self.pipe.unet.named_parameters():
            if 'lora_' in name:
                value.data = value.data.float()
        if parameter_fingerprint(self.pipe.unet) != before_base:
            raise RuntimeError('Inserting adapters changed base parameters')
        adapter_before = {n: p.detach().cpu().clone() for n, p in self.pipe.unet.named_parameters() if 'lora_' in n}
        instance_latents = self._latents(images, resolution)
        prior_latents = self._latents(prior_images, resolution)
        instance_hidden = self._embedding(data['manifest']['instance_prompt'])
        prior_hidden = self._embedding(data['manifest']['class_prompt'])
        scheduler = DDPMScheduler.from_config(self.pipe.scheduler.config)
        optimizer = torch.optim.AdamW([p for p in self.pipe.unet.parameters() if p.requires_grad], lr=learning_rate, weight_decay=0.)
        self.pipe.unet.enable_gradient_checkpointing()
        self.pipe.unet.train()
        g = torch.Generator().manual_seed(seed + 100)
        history, first_gradients = [], {}
        started = time.perf_counter()
        for step in range(steps):
            clean = torch.cat([instance_latents[step % len(instance_latents)][None], prior_latents[step % len(prior_latents)][None]])
            hidden = torch.cat([instance_hidden, prior_hidden])
            noise = torch.randn(clean.shape, generator=g, dtype=torch.float32).to(clean.device, dtype=clean.dtype)
            t = torch.randint(0, scheduler.config.num_train_timesteps, (2,), generator=g).to(clean.device)
            noisy = scheduler.add_noise(clean, noise, t)
            if scheduler.config.prediction_type == 'epsilon':
                target = noise
            elif scheduler.config.prediction_type == 'v_prediction':
                target = scheduler.get_velocity(clean, noise, t)
            else:
                raise ValueError('Unsupported scheduler prediction target')
            optimizer.zero_grad(set_to_none=True)
            predicted = self.pipe.unet(noisy, t, encoder_hidden_states=hidden).sample
            losses = prior_loss_components(predicted.float(), target.float(), 1, prior_weight)
            losses['total'].backward()
            trainables = [p for p in self.pipe.unet.parameters() if p.requires_grad]
            norm = float(torch.nn.utils.clip_grad_norm_(trainables, 1.))
            if not math.isfinite(norm) or norm == 0:
                raise RuntimeError('Expected finite nonzero LoRA gradients')
            if step == 0:
                for kind in ('lora_A', 'lora_B'):
                    first_gradients[kind] = float(torch.sqrt(sum(p.grad.float().square().sum()
                        for n, p in self.pipe.unet.named_parameters() if kind in n and p.grad is not None)))
            optimizer.step()
            history.append({'update': step + 1, **{k + '_loss': float(v.detach()) for k, v in losses.items()},
                            'gradient_norm': norm, 'seconds': time.perf_counter() - started})
            print('SD15_UPDATE', step + 1, 'loss', round(history[-1]['total_loss'], 5), flush=True)
        self.pipe.unet.eval()
        changes = {n: float((p.detach().cpu() - adapter_before[n]).norm())
                   for n, p in self.pipe.unet.named_parameters() if n in adapter_before}
        after_base = parameter_fingerprint(self.pipe.unet)
        self._probe = (noisy[:1].detach().clone(), t[:1].detach().clone(), hidden[:1].detach().clone())
        self.training = {'mode': 'actual_sd15_lora_training', 'state': self.state(), 'steps': steps,
            'resolution': resolution, 'rank': rank, 'alpha': rank, 'alpha_over_rank': 1.,
            'learning_rate': learning_rate, 'instance_batch': 1, 'prior_batch': 1,
            'prior_weight': prior_weight, 'instance_prompt': data['manifest']['instance_prompt'],
            'class_prompt': data['manifest']['class_prompt'], 'instance_view_ids': data['ids']['train'],
            'validation_view_ids': data['ids']['validation'], 'seed': seed,
            'vae_posterior': 'fixed mode, not stochastic posterior sampling',
            'base_before': before_base, 'base_after': after_base, 'base_unchanged': before_base == after_base,
            'parameter_changes': changes, 'first_gradients': first_gradients,
            'history': history, 'seconds': time.perf_counter() - started,
            'sdxl_training_executed': False}
        if not self.training['base_unchanged'] or not any(v > 0 for v in changes.values()):
            raise RuntimeError('Frozen-base/actual-adapter-update invariant failed')
        return self.training

    def save_reload_check(self, directory: str | Path) -> dict:
        """Save real Diffusers safetensors LoRA, unload/reload, compare U-Net outputs.

        Writes adapter weights and public base metadata only, never base weights.
        Restores the adapter on this pipeline under the course name. Does not
        merge it. Same base/revision is required for loading this artifact later.
        """
        if self.training is None or self._probe is None:
            raise ValueError('Actually fit before saving')
        from peft.utils import get_peft_model_state_dict
        from diffusers.utils import convert_state_dict_to_diffusers
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        filename = path/'pytorch_lora_weights.safetensors'
        if filename.exists():
            raise FileExistsError('Use a fresh adapter directory')
        self.pipe.set_adapters(self.adapter_name, adapter_weights=1.)
        self.pipe.unet.eval()
        with torch.no_grad():
            before = self.pipe.unet(self._probe[0], self._probe[1], encoder_hidden_states=self._probe[2]).sample.detach().float()
        weights = convert_state_dict_to_diffusers(get_peft_model_state_dict(self.pipe.unet, adapter_name=self.adapter_name))
        self.pipe.save_lora_weights(path, unet_lora_layers=weights, safe_serialization=True)
        meta = {'model_id': MODEL_ID, 'revision': MODEL_REVISION, 'rank': self.training['rank'],
                'sha256': hashlib.sha256(filename.read_bytes()).hexdigest()}
        (path/'base.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        self.pipe.unload_lora_weights()
        self.pipe.load_lora_weights(path, adapter_name='course')
        self.adapter_name = 'course'
        self.pipe.set_adapters('course', adapter_weights=1.)
        self.pipe.unet.eval()
        with torch.no_grad():
            after = self.pipe.unet(self._probe[0], self._probe[1], encoder_hidden_states=self._probe[2]).sample.detach().float()
        error = float((before - after).abs().max())
        tolerance = 1e-5 if self.device == 'cpu' else 5e-3
        if error > tolerance:
            raise RuntimeError('Reloaded adapter predictions differ')
        return {'max_abs_prediction_error': error, 'tolerance': tolerance,
                'adapter_sha256': meta['sha256'], 'bytes': filename.stat().st_size,
                'base_model_id': MODEL_ID, 'base_revision': MODEL_REVISION,
                'base_unchanged_by_training': self.training['base_unchanged'], 'format': 'Diffusers safetensors LoRA'}

    def fusion_check(self) -> dict:
        """Last-step actual float fusion check; fusion changes base weights in place.

        Compare noise prediction before/after within dtype tolerance. Do this
        only after saving the adapter. To continue controlled experiments load
        a fresh base; unfusing is not a promise of byte-identical original base.
        """
        if self._probe is None:
            raise ValueError('Fit and save an adapter first')
        self.pipe.set_adapters(self.adapter_name, adapter_weights=1.)
        self.pipe.unet.eval()
        with torch.no_grad():
            before = self.pipe.unet(self._probe[0], self._probe[1], encoder_hidden_states=self._probe[2]).sample.detach().float()
        self.pipe.fuse_lora(lora_scale=1., safe_fusing=True, adapter_names=[self.adapter_name])
        with torch.no_grad():
            after = self.pipe.unet(self._probe[0], self._probe[1], encoder_hidden_states=self._probe[2]).sample.detach().float()
        error = float((before - after).abs().max())
        tolerance = 5e-4 if self.device == 'cpu' else .02
        if error > tolerance:
            raise RuntimeError('Fusion equivalence check failed')
        return {'max_abs_prediction_error': error, 'tolerance': tolerance, 'base_weights_modified_by_fusion': True}
