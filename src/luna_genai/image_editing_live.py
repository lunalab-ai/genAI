"""Explicit optional full Stable Diffusion execution for W11A.

CPU records and a user's live GPU run share these loaders. No network, model
loading, training, paid API or fallback occurs on import. SD1.5 is deliberately
an alternate backend to the book's larger SDXL examples.
"""
from __future__ import annotations
import time
import torch
from .stable_diffusion import load_sd15, MODEL_ID, MODEL_REVISION

CONTROL_ID = 'lllyasviel/control_v11p_sd15_canny'
CONTROL_REVISION = '115a470d547982438f70198e353a921996e2e819'
IP_ID = 'h94/IP-Adapter'
IP_REVISION = '018e402774aeeddd60609b4ecdb7e298259dc729'
INSTRUCTION_ID = 'timbrooks/instruct-pix2pix'
INSTRUCTION_REVISION = '31519b5cb02a7fd89b906d88731cd4d6a7bbf88d'


def make_edit_pipeline(kind: str, *, device: str = 'cuda', base=None, offline: bool = False):
    """Load explicit kind: img2img/inpaint/semantic/inversion/instruction/control.

    First call may download multiple GB. CPU float32, CUDA float16. Keeps
    safety checking. Control uses SD1.5 Canny, not SDXL depth; instruction uses
    InstructPix2Pix, not CosXL. No silent alternative if memory/cache fails.
    """
    from diffusers import (StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline,
        SemanticStableDiffusionPipeline, LEditsPPPipelineStableDiffusion,
        StableDiffusionInstructPix2PixPipeline, StableDiffusionControlNetPipeline,
        ControlNetModel, DPMSolverMultistepScheduler)
    if device not in ('cpu', 'cuda') or (device == 'cuda' and not torch.cuda.is_available()):
        raise RuntimeError('Select an available device explicitly; no automatic fallback')
    dtype = torch.float16 if device == 'cuda' else torch.float32
    if kind == 'instruction':
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            INSTRUCTION_ID, revision=INSTRUCTION_REVISION, variant='fp16',
            torch_dtype=dtype, local_files_only=offline, low_cpu_mem_usage=True).to(device)
    else:
        base = base or load_sd15(device=device, local_files_only=offline)
        classes = {'img2img': StableDiffusionImg2ImgPipeline, 'inpaint': StableDiffusionInpaintPipeline,
                   'semantic': SemanticStableDiffusionPipeline, 'inversion': LEditsPPPipelineStableDiffusion}
        if kind == 'control':
            control = ControlNetModel.from_pretrained(CONTROL_ID, revision=CONTROL_REVISION,
                variant='fp16', torch_dtype=dtype, local_files_only=offline, use_safetensors=True)
            pipe = StableDiffusionControlNetPipeline.from_pipe(base, controlnet=control).to(device)
        elif kind in classes:
            pipe = classes[kind].from_pipe(base).to(device)
            if kind == 'inversion':
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(base.scheduler.config,
                    algorithm_type='sde-dpmsolver++', solver_order=2)
        else:
            raise ValueError('Unknown editing pipeline kind')
    if pipe.safety_checker is None:
        raise RuntimeError('Active safety checker required')
    pipe.enable_attention_slicing()
    return pipe


def attach_ip_adapter(pipe, *, offline: bool = False):
    """Install pinned trained SD1.5 image-attention weights and CLIP image encoder.

    Mutates pipeline attention processors. Downloads cached public weights;
    does not train them. SDXL block routing is not copied to SD1.5.
    """
    # Diffusers 0.40.0 does not forward revision to its internal CLIP loader.
    # Register the explicitly pinned encoder first; do not resolve mutable main.
    from transformers import CLIPVisionModelWithProjection
    # IP-Adapter installs its own attention processors. SlicedAttnProcessor
    # cannot be reconstructed by the 0.40.0 adapter converter without slice_size.
    pipe.disable_attention_slicing()
    encoder = CLIPVisionModelWithProjection.from_pretrained(IP_ID, revision=IP_REVISION,
        subfolder='models/image_encoder', local_files_only=offline,
        dtype=pipe.unet.dtype, use_safetensors=True).to(pipe.device)
    pipe.register_modules(image_encoder=encoder)
    pipe.load_ip_adapter(IP_ID, revision=IP_REVISION, subfolder='models',
        weight_name='ip-adapter_sd15.safetensors', local_files_only=offline,
        image_encoder_folder='models/image_encoder')
    return pipe


def checked_generate(pipe, *, seed: int = 42, **settings) -> dict:
    """Run a real pipeline and require an actually assessed, unblocked image.

    Returns image, seconds, pipeline class and JSON-compatible supplied settings.
    This method never returns a recorded image after an execution error.
    """
    if pipe.safety_checker is None:
        raise RuntimeError('Safety checker required')
    started = time.perf_counter()
    output = pipe(generator=torch.Generator(device=pipe.device).manual_seed(seed), **settings)
    flags = output.nsfw_content_detected
    if flags is None or any(flags) or not output.images:
        raise RuntimeError('Image blocked/unassessed; no replacement is shown')
    serial = {k: v for k, v in settings.items() if isinstance(v, (str, int, float, bool, type(None)))}
    return {'image': output.images[0], 'seconds': time.perf_counter() - started,
            'settings': serial, 'seed': seed, 'pipeline': type(pipe).__name__,
            'device': pipe.device.type, 'dtype': str(pipe.unet.dtype), 'safety_checked': True}
