"""W06B pinned SD 1.5 inspection, real generation, and honest recorded mode."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch

MODEL_ID='stable-diffusion-v1-5/stable-diffusion-v1-5'
MODEL_REVISION='451f4fe16113bff5a5d2269ed5ad43b0592e9a14'
ASSETS=Path(__file__).parent/'assets'/'w06b'
DEFAULT_PROMPT='a studio photograph of a red ceramic mug on a white table'

def cfg_combine(unconditional:torch.Tensor,conditional:torch.Tensor,scale:float)->torch.Tensor:
    """Pure CFG algebra u+s*(c-u); same-shaped finite tensors, finite scale>=0.

    Mathematical s=0 returns u, s=1 returns c. Diffusers SD pipeline skips
    its two-branch CFG for API guidance_scale<=1; that API does not use this
    formula to return unconditional generation at 0. No weights/state change.
    """
    if unconditional.shape!=conditional.shape or unconditional.dtype!=conditional.dtype or unconditional.device!=conditional.device:
        raise ValueError('CFG inputs must share shape, dtype and device')
    if not math.isfinite(float(scale)) or scale<0 or not torch.isfinite(unconditional).all() or not torch.isfinite(conditional).all():
        raise ValueError('CFG requires finite inputs and nonnegative scale')
    return unconditional+float(scale)*(conditional-unconditional)

def load_sd15(*,device:str='cpu',local_files_only:bool=False):
    """Load revision-pinned real SD 1.5 + DDIM; CPU float32 or CUDA float16.

    First call downloads ~2.6 GB; cache reused. Keeps safety checker active.
    Does not fine-tune. CUDA unavailability raises, with no CPU substitution.
    local_files_only=True raises on cache miss. Returns Diffusers pipeline.
    """
    from diffusers import StableDiffusionPipeline,DDIMScheduler
    if device not in ('cpu','cuda'):raise ValueError('Choose cpu or cuda')
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA GPU unavailable; select recorded mode explicitly')
    pipe=StableDiffusionPipeline.from_pretrained(MODEL_ID,revision=MODEL_REVISION,variant='fp16',
        torch_dtype=torch.float16 if device=='cuda' else torch.float32,low_cpu_mem_usage=False,
        local_files_only=local_files_only)
    pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    if pipe.safety_checker is None:raise RuntimeError('Expected active safety checker')
    return pipe

@torch.inference_mode()
def text_condition(pipe,prompt:str)->dict:
    """Inspect real token IDs and token-wise hidden states, not pooled CLIP search.

    Returns tokens table, input_ids [1,77], hidden [1,77,768] CPU tensors,
    unpadded_length and truncated flag. Does not mutate weights or generate.
    Boundary/padding tokens remain visible; length 77 is not 77 words.
    """
    if not isinstance(prompt,str) or not prompt.strip():raise ValueError('Enter a nonempty prompt')
    tokenizer=pipe.tokenizer
    raw=tokenizer(prompt,add_special_tokens=True)['input_ids']
    encoded=tokenizer(prompt,padding='max_length',max_length=tokenizer.model_max_length,truncation=True,return_tensors='pt')
    ids=encoded.input_ids.to(pipe.device)
    hidden=pipe.text_encoder(ids)[0]
    mask=encoded.attention_mask[0].tolist()
    tokens=[dict(position=i,token_id=int(token),token=tokenizer.convert_ids_to_tokens(int(token)),is_padding=not bool(mask[i])) for i,token in enumerate(ids[0])]
    return dict(tokens=tokens,input_ids=ids.cpu(),hidden=hidden.float().cpu(),unpadded_length=len(raw),truncated=len(raw)>tokenizer.model_max_length)

@torch.inference_mode()
def vae_roundtrip(pipe,image:Image.Image)->dict:
    """Actual encode/decode at 512 square; posterior mode for reproducible comparison.

    Input PIL RGB resized to 512x512. Returns source/reconstruction uint8,
    scaled_latent [1,4,64,64], range [0,1] image MSE and config scale.
    Decode divides by the SAME config scale. This is reconstruction, not new
    generation; four latent channels are not independently interpretable RGB.
    """
    source=np.asarray(image.convert('RGB').resize((512,512)),dtype=np.uint8).copy()
    x=torch.from_numpy(source).permute(2,0,1).unsqueeze(0).float()/127.5-1
    x=x.to(device=pipe.device,dtype=pipe.vae.dtype)
    scale=float(pipe.vae.config.scaling_factor)
    z=pipe.vae.encode(x).latent_dist.mode()*scale
    decoded=pipe.vae.decode(z/scale).sample
    restored=(decoded.float()/2+.5).clamp(0,1).cpu()
    mse=float(((restored-(x.float().cpu()/2+.5))**2).mean())
    reconstruction=np.rint(restored[0].permute(1,2,0).numpy()*255).astype(np.uint8)
    return dict(source=source,reconstruction=reconstruction,scaled_latent=z.float().cpu(),scaling_factor=scale,mse=mse,posterior='mode')

@torch.inference_mode()
def generate_sd(pipe,prompt:str=DEFAULT_PROMPT,*,seed:int=42,guidance:float=7.5,steps:int=20)->dict:
    """Real 512x512 DDIM generation with active safety checker and fixed weights.

    Returns PIL image, full settings/time and five latent snapshots. Same CPU
    initial-noise generator seed controls comparisons; cross-device bitwise
    identity is not promised. guidance>=1; 1 means plain conditional prediction.
    Safe blocked outputs raise, never silently replaced by cached pictures.
    """
    if not isinstance(prompt,str) or not prompt.strip():raise ValueError('Prompt is required')
    if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<2**32:raise ValueError('seed must be integer 0..2^32-1')
    if not math.isfinite(float(guidance)) or not 1<=guidance<=20:raise ValueError('guidance must lie in [1,20]')
    if isinstance(steps,bool) or not isinstance(steps,int) or not 2<=steps<=50:raise ValueError('steps must be 2..50')
    if pipe.safety_checker is None:raise RuntimeError('Safety checker must remain active')
    frames=[];indices={0,steps//4,steps//2,3*steps//4,steps-1}
    def capture(pipeline,index,timestep,kwargs):
        if index in indices:frames.append(dict(step=index+1,timestep=int(timestep),latent=kwargs['latents'].detach().float().cpu().clone()))
        return kwargs
    if pipe.device.type=='cuda':torch.cuda.synchronize()
    started=time.perf_counter()
    result=pipe(prompt,generator=torch.Generator(device='cpu').manual_seed(seed),height=512,width=512,
                num_inference_steps=steps,guidance_scale=float(guidance),callback_on_step_end=capture,
                callback_on_step_end_tensor_inputs=['latents'])
    if pipe.device.type=='cuda':torch.cuda.synchronize()
    if result.nsfw_content_detected is None or any(result.nsfw_content_detected):
        raise RuntimeError('Safety checker blocked or did not assess this output; choose another prompt')
    settings=dict(model_id=MODEL_ID,revision=MODEL_REVISION,prompt=prompt,seed=seed,guidance_scale=float(guidance),
                  steps=steps,scheduler=type(pipe.scheduler).__name__,height=512,width=512,device=pipe.device.type,
                  dtype=str(pipe.unet.dtype),seconds=time.perf_counter()-started,safety_checked=True,mode='live generation')
    return dict(image=result.images[0],settings=settings,frames=frames)

def load_recorded_sd()->tuple[dict,list[dict]]:
    """Read and verify actual generated records; does NOT run SD or need GPU.

    Returns metadata and gallery entries. Each entry includes exact settings
    plus image_path. All listed image/tensor hashes are validated on load.
    Missing or corrupt file raises. Never a fallback for failed live generation.
    """
    record=json.loads((ASSETS/'sd-records.json').read_text(encoding='utf-8'))
    for name,digest in record['files'].items():
        if hashlib.sha256((ASSETS/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('SD record checksum mismatch: '+name)
    gallery=[dict(item,image_path=str(ASSETS/item['file'])) for item in record['gallery']]
    return record,gallery

def load_recorded_tensors()->dict:
    """Return saved real SD component tensors as NumPy arrays; no inference.

    Provenance/settings in sd-records.json; use load_recorded_sd to inspect.
    Arrays include padded IDs, hidden states, scaled VAE latent, reconstruction,
    unconditional/conditional noise predictions at a fixed noisy latent state.
    """
    load_recorded_sd()
    with np.load(ASSETS/'sd-tensors.npz',allow_pickle=False) as data:return {k:data[k].copy() for k in data.files}
