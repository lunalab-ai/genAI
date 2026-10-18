"""W06A experiments: schedules, prediction targets, architecture and sampling.

Independent teaching code; existing W05B APIs and checkpoints are unchanged.
No download, training, server launch or random draw occurs on import.
"""
from __future__ import annotations

import gzip
import time
import numpy as np
import pandas as pd
import torch
from diffusers import DDPMScheduler, DDIMScheduler
from .autoencoder import MNISTSplit
from .diffusion import TinyTimeUNet, make_scheduler, training_step, sample_diffusion


def make_noise_schedule(name: str = 'cosine', *, steps: int = 100,
                        beta_end: float = .02) -> DDPMScheduler:
    """Create a FORWARD experiment schedule, without changing any trained model.

    name='cosine' or 'linear'; steps=100, beta_end=.02 (linear only).
    Returns a new scheduler; betas and alphas_cumprod are CPU float32 [steps].
    Linear beta_start=.0001. Changing training betas at inference is invalid.
    Example: make_noise_schedule('linear').alphas_cumprod[-1].item().
    """
    if name not in ('linear', 'cosine'):
        raise ValueError('name must be linear or cosine')
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 2:
        raise ValueError('steps must be an integer >= 2')
    if not .0001 <= beta_end < 1:
        raise ValueError('beta_end must lie in [.0001,1)')
    return DDPMScheduler(num_train_timesteps=steps, beta_start=.0001,
                         beta_end=beta_end, prediction_type='epsilon',
                         beta_schedule='linear' if name == 'linear' else 'squaredcos_cap_v2')


def coefficient_table(scheduler: DDPMScheduler) -> pd.DataFrame:
    """Return ordered t,beta,alpha_bar,signal,noise,snr rows; no mutation.

    t=0 denotes the first noisy level, not mathematical clean x0.
    snr=alpha_bar/(1-alpha_bar) assumes unit signal variance. With clean
    variance V and scale c, the data SNR is c**2*V*snr.
    Example: coefficient_table(make_noise_schedule()).iloc[[0,49,99]].
    """
    a = scheduler.alphas_cumprod.detach().cpu().numpy()
    return pd.DataFrame(dict(t=np.arange(len(a)), beta=scheduler.betas.numpy(),
                             alpha_bar=a, signal=np.sqrt(a), noise=np.sqrt(1-a),
                             snr=a/(1-a)))


def _pair(clean: torch.Tensor, noise: torch.Tensor) -> None:
    if (clean.shape != noise.shape or clean.ndim < 1 or clean.numel() == 0
            or clean.dtype != torch.float32 or noise.dtype != torch.float32
            or clean.device.type != 'cpu' or noise.device.type != 'cpu'
            or not torch.isfinite(clean).all() or not torch.isfinite(noise).all()):
        raise ValueError('Expected same-shaped, nonempty finite CPU float32 tensors')


def mix_at(clean: torch.Tensor, noise: torch.Tensor, t: int,
           scheduler: DDPMScheduler, *, scale: float = 1.) -> torch.Tensor:
    """Return noisy tensor a*(scale*clean)+b*noise at one code t.

    clean/noise: same-shaped finite CPU float32 (any image resolution).
    t is integral in [0,steps); scale is finite and >0, default1.
    No RNG, clipping, download, training or input mutation. noise is supplied
    standard normal. Example: mix_at(x,eps,49,make_noise_schedule()).
    """
    _pair(clean, noise)
    if isinstance(t, bool) or not isinstance(t, (int, np.integer)) or not 0 <= t < len(scheduler.betas):
        raise ValueError('t is outside the schedule')
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('scale must be finite and positive')
    a = scheduler.alphas_cumprod[int(t)]
    return a.sqrt() * (float(scale)*clean) + (1-a).sqrt() * noise


def prediction_targets(clean: torch.Tensor, noise: torch.Tensor,
                       alpha_bar: float) -> dict[str, torch.Tensor]:
    """Return known training targets epsilon, sample(x0), v_prediction.

    Finite same-shaped CPU float32 tensors; scalar 0<alpha_bar<1.
    v=a*epsilon-b*x0, with a=sqrt(alpha_bar), b=sqrt(1-alpha_bar).
    These are constructed ground truths, NOT outputs of a trained network.
    Does not alter inputs. Example: prediction_targets(x,eps,.64).
    """
    _pair(clean, noise)
    if not 0 < alpha_bar < 1:
        raise ValueError('Require 0 < alpha_bar < 1')
    a, b = float(alpha_bar)**.5, (1-float(alpha_bar))**.5
    return dict(epsilon=noise.clone(), sample=clean.clone(), v_prediction=a*noise-b*clean)


def recover_clean(noisy: torch.Tensor, prediction: torch.Tensor,
                  alpha_bar: float, prediction_type: str) -> torch.Tensor:
    """Convert a same-shaped prediction to an unclipped clean-image estimate.

    epsilon: (xt-b*prediction)/a; sample: prediction; v_prediction: a*xt-b*v.
    0<alpha_bar<1 excludes singular endpoints. No randomness/parameter update.
    Exact known targets reconstruct clean; learned errors do not disappear.
    Example: recover_clean(xt,targets['epsilon'],.64,'epsilon').
    """
    _pair(noisy, prediction)
    if not 0 < alpha_bar < 1:
        raise ValueError('Require 0 < alpha_bar < 1')
    a, b = float(alpha_bar)**.5, (1-float(alpha_bar))**.5
    if prediction_type == 'epsilon': return (noisy-b*prediction)/a
    if prediction_type == 'sample': return prediction.clone()
    if prediction_type == 'v_prediction': return a*noisy-b*prediction
    raise ValueError('prediction_type must be epsilon, sample or v_prediction')


def trace_unet(model: TinyTimeUNet, noisy: torch.Tensor,
               timesteps: torch.Tensor) -> pd.DataFrame:
    """Observe actual forward shapes with temporary hooks; remove them on exit.

    Inputs follow TinyTimeUNet.forward: CPU [B,1,28,28] and int64[B].
    Returns block,input_shape,output_shape rows in execution order.
    Disables gradients; preserves train/eval mode and model parameters.
    Example: trace_unet(model,xt,torch.tensor([49])).
    """
    rows, handles = [], []
    def observer(name):
        def hook(module, args, output):
            rows.append(dict(block=name,input_shape=str(tuple(args[0].shape)),
                             output_shape=str(tuple(output.shape))))
        return hook
    was_training = model.training
    try:
        for name in ('enc1','enc2','middle','dec2','dec1','out'):
            handles.append(getattr(model,name).register_forward_hook(observer(name)))
        model.eval()
        with torch.inference_mode(): model(noisy,timesteps)
    finally:
        for handle in handles: handle.remove()
        model.train(was_training)
    return pd.DataFrame(rows)


def select_digit(data: MNISTSplit, digit: int | None = None) -> dict:
    """Select a project dataset from an already integrity-checked MNISTSplit.

    digit=None keeps all digits; 0..9 filters EACH existing split independently.
    Reads cached official train labels (verified by load_mnist); no download.
    Returns train/validation/test in [-1,1], split IDs and counts. Labels select
    the population; they are not inputs to the unconditional diffusion model.
    Empty selections fail. Example: select_digit(load_mnist(),5).
    """
    if digit is not None and (isinstance(digit,bool) or not isinstance(digit,int) or digit not in range(10)):
        raise ValueError('digit must be None or an integer 0..9')
    from pathlib import Path
    payload=gzip.decompress((Path(data.cache_dir)/'train-labels-idx1-ubyte.gz').read_bytes())
    labels=np.frombuffer(payload,dtype=np.uint8,offset=8)
    if len(labels)!=60000: raise ValueError('Unexpected MNIST label count')
    result={'digit':digit}
    for split in ('train','validation','test'):
        ids=getattr(data,split+'_ids')
        ys=data.test_labels if split=='test' else labels[ids]
        keep=np.ones(len(ids),dtype=bool) if digit is None else ys==digit
        if not keep.any(): raise ValueError('Empty project split: '+split)
        result[split]=getattr(data,split)[torch.from_numpy(keep.copy())]*2-1
        result[split+'_ids']=ids[keep].copy()
    result['counts']={s:len(result[s]) for s in ('train','validation','test')}
    return result


def fit_diffusion_project(clean: torch.Tensor, *, steps: int = 32,
                          batch_size: int = 16, learning_rate: float = .001,
                          seed: int = 20261005) -> tuple[TinyTimeUNet, pd.DataFrame]:
    """Train a NEW time-conditioned epsilon predictor using real Adam updates.

    clean is nonempty CPU float32 [N,1,28,28] in [-1,1]. Defaults:32 steps,
    batch16, lr.001, seed20261005. Samples batches with replacement, random
    t=0..99 and Gaussian noise; fixed W05B cosine schedule. Does not touch old
    checkpoints. Returns model and per-step loss/gradient/change/timing rows.
    32 steps illustrate optimization, not a fully trained image generator.
    Example: model,history=fit_diffusion_project(project['train'],steps=8).
    """
    if clean.ndim!=4 or clean.shape[1:]!=(1,28,28) or not len(clean):
        raise ValueError('Expected nonempty [N,1,28,28] images')
    _pair(clean,clean)
    if clean.min() < -1 or clean.max() > 1: raise ValueError('Expected [-1,1] clean images')
    if any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in (steps,batch_size)):
        raise ValueError('steps and batch_size must be positive integers')
    if not np.isfinite(learning_rate) or learning_rate<=0: raise ValueError('learning_rate must be positive')
    model=TinyTimeUNet(seed=seed)
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
    generator=torch.Generator().manual_seed(seed)
    rows=[]; started=time.perf_counter()
    for step in range(steps):
        batch=clean[torch.randint(len(clean),(batch_size,),generator=generator)]
        t=torch.randint(100,(batch_size,),generator=generator)
        noise=torch.randn(batch.shape,generator=generator)
        row=training_step(model,optimizer,batch,t,noise)
        rows.append(dict(step=step+1,**row,elapsed_seconds=time.perf_counter()-started))
    return model.eval(),pd.DataFrame(rows)


@torch.inference_mode()
def sample_with_sampler(model: TinyTimeUNet, *, method: str = 'DDIM', steps: int = 20,
                         count: int = 8, seed: int = 42) -> dict:
    """Generate from Gaussian noise with the checkpoint's original trained betas.

    DDPM requires100 steps and delegates to the existing sampler. DDIM accepts
    2..100 steps, eta=0, trailing spacing (starts at code99). Both retain the
    100-step cosine/epsilon training definition. No clean-image input or fit.
    Returns images,initial,frames,steps,seed,method,seconds; frames contain
    input,predicted_clean,previous,t. Preserves model mode/parameters.
    Same seed pairs initial noise, not every later random operation across
    different samplers. Example: sample_with_sampler(model,steps=20,count=4).
    """
    if method not in ('DDPM','DDIM'): raise ValueError('method must be DDPM or DDIM')
    if isinstance(steps,bool) or not isinstance(steps,int) or not 2<=steps<=100:
        raise ValueError('steps must be an integer from 2 to 100')
    if isinstance(count,bool) or not isinstance(count,int) or not 1<=count<=64:
        raise ValueError('count must be an integer from 1 to 64')
    if method=='DDPM' and steps!=100: raise ValueError('This DDPM baseline uses all 100 steps')
    start=time.perf_counter()
    if method=='DDPM':
        result=sample_diffusion(model,count=count,seed=seed)
        return dict(result,method=method,seconds=time.perf_counter()-start)
    scheduler=DDIMScheduler.from_config(make_scheduler().config,timestep_spacing='trailing')
    scheduler.set_timesteps(steps)
    generator=torch.Generator().manual_seed(seed)
    x=torch.randn((count,1,28,28),generator=generator)
    initial=x.clone();frames=[]; was_training=model.training
    capture=set(np.linspace(0,steps-1,5,dtype=int).tolist())
    model.eval()
    try:
        for index,timestep in enumerate(scheduler.timesteps):
            t=int(timestep);eps=model(x,torch.full((count,),t,dtype=torch.long))
            update=scheduler.step(eps,timestep,x,eta=0,use_clipped_model_output=True)
            if index in capture:
                frames.append(dict(t=t,input=x.clone(),predicted_clean=update.pred_original_sample.clone(),previous=update.prev_sample.clone()))
            x=update.prev_sample
        if not torch.isfinite(x).all(): raise RuntimeError('Nonfinite generated images')
    finally: model.train(was_training)
    return dict(images=x,initial=initial,frames=frames,steps=steps,seed=seed,method=method,seconds=time.perf_counter()-start)
