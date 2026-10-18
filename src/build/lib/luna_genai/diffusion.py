"""Inspect a small, unconditional MNIST DDPM on CPU (W05B).

Images: CPU float32 [B,1,28,28], clean range [-1,1]. Code timesteps 0..99
refer to the first through hundredth noising levels; clean x0 is separate.
No downloads, training, or global RNG mutation occur on import.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from diffusers import DDPMScheduler

ASSETS = Path(__file__).parent / "assets" / "w05b"
TIMESTEPS = 100


def make_scheduler() -> DDPMScheduler:
    """Return a fresh 100-step cosine DDPM epsilon scheduler; no learned weights.

    clip_sample=True clips estimated clean images during sampling, not training
    noise. Its step() includes the scheduled mean and variance, not just x-eps.
    Example: scheduler=make_scheduler(); scheduler.alphas_cumprod.shape == (100,).
    The schedule is fixed for this lesson, not a schedule-comparison experiment.
    """
    return DDPMScheduler(num_train_timesteps=TIMESTEPS,
                         beta_schedule="squaredcos_cap_v2",
                         prediction_type="epsilon", clip_sample=True)


def _images(x: torch.Tensor, *, clean: bool = False) -> None:
    if x.device.type != 'cpu' or x.dtype != torch.float32 or x.ndim != 4 or x.shape[1:] != (1,28,28) or len(x) < 1:
        raise ValueError('Expected nonempty CPU float32 [B,1,28,28]')
    if not torch.isfinite(x).all() or (clean and (x.min() < -1 or x.max() > 1)):
        raise ValueError('Images must be finite; clean images must lie in [-1,1]')


def _times(t: torch.Tensor, n: int) -> None:
    if t.device.type != 'cpu' or t.dtype != torch.long or t.shape != (n,) or (t < 0).any() or (t >= TIMESTEPS).any():
        raise ValueError('Timesteps must be CPU int64 [B], each in 0..99')


def forward_noise(clean: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Apply sqrt(alpha_bar)*clean + sqrt(1-alpha_bar)*noise, preserving shape.

    All images are CPU float32 [B,1,28,28]; timesteps is int64 [B], 0..99.
    clean is in [-1,1]; noise is a supplied standard-normal draw, not a label.
    Does not clamp noisy output, draw randomness, or train a network.
    Example: forward_noise(x, torch.full((len(x),),49), torch.randn_like(x)).
    """
    _images(clean, clean=True); _images(noise); _times(timesteps,len(clean))
    if noise.shape != clean.shape: raise ValueError('Noise and clean shapes must match')
    return make_scheduler().add_noise(clean, noise, timesteps)


class TimeBlock(nn.Module):
    """Residual two-convolution block; add projected [B,64] time features.

    Internal U-Net component. Spatial size is unchanged; channels change from
    in_channels to out_channels. No down/up sampling or optimizer step here.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,3,padding=1)
        self.norm1=nn.GroupNorm(4,out_channels)
        self.time=nn.Linear(64,out_channels)
        self.conv2=nn.Conv2d(out_channels,out_channels,3,padding=1)
        self.norm2=nn.GroupNorm(4,out_channels)
        self.skip=nn.Conv2d(in_channels,out_channels,1) if in_channels!=out_channels else nn.Identity()

    def forward(self,x:torch.Tensor,time_features:torch.Tensor)->torch.Tensor:
        """Return residual features; broadcast time along height and width."""
        h=F.silu(self.norm1(self.conv1(x)))
        h=h+self.time(time_features)[:,:,None,None]
        h=self.norm2(self.conv2(F.silu(h)))
        return F.silu(h+self.skip(x))


class TinyTimeUNet(nn.Module):
    """CPU epsilon predictor: [B,1,28,28] and int64 [B] -> [B,1,28,28].

    Channels 16->32->64->32->16; sizes 28->14->7->14->28. Skip concatenation
    retains fine spatial features. Sinusoidal time features enter every block.
    seed=20260930 initializes weights locally without training or downloading.
    This small teaching model differs from the textbook RGB 64x64 U-Net.
    Example: model=TinyTimeUNet(); prediction=model(noisy, timesteps).
    """
    def __init__(self,seed:int=20260930):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.time_mlp=nn.Sequential(nn.Linear(32,64),nn.SiLU(),nn.Linear(64,64))
            self.enc1=TimeBlock(1,16); self.enc2=TimeBlock(16,32)
            self.middle=TimeBlock(32,64)
            self.dec2=TimeBlock(96,32); self.dec1=TimeBlock(48,16)
            self.out=nn.Conv2d(16,1,1)

    def forward(self,noisy:torch.Tensor,timesteps:torch.Tensor)->torch.Tensor:
        """Predict epsilon without parameter updates; do not pass true epsilon in.

        Input/output share the image shape but different meanings. The output
        is unbounded predicted noise, not a sigmoid image or a digit class.
        """
        _images(noisy); _times(timesteps,len(noisy))
        frequencies=torch.exp(-math.log(10000)*torch.arange(16)/15)
        angles=timesteps.float()[:,None]*frequencies[None,:]
        time=self.time_mlp(torch.cat([angles.sin(),angles.cos()],dim=1))
        h1=self.enc1(noisy,time)
        h2=self.enc2(F.avg_pool2d(h1,2),time)
        h3=self.middle(F.avg_pool2d(h2,2),time)
        u2=self.dec2(torch.cat([F.interpolate(h3,size=(14,14),mode='nearest'),h2],dim=1),time)
        u1=self.dec1(torch.cat([F.interpolate(u2,size=(28,28),mode='nearest'),h1],dim=1),time)
        return self.out(u1)


def training_step(model:TinyTimeUNet, optimizer:torch.optim.Optimizer,
                  clean:torch.Tensor, timesteps:torch.Tensor, noise:torch.Tensor)->dict:
    """Perform one real MSE/backward/optimizer step and return measured scalars.

    Mutates model parameters and optimizer state; sets model.train(). Inputs
    follow forward_noise(). target is the supplied epsilon, never a digit ID.
    Returns loss, gradient_norm, backward_parameter_change, parameter_change.
    Example: training_step(model, Adam(model.parameters()), x, t, epsilon).
    """
    model.train(); noisy=forward_noise(clean,timesteps,noise)
    first=next(model.parameters()); before=first.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    prediction=model(noisy,timesteps)
    loss=F.mse_loss(prediction,noise)
    if not torch.isfinite(loss): raise RuntimeError('Nonfinite diffusion loss')
    loss.backward()
    gradient=float(torch.sqrt(sum(p.grad.square().sum() for p in model.parameters() if p.grad is not None)))
    if not math.isfinite(gradient): raise RuntimeError('Nonfinite gradient')
    backward_change=float((first.detach()-before).abs().sum())
    optimizer.step()
    return dict(loss=float(loss.detach()),gradient_norm=gradient,
                backward_parameter_change=backward_change,
                parameter_change=float((first.detach()-before).abs().sum()))


@torch.inference_mode()
def evaluate_noise(model:TinyTimeUNet,clean:torch.Tensor,*,seed:int=901,
                    levels:tuple[int,...]=(0,24,49,74,99))->pd.DataFrame:
    """Evaluate fixed held-out images and fixed Gaussian noise at each code t.

    Default five levels, seed901. Returns columns t,mse,zero_mse,images. A zero
    epsilon predictor gives ~1 MSE. Same noise is reused across levels for a
    controlled marginal comparison, not a sampled forward Markov path.
    Preserves model training mode; no optimizer updates. Use validation/test,
    not the training batch, when comparing checkpoints.
    """
    _images(clean,clean=True)
    noise=torch.randn(clean.shape,generator=torch.Generator().manual_seed(seed))
    was_training=model.training; model.eval(); rows=[]
    try:
        for level in levels:
            t=torch.full((len(clean),),level,dtype=torch.long)
            noisy=forward_noise(clean,t,noise)
            prediction=model(noisy,t)
            rows.append(dict(t=level,mse=float(F.mse_loss(prediction,noise)),
                             zero_mse=float(noise.square().mean()),images=len(clean)))
    finally: model.train(was_training)
    return pd.DataFrame(rows)


@torch.inference_mode()
def sample_diffusion(model:TinyTimeUNet,*,count:int=8,seed:int=42,
                     capture:tuple[int,...]=(99,74,49,24,0))->dict:
    """Run all 100 DDPM reverse updates from Gaussian noise, no source image.

    count=8 (1..64), seed=42; returns images [N,1,28,28] in [-1,1] and frames.
    Each frame contains t, input (x_t), predicted_clean (clipped x0 estimate),
    and previous (scheduler's next state). Frames are CPU tensors; input and
    previous are not clamped. The model mode is preserved. Repeats with the
    same CPU version/model/seed reproduce the same trajectory.
    Example: result=sample_diffusion(model,count=4); result['images'].shape.
    """
    if isinstance(count,bool) or not isinstance(count,int) or not 1<=count<=64:
        raise ValueError('count must be an integer from 1 to 64')
    generator=torch.Generator().manual_seed(seed)
    scheduler=make_scheduler(); scheduler.set_timesteps(TIMESTEPS)
    x=torch.randn((count,1,28,28),generator=generator)
    initial=x.clone(); frames=[]; was_training=model.training; model.eval()
    try:
        for timestep in scheduler.timesteps:
            t=int(timestep); batch_t=torch.full((count,),t,dtype=torch.long)
            predicted_noise=model(x,batch_t)
            update=scheduler.step(predicted_noise,timestep,x,generator=generator)
            if t in capture:
                frames.append(dict(t=t,input=x.clone(),predicted_clean=update.pred_original_sample.clone(),
                                   previous=update.prev_sample.clone()))
            x=update.prev_sample
        if not torch.isfinite(x).all(): raise RuntimeError('Nonfinite sampling output')
    finally: model.train(was_training)
    return dict(images=x,initial=initial,frames=frames,steps=TIMESTEPS,seed=seed)


def load_diffusion_checkpoints()->tuple[dict[str,TinyTimeUNet],dict]:
    """Load bundled initial/early/trained CPU models after SHA-256 validation.

    NPZ allow_pickle=False; strict state-dict loading. Returns (models,metadata),
    models keyed initial/early/trained in eval mode. No download or training.
    Raises on corruption; never silently substitutes random weights.
    Example: models,record=load_diffusion_checkpoints(); models['trained'].
    """
    record=json.loads((ASSETS/'training.json').read_text(encoding='utf8'))
    models={}
    for name,item in record['checkpoints'].items():
        path=ASSETS/item['file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
            raise RuntimeError('Diffusion checkpoint integrity failure: '+name)
        model=TinyTimeUNet()
        with np.load(path,allow_pickle=False) as arrays:
            model.load_state_dict({key:torch.from_numpy(arrays[key].copy()) for key in arrays.files},strict=True)
        models[name]=model.eval()
    return models,record


def image_grid(images:torch.Tensor,*,columns:int=8,title:str='Images'):
    """Return a matplotlib Figure for [-1,1] images, clipping for display only.

    CPU float32 [B,1,28,28]; columns=8. Does not mutate tensors. Figures use
    a common -1..1 scale so contrast does not get rescaled per image.
    """
    import matplotlib.pyplot as plt
    _images(images)
    if not isinstance(columns,int) or columns<1: raise ValueError('columns must be positive')
    columns=min(columns,len(images)); rows=math.ceil(len(images)/columns)
    fig,axes=plt.subplots(rows,columns,figsize=(1.5*columns,1.65*rows),squeeze=False)
    for i,ax in enumerate(axes.flat):
        if i<len(images): ax.imshow(images[i,0].detach().numpy(),cmap='gray',vmin=-1,vmax=1)
        ax.axis('off')
    fig.suptitle(title); fig.tight_layout(); return fig
