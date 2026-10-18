"""W06B: real Fashion-MNIST class-conditional epsilon prediction on CPU.

No downloads or training on import. Inputs use [-1,1], Gaussian noise, and
the same 100-step cosine schedule as W06A. This is not Stable Diffusion.
"""
from __future__ import annotations
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import urllib.request
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from diffusers import DDIMScheduler
from .diffusion import TinyTimeUNet, _images, _times, forward_noise, make_scheduler

ASSETS = Path(__file__).parent / 'assets' / 'w06b'
CLASS_NAMES = ('T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot')
FILES = {
    'train-images-idx3-ubyte.gz':'8d4fb7e6c68d591d4c3dfef9ec88bf0d',
    'train-labels-idx1-ubyte.gz':'25c81989df183df01b3e8a0aad5dffbe',
    't10k-images-idx3-ubyte.gz':'bef4ecab320f06d8554ea6380940ec79',
    't10k-labels-idx1-ubyte.gz':'bb300cfdad3c16e7a12a480ee83cd310',
}

@dataclass
class FashionData:
    """CPU images [N,1,28,28] in [-1,1], labels int64 [N], disjoint indices."""
    train: torch.Tensor
    train_labels: torch.Tensor
    validation: torch.Tensor
    validation_labels: torch.Tensor
    test: torch.Tensor
    test_labels: torch.Tensor
    train_ids: np.ndarray
    validation_ids: np.ndarray
    downloads: list[str]

def load_fashion(*,train_size:int=12000,validation_size:int=1000,test_size:int=1000,
                 seed:int=1337,cache_dir:str|Path|None=None,offline:bool=False)->FashionData:
    """Download/check official Fashion-MNIST and make reproducible disjoint splits.

    Default train/validation/test=12000/1000/1000. train+validation<=60000;
    test<=10000. Writes only missing cache files, checks MD5 and IDX headers.
    offline=True raises when files are missing. No model training. Example:
    data=load_fashion(); data.train.shape == (12000,1,28,28).
    """
    for n in (train_size,validation_size,test_size):
        if isinstance(n,bool) or not isinstance(n,int) or n<1: raise ValueError('Positive integer split sizes required')
    if train_size+validation_size>60000 or test_size>10000: raise ValueError('Split exceeds official data size')
    cache=Path(cache_dir or Path.home()/'.cache/luna-genai/fashion-mnist'); cache.mkdir(parents=True,exist_ok=True)
    downloaded=[]; arrays={}
    for name,expected in FILES.items():
        path=cache/name
        if not path.exists():
            if offline: raise FileNotFoundError(path)
            url='https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/'+name
            with urllib.request.urlopen(url,timeout=120) as response: payload=response.read()
            if hashlib.md5(payload).hexdigest()!=expected: raise RuntimeError('Fashion-MNIST download checksum mismatch: '+name)
            path.write_bytes(payload); downloaded.append(name)
        payload=path.read_bytes()
        if hashlib.md5(payload).hexdigest()!=expected: raise RuntimeError('Corrupt Fashion-MNIST cache: '+str(path))
        raw=gzip.decompress(payload); is_image='images' in name
        magic=int.from_bytes(raw[:4],'big'); count=int.from_bytes(raw[4:8],'big')
        if magic!=(2051 if is_image else 2049): raise RuntimeError('Invalid IDX header')
        if is_image:
            if raw[8:16] != b'\x00\x00\x00\x1c\x00\x00\x00\x1c': raise RuntimeError('Expected 28x28')
            arrays[name]=np.frombuffer(raw,offset=16,dtype=np.uint8).reshape(count,1,28,28)
        else: arrays[name]=np.frombuffer(raw,offset=8,dtype=np.uint8).reshape(count)
    rng=np.random.default_rng(seed); order=rng.permutation(60000)
    ti=order[:train_size];vi=order[train_size:train_size+validation_size];si=rng.permutation(10000)[:test_size]
    def images(prefix,ids): return torch.from_numpy(arrays[prefix+'-images-idx3-ubyte.gz'][ids].copy()).float()/127.5-1
    def labels(prefix,ids): return torch.from_numpy(arrays[prefix+'-labels-idx1-ubyte.gz'][ids].copy()).long()
    return FashionData(images('train',ti),labels('train',ti),images('train',vi),labels('train',vi),images('t10k',si),labels('t10k',si),ti,vi,downloaded)

class TinyClassUNet(TinyTimeUNet):
    """Learned class embedding + time embedding -> tiny conditional U-Net.

    CPU noisy float32 [B,1,28,28], time int64 [B], class int64 [B] (0..9)
    -> epsilon float32 [B,1,28,28]. Calling forward never trains weights.
    This teaching design adds class/time vectors; SD uses token cross-attention.
    """
    def __init__(self,seed:int=20261007):
        super().__init__(seed)
        with torch.random.fork_rng():
            torch.manual_seed(seed+1);self.class_embedding=nn.Embedding(10,64)

    def forward(self,noisy:torch.Tensor,timesteps:torch.Tensor,labels:torch.Tensor)->torch.Tensor:
        """Predict Gaussian epsilon, not class probabilities; true epsilon is not an input."""
        _images(noisy);_times(timesteps,len(noisy))
        if labels.device.type!='cpu' or labels.dtype!=torch.long or labels.shape!=(len(noisy),) or (labels<0).any() or (labels>9).any():
            raise ValueError('Class labels must be CPU int64 [B], values 0..9')
        frequencies=torch.exp(-math.log(10000)*torch.arange(16)/15)
        angles=timesteps.float()[:,None]*frequencies[None,:]
        condition=self.time_mlp(torch.cat([angles.sin(),angles.cos()],1))+self.class_embedding(labels)
        h1=self.enc1(noisy,condition);h2=self.enc2(F.avg_pool2d(h1,2),condition)
        h3=self.middle(F.avg_pool2d(h2,2),condition)
        u2=self.dec2(torch.cat([F.interpolate(h3,size=(14,14),mode='nearest'),h2],1),condition)
        u1=self.dec1(torch.cat([F.interpolate(u2,size=(28,28),mode='nearest'),h1],1),condition)
        return self.out(u1)

def conditional_step(model:TinyClassUNet,optimizer:torch.optim.Optimizer,clean:torch.Tensor,
                     labels:torch.Tensor,timesteps:torch.Tensor,noise:torch.Tensor)->dict:
    """One actual MSE/backward/update; modifies model and optimizer, not input data.

    Images [B,1,28,28], class/time int64 [B]. Returns measured loss, gradient
    norm, parameter change and change before optimizer.step (should be zero).
    Target is noise; labels condition the predictor, not the MSE target.
    """
    model.train();noisy=forward_noise(clean,timesteps,noise)
    before=model.class_embedding.weight.detach().clone();optimizer.zero_grad(set_to_none=True)
    prediction=model(noisy,timesteps,labels);loss=F.mse_loss(prediction,noise)
    if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
    loss.backward(); gradient=float(torch.sqrt(sum(p.grad.square().sum() for p in model.parameters() if p.grad is not None)))
    if not math.isfinite(gradient): raise RuntimeError('Nonfinite gradient')
    before_step=float((model.class_embedding.weight.detach()-before).abs().sum());optimizer.step()
    return dict(loss=float(loss.detach()),gradient_norm=gradient,backward_parameter_change=before_step,
                parameter_change=float((model.class_embedding.weight.detach()-before).abs().sum()))

@torch.inference_mode()
def conditional_evaluation(model:TinyClassUNet,clean:torch.Tensor,labels:torch.Tensor,*,seed:int=901)->pd.DataFrame:
    """Held-out epsilon MSE at five times; same noise for correct/wrong labels.

    Wrong condition is (label+1)%10. This is a sensitivity probe, not accuracy
    or an assertion that every wrong class always increases MSE. No training.
    """
    _images(clean,clean=True);noise=torch.randn(clean.shape,generator=torch.Generator().manual_seed(seed))
    mode=model.training;model.eval();rows=[]
    try:
        for level in (0,24,49,74,99):
            t=torch.full((len(clean),),level,dtype=torch.long);noisy=forward_noise(clean,t,noise)
            pred=model(noisy,t,labels);wrong=model(noisy,t,(labels+1)%10)
            rows.append(dict(t=level,mse=float(F.mse_loss(pred,noise)),wrong_label_mse=float(F.mse_loss(wrong,noise)),zero_mse=float(noise.square().mean())))
    finally:model.train(mode)
    return pd.DataFrame(rows)

@torch.inference_mode()
def sample_classes(model:TinyClassUNet,labels:list[int]|tuple[int,...],*,seed:int=42,steps:int=30,
                   matched_noise:bool=True)->dict:
    """DDIM generation, fixed weights; labels choose classes, not clean targets.

    labels has 1..40 integer IDs. matched_noise=True repeats ONE initial draw
    across classes; False draws independent images. Returns images, initial,
    seed, labels, steps. eta=0 gives deterministic sampling in the same setup.
    Preserves model training mode. Does not download or optimize parameters.
    """
    if not 1<=len(labels)<=40 or any(isinstance(x,bool) or not isinstance(x,int) or not 0<=x<=9 for x in labels):
        raise ValueError('Provide 1..40 integer class IDs in 0..9')
    if isinstance(steps,bool) or not isinstance(steps,int) or not 2<=steps<=100: raise ValueError('steps must be 2..100')
    generator=torch.Generator().manual_seed(seed)
    x=torch.randn((1 if matched_noise else len(labels),1,28,28),generator=generator)
    if matched_noise:x=x.repeat(len(labels),1,1,1)
    initial=x.clone();y=torch.tensor(labels,dtype=torch.long)
    scheduler=DDIMScheduler.from_config(make_scheduler().config,timestep_spacing='trailing');scheduler.set_timesteps(steps)
    mode=model.training;model.eval()
    try:
        for t in scheduler.timesteps:
            predicted=model(x,torch.full((len(y),),int(t),dtype=torch.long),y)
            x=scheduler.step(predicted,t,x,eta=0).prev_sample
    finally:model.train(mode)
    if not torch.isfinite(x).all(): raise RuntimeError('Nonfinite generated image')
    return dict(images=x,initial=initial,seed=seed,labels=list(labels),steps=steps,matched_noise=matched_noise,timestep_spacing='trailing')

def load_conditional_checkpoint()->tuple[TinyClassUNet,dict]:
    """Load genuinely trained bundled NPZ after SHA-256 check; never random fallback.

    No network. Returns eval model and full training record. NPZ disallows
    pickle; missing/corrupt assets raise. Loading weights is not training.
    """
    record=json.loads((ASSETS/'conditional-training.json').read_text(encoding='utf-8'))
    path=ASSETS/'fashion-conditional.npz'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']: raise RuntimeError('Conditional checkpoint checksum mismatch')
    model=TinyClassUNet()
    with np.load(path,allow_pickle=False) as data:model.load_state_dict({k:torch.from_numpy(data[k].copy()) for k in data.files},strict=True)
    return model.eval(),record
