"""AE/VAE representation experiments for W05A. No work runs on import.

Example: models, metadata = load_checkpoints(); y = reconstruct_images(models['ae2'], x)
All images use CPU float32 [N,1,28,28], range [0,1]. Labels never enter training.
The dense backbone deliberately makes the shape path inspectable on a CPU.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from .autoencoder import _validate_images

ASSETS = Path(__file__).parent / 'assets' / 'w05a'


class DenseAE(nn.Module):
    """Deterministic autoencoder; [B,1,28,28] -> [B,d] -> original shape.

    latent_dim=2, seed=1337. Initializes weights (does not train/download).
    Backbone 784->256->128->d, decoder d->128->256->784; sigmoid output.
    seed is local to initialization: the caller's global RNG state is preserved.
    Example: model=DenseAE(2); z=model.encode(torch.zeros(1,1,28,28)).
    """
    def __init__(self, latent_dim: int = 2, seed: int = 1337):
        super().__init__()
        if isinstance(latent_dim, bool) or not isinstance(latent_dim, int) or latent_dim < 1:
            raise ValueError('latent_dim must be a positive integer')
        self.latent_dim = latent_dim
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.encoder = nn.Sequential(nn.Flatten(1), nn.Linear(784,256), nn.ReLU(),
                                         nn.Linear(256,128), nn.ReLU(), nn.Linear(128,latent_dim))
            self.decoder = nn.Sequential(nn.Linear(latent_dim,128), nn.ReLU(),
                                         nn.Linear(128,256), nn.ReLU(), nn.Linear(256,784),
                                         nn.Sigmoid(), nn.Unflatten(1,(1,28,28)))

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Map float32 [B,1,28,28] to unconstrained [B,d]; gradients retained."""
        return self.encoder(images)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Map float32 [B,d] to [B,1,28,28] in [0,1]; no inverse guarantee."""
        return self.decoder(z)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return differentiable reconstruction; does not update parameters."""
        return self.decode(self.encode(images))


class DenseVAE(nn.Module):
    """Diagonal Gaussian encoder with the same dense backbone as DenseAE.

    latent_dim=2, seed=1337; encoder predicts mu and log(sigma^2), each [B,d].
    Independent linear heads need not produce equal variances across coordinates.
    Initializing a model does not fit it. Prior is standard normal, not the encoder.
    """
    def __init__(self, latent_dim: int = 2, seed: int = 1337):
        super().__init__()
        if isinstance(latent_dim,bool) or not isinstance(latent_dim,int) or latent_dim < 1:
            raise ValueError('latent_dim must be a positive integer')
        self.latent_dim = latent_dim
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.encoder = nn.Sequential(nn.Flatten(1),nn.Linear(784,256),nn.ReLU(),
                                         nn.Linear(256,128),nn.ReLU())
            self.mu = nn.Linear(128,latent_dim)
            self.logvar = nn.Linear(128,latent_dim)
            self.decoder = nn.Sequential(nn.Linear(latent_dim,128),nn.ReLU(),
                                         nn.Linear(128,256),nn.ReLU(),nn.Linear(256,784),
                                         nn.Sigmoid(),nn.Unflatten(1,(1,28,28)))

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        """Return (mu, logvar), both [B,d]; logvar is log variance, not std."""
        features = self.encoder(images)
        return self.mu(features), self.logvar(features)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode [B,d] into pixel means [B,1,28,28] in [0,1]."""
        return self.decoder(z)

    def forward(self, images: torch.Tensor, *, sample: bool = True) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        """Return (reconstruction,mu,logvar); sample=True draws fresh epsilon.

        sample=False decodes mu deterministically; this is not the expected image
        under the nonlinear decoder. Neither mode performs an optimizer step.
        """
        mu, logvar = self.encode(images)
        z = reparameterize(mu,logvar) if sample else mu
        return self.decode(z), mu, logvar


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, epsilon: torch.Tensor | None = None) -> torch.Tensor:
    """Return mu + exp(0.5*logvar)*epsilon; all tensors must have equal shape.

    epsilon=None (default) draws standard-normal noise like mu. Supply epsilon
    to reproduce a calculation. Gradients flow through both mu and logvar.
    Example: reparameterize(torch.tensor([[1.]]),torch.log(torch.tensor([[4.]])),
                            torch.tensor([[0.5]])) yields [[2.]].
    """
    if mu.shape != logvar.shape or mu.ndim != 2 or not torch.isfinite(mu).all() or not torch.isfinite(logvar).all():
        raise ValueError('mu and logvar must be finite matching [B,d] tensors')
    if epsilon is None: epsilon = torch.randn_like(mu)
    if epsilon.shape != mu.shape or not torch.isfinite(epsilon).all():
        raise ValueError('epsilon shape/values invalid')
    result = mu + torch.exp(0.5*logvar)*epsilon
    if not torch.isfinite(result).all(): raise ValueError('variance overflow')
    return result


def vae_loss(prediction: torch.Tensor, target: torch.Tensor, mu: torch.Tensor,
             logvar: torch.Tensor, beta: float = 1.0) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    """Return scalar (SSE + beta*KL, SSE, KL), each averaged over images.

    SSE sums pixels per image; KL sums latent coordinates per image. Thus SSE
    equals 784*pixel-MSE for MNIST. beta>=0; beta=1 is the unweighted KL term.
    Shapes: prediction/target [B,1,28,28], mu/logvar [B,d]. No parameter update.
    This teaching Gaussian reconstruction objective omits fixed constants.
    """
    if prediction.shape != target.shape or prediction.ndim != 4 or len(target)==0:
        raise ValueError('reconstruction and target must match [B,C,H,W]')
    if mu.shape != logvar.shape or mu.ndim!=2 or len(mu)!=len(target) or beta<0 or not np.isfinite(beta):
        raise ValueError('latent shape or beta invalid')
    sse = (prediction-target).square().flatten(1).sum(1).mean()
    kl = 0.5*(mu.square()+logvar.exp()-1-logvar).sum(1).mean()
    total = sse+beta*kl
    if not torch.isfinite(total): raise ValueError('non-finite VAE loss')
    return total,sse,kl


def latent_means(model: DenseAE | DenseVAE, images: torch.Tensor) -> torch.Tensor:
    """Encode CPU images to detached [N,d]; use mu for VAE; batches of 256."""
    _validate_images(images); model.eval()
    with torch.inference_mode():
        chunks=[model.encode(x) for x in images.split(256)]
        return torch.cat([v[0] if isinstance(model,DenseVAE) else v for v in chunks])


def reconstruct_images(model: DenseAE | DenseVAE, images: torch.Tensor) -> torch.Tensor:
    """Return detached [N,1,28,28] in original order, deterministically via z/mu.

    Sets eval mode, disables gradient recording, never changes parameters.
    For VAE this decodes mu; stochastic training loss has a different meaning.
    """
    _validate_images(images); model.eval()
    with torch.inference_mode():
        return torch.cat([model(x,sample=False)[0] if isinstance(model,DenseVAE) else model(x)
                          for x in images.split(256)])


def fit_representation(model: DenseAE | DenseVAE, train: torch.Tensor, validation: torch.Tensor,
                       *, epochs: int = 5, beta: float = 1.0, batch_size: int = 256,
                       lr: float = 0.001, seed: int = 1337) -> list[dict]:
    """Train in place with Adam and return per-epoch loss and validation MSE.

    train/validation CPU images, no labels or test. epochs=5, beta=1,
    batch_size=256, lr=.001, seed=1337. AE minimizes pixel-MSE; VAE minimizes
    image-SSE+beta*KL. Validation decodes z/mu. Fixed epochs, no test selection.
    Each call creates a new optimizer; global RNG is restored after training.
    Caller controls torch CPU thread count. Example: fit_representation(m,x,v).
    """
    _validate_images(train); _validate_images(validation)
    if epochs<1 or batch_size<1 or lr<=0 or beta<0: raise ValueError('invalid training setting')
    history=[]
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train),batch_size=batch_size,
                                          shuffle=True,generator=torch.Generator().manual_seed(seed))
        optimizer=torch.optim.Adam(model.parameters(),lr=lr)
        for epoch in range(1,epochs+1):
            started=time.perf_counter();model.train(); totals=np.zeros(3);n=0
            for (x,) in loader:
                optimizer.zero_grad(set_to_none=True)
                if isinstance(model,DenseVAE):
                    pred,mu,lv=model(x);loss,sse,kl=vae_loss(pred,x,mu,lv,beta)
                else:
                    loss=nn.functional.mse_loss(model(x),x);sse=loss*784;kl=loss.new_zeros(())
                loss.backward();optimizer.step()
                totals+=np.array([loss.item(),sse.item(),kl.item()])*len(x);n+=len(x)
            mse=float((reconstruct_images(model,validation)-validation).square().mean())
            history.append(dict(epoch=epoch,train_loss=totals[0]/n,train_sse=totals[1]/n,
                                train_kl=totals[2]/n,val_mse=mse,seconds=time.perf_counter()-started))
    return history


def load_checkpoints() -> tuple[dict[str,DenseAE | DenseVAE],dict]:
    """Verify bundled NPZ SHA256 and load five classroom checkpoints on CPU.

    Package installation automatically downloads these small teaching weights.
    Returns ({ae2,ae16,vae01,vae1,vae4}, provenance), all in eval mode.
    NPZ is read with allow_pickle=False. Raises on corruption; never substitutes
    random weights. No training or network call. See assets/w05a/models.json.
    """
    metadata=json.loads((ASSETS/'models.json').read_text(encoding='utf8'));models={}
    for name,entry in metadata['models'].items():
        path=ASSETS/entry['file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
            raise RuntimeError('Checkpoint integrity failure: '+name+'; reinstall the fixed course version.')
        model=(DenseAE if entry['kind']=='AE' else DenseVAE)(entry['latent_dim'],metadata['seed'])
        with np.load(path,allow_pickle=False) as arrays:
            model.load_state_dict({k:torch.from_numpy(arrays[k].copy()) for k in arrays.files},strict=True)
        model.eval();models[name]=model
    return models,metadata


def reconstruction_figure(models: dict, images: torch.Tensor, count: int = 8):
    """Return a matplotlib Figure of identical images and model reconstructions.

    count=8 first images, fixed gray [0,1]; no cherry-picking. Caller closes figure.
    """
    import matplotlib.pyplot as plt
    count=min(count,len(images));fig,axes=plt.subplots(len(models)+1,count,figsize=(count*1.25,(len(models)+1)*1.35),squeeze=False)
    for row,(name,x) in enumerate([('Original',images[:count])]+[(k,reconstruct_images(m,images[:count])) for k,m in models.items()]):
        for col in range(count):
            axes[row,col].imshow(x[col,0],cmap='gray',vmin=0,vmax=1);axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
        axes[row,0].set_ylabel(name,fontsize=10)
    fig.tight_layout();return fig


def latent_figure(models: dict, images: torch.Tensor, labels: np.ndarray):
    """Return 2-D z/mu scatter figure; labels color points only, not training."""
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,len(models),figsize=(5*len(models),4.4),squeeze=False)
    for ax,(name,model) in zip(axes[0],models.items()):
        z=latent_means(model,images).numpy()
        if z.shape[1]!=2 or len(labels)!=len(z): raise ValueError('requires 2D and aligned labels')
        dots=ax.scatter(z[:,0],z[:,1],c=labels,s=8,cmap='tab10',vmin=-.5,vmax=9.5,alpha=.6)
        ax.set(title=name,xlabel='z1 / mu1',ylabel='z2 / mu2');ax.grid(alpha=.15)
    fig.colorbar(dots,ax=axes[0].tolist(),ticks=range(10),label='MNIST label (observation only)',shrink=.8)
    return fig


def latent_grid(model: DenseAE | DenseVAE, low: float = -3, high: float = 3, steps: int = 9) -> np.ndarray:
    """Return grayscale tiled image from 2D grid; y increases upwards.

    low=-3,high=3,steps=9. A uniform grid is a visualization, not a normal sample.
    AE may have a very different coordinate scale. No learning or RNG changes.
    """
    if model.latent_dim!=2 or steps<2 or not low<high: raise ValueError('invalid 2D grid')
    coordinates=torch.tensor([[x,y] for y in np.linspace(high,low,steps) for x in np.linspace(low,high,steps)],dtype=torch.float32)
    with torch.inference_mode(): output=model.decode(coordinates)[:,0].numpy()
    return output.reshape(steps,steps,28,28).transpose(0,2,1,3).reshape(steps*28,steps*28)


def interpolation(model: DenseAE | DenseVAE, first: torch.Tensor, second: torch.Tensor, steps: int = 9) -> torch.Tensor:
    """Return [steps,1,28,28] decoded linear interpolation between z/mu endpoints.

    Each endpoint must contain exactly one image. steps=9 includes both endpoints;
    interpolation does not establish that all intermediate images are plausible.
    """
    if len(first)!=1 or len(second)!=1 or steps<2: raise ValueError('one image per endpoint required')
    z=latent_means(model,torch.cat([first,second]));alpha=torch.linspace(0,1,steps)[:,None]
    with torch.inference_mode(): return model.decode((1-alpha)*z[:1]+alpha*z[1:])
