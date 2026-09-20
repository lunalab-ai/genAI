"""CPU autoencoder experiments. Independent teaching implementation for W04A.

Example: data = load_mnist(); model = ConvAutoencoder(16)
         history = fit_autoencoder(model, data.train, data.validation)
No download, training, or server starts on import. MNIST uses the torchvision
project's public HTTPS mirror and published integrity checks, not torchvision.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import gzip
import hashlib
import os
import struct
import tempfile
import time
from urllib.request import urlopen

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

MNIST_CSS = ".mnist-pixels img { width:224px !important; height:224px !important; object-fit:contain; image-rendering:pixelated; }"

MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
RESOURCES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


@dataclass
class MNISTSplit:
    """Image tensors are float32 [N,1,28,28] in [0,1], on CPU.

    train_ids/validation_ids refer to the official train set; test_ids refer
    to the separate official test set. Labels are for plotting, never fit.
    downloads lists files fetched in this call; an empty list means cache use.
    """
    train: torch.Tensor
    validation: torch.Tensor
    test: torch.Tensor
    test_labels: np.ndarray
    train_ids: np.ndarray
    validation_ids: np.ndarray
    test_ids: np.ndarray
    downloads: list[str]
    cache_dir: str


def _cached_file(root: Path, name: str, expected: str, offline: bool) -> tuple[Path, bool]:
    path = root / name
    if path.exists() and hashlib.md5(path.read_bytes()).hexdigest() == expected:
        return path, False
    if offline:
        raise RuntimeError(f"MNIST 캐시가 없거나 손상됨: {path}. 온라인에서 load_mnist를 다시 실행하세요.")
    temporary = None
    try:
        with urlopen(MNIST_URL + name, timeout=60) as response:
            payload = response.read()
        if hashlib.md5(payload).hexdigest() != expected:
            raise ValueError("MNIST checksum mismatch")
        with tempfile.NamedTemporaryFile(dir=root, delete=False) as out:
            temporary = Path(out.name)
            out.write(payload)
        os.replace(temporary, path)
        return path, True
    except Exception as exc:
        raise RuntimeError(f"MNIST 다운로드/검증 실패: {name}. 네트워크와 캐시 쓰기 권한을 확인한 뒤 재실행하세요.") from exc
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _idx(path: Path) -> np.ndarray:
    raw = gzip.decompress(path.read_bytes())
    magic, count = struct.unpack(">II", raw[:8])
    if magic == 2051:
        rows, cols = struct.unpack(">II", raw[8:16])
        if (rows, cols) != (28, 28) or len(raw) != 16 + count * 784:
            raise ValueError("Invalid MNIST image dimensions")
        return np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(count, 28, 28).copy()
    if magic == 2049 and len(raw) == 8 + count:
        return np.frombuffer(raw, dtype=np.uint8, offset=8).copy()
    raise ValueError("Invalid MNIST IDX header")


def load_mnist(cache_dir: str | Path | None = None, *, train_size: int = 6000,
               validation_size: int = 1000, test_size: int = 1000,
               seed: int = 1337, offline: bool = False) -> MNISTSplit:
    """Download ~11.6 MB once, verify MD5 each call, then return fixed splits.

    Defaults: 6000/1000 selected without overlap from official 60000 train;
    1000 selected from official 10000 test. Cache defaults to
    ~/.cache/luna-genai/mnist, or LUNA_MNIST_CACHE if set. Raises on network,
    checksum, IDX, or size errors; never substitutes synthetic data.
    offline=True forbids downloads. Seed affects selection, not the files.
    """
    sizes = (train_size, validation_size, test_size)
    if any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in sizes):
        raise ValueError("split sizes must be positive integers")
    if train_size + validation_size > 60000 or test_size > 10000:
        raise ValueError("requested split exceeds official MNIST split")
    root = Path(cache_dir or os.environ.get("LUNA_MNIST_CACHE", Path.home() / ".cache/luna-genai/mnist"))
    root.mkdir(parents=True, exist_ok=True)
    arrays, downloads = {}, []
    for name, expected in RESOURCES.items():
        path, downloaded = _cached_file(root, name, expected, offline)
        arrays[name] = _idx(path)
        if downloaded:
            downloads.append(name)
    if len(arrays["train-images-idx3-ubyte.gz"]) != 60000 or len(arrays["t10k-images-idx3-ubyte.gz"]) != 10000:
        raise ValueError("Unexpected official MNIST split sizes")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(60000)
    train_ids = indices[:train_size]
    validation_ids = indices[train_size:train_size + validation_size]
    test_ids = rng.permutation(10000)[:test_size]
    def images(name: str, ids: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arrays[name][ids]).unsqueeze(1).float() / 255.0
    return MNISTSplit(images("train-images-idx3-ubyte.gz", train_ids),
                      images("train-images-idx3-ubyte.gz", validation_ids),
                      images("t10k-images-idx3-ubyte.gz", test_ids),
                      arrays["t10k-labels-idx1-ubyte.gz"][test_ids],
                      train_ids, validation_ids, test_ids, downloads, str(root))


class ConvAutoencoder(nn.Module):
    """Small CPU CNN, distinct from the textbook's large model.

    Input [B,1,28,28] -> [B,8,14,14] -> [B,16,7,7] -> [B,d].
    Decoder: [B,d] -> [B,16,7,7] -> [B,8,14,14] -> [B,1,28,28].
    latent_dim defaults to 16; seed resets torch's RNG for initialization.
    Output uses sigmoid (continuous values), latent code is unconstrained.
    encode/decode/forward keep gradients unless called in inference_mode.
    """
    def __init__(self, latent_dim: int = 16, seed: int = 1337):
        super().__init__()
        if not isinstance(latent_dim, int) or isinstance(latent_dim, bool) or latent_dim < 1:
            raise ValueError("latent_dim must be a positive integer")
        torch.manual_seed(seed)
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(nn.Conv2d(1, 8, 4, 2, 1), nn.ReLU(),
                                     nn.Conv2d(8, 16, 4, 2, 1), nn.ReLU(),
                                     nn.Flatten(start_dim=1), nn.Linear(784, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 784), nn.Unflatten(1, (16, 7, 7)),
                                     nn.ConvTranspose2d(16, 8, 4, 2, 1), nn.ReLU(),
                                     nn.ConvTranspose2d(8, 1, 4, 2, 1), nn.Sigmoid())

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def _validate_images(x: torch.Tensor) -> None:
    if x.ndim != 4 or x.shape[1:] != (1, 28, 28) or len(x) == 0:
        raise ValueError("Expected nonempty [N,1,28,28] tensor")
    if x.dtype != torch.float32 or x.device.type != "cpu":
        raise ValueError("CPU float32 images required")
    if not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise ValueError("Image values must be finite and in [0,1]")


def reconstruct(model: ConvAutoencoder, images: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    """Return detached CPU reconstructions; set model.eval(), disable autograd.

    Does not change parameters. images must be CPU float32 [N,1,28,28] in
    [0,1]. batch_size=256 limits working memory; input order is preserved.
    """
    _validate_images(images)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model.eval()
    with torch.inference_mode():
        return torch.cat([model(x) for x in images.split(batch_size)])


def reconstruction_mse(model: ConvAutoencoder, images: torch.Tensor) -> float:
    """Mean squared error over every image/channel/pixel; lower is better."""
    return float((reconstruct(model, images) - images).square().mean())


def fit_autoencoder(model: ConvAutoencoder, train: torch.Tensor, validation: torch.Tensor,
                    *, epochs: int = 5, batch_size: int = 128, lr: float = 0.001,
                    seed: int = 1337, threads: int = 2) -> list[dict[str, float]]:
    """Mutate model by Adam training; return epoch/train_mse/val_mse/seconds.

    Uses inputs as targets, no labels and no test set. Fixed epochs; validation
    monitors training, does not select a checkpoint. train_mse accumulates
    pre-update minibatch losses, val_mse uses epoch-end weights, so they are
    not measurements of exactly the same model. Limits torch CPU threads to
    threads (process-wide). Repeated calls continue from current parameters.
    """
    _validate_images(train)
    _validate_images(validation)
    if epochs < 1 or batch_size < 1 or lr <= 0 or threads < 1:
        raise ValueError("epochs/batch_size/lr/threads must be positive")
    torch.set_num_threads(threads)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(train), batch_size=batch_size, shuffle=True, generator=generator)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        total, count = 0.0, 0
        for (x,) in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(x), x)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite reconstruction loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(x)
            count += len(x)
        history.append({"epoch": epoch, "train_mse": total / count,
                        "val_mse": reconstruction_mse(model, validation),
                        "seconds": time.perf_counter() - started})
    return history


def comparison_rows(models: dict[int, ConvAutoencoder], data: MNISTSplit) -> list[dict[str, float | str]]:
    """Evaluate once on the same held-out test set, plus train-mean baseline.

    Baseline predicts the per-pixel mean of training images for every test
    image. No test image contributes to that mean. Returns list of records.
    """
    mean = data.train.mean(dim=0, keepdim=True)
    rows = [{"model": "train-mean baseline", "test_mse": float((data.test - mean).square().mean()), "parameters": 0}]
    rows.extend({"model": f"AE d={d}", "test_mse": reconstruction_mse(m, data.test),
                 "parameters": sum(p.numel() for p in m.parameters())} for d, m in models.items())
    return rows


def reconstruction_figure(models: dict[int, ConvAutoencoder], data: MNISTSplit, count: int = 8):
    """Matplotlib Figure: same first count test images, mean, and each model.

    Fixed gray range [0,1]. Does not cherry-pick examples by reconstruction.
    Caller owns the figure and can savefig/close it. No disk writes here.
    """
    import matplotlib.pyplot as plt
    count = min(count, len(data.test))
    if count < 1:
        raise ValueError("count must be positive")
    rows = [("input", data.test[:count]), ("train mean", data.train.mean(0, keepdim=True).repeat(count, 1, 1, 1))]
    rows += [(f"AE d={d}", reconstruct(m, data.test[:count])) for d, m in models.items()]
    fig, axes = plt.subplots(len(rows), count, figsize=(count * 1.2, len(rows) * 1.3), squeeze=False)
    for r, (name, values) in enumerate(rows):
        for c in range(count):
            axes[r, c].imshow(values[c, 0], cmap="gray", vmin=0, vmax=1)
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if c == 0:
                axes[r, c].set_ylabel(name)
    fig.tight_layout()
    return fig


def latent_figure(model: ConvAutoencoder, images: torch.Tensor, labels: np.ndarray):
    """2D latent scatter, digit labels used only as colors. Return Figure.

    Axes have no predefined semantic meaning. Requires latent_dim=2 and one
    label per image. Runs eval/inference_mode; does not alter weights.
    """
    import matplotlib.pyplot as plt
    _validate_images(images)
    if model.latent_dim != 2 or len(labels) != len(images):
        raise ValueError("Need a 2D model and one label per image")
    model.eval()
    with torch.inference_mode():
        z = model.encode(images).numpy()
    fig, ax = plt.subplots(figsize=(7, 5))
    dots = ax.scatter(z[:, 0], z[:, 1], c=labels, s=9, alpha=0.65, cmap="tab10", vmin=-0.5, vmax=9.5)
    fig.colorbar(dots, ax=ax, ticks=range(10), label="digit label (display only)")
    ax.set(xlabel="z1 (learned coordinate)", ylabel="z2 (learned coordinate)", title="Held-out MNIST encodings")
    fig.tight_layout()
    return fig


def reconstruction_callback(index: int, latent_dim: int, models: dict[int, ConvAutoencoder],
                            images: torch.Tensor) -> tuple[np.ndarray, np.ndarray, str]:
    """Return input, reconstruction (uint8 28x28), and MSE text for Gradio.

    Validates index and dimension; inference only, no downloads or training.
    This is the supplied reconstruction feature. Students add latent controls.
    """
    i, d = int(index), int(latent_dim)
    if i != index or d != latent_dim or not 0 <= i < len(images) or d not in models:
        raise ValueError("Invalid image index or latent dimension")
    x = images[i:i + 1]
    y = reconstruct(models[d], x)
    mse = float((x - y).square().mean())
    return (x[0, 0].numpy() * 255).round().astype(np.uint8), (y[0, 0].numpy() * 255).round().astype(np.uint8), f"test index={i} · d={d} · MSE={mse:.6f}"


def build_autoencoder_app(models: dict[int, ConvAutoencoder], images: torch.Tensor):
    """Build supplied reconstruction-only Blocks; no server starts.

    Returns (app, callback). Student notebook adds a separate latent explorer
    rather than hiding that exercise's completed wiring in this public API.
    app.launch(share=False, css=MNIST_CSS) starts locally. Colab share=True is opt-in and
    exposes the app while that runtime is alive. Models are reused in memory.
    """
    import gradio as gr
    def callback(index, dimension):
        return reconstruction_callback(index, dimension, models, images)
    with gr.Blocks() as app:
        gr.Markdown("# 오토인코더 복원 관찰실\n같은 테스트 이미지의 16/2차원 복원을 비교하세요. MSE는 낮을수록 작게 틀립니다.")
        index = gr.Slider(0, len(images) - 1, value=0, step=1, label="테스트 이미지 번호")
        dimension = gr.Dropdown(sorted(models, reverse=True), value=16 if 16 in models else next(iter(models)), label="잠재 차원")
        button = gr.Button("복원 비교")
        with gr.Row():
            original = gr.Image(label="원본", image_mode="L", format="png", height=280, interactive=False, elem_classes="mnist-pixels")
            reconstruction = gr.Image(label="복원", image_mode="L", format="png", height=280, interactive=False, elem_classes="mnist-pixels")
        report = gr.Textbox(label="실측 오차")
        button.click(callback, [index, dimension], [original, reconstruction, report], api_name="reconstruct")
    return app, callback
