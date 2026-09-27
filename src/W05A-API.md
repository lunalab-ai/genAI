# W05A · 공통 API와 소스 바로가기

패키지 0.1.7 / `2026-fall-w05a-v2`. 링크는 설치한 고정 버전의 실제 선언 줄을 가리킨다. 각 메소드의 입력·출력과 부작용을 확인하고 사용한다.

## 빠른 사용

```python
from luna_genai.representation import load_checkpoints
models, metadata = load_checkpoints()
from luna_genai.semantic_search import CLIPSearch
lab = CLIPSearch()
lab.index_images()
result = lab.search("a cat in the snow", 3)
```

## MNISTSplit

[`MNISTSplit`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/autoencoder.py#L37)

Image tensors are float32 [N,1,28,28] in [0,1], on CPU.

train_ids/validation_ids refer to the official train set; test_ids refer
to the separate official test set. Labels are for plotting, never fit.
downloads lists files fetched in this call; an empty list means cache use.

## load_mnist

[`load_mnist`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/autoencoder.py#L92)

Download ~11.6 MB once, verify MD5 each call, then return fixed splits.

Defaults: 6000/1000 selected without overlap from official 60000 train;
1000 selected from official 10000 test. Cache defaults to
~/.cache/luna-genai/mnist, or LUNA_MNIST_CACHE if set. Raises on network,
checksum, IDX, or size errors; never substitutes synthetic data.
offline=True forbids downloads. Seed affects selection, not the files.

## reconstruction_figure

[`reconstruction_figure`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/representation.py#L210)

Return a matplotlib Figure of identical images and model reconstructions.

count=8 first images, fixed gray [0,1]; no cherry-picking. Caller closes figure.

## latent_figure

[`latent_figure`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/representation.py#L224)

Return 2-D z/mu scatter figure; labels color points only, not training.

## DenseAE

[`DenseAE`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/representation.py#L21)

Deterministic autoencoder; [B,1,28,28] -> [B,d] -> original shape.

latent_dim=2, seed=1337. Initializes weights (does not train/download).
Backbone 784->256->128->d, decoder d->128->256->784; sigmoid output.
seed is local to initialization: the caller's global RNG state is preserved.
Example: model=DenseAE(2); z=model.encode(torch.zeros(1,1,28,28)).

## DenseAE.__init__

[`DenseAE.__init__`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/representation.py#L29)



## DenseAE.encode

[`DenseAE.encode`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/representation.py#L42)

Map float32 [B,1,28,28] to unconstrained [B,d]; gradients retained.

## DenseAE.decode

[`DenseAE.decode`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/representation.py#L46)

Map float32 [B,d] to [B,1,28,28] in [0,1]; no inverse guarantee.

## DenseAE.forward

[`DenseAE.forward`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/representation.py#L50)

Return differentiable reconstruction; does not update parameters.

## DenseVAE

[`DenseVAE`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/representation.py#L55)

Diagonal Gaussian encoder with the same dense backbone as DenseAE.

latent_dim=2, seed=1337; encoder predicts mu and log(sigma^2), each [B,d].
Independent linear heads need not produce equal variances across coordinates.
Initializing a model does not fit it. Prior is standard normal, not the encoder.

## DenseVAE.__init__

[`DenseVAE.__init__`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/representation.py#L62)



## DenseVAE.encode

[`DenseVAE.encode`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/representation.py#L77)

Return (mu, logvar), both [B,d]; logvar is log variance, not std.

## DenseVAE.decode

[`DenseVAE.decode`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/representation.py#L82)

Decode [B,d] into pixel means [B,1,28,28] in [0,1].

## DenseVAE.forward

[`DenseVAE.forward`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/representation.py#L86)

Return (reconstruction,mu,logvar); sample=True draws fresh epsilon.

sample=False decodes mu deterministically; this is not the expected image
under the nonlinear decoder. Neither mode performs an optimizer step.

## reparameterize

[`reparameterize`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/representation.py#L97)

Return mu + exp(0.5*logvar)*epsilon; all tensors must have equal shape.

epsilon=None (default) draws standard-normal noise like mu. Supply epsilon
to reproduce a calculation. Gradients flow through both mu and logvar.
Example: reparameterize(torch.tensor([[1.]]),torch.log(torch.tensor([[4.]])),
                        torch.tensor([[0.5]])) yields [[2.]].

## vae_loss

[`vae_loss`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/representation.py#L115)

Return scalar (SSE + beta*KL, SSE, KL), each averaged over images.

SSE sums pixels per image; KL sums latent coordinates per image. Thus SSE
equals 784*pixel-MSE for MNIST. beta>=0; beta=1 is the unweighted KL term.
Shapes: prediction/target [B,1,28,28], mu/logvar [B,d]. No parameter update.
This teaching Gaussian reconstruction objective omits fixed constants.

## latent_means

[`latent_means`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/representation.py#L135)

Encode CPU images to detached [N,d]; use mu for VAE; batches of 256.

## reconstruct_images

[`reconstruct_images`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/representation.py#L143)

Return detached [N,1,28,28] in original order, deterministically via z/mu.

Sets eval mode, disables gradient recording, never changes parameters.
For VAE this decodes mu; stochastic training loss has a different meaning.

## fit_representation

[`fit_representation`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/representation.py#L155)

Train in place with Adam and return per-epoch loss and validation MSE.

train/validation CPU images, no labels or test. epochs=5, beta=1,
batch_size=256, lr=.001, seed=1337. AE minimizes pixel-MSE; VAE minimizes
image-SSE+beta*KL. Validation decodes z/mu. Fixed epochs, no test selection.
Each call creates a new optimizer; global RNG is restored after training.
Caller controls torch CPU thread count. Example: fit_representation(m,x,v).

## load_checkpoints

[`load_checkpoints`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/representation.py#L190)

Verify bundled NPZ SHA256 and load five classroom checkpoints on CPU.

Package installation automatically downloads these small teaching weights.
Returns ({ae2,ae16,vae01,vae1,vae4}, provenance), all in eval mode.
NPZ is read with allow_pickle=False. Raises on corruption; never substitutes
random weights. No training or network call. See assets/w05a/models.json.

## latent_grid

[`latent_grid`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/representation.py#L237)

Return grayscale tiled image from 2D grid; y increases upwards.

low=-3,high=3,steps=9. A uniform grid is a visualization, not a normal sample.
AE may have a very different coordinate scale. No learning or RNG changes.

## interpolation

[`interpolation`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/representation.py#L249)

Return [steps,1,28,28] decoded linear interpolation between z/mu endpoints.

Each endpoint must contain exactly one image. steps=9 includes both endpoints;
interpolation does not establish that all intermediate images are plausible.

## unit_rows

[`unit_rows`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/semantic_search.py#L23)

Normalize finite nonzero [N,d] rows by L2 norm; return same-shape tensor.

Reject zero vectors instead of pretending their cosine is defined. Preserves
device, dtype and autograd. Example: unit_rows(tensor([[3.,4.]])) -> [[.6,.8]].

## load_gallery

[`load_gallery`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/semantic_search.py#L36)

Verify and return the 40 bundled Commons photos in fixed P01..P40 order.

Columns include id,category,title,source,author,license,license_url,sha256,path.
Photos arrive with the fixed package install; no login/upload required. Paths
are local filenames usable by PIL/Gradio. Raises on missing/corrupted images.
Attribution lives in gallery.json and GALLERY-CREDITS.md; do not strip it.

## precision_at_k

[`precision_at_k`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/semantic_search.py#L53)

Count relevant unique IDs among first k results / k; missing ranks count 0.

k=3 positive integer; duplicate IDs are invalid. Relevance must be defined
before inspecting scores. An absent-target query can have empty relevance.
Example: precision_at_k(['A','B','C'],{'A','C'},3) == 2/3.

## CLIPSearch

[`CLIPSearch`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/semantic_search.py#L65)

Load fixed CLIP on CPU, then separately index pixels and query by text.

gallery=None loads checked classroom photos; cache_dir=None uses
~/.cache/luna-genai/clip-w05a. local_files_only=False permits initial model
download (~605 MB); True requires existing complete HF cache. First call
loads the model, index_images builds/reuses the feature cache. search never
trains or re-encodes stored images. English is the primary query language.

## CLIPSearch.__init__

[`CLIPSearch.__init__`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/semantic_search.py#L74)



## CLIPSearch.encode_images

[`CLIPSearch.encode_images`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/semantic_search.py#L106)

Read RGB images; return normalized CPU [N,512], preserving path order.

batch_size=8 limits memory. CLIP processor resizes/crops to 224x224 and
applies its pretrained pixel normalization. Image filenames never enter
the model. Transformers 5.15.1 returns projected features in pooler_output.

## CLIPSearch.encode_texts

[`CLIPSearch.encode_texts`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/semantic_search.py#L125)

Return normalized [N,512] for nonempty strings, without training.

Tokenization pads a batch and truncates at the model's 77-token limit.
Short English descriptions are recommended; Korean is exploratory.

## CLIPSearch.index_images

[`CLIPSearch.index_images`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/semantic_search.py#L137)

Compute/reuse normalized [N,512] pixel embeddings; return the matrix.

force=False reuses matching model revision, library version, ordered
image hashes and IDs. A corrupt cache raises; force=True rebuilds that
derived file. Writes one local NPZ atomically, never changes weights.

## CLIPSearch.search

[`CLIPSearch.search`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/semantic_search.py#L159)

Return descending cosine top-k rows with rank,id,cosine and metadata.

Requires index_images first. k=3, range 1..gallery size. Ties preserve
gallery order. Cosine is not a calibrated probability. A result is always
returned even when the requested object is absent from the gallery.

## search_view

[`search_view`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/semantic_search.py#L176)

Gradio callback: (query,k,prepared lab) -> (gallery,score table,status).

Gallery captions preserve author/license/source URLs; table includes IDs and
cosine. k may be a whole-number slider value. No new model/index is created.

## reconstruction_view

[`reconstruction_view`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/embedding_app.py#L12)

Return original/reconstruction/absolute residual uint8 images and metrics.

name is a prepared-model key, index is an integral test row. Residual has
fixed scale [0,1], not independent contrast enhancement. VAE decodes mu.
Input order to bind: name,index; models/images are fixed callback context.

## latent_view

[`latent_view`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/embedding_app.py#L28)

Decode one user-selected 2D coordinate and describe sampling context.

Returns uint8 [28,28], status. Sliders do not retrain; beta choices switch
separately trained checkpoints. Coordinate limits are UI choices, not bounds
on the Gaussian distribution or guarantees of realistic output.

## build_embedding_app

[`build_embedding_app`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/embedding_app.py#L42)

Return Gradio Blocks with reconstruction, 2D latent, optional real search.

models: loaded checkpoint mapping; images: held-out CPU images; search_lab:
prepared CLIPSearch with index_images already run, or None for AE/VAE-only.
Builds UI/events but does not download/train/launch. Launch via
app.launch(css=APP_CSS,share=True) in Colab; link expires with runtime.
For search, also pass allowed_paths=search_lab.gallery['path'].tolist()
to launch: only the verified public photo files need to be served.
The same numerical callbacks are independently callable for verification.
