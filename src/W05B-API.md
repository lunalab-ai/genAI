# W05B · 클래스·함수·메소드 정의 바로가기

설치 버전 0.1.8 / 태그 `2026-fall-w05b`. 각 링크는 이 버전의 실제 선언 줄입니다. 입력·출력·부작용을 확인한 뒤 실습 셀로 돌아오세요.

## MNISTSplit

[`MNISTSplit`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05b/src/luna_genai/autoencoder.py#L37)

```text
Image tensors are float32 [N,1,28,28] in [0,1], on CPU.

train_ids/validation_ids refer to the official train set; test_ids refer
to the separate official test set. Labels are for plotting, never fit.
downloads lists files fetched in this call; an empty list means cache use.
```

## load_mnist

[`load_mnist`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05b/src/luna_genai/autoencoder.py#L92)

```text
Download ~11.6 MB once, verify MD5 each call, then return fixed splits.

Defaults: 6000/1000 selected without overlap from official 60000 train;
1000 selected from official 10000 test. Cache defaults to
~/.cache/luna-genai/mnist, or LUNA_MNIST_CACHE if set. Raises on network,
checksum, IDX, or size errors; never substitutes synthetic data.
offline=True forbids downloads. Seed affects selection, not the files.
```

## ConvAutoencoder

[`ConvAutoencoder`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05b/src/luna_genai/autoencoder.py#L132)

```text
Small CPU CNN, distinct from the textbook's large model.

Input [B,1,28,28] -> [B,8,14,14] -> [B,16,7,7] -> [B,d].
Decoder: [B,d] -> [B,16,7,7] -> [B,8,14,14] -> [B,1,28,28].
latent_dim defaults to 16; seed resets torch's RNG for initialization.
Output uses sigmoid (continuous values), latent code is unconstrained.
encode/decode/forward keep gradients unless called in inference_mode.
```

## ConvAutoencoder.encode

[`ConvAutoencoder.encode`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05b/src/luna_genai/autoencoder.py#L154)

```text

```

## ConvAutoencoder.decode

[`ConvAutoencoder.decode`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05b/src/luna_genai/autoencoder.py#L157)

```text

```

## ConvAutoencoder.forward

[`ConvAutoencoder.forward`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05b/src/luna_genai/autoencoder.py#L160)

```text

```

## reconstruct

[`reconstruct`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05b/src/luna_genai/autoencoder.py#L173)

```text
Return detached CPU reconstructions; set model.eval(), disable autograd.

Does not change parameters. images must be CPU float32 [N,1,28,28] in
[0,1]. batch_size=256 limits working memory; input order is preserved.
```

## reconstruction_mse

[`reconstruction_mse`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05b/src/luna_genai/autoencoder.py#L187)

```text
Mean squared error over every image/channel/pixel; lower is better.
```

## fit_autoencoder

[`fit_autoencoder`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05b/src/luna_genai/autoencoder.py#L192)

```text
Mutate model by Adam training; return epoch/train_mse/val_mse/seconds.

Uses inputs as targets, no labels and no test set. Fixed epochs; validation
monitors training, does not select a checkpoint. train_mse accumulates
pre-update minibatch losses, val_mse uses epoch-end weights, so they are
not measurements of exactly the same model. Limits torch CPU threads to
threads (process-wide). Repeated calls continue from current parameters.
```

## comparison_rows

[`comparison_rows`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05b/src/luna_genai/autoencoder.py#L231)

```text
Evaluate once on the same held-out test set, plus train-mean baseline.

Baseline predicts the per-pixel mean of training images for every test
image. No test image contributes to that mean. Returns list of records.
```

## reconstruction_figure

[`reconstruction_figure`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05b/src/luna_genai/autoencoder.py#L244)

```text
Matplotlib Figure: same first count test images, mean, and each model.

Fixed gray range [0,1]. Does not cherry-pick examples by reconstruction.
Caller owns the figure and can savefig/close it. No disk writes here.
```

## latent_figure

[`latent_figure`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05b/src/luna_genai/autoencoder.py#L267)

```text
2D latent scatter, digit labels used only as colors. Return Figure.

Axes have no predefined semantic meaning. Requires latent_dim=2 and one
label per image. Runs eval/inference_mode; does not alter weights.
```

## reconstruction_callback

[`reconstruction_callback`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05b/src/luna_genai/autoencoder.py#L288)

```text
Return input, reconstruction (uint8 28x28), and MSE text for Gradio.

Validates index and dimension; inference only, no downloads or training.
This is the supplied reconstruction feature. Students add latent controls.
```

## build_autoencoder_app

[`build_autoencoder_app`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05b/src/luna_genai/autoencoder.py#L304)

```text
Build supplied reconstruction-only Blocks; no server starts.

Returns (app, callback). Student notebook adds a separate latent explorer
rather than hiding that exercise's completed wiring in this public API.
app.launch(share=False, css=MNIST_CSS) starts locally. Colab share=True is opt-in and
exposes the app while that runtime is alive. Models are reused in memory.
```

## unit_rows

[`unit_rows`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05b/src/luna_genai/semantic_search.py#L23)

```text
Normalize finite nonzero [N,d] rows by L2 norm; return same-shape tensor.

Reject zero vectors instead of pretending their cosine is defined. Preserves
device, dtype and autograd. Example: unit_rows(tensor([[3.,4.]])) -> [[.6,.8]].
```

## load_gallery

[`load_gallery`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05b/src/luna_genai/semantic_search.py#L36)

```text
Verify and return the 40 bundled Commons photos in fixed P01..P40 order.

Columns include id,category,title,source,author,license,license_url,sha256,path.
Photos arrive with the fixed package install; no login/upload required. Paths
are local filenames usable by PIL/Gradio. Raises on missing/corrupted images.
Attribution lives in gallery.json and GALLERY-CREDITS.md; do not strip it.
```

## precision_at_k

[`precision_at_k`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05b/src/luna_genai/semantic_search.py#L53)

```text
Count relevant unique IDs among first k results / k; missing ranks count 0.

k=3 positive integer; duplicate IDs are invalid. Relevance must be defined
before inspecting scores. An absent-target query can have empty relevance.
Example: precision_at_k(['A','B','C'],{'A','C'},3) == 2/3.
```

## CLIPSearch

[`CLIPSearch`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05b/src/luna_genai/semantic_search.py#L65)

```text
Load fixed CLIP on CPU, then separately index pixels and query by text.

gallery=None loads checked classroom photos; cache_dir=None uses
~/.cache/luna-genai/clip-w05a. local_files_only=False permits initial model
download (~605 MB); True requires existing complete HF cache. First call
loads the model, index_images builds/reuses the feature cache. search never
trains or re-encodes stored images. English is the primary query language.
```

## CLIPSearch.encode_images

[`CLIPSearch.encode_images`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05b/src/luna_genai/semantic_search.py#L106)

```text
Read RGB images; return normalized CPU [N,512], preserving path order.

batch_size=8 limits memory. CLIP processor resizes/crops to 224x224 and
applies its pretrained pixel normalization. Image filenames never enter
the model. Transformers 5.15.1 returns projected features in pooler_output.
```

## CLIPSearch.encode_texts

[`CLIPSearch.encode_texts`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05b/src/luna_genai/semantic_search.py#L125)

```text
Return normalized [N,512] for nonempty strings, without training.

Tokenization pads a batch and truncates at the model's 77-token limit.
Short English descriptions are recommended; Korean is exploratory.
```

## CLIPSearch.index_images

[`CLIPSearch.index_images`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05b/src/luna_genai/semantic_search.py#L137)

```text
Compute/reuse normalized [N,512] pixel embeddings; return the matrix.

force=False reuses matching model revision, library version, ordered
image hashes and IDs. A corrupt cache raises; force=True rebuilds that
derived file. Writes one local NPZ atomically, never changes weights.
```

## CLIPSearch.search

[`CLIPSearch.search`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05b/src/luna_genai/semantic_search.py#L159)

```text
Return descending cosine top-k rows with rank,id,cosine and metadata.

Requires index_images first. k=3, range 1..gallery size. Ties preserve
gallery order. Cosine is not a calibrated probability. A result is always
returned even when the requested object is absent from the gallery.
```

## search_view

[`search_view`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05b/src/luna_genai/semantic_search.py#L176)

```text
Gradio callback: (query,k,prepared lab) -> (gallery,score table,status).

Gallery captions preserve author/license/source URLs; table includes IDs and
cosine. k may be a whole-number slider value. No new model/index is created.
```

## contrastive_table

[`contrastive_table`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05b/src/luna_genai/clip_teaching.py#L9)

```text
Compute paired N-by-N cosine/logits/probabilities and symmetric CE loss.

Finite, nonzero [N,d] matrices; matching diagonal pairs, N>=2. scale=10 is
an explicit toy setting; use model.logit_scale.exp() for a real CLIP batch.
Normalizes rows; returns cosine, logits, image_to_text, text_to_image, loss.
text_to_image rows are texts (transpose logits). Preserves autograd; does
not train. Example: contrastive_table(torch.eye(3),torch.eye(3),scale=2).
```

## compare_search

[`compare_search`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05b/src/luna_genai/clip_teaching.py#L28)

```text
Two independent real searches -> two galleries, combined table, message.

Inputs: two nonempty strings, integral k, already indexed CLIPSearch.
Returns (gallery_a,gallery_b,DataFrame,status). No training/download occurs;
table has query,rank,id,cosine,category,source. Gallery keeps attribution.
Example: compare_search('a cat in the snow','a cat indoors',3,lab).
```

## make_scheduler

[`make_scheduler`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05b/src/luna_genai/diffusion.py#L24)

```text
Return a fresh 100-step cosine DDPM epsilon scheduler; no learned weights.

clip_sample=True clips estimated clean images during sampling, not training
noise. Its step() includes the scheduled mean and variance, not just x-eps.
Example: scheduler=make_scheduler(); scheduler.alphas_cumprod.shape == (100,).
The schedule is fixed for this lesson, not a schedule-comparison experiment.
```

## forward_noise

[`forward_noise`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05b/src/luna_genai/diffusion.py#L49)

```text
Apply sqrt(alpha_bar)*clean + sqrt(1-alpha_bar)*noise, preserving shape.

All images are CPU float32 [B,1,28,28]; timesteps is int64 [B], 0..99.
clean is in [-1,1]; noise is a supplied standard-normal draw, not a label.
Does not clamp noisy output, draw randomness, or train a network.
Example: forward_noise(x, torch.full((len(x),),49), torch.randn_like(x)).
```

## TimeBlock

[`TimeBlock`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05b/src/luna_genai/diffusion.py#L62)

```text
Residual two-convolution block; add projected [B,64] time features.

Internal U-Net component. Spatial size is unchanged; channels change from
in_channels to out_channels. No down/up sampling or optimizer step here.
```

## TinyTimeUNet

[`TinyTimeUNet`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05b/src/luna_genai/diffusion.py#L85)

```text
CPU epsilon predictor: [B,1,28,28] and int64 [B] -> [B,1,28,28].

Channels 16->32->64->32->16; sizes 28->14->7->14->28. Skip concatenation
retains fine spatial features. Sinusoidal time features enter every block.
seed=20260930 initializes weights locally without training or downloading.
This small teaching model differs from the textbook RGB 64x64 U-Net.
Example: model=TinyTimeUNet(); prediction=model(noisy, timesteps).
```

## TinyTimeUNet.forward

[`TinyTimeUNet.forward`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05b/src/luna_genai/diffusion.py#L104)

```text
Predict epsilon without parameter updates; do not pass true epsilon in.

Input/output share the image shape but different meanings. The output
is unbounded predicted noise, not a sigmoid image or a digit class.
```

## training_step

[`training_step`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05b/src/luna_genai/diffusion.py#L122)

```text
Perform one real MSE/backward/optimizer step and return measured scalars.

Mutates model parameters and optimizer state; sets model.train(). Inputs
follow forward_noise(). target is the supplied epsilon, never a digit ID.
Returns loss, gradient_norm, backward_parameter_change, parameter_change.
Example: training_step(model, Adam(model.parameters()), x, t, epsilon).
```

## evaluate_noise

[`evaluate_noise`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05b/src/luna_genai/diffusion.py#L148)

```text
Evaluate fixed held-out images and fixed Gaussian noise at each code t.

Default five levels, seed901. Returns columns t,mse,zero_mse,images. A zero
epsilon predictor gives ~1 MSE. Same noise is reused across levels for a
controlled marginal comparison, not a sampled forward Markov path.
Preserves model training mode; no optimizer updates. Use validation/test,
not the training batch, when comparing checkpoints.
```

## sample_diffusion

[`sample_diffusion`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05b/src/luna_genai/diffusion.py#L173)

```text
Run all 100 DDPM reverse updates from Gaussian noise, no source image.

count=8 (1..64), seed=42; returns images [N,1,28,28] in [-1,1] and frames.
Each frame contains t, input (x_t), predicted_clean (clipped x0 estimate),
and previous (scheduler's next state). Frames are CPU tensors; input and
previous are not clamped. The model mode is preserved. Repeats with the
same CPU version/model/seed reproduce the same trajectory.
Example: result=sample_diffusion(model,count=4); result['images'].shape.
```

## load_diffusion_checkpoints

[`load_diffusion_checkpoints`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05b/src/luna_genai/diffusion.py#L204)

```text
Load bundled initial/early/trained CPU models after SHA-256 validation.

NPZ allow_pickle=False; strict state-dict loading. Returns (models,metadata),
models keyed initial/early/trained in eval mode. No download or training.
Raises on corruption; never silently substitutes random weights.
Example: models,record=load_diffusion_checkpoints(); models['trained'].
```

## image_grid

[`image_grid`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05b/src/luna_genai/diffusion.py#L225)

```text
Return a matplotlib Figure for [-1,1] images, clipping for display only.

CPU float32 [B,1,28,28]; columns=8. Does not mutate tensors. Figures use
a common -1..1 scale so contrast does not get rescaled per image.
```

## noise_view

[`noise_view`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05b/src/luna_genai/diffusion_app.py#L13)

```text
Return original/noisy uint8 images and coefficient/range measurements.

images: prepared clean [-1,1] CPU [N,1,28,28]. UI inputs index,t,seed must
be integers. A fixed seed uses identical epsilon when t changes: marginal
comparisons, not a single Markov trajectory. Display alone clips [-1,1].
No training or downloads. Returns (original,noisy,dict).
```

## generation_view

[`generation_view`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05b/src/luna_genai/diffusion_app.py#L35)

```text
Run 100 reverse steps for eight images and return grid/trajectory/status.

name: initial/early/trained; seed integral. Uses prepared models; switching
models does not train. Trajectory shows sample 0's x_t and predicted x0.
Returns matplotlib figures and Markdown. Changing seed changes the complete
random sequence; fixed seed permits paired checkpoint comparisons.
```

## build_study_app

[`build_study_app`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05b/src/luna_genai/diffusion_app.py#L57)

```text
Build Gradio Blocks for prepared CLIP and/or diffusion, without launch.

A supplies indexed search_lab only. B supplies models and clean test images;
pass all three for the cumulative lab. Explicitly absent components are not
fallback results. launch(share=True,allowed_paths=gallery.path.tolist()) in
Colab; allow only the checked photo files. Share URL expires with runtime.
```

## 외부 라이브러리 공식 설명

- [CLIPModel](https://huggingface.co/docs/transformers/model_doc/clip)
- [DDPMScheduler](https://huggingface.co/docs/diffusers/api/schedulers/ddpm)
- [PyTorch MSELoss](https://docs.pytorch.org/docs/2.8/generated/torch.nn.MSELoss.html)
- [Gradio 이벤트](https://www.gradio.app/guides/blocks-and-event-listeners)
