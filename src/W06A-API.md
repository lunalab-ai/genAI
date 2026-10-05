# W06A · 함수·클래스·메소드 정의 바로가기

설치 버전 **luna-genai 0.1.9 / 2026-fall-w06a**. 아래 링크는 이 버전의 실제 선언 줄을 가리킵니다. 이전 차시 구현과 체크포인트를 보존하며 새 확산 실험을 추가했습니다.

## 사용 순서

`load_mnist` → `mix_at` / `prediction_targets` → `trace_unet` → `select_digit` / `fit_diffusion_project` → `sample_with_sampler` → `build_diffusion_lab`. 단순 함수 호출, 실제 학습, 파일 읽기, 서버 시작을 구별합니다.

## autoencoder.load_mnist

[`autoencoder.load_mnist`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06a/src/luna_genai/autoencoder.py#L92)

Download ~11.6 MB once, verify MD5 each call, then return fixed splits.

Defaults: 6000/1000 selected without overlap from official 60000 train;
1000 selected from official 10000 test. Cache defaults to
~/.cache/luna-genai/mnist, or LUNA_MNIST_CACHE if set. Raises on network,
checksum, IDX, or size errors; never substitutes synthetic data.
offline=True forbids downloads. Seed affects selection, not the files.

## diffusion.make_scheduler

[`diffusion.make_scheduler`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06a/src/luna_genai/diffusion.py#L24)

Return a fresh 100-step cosine DDPM epsilon scheduler; no learned weights.

clip_sample=True clips estimated clean images during sampling, not training
noise. Its step() includes the scheduled mean and variance, not just x-eps.
Example: scheduler=make_scheduler(); scheduler.alphas_cumprod.shape == (100,).
The schedule is fixed for this lesson, not a schedule-comparison experiment.

## diffusion.forward_noise

[`diffusion.forward_noise`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06a/src/luna_genai/diffusion.py#L49)

Apply sqrt(alpha_bar)*clean + sqrt(1-alpha_bar)*noise, preserving shape.

All images are CPU float32 [B,1,28,28]; timesteps is int64 [B], 0..99.
clean is in [-1,1]; noise is a supplied standard-normal draw, not a label.
Does not clamp noisy output, draw randomness, or train a network.
Example: forward_noise(x, torch.full((len(x),),49), torch.randn_like(x)).

## diffusion.TimeBlock

[`diffusion.TimeBlock`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06a/src/luna_genai/diffusion.py#L62)

Residual two-convolution block; add projected [B,64] time features.

Internal U-Net component. Spatial size is unchanged; channels change from
in_channels to out_channels. No down/up sampling or optimizer step here.

## diffusion.TimeBlock.__init__

[`diffusion.TimeBlock.__init__`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06a/src/luna_genai/diffusion.py#L68)

See class contract.

## diffusion.TimeBlock.forward

[`diffusion.TimeBlock.forward`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06a/src/luna_genai/diffusion.py#L77)

Return residual features; broadcast time along height and width.

## diffusion.TinyTimeUNet

[`diffusion.TinyTimeUNet`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06a/src/luna_genai/diffusion.py#L85)

CPU epsilon predictor: [B,1,28,28] and int64 [B] -> [B,1,28,28].

Channels 16->32->64->32->16; sizes 28->14->7->14->28. Skip concatenation
retains fine spatial features. Sinusoidal time features enter every block.
seed=20260930 initializes weights locally without training or downloading.
This small teaching model differs from the textbook RGB 64x64 U-Net.
Example: model=TinyTimeUNet(); prediction=model(noisy, timesteps).

## diffusion.TinyTimeUNet.__init__

[`diffusion.TinyTimeUNet.__init__`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06a/src/luna_genai/diffusion.py#L94)

See class contract.

## diffusion.TinyTimeUNet.forward

[`diffusion.TinyTimeUNet.forward`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06a/src/luna_genai/diffusion.py#L104)

Predict epsilon without parameter updates; do not pass true epsilon in.

Input/output share the image shape but different meanings. The output
is unbounded predicted noise, not a sigmoid image or a digit class.

## diffusion.training_step

[`diffusion.training_step`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06a/src/luna_genai/diffusion.py#L122)

Perform one real MSE/backward/optimizer step and return measured scalars.

Mutates model parameters and optimizer state; sets model.train(). Inputs
follow forward_noise(). target is the supplied epsilon, never a digit ID.
Returns loss, gradient_norm, backward_parameter_change, parameter_change.
Example: training_step(model, Adam(model.parameters()), x, t, epsilon).

## diffusion.evaluate_noise

[`diffusion.evaluate_noise`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06a/src/luna_genai/diffusion.py#L148)

Evaluate fixed held-out images and fixed Gaussian noise at each code t.

Default five levels, seed901. Returns columns t,mse,zero_mse,images. A zero
epsilon predictor gives ~1 MSE. Same noise is reused across levels for a
controlled marginal comparison, not a sampled forward Markov path.
Preserves model training mode; no optimizer updates. Use validation/test,
not the training batch, when comparing checkpoints.

## diffusion.sample_diffusion

[`diffusion.sample_diffusion`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06a/src/luna_genai/diffusion.py#L173)

Run all 100 DDPM reverse updates from Gaussian noise, no source image.

count=8 (1..64), seed=42; returns images [N,1,28,28] in [-1,1] and frames.
Each frame contains t, input (x_t), predicted_clean (clipped x0 estimate),
and previous (scheduler's next state). Frames are CPU tensors; input and
previous are not clamped. The model mode is preserved. Repeats with the
same CPU version/model/seed reproduce the same trajectory.
Example: result=sample_diffusion(model,count=4); result['images'].shape.

## diffusion.load_diffusion_checkpoints

[`diffusion.load_diffusion_checkpoints`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06a/src/luna_genai/diffusion.py#L204)

Load bundled initial/early/trained CPU models after SHA-256 validation.

NPZ allow_pickle=False; strict state-dict loading. Returns (models,metadata),
models keyed initial/early/trained in eval mode. No download or training.
Raises on corruption; never silently substitutes random weights.
Example: models,record=load_diffusion_checkpoints(); models['trained'].

## diffusion.image_grid

[`diffusion.image_grid`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06a/src/luna_genai/diffusion.py#L225)

Return a matplotlib Figure for [-1,1] images, clipping for display only.

CPU float32 [B,1,28,28]; columns=8. Does not mutate tensors. Figures use
a common -1..1 scale so contrast does not get rescaled per image.

## diffusion_lab.make_noise_schedule

[`diffusion_lab.make_noise_schedule`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06a/src/luna_genai/diffusion_lab.py#L18)

Create a FORWARD experiment schedule, without changing any trained model.

name='cosine' or 'linear'; steps=100, beta_end=.02 (linear only).
Returns a new scheduler; betas and alphas_cumprod are CPU float32 [steps].
Linear beta_start=.0001. Changing training betas at inference is invalid.
Example: make_noise_schedule('linear').alphas_cumprod[-1].item().

## diffusion_lab.coefficient_table

[`diffusion_lab.coefficient_table`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06a/src/luna_genai/diffusion_lab.py#L38)

Return ordered t,beta,alpha_bar,signal,noise,snr rows; no mutation.

t=0 denotes the first noisy level, not mathematical clean x0.
snr=alpha_bar/(1-alpha_bar) assumes unit signal variance. With clean
variance V and scale c, the data SNR is c**2*V*snr.
Example: coefficient_table(make_noise_schedule()).iloc[[0,49,99]].

## diffusion_lab.mix_at

[`diffusion_lab.mix_at`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06a/src/luna_genai/diffusion_lab.py#L60)

Return noisy tensor a*(scale*clean)+b*noise at one code t.

clean/noise: same-shaped finite CPU float32 (any image resolution).
t is integral in [0,steps); scale is finite and >0, default1.
No RNG, clipping, download, training or input mutation. noise is supplied
standard normal. Example: mix_at(x,eps,49,make_noise_schedule()).

## diffusion_lab.prediction_targets

[`diffusion_lab.prediction_targets`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06a/src/luna_genai/diffusion_lab.py#L78)

Return known training targets epsilon, sample(x0), v_prediction.

Finite same-shaped CPU float32 tensors; scalar 0<alpha_bar<1.
v=a*epsilon-b*x0, with a=sqrt(alpha_bar), b=sqrt(1-alpha_bar).
These are constructed ground truths, NOT outputs of a trained network.
Does not alter inputs. Example: prediction_targets(x,eps,.64).

## diffusion_lab.recover_clean

[`diffusion_lab.recover_clean`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06a/src/luna_genai/diffusion_lab.py#L94)

Convert a same-shaped prediction to an unclipped clean-image estimate.

epsilon: (xt-b*prediction)/a; sample: prediction; v_prediction: a*xt-b*v.
0<alpha_bar<1 excludes singular endpoints. No randomness/parameter update.
Exact known targets reconstruct clean; learned errors do not disappear.
Example: recover_clean(xt,targets['epsilon'],.64,'epsilon').

## diffusion_lab.trace_unet

[`diffusion_lab.trace_unet`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06a/src/luna_genai/diffusion_lab.py#L113)

Observe actual forward shapes with temporary hooks; remove them on exit.

Inputs follow TinyTimeUNet.forward: CPU [B,1,28,28] and int64[B].
Returns block,input_shape,output_shape rows in execution order.
Disables gradients; preserves train/eval mode and model parameters.
Example: trace_unet(model,xt,torch.tensor([49])).

## diffusion_lab.select_digit

[`diffusion_lab.select_digit`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06a/src/luna_genai/diffusion_lab.py#L140)

Select a project dataset from an already integrity-checked MNISTSplit.

digit=None keeps all digits; 0..9 filters EACH existing split independently.
Reads cached official train labels (verified by load_mnist); no download.
Returns train/validation/test in [-1,1], split IDs and counts. Labels select
the population; they are not inputs to the unconditional diffusion model.
Empty selections fail. Example: select_digit(load_mnist(),5).

## diffusion_lab.fit_diffusion_project

[`diffusion_lab.fit_diffusion_project`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06a/src/luna_genai/diffusion_lab.py#L167)

Train a NEW time-conditioned epsilon predictor using real Adam updates.

clean is nonempty CPU float32 [N,1,28,28] in [-1,1]. Defaults:32 steps,
batch16, lr.001, seed20261005. Samples batches with replacement, random
t=0..99 and Gaussian noise; fixed W05B cosine schedule. Does not touch old
checkpoints. Returns model and per-step loss/gradient/change/timing rows.
32 steps illustrate optimization, not a fully trained image generator.
Example: model,history=fit_diffusion_project(project['train'],steps=8).

## diffusion_lab.sample_with_sampler

[`diffusion_lab.sample_with_sampler`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06a/src/luna_genai/diffusion_lab.py#L200)

Generate from Gaussian noise with the checkpoint's original trained betas.

DDPM requires100 steps and delegates to the existing sampler. DDIM accepts
2..100 steps, eta=0, trailing spacing (starts at code99). Both retain the
100-step cosine/epsilon training definition. No clean-image input or fit.
Returns images,initial,frames,steps,seed,method,seconds; frames contain
input,predicted_clean,previous,t. Preserves model mode/parameters.
Same seed pairs initial noise, not every later random operation across
different samplers. Example: sample_with_sampler(model,steps=20,count=4).

## diffusion_lab_app.noise_view

[`diffusion_lab_app.noise_view`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06a/src/luna_genai/diffusion_lab_app.py#L18)

Return original/noisy uint8 images and coefficients for a button click.

images: prepared CPU float32 [N,1,28,28] in [-1,1]; no downloads.
index,t,seed integral; schedule linear/cosine; scale .1..1.
Reuses the same Gaussian draw for the same seed, not a Markov path.
Only display values are clipped. No model/parameter mutation.
Example: noise_view(0,49,42,'cosine',1.,images=test).

## diffusion_lab_app.target_view

[`diffusion_lab_app.target_view`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06a/src/luna_genai/diffusion_lab_app.py#L43)

Plot the known target and algebraic reconstruction, NOT learned quality.

index,t,seed integral; kind epsilon/sample/v_prediction. Uses fixed cosine
training schedule. Returns matplotlib Figure and float reconstruction error.
All target fields share one symmetric scale; image fields use [-1,1].
Example: target_view(0,49,42,'v_prediction',images=test).

## diffusion_lab_app.generation_view

[`diffusion_lab_app.generation_view`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06a/src/luna_genai/diffusion_lab_app.py#L71)

Generate eight images with prepared models; return grid, trajectory, facts.

name is a key of models, setting DDPM100/DDIM20/DDIM50/DDIM100; seed integral.
Preserves parameters; no training, download, clean image or text prompt.
Example: generation_view('trained','DDIM20',42,models=models).

## diffusion_lab_app.build_diffusion_lab

[`diffusion_lab_app.build_diffusion_lab`](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06a/src/luna_genai/diffusion_lab_app.py#L95)

Build (not launch) Gradio Blocks from prepared clean images and models.

images: CPU [N,1,28,28] [-1,1]. models=None gives A's two-tab calculator;
prepared initial/early/trained models add B's generation tab. No CLIP load.
Each button maps input widgets -> numerical callback -> output widgets.
Returns Blocks; caller launches and owns server lifetime.
Example: app=build_diffusion_lab(test,models); app.launch(share=True).

## 외부 API

[DDPMScheduler](https://huggingface.co/docs/diffusers/api/schedulers/ddpm) · [DDIMScheduler](https://huggingface.co/docs/diffusers/api/schedulers/ddim) · [MSELoss](https://docs.pytorch.org/docs/2.8/generated/torch.nn.MSELoss.html) · [Adam](https://docs.pytorch.org/docs/2.8/generated/torch.optim.Adam.html).

외부 Diffusers는0.40.0, Gradio는6.26.0에 고정합니다. 노트북에서 입출력·차원·사용 목적을 확인한 뒤 정의를 열어 내부 연산을 추적하세요.
