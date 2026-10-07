# W06B API · 구현 바로가기

패키지0.1.10 · 태그2026-fall-w06b. 설치 ref와 아래 소스 ref는 같습니다. 클래스·메소드·함수의 입력/출력·상태 변경 계약을 확인하세요.

| API | 정의 |

| --- | --- |

| `conditional.FashionData` | [정의 L32](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/conditional.py#L32) |

| `conditional.load_fashion` | [정의 L44](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/conditional.py#L44) |

| `conditional.TinyClassUNet` | [정의 L81](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/conditional.py#L81) |

| `conditional.TinyClassUNet.__init__` | [정의 L88](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/conditional.py#L88) |

| `conditional.TinyClassUNet.forward` | [정의 L93](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/conditional.py#L93) |

| `conditional.conditional_step` | [정의 L107](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/conditional.py#L107) |

| `conditional.conditional_evaluation` | [정의 L126](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/conditional.py#L126) |

| `conditional.sample_classes` | [정의 L143](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/conditional.py#L143) |

| `conditional.load_conditional_checkpoint` | [정의 L169](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/conditional.py#L169) |

| `stable_diffusion.cfg_combine` | [정의 L17](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/stable_diffusion.py#L17) |

| `stable_diffusion.load_sd15` | [정의 L30](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/stable_diffusion.py#L30) |

| `stable_diffusion.text_condition` | [정의 L49](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/stable_diffusion.py#L49) |

| `stable_diffusion.vae_roundtrip` | [정의 L67](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/stable_diffusion.py#L67) |

| `stable_diffusion.generate_sd` | [정의 L87](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/stable_diffusion.py#L87) |

| `stable_diffusion.load_recorded_sd` | [정의 L117](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/stable_diffusion.py#L117) |

| `stable_diffusion.load_recorded_tensors` | [정의 L130](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/stable_diffusion.py#L130) |

| `conditional_app.class_view` | [정의 L17](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/conditional_app.py#L17) |

| `conditional_app.recorded_view` | [정의 L29](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/conditional_app.py#L29) |

| `conditional_app.cfg_view` | [정의 L40](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/conditional_app.py#L40) |

| `conditional_app.live_view` | [정의 L55](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/conditional_app.py#L55) |

| `conditional_app.build_conditional_app` | [정의 L64](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/conditional_app.py#L64) |

| `diffusion.make_scheduler` | [정의 L24](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/diffusion.py#L24) |

| `diffusion.forward_noise` | [정의 L49](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/diffusion.py#L49) |

| `diffusion.TimeBlock` | [정의 L62](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/diffusion.py#L62) |

| `diffusion.TimeBlock.__init__` | [정의 L68](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/diffusion.py#L68) |

| `diffusion.TimeBlock.forward` | [정의 L77](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/diffusion.py#L77) |

| `diffusion.TinyTimeUNet` | [정의 L85](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/diffusion.py#L85) |

| `diffusion.TinyTimeUNet.__init__` | [정의 L94](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/diffusion.py#L94) |

| `diffusion.TinyTimeUNet.forward` | [정의 L104](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/diffusion.py#L104) |

| `diffusion.image_grid` | [정의 L225](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/diffusion.py#L225) |

## conditional.FashionData

[정의 직접 보기](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/conditional.py#L32)

CPU images [N,1,28,28] in [-1,1], labels int64 [N], disjoint indices.

## conditional.load_fashion

[정의 직접 보기](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/conditional.py#L44)

Download/check official Fashion-MNIST and make reproducible disjoint splits.

Default train/validation/test=12000/1000/1000. train+validation<=60000;
test<=10000. Writes only missing cache files, checks MD5 and IDX headers.
offline=True raises when files are missing. No model training. Example:
data=load_fashion(); data.train.shape == (12000,1,28,28).

## conditional.TinyClassUNet

[정의 직접 보기](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/conditional.py#L81)

Learned class embedding + time embedding -> tiny conditional U-Net.

CPU noisy float32 [B,1,28,28], time int64 [B], class int64 [B] (0..9)
-> epsilon float32 [B,1,28,28]. Calling forward never trains weights.
This teaching design adds class/time vectors; SD uses token cross-attention.

## conditional.TinyClassUNet.__init__

[정의 직접 보기](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/conditional.py#L88)

Constructor: see class contract above.

## conditional.TinyClassUNet.forward

[정의 직접 보기](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/conditional.py#L93)

Predict Gaussian epsilon, not class probabilities; true epsilon is not an input.

## conditional.conditional_step

[정의 직접 보기](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/conditional.py#L107)

One actual MSE/backward/update; modifies model and optimizer, not input data.

Images [B,1,28,28], class/time int64 [B]. Returns measured loss, gradient
norm, parameter change and change before optimizer.step (should be zero).
Target is noise; labels condition the predictor, not the MSE target.

## conditional.conditional_evaluation

[정의 직접 보기](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/conditional.py#L126)

Held-out epsilon MSE at five times; same noise for correct/wrong labels.

Wrong condition is (label+1)%10. This is a sensitivity probe, not accuracy
or an assertion that every wrong class always increases MSE. No training.

## conditional.sample_classes

[정의 직접 보기](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/conditional.py#L143)

DDIM generation, fixed weights; labels choose classes, not clean targets.

labels has 1..40 integer IDs. matched_noise=True repeats ONE initial draw
across classes; False draws independent images. Returns images, initial,
seed, labels, steps. eta=0 gives deterministic sampling in the same setup.
Preserves model training mode. Does not download or optimize parameters.

## conditional.load_conditional_checkpoint

[정의 직접 보기](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/conditional.py#L169)

Load genuinely trained bundled NPZ after SHA-256 check; never random fallback.

No network. Returns eval model and full training record. NPZ disallows
pickle; missing/corrupt assets raise. Loading weights is not training.

## stable_diffusion.cfg_combine

[정의 직접 보기](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/stable_diffusion.py#L17)

Pure CFG algebra u+s*(c-u); same-shaped finite tensors, finite scale>=0.

Mathematical s=0 returns u, s=1 returns c. Diffusers SD pipeline skips
its two-branch CFG for API guidance_scale<=1; that API does not use this
formula to return unconditional generation at 0. No weights/state change.

## stable_diffusion.load_sd15

[정의 직접 보기](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/stable_diffusion.py#L30)

Load revision-pinned real SD 1.5 + DDIM; CPU float32 or CUDA float16.

First call downloads ~2.6 GB; cache reused. Keeps safety checker active.
Does not fine-tune. CUDA unavailability raises, with no CPU substitution.
local_files_only=True raises on cache miss. Returns Diffusers pipeline.

## stable_diffusion.text_condition

[정의 직접 보기](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/stable_diffusion.py#L49)

Inspect real token IDs and token-wise hidden states, not pooled CLIP search.

Returns tokens table, input_ids [1,77], hidden [1,77,768] CPU tensors,
unpadded_length and truncated flag. Does not mutate weights or generate.
Boundary/padding tokens remain visible; length 77 is not 77 words.

## stable_diffusion.vae_roundtrip

[정의 직접 보기](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/stable_diffusion.py#L67)

Actual encode/decode at 512 square; posterior mode for reproducible comparison.

Input PIL RGB resized to 512x512. Returns source/reconstruction uint8,
scaled_latent [1,4,64,64], range [0,1] image MSE and config scale.
Decode divides by the SAME config scale. This is reconstruction, not new
generation; four latent channels are not independently interpretable RGB.

## stable_diffusion.generate_sd

[정의 직접 보기](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/stable_diffusion.py#L87)

Real 512x512 DDIM generation with active safety checker and fixed weights.

Returns PIL image, full settings/time and five latent snapshots. Same CPU
initial-noise generator seed controls comparisons; cross-device bitwise
identity is not promised. guidance>=1; 1 means plain conditional prediction.
Safe blocked outputs raise, never silently replaced by cached pictures.

## stable_diffusion.load_recorded_sd

[정의 직접 보기](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/stable_diffusion.py#L117)

Read and verify actual generated records; does NOT run SD or need GPU.

Returns metadata and gallery entries. Each entry includes exact settings
plus image_path. All listed image/tensor hashes are validated on load.
Missing or corrupt file raises. Never a fallback for failed live generation.

## stable_diffusion.load_recorded_tensors

[정의 직접 보기](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/stable_diffusion.py#L130)

Return saved real SD component tensors as NumPy arrays; no inference.

Provenance/settings in sd-records.json; use load_recorded_sd to inspect.
Arrays include padded IDs, hidden states, scaled VAE latent, reconstruction,
unconditional/conditional noise predictions at a fixed noisy latent state.

## conditional_app.class_view

[정의 직접 보기](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/conditional_app.py#L17)

Actual tiny-model DDIM30 generation, four independent draws; no training.

Inputs class 0..9 and integer seed. Returns uint8 2x2 image and settings.
Changing class with same seed reuses the same four initial noise draws.

## conditional_app.recorded_view

[정의 직접 보기](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/conditional_app.py#L29)

Read a hash-verified previously generated SD image and exact settings.

key: cfg-1/cfg-4/cfg-7.5/cfg-12/blue/seed-43. Does not run a model.
Returns PIL image and record explicitly labelled as playback.

## conditional_app.cfg_view

[정의 직접 보기](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/conditional_app.py#L40)

Illustrative 2D epsilon vectors, not measured SD attention or image quality.

Returns a vector plot and the algebra result for u=(.2,-.1), c=(.5,.1).

## conditional_app.live_view

[정의 직접 보기](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/conditional_app.py#L55)

Actual SD generation callback; forwards all widget parameters, DDIM20.

Returns generated PIL image and measured settings. Errors remain visible;
never replaces failed inference with recorded images.

## conditional_app.build_conditional_app

[정의 직접 보기](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/conditional_app.py#L64)

Build Gradio Blocks without launch/download/training.

model: loaded TinyClassUNet. Optional prepared pipe adds live SD tab;
omission means recorded SD only. Caller owns server and uses .launch().
Class generation and recorded image settings are labelled separately.

## diffusion.make_scheduler

[정의 직접 보기](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/diffusion.py#L24)

Return a fresh 100-step cosine DDPM epsilon scheduler; no learned weights.

clip_sample=True clips estimated clean images during sampling, not training
noise. Its step() includes the scheduled mean and variance, not just x-eps.
Example: scheduler=make_scheduler(); scheduler.alphas_cumprod.shape == (100,).
The schedule is fixed for this lesson, not a schedule-comparison experiment.

## diffusion.forward_noise

[정의 직접 보기](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/diffusion.py#L49)

Apply sqrt(alpha_bar)*clean + sqrt(1-alpha_bar)*noise, preserving shape.

All images are CPU float32 [B,1,28,28]; timesteps is int64 [B], 0..99.
clean is in [-1,1]; noise is a supplied standard-normal draw, not a label.
Does not clamp noisy output, draw randomness, or train a network.
Example: forward_noise(x, torch.full((len(x),),49), torch.randn_like(x)).

## diffusion.TimeBlock

[정의 직접 보기](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/diffusion.py#L62)

Residual two-convolution block; add projected [B,64] time features.

Internal U-Net component. Spatial size is unchanged; channels change from
in_channels to out_channels. No down/up sampling or optimizer step here.

## diffusion.TimeBlock.__init__

[정의 직접 보기](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/diffusion.py#L68)

Constructor: see class contract above.

## diffusion.TimeBlock.forward

[정의 직접 보기](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/diffusion.py#L77)

Return residual features; broadcast time along height and width.

## diffusion.TinyTimeUNet

[정의 직접 보기](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/diffusion.py#L85)

CPU epsilon predictor: [B,1,28,28] and int64 [B] -> [B,1,28,28].

Channels 16->32->64->32->16; sizes 28->14->7->14->28. Skip concatenation
retains fine spatial features. Sinusoidal time features enter every block.
seed=20260930 initializes weights locally without training or downloading.
This small teaching model differs from the textbook RGB 64x64 U-Net.
Example: model=TinyTimeUNet(); prediction=model(noisy, timesteps).

## diffusion.TinyTimeUNet.__init__

[정의 직접 보기](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/diffusion.py#L94)

Constructor: see class contract above.

## diffusion.TinyTimeUNet.forward

[정의 직접 보기](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/diffusion.py#L104)

Predict epsilon without parameter updates; do not pass true epsilon in.

Input/output share the image shape but different meanings. The output
is unbounded predicted noise, not a sigmoid image or a digit class.

## diffusion.image_grid

[정의 직접 보기](https://github.com/lunalab-ai/genAI/blob/2026-fall-w06b/src/luna_genai/diffusion.py#L225)

Return a matplotlib Figure for [-1,1] images, clipping for display only.

CPU float32 [B,1,28,28]; columns=8. Does not mutate tensors. Figures use
a common -1..1 scale so contrast does not get rescaled per image.

## 의존 라이브러리 문서

[Diffusers SD pipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img) · [PyTorch Embedding](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html) · [Gradio Blocks](https://www.gradio.app/docs/gradio/blocks)

## 실행 경계

load 함수의 파일 다운로드/읽기, training step의 모수 갱신, sampling의 상태 갱신, build_app의 UI 구성, launch의 서버 시작을 구별합니다. 모델 오류를 기록 재생으로 자동 대체하지 않습니다.
