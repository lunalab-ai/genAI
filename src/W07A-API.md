# W07A API · 함수 정의 바로가기

패키지0.1.11 · 설치/소스 태그2026-fall-w07a. 링크는 실제 AST의 def 선언 줄에서 생성했습니다.

| 함수 | 직접 정의 |

| --- | --- |

| `sampling.ddim_scalar` | [L17](https://github.com/lunalab-ai/genAI/blob/2026-fall-w07a/src/luna_genai/sampling.py#L17) |

| `sampling.make_initial_latents` | [L42](https://github.com/lunalab-ai/genAI/blob/2026-fall-w07a/src/luna_genai/sampling.py#L42) |

| `sampling.decode_checked` | [L56](https://github.com/lunalab-ai/genAI/blob/2026-fall-w07a/src/luna_genai/sampling.py#L56) |

| `sampling.manual_sample` | [L76](https://github.com/lunalab-ai/genAI/blob/2026-fall-w07a/src/luna_genai/sampling.py#L76) |

| `sampling.compare_sampling_paths` | [L161](https://github.com/lunalab-ai/genAI/blob/2026-fall-w07a/src/luna_genai/sampling.py#L161) |

| `sampling.load_sampling_records` | [L188](https://github.com/lunalab-ai/genAI/blob/2026-fall-w07a/src/luna_genai/sampling.py#L188) |

| `sampling_app.sampling_record_view` | [L20](https://github.com/lunalab-ai/genAI/blob/2026-fall-w07a/src/luna_genai/sampling_app.py#L20) |

| `sampling_app.live_sampling_view` | [L37](https://github.com/lunalab-ai/genAI/blob/2026-fall-w07a/src/luna_genai/sampling_app.py#L37) |

| `sampling_app.build_sampling_app` | [L49](https://github.com/lunalab-ai/genAI/blob/2026-fall-w07a/src/luna_genai/sampling_app.py#L49) |

| `stable_diffusion.cfg_combine` | [L17](https://github.com/lunalab-ai/genAI/blob/2026-fall-w07a/src/luna_genai/stable_diffusion.py#L17) |

| `stable_diffusion.load_sd15` | [L30](https://github.com/lunalab-ai/genAI/blob/2026-fall-w07a/src/luna_genai/stable_diffusion.py#L30) |

| `conditional_app.recorded_view` | [L29](https://github.com/lunalab-ai/genAI/blob/2026-fall-w07a/src/luna_genai/conditional_app.py#L29) |

| `conditional_app.cfg_view` | [L40](https://github.com/lunalab-ai/genAI/blob/2026-fall-w07a/src/luna_genai/conditional_app.py#L40) |

## sampling.ddim_scalar

[정의 보기](https://github.com/lunalab-ai/genAI/blob/2026-fall-w07a/src/luna_genai/sampling.py#L17)

One illustrative deterministic DDIM step, with cumulative alpha values.

Returns predicted clean x0 and the next less-noisy state. This scalar
example has eta=0, no clipping; it neither calls SD nor measures quality.
Example: ddim_scalar(.6,.2,.25,.64)['previous'] is about .802872.

## sampling.make_initial_latents

[정의 보기](https://github.com/lunalab-ai/genAI/blob/2026-fall-w07a/src/luna_genai/sampling.py#L42)

New CPU generator on EACH call; return SD1.5 raw noise [1,4,64,64].

Draw float32 noise on CPU, then convert to pipe device/U-Net dtype.
Does not multiply scheduler.init_noise_sigma yet. No model execution.
Identical calls in this environment return equal tensors. Cross-device
generated images are not promised bitwise identical.

## sampling.decode_checked

[정의 보기](https://github.com/lunalab-ai/genAI/blob/2026-fall-w07a/src/luna_genai/sampling.py#L56)

Decode one scaled latent using VAE config and the active safety checker.

Returns PIL RGB. Raises on missing/final blocked assessment. For a blocked
preview only, returns the checker's BLACK image, labelled in image.info;
never returns the blocked pixels. Intermediate input
is noisy z, not the scheduler's predicted x0; label previews accordingly.

## sampling.manual_sample

[정의 보기](https://github.com/lunalab-ai/genAI/blob/2026-fall-w07a/src/luna_genai/sampling.py#L76)

Real single-image SD1.5 DDIM loop with visible intermediate tensors.

Input: loaded SD1.5 epsilon-prediction pipeline, English prompt/negative
strings, seed, CFG in [1,20], steps in [2,50]. Resolution fixed at 512.
Optional raw initial_latents [1,4,64,64] is cloned, never mutated.
Returns image, final_latent, trace (one row/step), frames and settings.
No optimizer/backward/weight updates. Uses independent scheduler state.
CFG=1 uses the conditional branch only, so negative prompt is inactive.
Call load_sd15(device='cuda') once before repeated calls.

## sampling.compare_sampling_paths

[정의 보기](https://github.com/lunalab-ai/genAI/blob/2026-fall-w07a/src/luna_genai/sampling.py#L161)

Compare real manual vs pipeline final latents from the SAME raw noise.

Fix DDIM, eta=0, resolution, prompt, negative prompt, dtype and steps.
Pipeline output_type='latent' is for numerical comparison, not display;
the manual final image is safety-checked. Reports measured errors and
allclose(atol=.002,rtol=.002). A false result is returned, not concealed.

## sampling.load_sampling_records

[정의 보기](https://github.com/lunalab-ai/genAI/blob/2026-fall-w07a/src/luna_genai/sampling.py#L188)

Read hash-checked W07A actual T4 records. No model load or new inference.

Returns metadata, runs, parity evidence and relative file names. Array
archive and every picture are verified before use; errors stay visible.
Use ASSETS/run['file'] for a final image. No network is performed here.

## sampling_app.sampling_record_view

[정의 보기](https://github.com/lunalab-ai/genAI/blob/2026-fall-w07a/src/luna_genai/sampling_app.py#L20)

Read one measured T4 run by key; NEVER generates an image.

Keys: base, negative-red, cfg-4, steps-10. Return PIL final image,
list[(PIL preview, caption)], settings dict and downloadable JSON path.
All underlying files are hash-checked. Preview is direct noisy-z decode.

## sampling_app.live_sampling_view

[정의 보기](https://github.com/lunalab-ai/genAI/blob/2026-fall-w07a/src/luna_genai/sampling_app.py#L37)

Five UI inputs go to manual_sample; reuse the already-loaded pipeline.

Return final image, preview gallery, measured settings and JSON download.
Errors propagate visibly. Fractional seed/steps are rejected, not rounded.

## sampling_app.build_sampling_app

[정의 보기](https://github.com/lunalab-ai/genAI/blob/2026-fall-w07a/src/luna_genai/sampling_app.py#L49)

Build, but do not launch/download/train, the W07A Gradio Blocks app.

pipe=None: explicit recorded mode; a prepared pipe adds live generation.
Optional summary_fn(setting,observation)->str is the student B4 extension.
Reuses W06B CFG and stored-image APIs. Single GPU request at a time.
Example: app=build_sampling_app(); app.launch(share=True).

## stable_diffusion.cfg_combine

[정의 보기](https://github.com/lunalab-ai/genAI/blob/2026-fall-w07a/src/luna_genai/stable_diffusion.py#L17)

Pure CFG algebra u+s*(c-u); same-shaped finite tensors, finite scale>=0.

Mathematical s=0 returns u, s=1 returns c. Diffusers SD pipeline skips
its two-branch CFG for API guidance_scale<=1; that API does not use this
formula to return unconditional generation at 0. No weights/state change.

## stable_diffusion.load_sd15

[정의 보기](https://github.com/lunalab-ai/genAI/blob/2026-fall-w07a/src/luna_genai/stable_diffusion.py#L30)

Load revision-pinned real SD 1.5 + DDIM; CPU float32 or CUDA float16.

First call downloads ~2.6 GB; cache reused. Keeps safety checker active.
Does not fine-tune. CUDA unavailability raises, with no CPU substitution.
local_files_only=True raises on cache miss. Returns Diffusers pipeline.

## conditional_app.recorded_view

[정의 보기](https://github.com/lunalab-ai/genAI/blob/2026-fall-w07a/src/luna_genai/conditional_app.py#L29)

Read a hash-verified previously generated SD image and exact settings.

key: cfg-1/cfg-4/cfg-7.5/cfg-12/blue/seed-43. Does not run a model.
Returns PIL image and record explicitly labelled as playback.

## conditional_app.cfg_view

[정의 보기](https://github.com/lunalab-ai/genAI/blob/2026-fall-w07a/src/luna_genai/conditional_app.py#L40)

Illustrative 2D epsilon vectors, not measured SD attention or image quality.

Returns a vector plot and the algebra result for u=(.2,-.1), c=(.5,.1).

## 외부 API 문서

[torch.Generator](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html) · [torch.randn](https://docs.pytorch.org/docs/stable/generated/torch.randn.html) · [DDIMScheduler](https://huggingface.co/docs/diffusers/api/schedulers/ddim) · [Gradio Blocks](https://www.gradio.app/docs/gradio/blocks)

## 실행 경계

load_sd15는 모델 다운로드/로드, manual_sample은 실제 추론, load_sampling_records는 파일 읽기, build_sampling_app은 UI 구성, launch는 서버 시작이다. 가중치 학습은 수행하지 않는다.
