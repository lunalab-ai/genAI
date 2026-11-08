# W11A 공통 API · 0.1.18 / 2026-fall-w11a

노트북 본문에 동일 구현의 핵심 코드를 펼쳐 제공합니다. 아래 소스가 단일 실행 구현입니다. source 발췌의 SHA는 제작 단계에서 대조합니다. import만으로 학습·다운로드하지 않습니다.

## image_editing

### blend_mask

[실제 정의](https://github.com/lunalab-ai/genAI/blob/2026-fall-w11a/src/luna_genai/image_editing.py#L20)

Float arrays [H,W,C], mask [H,W] in [0,1]; white selects edited pixels.

Returns a new float array. This explicit output overlay preserves black
pixels exactly; a diffusion inpainting pipeline by itself need not do so.

### region_mse

[실제 정의](https://github.com/lunalab-ai/genAI/blob/2026-fall-w11a/src/luna_genai/image_editing.py#L34)

Mean per-channel squared error in binary inside/outside regions.

Inputs HWC float images and HW binary mask; empty region returns None,
never zero evidence for a missing region. Scale depends on input units.

### weighted_embedding

[실제 정의](https://github.com/lunalab-ai/genAI/blob/2026-fall-w11a/src/luna_genai/image_editing.py#L48)

Teaching linear combination [K,T,D], [K] -> [T,D], not a Compel parser.

Weights are coefficients, not probabilities or object-area percentages.
No normalization is applied, no tokens parsed, no model state changed.

### semantic_delta

[실제 정의](https://github.com/lunalab-ai/genAI/blob/2026-fall-w11a/src/luna_genai/image_editing.py#L61)

One-concept SEGA-inspired local calculation; NOT the full SEGA scheduler.

Equal float tensors -> delta and guided prediction. Keep magnitudes >=
quantile, optionally reverse, then add scale*delta. Omits warmup/momentum.

### separate_attention

[실제 정의](https://github.com/lunalab-ai/genAI/blob/2026-fall-w11a/src/luna_genai/image_editing.py#L81)

IP-Adapter-inspired single-head algebra, not a trained IP-Adapter.

Q[N,D], Kt[T,D], Vt[T,V], Ki[I,D], Vi[I,V] -> separate softmax maps and
text + scale*image [N,V]. Concatenating tokens changes the denominator.

### add_control

[실제 정의](https://github.com/lunalab-ai/genAI/blob/2026-fall-w11a/src/luna_genai/image_editing.py#L105)

ControlNet-inspired residual addition only; neither trains nor encodes an image.

### tiny_edit

[실제 정의](https://github.com/lunalab-ai/genAI/blob/2026-fall-w11a/src/luna_genai/image_editing.py#L113)

Real CPU image2image with pretrained 28x28 class-conditioned denoiser.

clean [B,1,28,28] in [-1,1], mask same shape with 0/1. strength picks a
suffix of DDIM steps. With mask, black pixels are explicitly restored at
each noise level and in final output. This is pixel-space teaching, not
SDXL's nine-channel latent inpainting. Model weights/training mode preserved.

### load_edit_records

[실제 정의](https://github.com/lunalab-ai/genAI/blob/2026-fall-w11a/src/luna_genai/image_editing.py#L154)

Load checked actual full-SD execution records; never execute/download SD.

Returns manifest and asset directory. Verify every recorded SHA before any
image is shown. Missing/tampered records raise; no synthetic replacement.

## image_editing_live

### make_edit_pipeline

[실제 정의](https://github.com/lunalab-ai/genAI/blob/2026-fall-w11a/src/luna_genai/image_editing_live.py#L20)

Load explicit kind: img2img/inpaint/semantic/inversion/instruction/control.

First call may download multiple GB. CPU float32, CUDA float16. Keeps
safety checking. Control uses SD1.5 Canny, not SDXL depth; instruction uses
InstructPix2Pix, not CosXL. No silent alternative if memory/cache fails.

### attach_ip_adapter

[실제 정의](https://github.com/lunalab-ai/genAI/blob/2026-fall-w11a/src/luna_genai/image_editing_live.py#L59)

Install pinned trained SD1.5 image-attention weights and CLIP image encoder.

Mutates pipeline attention processors. Downloads cached public weights;
does not train them. SDXL block routing is not copied to SD1.5.

### checked_generate

[실제 정의](https://github.com/lunalab-ai/genAI/blob/2026-fall-w11a/src/luna_genai/image_editing_live.py#L81)

Run a real pipeline and require an actually assessed, unblocked image.

Returns image, seconds, pipeline class and JSON-compatible supplied settings.
This method never returns a recorded image after an execution error.

## image_editing_app

### tiny_view

[실제 정의](https://github.com/lunalab-ai/genAI/blob/2026-fall-w11a/src/luna_genai/image_editing_app.py#L15)

Actual sampling callback. B1x28x28 -> PIL image, trace table, settings dict.

### record_view

[실제 정의](https://github.com/lunalab-ai/genAI/blob/2026-fall-w11a/src/luna_genai/image_editing_app.py#L28)

Select a measured 2x2 cell. Booleans do NOT generate a new full SD image.

### observation_note

[실제 정의](https://github.com/lunalab-ai/genAI/blob/2026-fall-w11a/src/luna_genai/image_editing_app.py#L39)

Default four-field note; students replace this callback in the last activity.

### export_observation

[실제 정의](https://github.com/lunalab-ai/genAI/blob/2026-fall-w11a/src/luna_genai/image_editing_app.py#L44)

Create new JSON+CSV local downloads; no external upload or overwrite.

### build_edit_app

[실제 정의](https://github.com/lunalab-ai/genAI/blob/2026-fall-w11a/src/luna_genai/image_editing_app.py#L55)

Build a real callback graph; launching is explicit and no SD is downloaded.

## 작은 호출 예

```python
import torch
from luna_genai.image_editing import separate_attention
q=torch.zeros(1,1); k=torch.zeros(2,1)
r=separate_attention(q,k,torch.tensor([[2.],[4.]]),k,torch.tensor([[10.],[14.]]),.5)
print(r["combined"])  # tensor([[9.]])
```

실제 SD 현재 실행은 명시적 make_edit_pipeline/checked_generate가 필요합니다. record_view는 기존 실행 기록을 읽습니다. export_observation만 새 임시 다운로드 파일을 씁니다.
