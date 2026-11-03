# W10B · API와 실제 정의

패키지0.1.17 / 공개 고정 tag2026-fall-w10b. 노트북은 실험을 조율하고 아래 모듈은 계산·학습·앱을 재사용합니다. import만으로 다운로드나 학습을 시작하지 않습니다.

| API | 입력과 반환/변경 | 실제 정의 |
|---|---|---|
| `load_teaching_images` | manifest/hash를 확인하고 RGB8개·CPU train6/validation2 반환. 다운로드 없음 | [정의 L45](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10b/src/luna_genai/diffusion_adaptation.py#L45) |
| `LowRankLinear` | 고정 Linear와 rank/alpha → A/B 학습 경로. base 동결 | [정의 L109](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10b/src/luna_genai/diffusion_adaptation.py#L109) |
| `prior_loss_components` | prediction/target·instance수·lambda → 미분 가능한 두 평균과 total | [정의 L131](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10b/src/luna_genai/diffusion_adaptation.py#L131) |
| `TinyAdaptationLab` | 고정 작은 checkpoint 복사. fit은 실제16회 기본 학습; state/evaluate/sample/save_reload_check 사용 | [정의 L150](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10b/src/luna_genai/diffusion_adaptation.py#L150) |
| `run_tiny_comparison` | 같은 시작점의 base+full/LoRA×prior0/1. 표·학습 객체·분리 데이터 반환 | [정의 L343](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10b/src/luna_genai/diffusion_adaptation.py#L343) |
| `inpainting_channels` | 4채널·1mask·4채널 → 검증 후9채널. 모델 실행 아님 | [정의 L373](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10b/src/luna_genai/diffusion_adaptation.py#L373) |
| `sdxl_condition_shapes` | SDXL base 조건 shape 계산. 가중치 다운로드·학습 없음 | [정의 L395](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10b/src/luna_genai/diffusion_adaptation.py#L395) |
| `load_actual_sd_record` | 실제 SD1.5 기록과 이미지 hash 검증. 새 생성 없음 | [정의 L409](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10b/src/luna_genai/diffusion_adaptation.py#L409) |
| `identifier_tokens` | 고정 SD tokenizer 자동 준비/캐시 → 실제 piece·ID. 새 토큰 추가 아님 | [정의 L32](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10b/src/luna_genai/sd_adaptation.py#L32) |
| `StableDiffusionAdapterLab` | 고정 SD1.5 로드. fit(prior_images,...) 실제 attention LoRA; generate/save_reload_check/fusion_check 명시 호출 | [정의 L48](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10b/src/luna_genai/sd_adaptation.py#L48) |
| `prepare_project_files` | 새 디렉터리에 train6/validation2·metadata. SDXL 실행 없음 | [정의 L23](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10b/src/luna_genai/sdxl_project.py#L23) |
| `sdxl_command` | 고정 공식 script/base 인자 목록 반환. 실행 없음 | [정의 L49](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10b/src/luna_genai/sdxl_project.py#L49) |
| `download_reviewed_script` | 공식v0.40.0 코드의 hash 확인 후 캐시. 실행 없음 | [정의 L76](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10b/src/luna_genai/sdxl_project.py#L76) |
| `run_optional_sdxl` | 명시 opt-in GPU 학습. 실패시 예외; 미검증 장치는 성공 보장 없음 | [정의 L94](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10b/src/luna_genai/sdxl_project.py#L94) |
| `tiny_view` | 이미 학습한 동일 객체에서 실제 작은 모델 표본/표/JSON | [정의 L20](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10b/src/luna_genai/diffusion_adaptation_app.py#L20) |
| `recorded_view` | 존재하는 prompt·seed만 사전 SD 그림·loss·JSON. 차단은 이미지 없음 | [정의 L37](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10b/src/luna_genai/diffusion_adaptation_app.py#L37) |
| `channel_view` | 마스크/반전 → 이미지·9채널/SDXL 구성 JSON | [정의 L55](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10b/src/luna_genai/diffusion_adaptation_app.py#L55) |
| `export_observation` | 실험payload+학생설명 → 새 임시 JSON 다운로드 경로 | [정의 L67](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10b/src/luna_genai/diffusion_adaptation_app.py#L67) |
| `build_tiny_app` | 준비한 비교 객체를 재사용하는 Gradio 앱. launch는 별도 | [정의 L89](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10b/src/luna_genai/diffusion_adaptation_app.py#L89) |
| `build_record_app` | 검증된 기록/채널/설명 앱. 새 SD 학습 없음 | [정의 L109](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10b/src/luna_genai/diffusion_adaptation_app.py#L109) |
