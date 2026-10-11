"""W07A sampling observation app: recorded evidence and live inference separated."""
from __future__ import annotations
import json
import tempfile
from functools import partial
from pathlib import Path
from typing import Any, Callable
from PIL import Image
from .sampling import ASSETS, load_sampling_records, manual_sample
from .stable_diffusion import DEFAULT_PROMPT
from .conditional_app import cfg_view, recorded_view


def _download(settings: dict) -> str:
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', prefix='w07a-', encoding='utf-8', delete=False) as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)
        return f.name


def sampling_record_view(key: str) -> tuple:
    """Read one measured T4 run by key; NEVER generates an image.

    Keys: base, negative-red, cfg-4, steps-10. Return PIL final image,
    list[(PIL preview, caption)], settings dict and downloadable JSON path.
    All underlying files are hash-checked. Preview is direct noisy-z decode.
    """
    record = load_sampling_records()
    matches = [x for x in record['runs'] if x['key']==key]
    if len(matches)!=1: raise ValueError('Select a prepared record')
    run = matches[0]
    frames = [(Image.open(ASSETS/f['file']).convert('RGB'),f"step {f['step']} · t={f['input_t']} · {f['safety_status']}") for f in run['frames']]
    settings = {k:v for k,v in run.items() if k not in ('frames','trace','file')}
    settings.update(mode='사전 실행 결과 재생 · 지금 SD를 생성하지 않음', trace=run['trace'])
    return Image.open(ASSETS/run['file']).convert('RGB'), frames, settings, _download(settings)


def live_sampling_view(prompt: str, negative: str, seed: float, guidance: float, steps: float, *, pipe: Any) -> tuple:
    """Five UI inputs go to manual_sample; reuse the already-loaded pipeline.

    Return final image, preview gallery, measured settings and JSON download.
    Errors propagate visibly. Fractional seed/steps are rejected, not rounded.
    """
    result = manual_sample(pipe, prompt, negative_prompt=negative, seed=seed, guidance=guidance, steps=steps)
    gallery = [(f['image'],f"step {f['step']} · t={f['input_t']} · {f['safety_status']}") for f in result['frames']]
    settings = dict(result['settings'], trace=result['trace'])
    return result['image'], gallery, settings, _download(settings)


def build_sampling_app(*, pipe: Any = None, summary_fn: Callable | None = None) -> Any:
    """Build, but do not launch/download/train, the W07A Gradio Blocks app.

    pipe=None: explicit recorded mode; a prepared pipe adds live generation.
    Optional summary_fn(setting,observation)->str is the student B4 extension.
    Reuses W06B CFG and stored-image APIs. Single GPU request at a time.
    Example: app=build_sampling_app(); app.launch(share=True).
    """
    import gradio as gr
    record = load_sampling_records()
    with gr.Blocks(title='W07A · 샘플링 과정 관찰실') as app:
        gr.Markdown('# 샘플링 과정 관찰실\n먼저 예상 → 조건 하나만 변경 → 이미지와 설정 확인 → 관찰의 한계 기록')
        with gr.Tab('1 · 실제 생성 기록 분석'):
            gr.Markdown('**저장된 T4 실행 결과입니다. 이 탭은 새 이미지를 생성하지 않습니다.**\nbase와 negative-red, cfg-4, steps-10을 비교하세요. 미리보기는 갱신된 noisy latent를 직접 디코딩한 결과이며 예측 x₀가 아닙니다.')
            choice=gr.Dropdown([r['key'] for r in record['runs']], value='base', label='비교할 실제 기록')
            show=gr.Button('기록과 중간 과정 보기',variant='primary')
            final=gr.Image(label='선택한 기록의 최종 이미지')
            gallery=gr.Gallery(label='중간 noisy latent 직접 복원',columns=5,height=260)
            settings=gr.JSON(label='기록의 설정·실행시간·단계별 수치')
            download=gr.File(label='실험 설정 JSON 내려받기')
            show.click(sampling_record_view, choice, [final,gallery,settings,download],api_name='sampling_record')
        with gr.Tab('2 · SD 지금 생성'):
            if pipe is None:
                gr.Markdown('현재 **기록 분석 모드**입니다. GPU 실습 셀에서 모델을 준비한 후 앱을 다시 구성하면 생성 버튼이 나타납니다. 기록을 새 생성으로 표시하지 않습니다.')
            else:
                gr.Markdown('**실제 모델 실행 · 512×512 · DDIM · 가중치 고정**. seed를 유지하고 한 조건씩 바꾸세요.')
                prompt=gr.Textbox(value=DEFAULT_PROMPT,label='positive prompt · 만들고 싶은 장면')
                negative=gr.Textbox(value='',label='negative prompt · CFG 기준 조건 (제거 보장 없음)')
                with gr.Row():
                    seed=gr.Number(value=42,precision=0,label='seed')
                    guidance=gr.Slider(1,12,value=7.5,step=.5,label='guidance scale · 1이면 negative 미사용')
                    steps=gr.Slider(2,30,value=20,step=1,label='sampling steps')
                generate=gr.Button('이 설정으로 실제 생성',variant='primary')
                image=gr.Image(label='지금 생성한 최종 이미지')
                frames=gr.Gallery(label='noisy latent 복원 과정 · x₀ 예측 아님',columns=5,height=260)
                measured=gr.JSON(label='실측 설정·시간');save=gr.File(label='설정 JSON 내려받기')
                generate.click(partial(live_sampling_view,pipe=pipe),[prompt,negative,seed,guidance,steps],
                               [image,frames,measured,save],api_name='sampling_live',concurrency_limit=1,concurrency_id='sd')
        with gr.Tab('3 · W06B 연결: CFG와 이전 기록'):
            gr.Markdown('2차원 수계산과 지난 차시 SD 기록을 다시 확인합니다. 수학식의 s=0과 pipeline API의 guidance≤1 분기를 구별하세요.')
            scale=gr.Slider(0,12,value=2,step=.5,label='수학식 s');calc=gr.Button('CFG 벡터 계산')
            plot=gr.Plot();numbers=gr.JSON(label='수계산 결과')
            calc.click(cfg_view,scale,[plot,numbers],api_name='cfg_vector')
            old=gr.Dropdown(['cfg-1','cfg-4','cfg-7.5','cfg-12','blue','seed-43'],value='cfg-7.5',label='지난 차시 기록')
            old_button=gr.Button('지난 기록 보기');old_image=gr.Image();old_settings=gr.JSON()
            old_button.click(recorded_view,old,[old_image,old_settings],api_name='previous_record')
        with gr.Tab('4 · 나의 실험 해석'):
            gr.Markdown('B4 활동에서 만든 함수가 있다면 summary_fn으로 연결하세요. 설정·관찰을 작성하고 원인을 단정하지 않는 요약을 만듭니다.')
            setting=gr.Textbox(label='바꾼 조건과 고정한 조건');observation=gr.Textbox(label='그림에서 실제로 본 변화')
            summarize=gr.Button('나의 요약 함수 실행');summary=gr.Textbox(label='실험 해석')
            callback=summary_fn if summary_fn is not None else lambda s,o: 'B4에서 작성한 summary_fn을 연결한 뒤 실행하세요.'
            summarize.click(callback,[setting,observation],summary,api_name='student_summary')
        gr.Markdown('공유 URL은 현재 Colab 런타임 동안 유지됩니다. 이미지 한 장의 차이를 일반적인 성능 향상으로 해석하지 마세요.')
    return app
