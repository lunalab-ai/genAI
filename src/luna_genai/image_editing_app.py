"""W11A cumulative app: real tiny denoising + explicit full-SD record analysis."""
from __future__ import annotations
import csv
import json
import tempfile
from pathlib import Path
from typing import Callable
import numpy as np
from PIL import Image
import pandas as pd
import torch
from .image_editing import tiny_edit, load_edit_records, region_mse


def tiny_view(model, clean: torch.Tensor, strength: float = .5, reverse: bool = False) -> tuple:
    """Actual sampling callback. B1x28x28 -> PIL image, trace table, settings dict."""
    mask = torch.zeros_like(clean); mask[:, :, 8:23, 8:23] = 1
    if reverse:
        mask = 1 - mask
    result = tiny_edit(model, clean, strength=strength, mask=mask, steps=20)
    image = np.rint((result['images'][0, 0].clamp(-1, 1).numpy() + 1) * 127.5).astype('uint8')
    metrics = region_mse(clean[0].permute(1, 2, 0).numpy(), result['images'][0].permute(1, 2, 0).numpy(), mask[0, 0].numpy())
    settings = {k: result[k] for k in ('mode', 'strength', 'seed', 'executed_steps')}
    settings.update(region_mse=metrics, reverse=bool(reverse), units='tiny image [-1,1]')
    return Image.fromarray(image).resize((280, 280)), pd.DataFrame(result['trace']), settings


def record_view(control: bool = True, reference: bool = True) -> tuple:
    """Select a measured 2x2 cell. Booleans do NOT generate a new full SD image."""
    record = load_edit_records()
    name = f'control{int(bool(control))}-ip{int(bool(reference))}'
    row = next(x for x in record['manifest']['records'] if x['id'] == name)
    return Image.open(record['directory'] / row['file']).copy(), {
        'mode': 'actual SD CPU execution record analysis; no current SD generation',
        'record': row, 'base': record['manifest']['base'], 'size': record['manifest']['size'],
        'limitations': record['manifest']['limitations']}


def observation_note(prediction: str, observed: str, held_fixed: str, limitation: str) -> str:
    """Default four-field note; students replace this callback in the last activity."""
    return f'예측: {prediction}\n관측: {observed}\n통제: {held_fixed}\n한계와 다음 실험: {limitation}'


def export_observation(payload: dict, note: str) -> tuple[str, str]:
    """Create new JSON+CSV local downloads; no external upload or overwrite."""
    directory = Path(tempfile.mkdtemp(prefix='luna-genai-w11a-'))
    obj = {'experiment': payload, 'learner_note': note}
    (directory/'experiment.json').write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    with (directory/'experiment.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle); writer.writerow(['field', 'value'])
        writer.writerows([['experiment', json.dumps(payload, ensure_ascii=False)], ['learner_note', note]])
    return str(directory/'experiment.json'), str(directory/'experiment.csv')


def build_edit_app(model, clean: torch.Tensor, *, note_fn: Callable = observation_note):
    """Build a real callback graph; launching is explicit and no SD is downloaded."""
    import gradio as gr
    initial = tiny_view(model, clean)
    full = record_view()
    with gr.Blocks(title='W11A 이미지 편집 원리', analytics_enabled=False) as app:
        gr.Markdown('# W11A · 학습한 모델에서 편집 조건으로\n7장의 학습은 가중치를 바꿉니다. 아래 8장 버튼은 준비된 가중치와 실행 기록을 사용합니다.')
        with gr.Tab('현재 소형 모델 실행'):
            strength = gr.Slider(0, 1, value=.5, step=.05, label='strength · 시작 noise/실행 구간')
            reverse = gr.Checkbox(label='마스크 반전')
            run = gr.Button('실제 소형 denoising 실행', variant='primary')
            image = gr.Image(value=initial[0], label='현재 CPU 실행 · SD 아님')
            table = gr.Dataframe(value=initial[1], label='t · 예측 noise RMS · 다음 상태 RMS', interactive=False)
            tiny_payload = gr.JSON(value=initial[2], label='현재 실행 조건·영역별 MSE')
            run.click(lambda s, r: tiny_view(model, clean, s, r), [strength, reverse], [image, table, tiny_payload], api_name='tiny_edit')
        with gr.Tab('구조 × 참조 · 실제 SD 기록'):
            control = gr.Checkbox(value=True, label='Canny ControlNet .8')
            reference = gr.Checkbox(value=True, label='IP-Adapter .6')
            show = gr.Button('저장된 2×2 실험 셀 보기')
            sd_image = gr.Image(value=full[0], label='실제 SD1.5 사전 실행 기록 · 현재 생성 아님')
            payload = gr.JSON(value=full[1], label='모델·seed·설정·실측 시간')
            show.click(record_view, [control, reference], [sd_image, payload], api_name='record_cell')
        with gr.Tab('내 설명 함수와 다운로드'):
            prediction = gr.Textbox(label='예측'); observed = gr.Textbox(label='관측')
            fixed = gr.Textbox(label='고정한 조건'); limitation = gr.Textbox(label='한계·다음 실험')
            explain = gr.Button('내 callback 실행'); note = gr.Textbox(label='내 설명', lines=5)
            explain.click(note_fn, [prediction, observed, fixed, limitation], note, api_name='learner_note')
            export = gr.Button('현재 SD 기록 설정과 내 설명 저장')
            js = gr.File(label='JSON'); cs = gr.File(label='CSV')
            export.click(export_observation, [payload, note], [js, cs], api_name='export')
        gr.Markdown('[7장 학습·LoRA 비교로 돌아가기](https://colab.research.google.com/github/lunalab-ai/genAI/blob/2026-fall-w11a/notebooks/student/w10b_diffusion_adaptation.ipynb)\n앱 주소는 현재 런타임이 켜진 동안만 유효합니다.')
    return app
