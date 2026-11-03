"""W10B inspectable Gradio callbacks: live small model vs actual saved SD records."""
from __future__ import annotations
import datetime as dt
import json
from pathlib import Path
import tempfile
from typing import Callable
import numpy as np
import pandas as pd
from PIL import Image
import torch
from .diffusion_adaptation import ASSETS, inpainting_channels, load_actual_sd_record, sdxl_condition_shapes


def observation_note(prediction: str, observation: str, controlled: str, limitation: str) -> str:
    """Format four learner-authored observations; no automatic correctness grade."""
    return f'예측: {prediction}\n관측: {observation}\n통제 조건: {controlled}\n근거와 한계: {limitation}'


def tiny_view(comparison: dict, case: str, seed: int = 42) -> tuple:
    """Read a previously fitted SAME lab and actually sample; never retrain/download."""
    if case not in comparison['cases']:raise ValueError('Unknown trained comparison case')
    if float(seed)!=int(seed):raise ValueError('Seed must be an integer')
    lab=comparison['cases'][case]
    result=lab.sample(seed=int(seed),count=2,steps=20)
    gallery=[]
    for i,raw in enumerate(result['images']):
        pixels=((raw[0].clamp(-1,1).numpy()+1)*127.5).round().astype(np.uint8)
        gallery.append((Image.fromarray(pixels).resize((224,224),Image.Resampling.NEAREST),f'{case} · 실제 소형 모델 표본{i+1}'))
    payload={'mode':'live_small_conditional_denoiser, not Stable Diffusion','case':case,
             'state':lab.state(),'sampling_seed':int(seed),'sampling_steps':20,
             'common_evaluation':comparison['table'].to_dict('records'),
             'quality':'noise MSE and parameter updates do not establish subject fidelity'}
    return comparison['table'].copy(),gallery,payload


def recorded_view(record: dict, comparison_id: str) -> tuple:
    """Display only an EXACT observed prompt/seed pair; no live SD execution."""
    rows=[r for r in record['comparisons'] if r['id']==comparison_id]
    if len(rows)!=1:raise ValueError('No saved run for this ID; select an existing record')
    row=rows[0]
    gallery=[]
    for phase,label in [('before','적응 전'),('after','LoRA 학습 후')]:
        if row.get(phase+'_status')=='blocked':
            continue
        with Image.open(ASSETS/row[phase]) as im:image=im.convert('RGB').copy()
        gallery.append((image,f'{label} · {row[phase+"_settings"]["resolution"]}px · seed{row["seed"]} · 전후 상태 {row.get("before_status")}/{row.get("after_status")}'))
    payload={'mode':'record analysis only; SD is NOT generated in this callback',
        'actual_source_run_at':record['executed_at'],'comparison':row,
        'base_unchanged_by_training':record['training']['base_unchanged'],
        'reload':record['reload'],'sdxl_training_executed':record['sdxl_training_executed']}
    return gallery,pd.DataFrame(record['training']['history']),payload


def channel_view(edit_size: int = 3, reverse: bool = False) -> tuple:
    """Assemble a tiny9-channel condition and show the mask; no inpainting model."""
    if float(edit_size)!=int(edit_size) or not 1<=int(edit_size)<=6:raise ValueError('Edit size must be an integer1..6')
    size=int(edit_size);mask=torch.zeros(1,1,8,8);mask[:,:,1:1+size,1:1+size]=1
    if reverse:mask=1-mask
    joined=inpainting_channels(torch.zeros(1,4,8,8),mask,torch.zeros(1,4,8,8))
    image=Image.fromarray((mask[0,0].numpy()*255).astype(np.uint8)).resize((256,256),Image.Resampling.NEAREST)
    return image,{'mode':'channel/configuration calculation, no inpainting or SDXL training',
        'joined_shape':list(joined.shape),'white_edit_pixels':int(mask.sum()),
        'mask_convention':'white=edit, black=keep','sdxl_configuration':sdxl_condition_shapes()}


def export_observation(payload: dict, note: str) -> str:
    """Write learner's experiment JSON to a new local temporary file for download."""
    root=Path(tempfile.mkdtemp(prefix='luna-genai-w10b-'))
    path=root/'w10b-experiment.json'
    path.write_text(json.dumps({'created_at':dt.datetime.now(dt.timezone.utc).isoformat(),
                               'observation':payload,'learner_note':note},ensure_ascii=False,indent=2),encoding='utf-8')
    return str(path)


def _notes(note_fn: Callable, payload):
    import gradio as gr
    prediction=gr.Textbox(label='실행 전 예측')
    observation=gr.Textbox(label='실제 관측과 중간값')
    controlled=gr.Textbox(label='바꾼 조건과 고정한 조건')
    limitation=gr.Textbox(label='근거·한계·다음 실험')
    note=gr.Textbox(label='나의 설명',lines=5)
    button=gr.Button('내 설명 함수 연결',variant='primary')
    button.click(note_fn,[prediction,observation,controlled,limitation],note,api_name='learner_note')
    download=gr.Button('실험 JSON 만들기');file=gr.File(label='실험 기록 다운로드')
    download.click(export_observation,[payload,note],file,api_name='export_observation')


def build_tiny_app(comparison: dict, *, note_fn: Callable = observation_note):
    """Build UI around already fitted comparison objects; launch() is explicit."""
    import gradio as gr
    default='lora_prior1'
    initial=tiny_view(comparison,default)
    with gr.Blocks(title='W10B · 작은 확산 모델 적응',analytics_enabled=False) as app:
        gr.Markdown('# W10B · 학습된 작은 확산 모델을 비교하기\nCPU 실제 학습·샘플링입니다. 이 모델은 Stable Diffusion이 아닙니다. 버튼은 준비한 모델을 재사용합니다.')
        with gr.Tab('가중치 갱신과 보존'):
            with gr.Row():
                choice=gr.Dropdown(choices=list(comparison['cases']),value=default,label='비교할 학습 조건')
                seed=gr.Number(value=42,precision=0,label='샘플링 seed')
            run=gr.Button('같은 모델에서 표본 관찰',variant='primary')
            table=gr.Dataframe(value=initial[0],label='공통 조건의 평가 · MSE는 이미지 품질 점수가 아닙니다',interactive=False)
            gallery=gr.Gallery(value=initial[1],columns=2,label='실제 소형 모델 표본')
            payload=gr.JSON(value=initial[2],label='현재 실험의 조건과 근거')
            run.click(lambda case,seed:tiny_view(comparison,case,seed),[choice,seed],[table,gallery,payload],api_name='compare')
        with gr.Tab('내가 설명하고 기록하기'):_notes(note_fn,payload)
    return app


def build_record_app(record: dict | None = None, *, note_fn: Callable = observation_note):
    """Build record-analysis/condition UI; it never downloads or runs SD weights."""
    import gradio as gr
    record=load_actual_sd_record() if record is None else record
    first=record['comparisons'][0]['id'];initial=recorded_view(record,first)
    mask,channels=channel_view()
    with gr.Blocks(title='W10B · SD 기록과 새 조건',analytics_enabled=False) as app:
        gr.Markdown('# W10B · 실제 SD 실행 기록을 읽고 조건을 연결하기\n이 앱의 그림은 검증된 사전 실행 기록입니다. 여기서 SD나 SDXL을 새로 학습·생성하지 않습니다.')
        with gr.Tab('실제 SD1.5 전후 기록'):
            choice=gr.Dropdown(choices=[r['id'] for r in record['comparisons']],value=first,label='실제로 실행된 prompt·seed 조합')
            show=gr.Button('저장된 쌍 비교',variant='primary')
            gallery=gr.Gallery(value=initial[0],columns=2,label='같은 시작 조건의 실행 기록 · 차단된 항목은 그림 없음/평가 제외')
            table=gr.Dataframe(value=initial[1],label='실제 instance/prior 학습 기록',interactive=False)
            payload=gr.JSON(value=initial[2],label='기록 출처·설정·재로딩 검사')
            show.click(lambda name:recorded_view(record,name),choice,[gallery,table,payload],api_name='recorded_pair')
        with gr.Tab('인페인팅 채널과 SDXL 구성'):
            size=gr.Slider(1,6,value=3,step=1,label='마스크 한 변의 길이')
            reverse=gr.Checkbox(label='마스크 반전')
            assemble=gr.Button('채널 조립 확인')
            image=gr.Image(value=mask,label='흰색이 편집 대상인 마스크',interactive=False)
            info=gr.JSON(value=channels,label='9채널/SDXL 구성 계산 · 모델 실행 아님')
            assemble.click(channel_view,[size,reverse],[image,info],api_name='condition_channels')
        with gr.Tab('내가 설명하고 기록하기'):_notes(note_fn,payload)
    return app
