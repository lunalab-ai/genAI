"""W06B web laboratory: separate class generation, stored SD, and live SD."""
from __future__ import annotations
from functools import partial
import numpy as np
import torch
from PIL import Image
from .conditional import CLASS_NAMES,sample_classes
from .stable_diffusion import cfg_combine,load_recorded_sd,generate_sd,DEFAULT_PROMPT

APP_CSS='.gradio-container {max-width:1100px!important} .fashion img {image-rendering:pixelated;object-fit:contain}'

def _integer(value,name):
    number=float(value)
    if not np.isfinite(number) or not number.is_integer() or number<0 or number>=2**32:raise ValueError(name+' must be a nonnegative integer below 2^32')
    return int(number)

def class_view(class_id,seed,*,model):
    """Actual tiny-model DDIM30 generation, four independent draws; no training.

    Inputs class 0..9 and integer seed. Returns uint8 2x2 image and settings.
    Changing class with same seed reuses the same four initial noise draws.
    """
    class_id=_integer(class_id,'class');seed=_integer(seed,'seed')
    result=sample_classes(model,[class_id]*4,seed=seed,steps=30,matched_noise=False)
    pictures=np.rint((result['images'][:,0].numpy().clip(-1,1)+1)*127.5).astype(np.uint8)
    grid=np.concatenate([np.concatenate(pictures[:2],axis=1),np.concatenate(pictures[2:],axis=1)],axis=0)
    return grid,dict(mode='소형 모델 실제 생성 · 학습 없음',class_id=class_id,class_name=CLASS_NAMES[class_id],seed=seed,steps=30,scheduler='DDIM')

def recorded_view(key):
    """Read a hash-verified previously generated SD image and exact settings.

    key: cfg-1/cfg-4/cfg-7.5/cfg-12/blue/seed-43. Does not run a model.
    Returns PIL image and record explicitly labelled as playback.
    """
    _,gallery=load_recorded_sd();matches=[x for x in gallery if x['key']==key]
    if not matches:raise ValueError('Choose a prepared record')
    record=matches[0];image=Image.open(record['image_path']).convert('RGB')
    return image,{**{k:v for k,v in record.items() if k!='image_path'},'mode':'사전 실행 결과 재생 · 지금 SD를 실행하지 않음'}

def cfg_view(scale):
    """Illustrative 2D epsilon vectors, not measured SD attention or image quality.

    Returns a vector plot and the algebra result for u=(.2,-.1), c=(.5,.1).
    """
    import matplotlib.pyplot as plt
    scale=float(scale);u=torch.tensor([.2,-.1]);c=torch.tensor([.5,.1]);guided=cfg_combine(u,c,scale)
    fig,ax=plt.subplots(figsize=(5,3.5))
    for name,vector,color in [('u',u,'#718096'),('c',c,'#168aad'),('u+s(c-u)',guided,'#e76f51')]:
        x,y=vector.tolist();ax.annotate('',xy=(x,y),xytext=(0,0),arrowprops={'arrowstyle':'->','color':color,'lw':2});ax.text(x,y,name,color=color)
    ax.set_xlim(-.2,max(.8,float(guided[0])+.5));ax.set_ylim(-.3,max(.4,float(guided[1])+.4))
    ax.axhline(0,color='#cccccc');ax.axvline(0,color='#cccccc');ax.grid(alpha=.2);ax.set_title('Illustrative noise-prediction vectors');fig.tight_layout()
    return fig,dict(scale=scale,unconditional=u.tolist(),conditional=c.tolist(),guided=guided.tolist(),
                    note='수계산 예시. pipeline guidance_scale<=1의 분기와 구별하세요.')

def live_view(prompt,seed,guidance,*,pipe):
    """Actual SD generation callback; forwards all widget parameters, DDIM20.

    Returns generated PIL image and measured settings. Errors remain visible;
    never replaces failed inference with recorded images.
    """
    result=generate_sd(pipe,prompt,seed=_integer(seed,'seed'),guidance=float(guidance),steps=20)
    return result['image'],result['settings']

def build_conditional_app(model,*,pipe=None):
    """Build Gradio Blocks without launch/download/training.

    model: loaded TinyClassUNet. Optional prepared pipe adds live SD tab;
    omission means recorded SD only. Caller owns server and uses .launch().
    Class generation and recorded image settings are labelled separately.
    """
    import gradio as gr
    _,gallery=load_recorded_sd()
    with gr.Blocks(title='W06B · 조건부 생성 관찰실') as app:
        gr.Markdown('# 조건부 생성 관찰실\n먼저 예상하고 조건 하나만 바꿔 보세요. 같은 seed로 비교하고 설정과 관찰을 함께 기록합니다.')
        with gr.Tab('1 · 의류 클래스 실제 생성'):
            gr.Markdown('실제로 학습한 소형 Fashion-MNIST 모델입니다. SD와 다른 모델이며 입력은 클래스 번호입니다.')
            with gr.Row():
                cls=gr.Dropdown([(f'{i} · {name}',i) for i,name in enumerate(CLASS_NAMES)],value=7,label='클래스')
                seed=gr.Number(value=42,precision=0,label='seed')
            button=gr.Button('이 클래스로 4장 생성',variant='primary');out=gr.Image(label='실제 소형 모델 생성',elem_classes='fashion');info=gr.JSON(label='설정')
            button.click(partial(class_view,model=model),[cls,seed],[out,info],api_name='class_generate')
        with gr.Tab('2 · SD 실제 기록 비교'):
            gr.Markdown('**사전 실행 결과 재생입니다. 이 탭은 GPU나 SD 모델을 실행하지 않습니다.** CFG 비교는 cfg-1/4/7.5/12, 색상 비교는 cfg-7.5/blue, seed 비교는 cfg-7.5/seed-43을 고르세요.')
            with gr.Row():
                left=gr.Dropdown([x['key'] for x in gallery],value='cfg-1',label='왼쪽 기록')
                right=gr.Dropdown([x['key'] for x in gallery],value='cfg-7.5',label='오른쪽 기록')
            show=gr.Button('두 기록 불러오기',variant='primary')
            with gr.Row():a=gr.Image(label='왼쪽 사전 생성');b=gr.Image(label='오른쪽 사전 생성')
            with gr.Row():ai=gr.JSON(label='왼쪽 설정');bi=gr.JSON(label='오른쪽 설정')
            def compare(l,r):
                li,lm=recorded_view(l);ri,rm=recorded_view(r);return li,ri,lm,rm
            show.click(compare,[left,right],[a,b,ai,bi],api_name='record_compare')
        with gr.Tab('3 · CFG 벡터 수계산'):
            gr.Markdown('작은 두 성분의 잡음 예측 예시입니다. 출력은 이미지나 품질 점수가 아닙니다.')
            scale=gr.Slider(0,12,value=4,step=.5,label='수학식의 s');calculate=gr.Button('u + s(c − u) 계산')
            plot=gr.Plot(label='방향과 크기');numbers=gr.JSON(label='계산값')
            calculate.click(cfg_view,scale,[plot,numbers],api_name='cfg_calculate')
        if pipe is not None:
            with gr.Tab('4 · SD 지금 생성'):
                gr.Markdown('**실제 SD 추론** · 512×512 · DDIM20 · 가중치는 고정합니다. GPU 종류와 상황에 따라 시간이 달라집니다.')
                prompt=gr.Textbox(value=DEFAULT_PROMPT,label='영어 프롬프트')
                with gr.Row():s=gr.Number(value=42,precision=0,label='seed');g=gr.Slider(1,12,value=7.5,step=.5,label='guidance_scale')
                run=gr.Button('SD 이미지 실제 생성',variant='primary');im=gr.Image(label='지금 생성한 이미지');settings=gr.JSON(label='측정한 실행 기록')
                run.click(partial(live_view,pipe=pipe),[prompt,s,g],[im,settings],api_name='sd_generate')
    return app
