"""W05B cumulative browser lab; UI callbacks reuse the tested numerical core."""
from __future__ import annotations
from functools import partial
import numpy as np
import pandas as pd
import torch
from .diffusion import forward_noise,make_scheduler,sample_diffusion,image_grid
from .clip_teaching import compare_search

APP_CSS='.gradio-container {max-width:1150px!important} .digit img {image-rendering:pixelated;object-fit:contain;}'


def noise_view(index:int|float,t:int|float,seed:int|float,images:torch.Tensor)->tuple:
    """Return original/noisy uint8 images and coefficient/range measurements.

    images: prepared clean [-1,1] CPU [N,1,28,28]. UI inputs index,t,seed must
    be integers. A fixed seed uses identical epsilon when t changes: marginal
    comparisons, not a single Markov trajectory. Display alone clips [-1,1].
    No training or downloads. Returns (original,noisy,dict).
    """
    if any(isinstance(v,bool) or not float(v).is_integer() for v in (index,t,seed)):
        raise ValueError('index, t, seed must be integers')
    index,t,seed=int(index),int(t),int(seed)
    if not 0<=index<len(images) or not 0<=t<100: raise ValueError('index or t outside range')
    x=images[index:index+1]
    noise=torch.randn(x.shape,generator=torch.Generator().manual_seed(seed))
    noisy=forward_noise(x,torch.tensor([t]),noise)
    alpha=float(make_scheduler().alphas_cumprod[t])
    def picture(a):return np.rint((a[0,0].numpy().clip(-1,1)+1)*127.5).astype(np.uint8)
    return picture(x),picture(noisy),dict(code_t=t,signal_coefficient=alpha**.5,
            noise_coefficient=(1-alpha)**.5,unclipped_min=float(noisy.min()),
            unclipped_max=float(noisy.max()),display='[-1,1] clipping only for viewing')


def generation_view(name:str,seed:int|float,models:dict)->tuple:
    """Run 100 reverse steps for eight images and return grid/trajectory/status.

    name: initial/early/trained; seed integral. Uses prepared models; switching
    models does not train. Trajectory shows sample 0's x_t and predicted x0.
    Returns matplotlib figures and Markdown. Changing seed changes the complete
    random sequence; fixed seed permits paired checkpoint comparisons.
    """
    import matplotlib.pyplot as plt
    if name not in models or isinstance(seed,bool) or not float(seed).is_integer():
        raise ValueError('Choose a prepared model and integer seed')
    result=sample_diffusion(models[name],count=8,seed=int(seed))
    grid=image_grid(result['images'],title=f'{name}: seed {int(seed)}, 100 reverse updates')
    fig,axes=plt.subplots(2,len(result['frames']),figsize=(10,4))
    for j,frame in enumerate(result['frames']):
        for row,key in enumerate(('input','predicted_clean')):
            axes[row,j].imshow(frame[key][0,0],cmap='gray',vmin=-1,vmax=1)
            axes[row,j].set_title(f'{key}\nt={frame["t"]}');axes[row,j].axis('off')
    fig.tight_layout()
    return grid,fig,f'{name} · seed {int(seed)} · 실제 역방향 갱신 100회. 위 행은 현재 상태, 아래 행은 원본 추정입니다. 깨끗한 원본 이미지는 입력하지 않았습니다.'


def build_study_app(*,search_lab=None,models:dict|None=None,images:torch.Tensor|None=None):
    """Build Gradio Blocks for prepared CLIP and/or diffusion, without launch.

    A supplies indexed search_lab only. B supplies models and clean test images;
    pass all three for the cumulative lab. Explicitly absent components are not
    fallback results. launch(share=True,allowed_paths=gallery.path.tolist()) in
    Colab; allow only the checked photo files. Share URL expires with runtime.
    """
    import gradio as gr
    if search_lab is None and models is None: raise ValueError('Provide prepared models')
    if search_lab is not None and search_lab.image_embeddings is None: raise ValueError('Index images first')
    if models is not None and images is None: raise ValueError('Diffusion requires clean images')
    with gr.Blocks(title='W05B · 검색과 확산 관찰실') as app:
        gr.Markdown('# 검색과 확산 관찰실\n한 번에 조건 하나를 바꾸고, 예상한 변화와 실제 결과를 비교하세요.')
        if search_lab is not None:
            with gr.Tab('1 · 문장 두 개로 검색'):
                gr.Markdown('대상은 고정하고 속성을 바꿔 보세요. 없는 대상도 순위는 반환됩니다. 사진의 출처·이용 조건은 캡션에 있습니다.')
                with gr.Row():
                    qa=gr.Textbox(value='a cat in the snow',label='문장 A')
                    qb=gr.Textbox(value='a cat indoors',label='문장 B')
                k=gr.Slider(1,8,value=3,step=1,label='각 문장의 결과 수 k')
                button=gr.Button('두 검색 결과 비교',variant='primary')
                with gr.Row():
                    ga=gr.Gallery(label='문장 A 결과',columns=3,height=330)
                    gb=gr.Gallery(label='문장 B 결과',columns=3,height=330)
                table=gr.Dataframe(label='실제 코사인 점수',interactive=False);message=gr.Markdown()
                button.click(partial(compare_search,lab=search_lab),[qa,qb,k],[ga,gb,table,message],api_name='compare_search')
        if models is not None:
            with gr.Tab('2 · 노이즈 관찰'):
                gr.Markdown('같은 seed에서 t만 바꾸면 같은 기본 잡음을 사용합니다. 이는 시점별 주변분포 비교입니다. 코드 t=0에도 소량의 잡음이 있습니다.')
                with gr.Row():
                    index=gr.Slider(0,len(images)-1,value=0,step=1,label='테스트 이미지 번호')
                    t=gr.Slider(0,99,value=49,step=1,label='코드 시점 t')
                    seed=gr.Number(value=42,precision=0,label='잡음 seed')
                add=gr.Button('노이즈 관찰',variant='primary')
                with gr.Row():
                    original=gr.Image(label='깨끗한 이미지',height=260,elem_classes=['digit'])
                    noisy=gr.Image(label='노이즈가 섞인 이미지',height=260,elem_classes=['digit'])
                stats=gr.JSON(label='계수와 자르기 전 실제 범위')
                add.click(partial(noise_view,images=images),[index,t,seed],[original,noisy,stats],api_name='noise')
            with gr.Tab('3 · 학습 전후와 역확산'):
                gr.Markdown('먼저 initial을 실행한 뒤 seed를 유지하고 trained로 바꿔 보세요. 체크포인트 선택은 재학습이 아닙니다. CPU에서 100회 갱신하므로 완료될 때까지 기다립니다.')
                with gr.Row():
                    choice=gr.Dropdown(list(models),value='trained',label='학습 상태')
                    gen_seed=gr.Number(value=42,precision=0,label='생성 seed')
                generate=gr.Button('노이즈에서 8장 생성',variant='primary')
                output=gr.Plot(label='실제 생성 결과');trajectory=gr.Plot(label='현재 상태와 원본 추정');status=gr.Markdown()
                generate.click(partial(generation_view,models=models),[choice,gen_seed],[output,trajectory,status],api_name='generate')
        gr.Markdown('CLIP 검색과 무조건부 확산 생성은 별개의 실험입니다. 이 앱은 문장으로 숫자를 생성하는 모델이 아닙니다.')
    return app
