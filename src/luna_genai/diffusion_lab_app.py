"""W06A diffusion-only browser experiments, without a CLIP model download."""
from __future__ import annotations
from functools import partial
import numpy as np
import torch
from .diffusion import image_grid
from .diffusion_lab import make_noise_schedule, mix_at, prediction_targets, recover_clean, sample_with_sampler

APP_CSS='.gradio-container {max-width:1100px!important} .digit img {image-rendering:pixelated;object-fit:contain}'


def _integer(value, name):
    if isinstance(value,bool) or not np.isfinite(float(value)) or not float(value).is_integer():
        raise ValueError(name+' must be an integer')
    return int(value)


def noise_view(index, t, seed, schedule, scale, *, images: torch.Tensor) -> tuple:
    """Return original/noisy uint8 images and coefficients for a button click.

    images: prepared CPU float32 [N,1,28,28] in [-1,1]; no downloads.
    index,t,seed integral; schedule linear/cosine; scale .1..1.
    Reuses the same Gaussian draw for the same seed, not a Markov path.
    Only display values are clipped. No model/parameter mutation.
    Example: noise_view(0,49,42,'cosine',1.,images=test).
    """
    index,t,seed=(_integer(v,n) for v,n in zip((index,t,seed),('index','t','seed')))
    if not 0<=index<len(images): raise ValueError('image index outside range')
    if not .1<=float(scale)<=1: raise ValueError('scale must lie in [.1,1]')
    clean=images[index:index+1]
    noise=torch.randn(clean.shape,generator=torch.Generator().manual_seed(seed))
    scheduler=make_noise_schedule(schedule)
    noisy=mix_at(clean,noise,t,scheduler,scale=float(scale))
    a=float(scheduler.alphas_cumprod[t])
    def picture(x): return np.rint((x[0,0].numpy().clip(-1,1)+1)*127.5).astype(np.uint8)
    return picture(clean*float(scale)),picture(noisy),dict(t=t,schedule=schedule,scale=float(scale),
        signal_coefficient=a**.5,noise_coefficient=(1-a)**.5,unit_variance_snr=a/(1-a),
        actual_signal_variance=float((clean*float(scale)).var(unbiased=False)),
        noisy_min=float(noisy.min()),noisy_max=float(noisy.max()),
        interpretation='전방 과정: 학습된 모델의 설정을 변경한 결과가 아닙니다.')


def target_view(index, t, seed, kind, *, images: torch.Tensor) -> tuple:
    """Plot the known target and algebraic reconstruction, NOT learned quality.

    index,t,seed integral; kind epsilon/sample/v_prediction. Uses fixed cosine
    training schedule. Returns matplotlib Figure and float reconstruction error.
    All target fields share one symmetric scale; image fields use [-1,1].
    Example: target_view(0,49,42,'v_prediction',images=test).
    """
    import matplotlib.pyplot as plt
    index,t,seed=(_integer(v,n) for v,n in zip((index,t,seed),('index','t','seed')))
    if not 0<=index<len(images) or not 0<=t<100: raise ValueError('index or t outside range')
    clean=images[index:index+1];scheduler=make_noise_schedule()
    noise=torch.randn(clean.shape,generator=torch.Generator().manual_seed(seed))
    a=float(scheduler.alphas_cumprod[t]); targets=prediction_targets(clean,noise,a)
    if kind not in targets: raise ValueError('Unknown prediction type')
    noisy=mix_at(clean,noise,t,scheduler)
    reconstructed=recover_clean(noisy,targets[kind],a,kind)
    fig,axes=plt.subplots(1,3,figsize=(9,3))
    limit=max(float(x.abs().max()) for x in targets.values())
    axes[0].imshow(noisy[0,0],cmap='gray',vmin=-1,vmax=1);axes[0].set_title('Noisy input')
    axes[1].imshow(targets[kind][0,0],cmap='coolwarm',vmin=-limit,vmax=limit);axes[1].set_title('Known target: '+kind)
    axes[2].imshow(reconstructed[0,0],cmap='gray',vmin=-1,vmax=1);axes[2].set_title('Algebraic reconstruction')
    for ax in axes:ax.axis('off')
    fig.tight_layout()
    return fig,dict(max_abs_error=float((reconstructed-clean).abs().max()),
                    meaning='정답 target을 이미 알고 계산한 검산입니다. 학습된 모델의 복원 성능이 아닙니다.')


def generation_view(name, setting, seed, *, models: dict) -> tuple:
    """Generate eight images with prepared models; return grid, trajectory, facts.

    name is a key of models, setting DDPM100/DDIM20/DDIM50/DDIM100; seed integral.
    Preserves parameters; no training, download, clean image or text prompt.
    Example: generation_view('trained','DDIM20',42,models=models).
    """
    import matplotlib.pyplot as plt
    settings={'DDPM100':('DDPM',100),'DDIM20':('DDIM',20),'DDIM50':('DDIM',50),'DDIM100':('DDIM',100)}
    if name not in models or setting not in settings: raise ValueError('Choose a prepared model and sampler')
    method,steps=settings[setting];seed=_integer(seed,'seed')
    result=sample_with_sampler(models[name],method=method,steps=steps,count=8,seed=seed)
    grid=image_grid(result['images'],title=f'{name} / {setting} / seed {seed}')
    fig,axes=plt.subplots(2,len(result['frames']),figsize=(10,4),squeeze=False)
    for j,frame in enumerate(result['frames']):
        for row,key in enumerate(('input','predicted_clean')):
            axes[row,j].imshow(frame[key][0,0],cmap='gray',vmin=-1,vmax=1)
            axes[row,j].set_title(f'{key}\nt={frame["t"]}');axes[row,j].axis('off')
    fig.tight_layout()
    return grid,fig,dict(model=name,method=method,steps=steps,seed=seed,seconds=round(result['seconds'],3),
                         image_mean=float(result['images'].mean()),image_std=float(result['images'].std()),
                         interpretation='픽셀 평균·표준편차는 상태 확인값이며 생성 품질 점수가 아닙니다.')


def build_diffusion_lab(images: torch.Tensor, models: dict | None = None):
    """Build (not launch) Gradio Blocks from prepared clean images and models.

    images: CPU [N,1,28,28] [-1,1]. models=None gives A's two-tab calculator;
    prepared initial/early/trained models add B's generation tab. No CLIP load.
    Each button maps input widgets -> numerical callback -> output widgets.
    Returns Blocks; caller launches and owns server lifetime.
    Example: app=build_diffusion_lab(test,models); app.launch(share=True).
    """
    import gradio as gr
    if len(images)<1: raise ValueError('Provide at least one clean image')
    with gr.Blocks(title='W06A · 확산 모델 관찰실') as app:
        gr.Markdown('# 확산 모델 관찰실\n예상 → 조건 하나 변경 → 실행 → 관찰 → 이유 설명. 같은 seed로 비교하세요.')
        with gr.Tab('1 · 노이즈와 스케줄'):
            gr.Markdown('전방 손상만 관찰합니다. 스케줄 변경이 기존 생성 모델을 재학습하는 것은 아닙니다.')
            with gr.Row():
                index=gr.Slider(0,len(images)-1,value=0,step=1,label='이미지 번호')
                t=gr.Slider(0,99,value=49,step=1,label='코드 시점 t')
                seed=gr.Number(value=42,precision=0,label='잡음 seed')
            with gr.Row():
                schedule=gr.Dropdown(['cosine','linear'],value='cosine',label='전방 스케줄')
                scale=gr.Slider(.1,1,value=1,step=.1,label='원본 입력 척도')
            button=gr.Button('노이즈 관찰',variant='primary')
            with gr.Row():
                original=gr.Image(label='척도 적용 원본',height=250,elem_classes=['digit'])
                noisy=gr.Image(label='노이즈가 섞인 이미지',height=250,elem_classes=['digit'])
            facts=gr.JSON(label='계수와 자르기 전 범위')
            button.click(partial(noise_view,images=images),[index,t,seed,schedule,scale],[original,noisy,facts],api_name='noise')
        with gr.Tab('2 · 무엇을 정답으로 예측하나'):
            gr.Markdown('직접 뽑은 정답 잡음을 사용한 대수적 검산입니다. 실제 신경망의 예측과 구별하세요.')
            with gr.Row():
                ti=gr.Slider(0,len(images)-1,value=0,step=1,label='목표 비교 이미지')
                tt=gr.Slider(0,99,value=49,step=1,label='목표 비교 시점')
                ts=gr.Number(value=42,precision=0,label='목표 비교 seed')
                kind=gr.Dropdown(['epsilon','sample','v_prediction'],value='v_prediction',label='정답의 종류')
            target_button=gr.Button('정답과 역변환 관찰',variant='primary')
            target_plot=gr.Plot(label='입력 · 목표 · 검산 결과');target_facts=gr.JSON(label='수치 오차')
            target_button.click(partial(target_view,images=images),[ti,tt,ts,kind],[target_plot,target_facts],api_name='target')
        if models is not None:
            with gr.Tab('3 · 생성과 결과 해석'):
                gr.Markdown('initial → trained를 같은 seed로 비교하세요. 이어서 DDPM100 → DDIM20을 비교하세요. 모든 설정은 학습된 cosine betas와 epsilon 목표를 유지합니다.')
                with gr.Row():
                    name=gr.Dropdown(list(models),value='trained',label='학습 상태')
                    setting=gr.Dropdown(['DDPM100','DDIM20','DDIM50','DDIM100'],value='DDIM20',label='추론 방법과 호출 횟수')
                    gs=gr.Number(value=42,precision=0,label='생성 seed')
                generate=gr.Button('새 이미지 8장 생성',variant='primary')
                output=gr.Plot(label='생성 이미지');trajectory=gr.Plot(label='현재 상태와 원본 추정');stats=gr.JSON(label='실행 조건과 소요 시간')
                generate.click(partial(generation_view,models=models),[name,setting,gs],[output,trajectory,stats],api_name='generate')
        gr.Markdown('공유 URL은 이 런타임이 실행되는 동안만 유지됩니다. 결과에서 잘 된 사례와 모호한 사례를 함께 기록하세요.')
    return app


if __name__=='__main__':
    from .autoencoder import load_mnist
    from .diffusion import load_diffusion_checkpoints
    torch.set_num_threads(2)
    data=load_mnist();models,_=load_diffusion_checkpoints()
    build_diffusion_lab(data.test[:32]*2-1,models).launch(css=APP_CSS)
