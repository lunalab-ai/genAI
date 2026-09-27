"""Cumulative AE/VAE and CLIP lab UI; caller prepares actual data/models first."""
from __future__ import annotations
from functools import partial
import numpy as np
import torch
from .representation import DenseVAE,reconstruct_images,latent_means
from .semantic_search import search_view

APP_CSS='.digit img {image-rendering:pixelated; object-fit:contain;} .gradio-container {max-width:1100px!important;}'


def reconstruction_view(name: str, index: int | float, models: dict, images: torch.Tensor) -> tuple:
    """Return original/reconstruction/absolute residual uint8 images and metrics.

    name is a prepared-model key, index is an integral test row. Residual has
    fixed scale [0,1], not independent contrast enhancement. VAE decodes mu.
    Input order to bind: name,index; models/images are fixed callback context.
    """
    if name not in models or isinstance(index,bool) or not float(index).is_integer() or not 0<=index<len(images):
        raise ValueError('Choose a valid model and test image index')
    x=images[int(index):int(index)+1];model=models[name];pred=reconstruct_images(model,x)
    residual=(x-pred).abs();z=latent_means(model,x)[0].tolist()
    def picture(t): return np.rint(t[0,0].numpy()*255).astype(np.uint8)
    return picture(x),picture(pred),picture(residual),dict(pixel_mse=float((x-pred).square().mean()),
            latent=z,mode='VAE: decode mean mu' if isinstance(model,DenseVAE) else 'AE: decode encoded z')


def latent_view(name: str, z1: float, z2: float, models: dict) -> tuple[np.ndarray,str]:
    """Decode one user-selected 2D coordinate and describe sampling context.

    Returns uint8 [28,28], status. Sliders do not retrain; beta choices switch
    separately trained checkpoints. Coordinate limits are UI choices, not bounds
    on the Gaussian distribution or guarantees of realistic output.
    """
    if name not in models or models[name].latent_dim!=2 or not np.isfinite([z1,z2]).all():
        raise ValueError('Choose a 2D model and finite coordinates')
    model=models[name];model.eval()
    with torch.inference_mode(): output=model.decode(torch.tensor([[z1,z2]],dtype=torch.float32))[0,0].numpy()
    return np.rint(output*255).astype(np.uint8),f'{name}: z=({z1:.2f}, {z2:.2f}). 직접 지정한 좌표이며 재학습하지 않습니다.'


def build_embedding_app(models: dict, images: torch.Tensor, search_lab=None):
    """Return Gradio Blocks with reconstruction, 2D latent, optional real search.

    models: loaded checkpoint mapping; images: held-out CPU images; search_lab:
    prepared CLIPSearch with index_images already run, or None for AE/VAE-only.
    Builds UI/events but does not download/train/launch. Launch via
    app.launch(css=APP_CSS,share=True) in Colab; link expires with runtime.
    The same numerical callbacks are independently callable for verification.
    """
    import gradio as gr
    from .representation import _validate_images
    _validate_images(images)
    if not models: raise ValueError('prepared models required')
    if search_lab is not None and search_lab.image_embeddings is None: raise ValueError('Index photos first')
    with gr.Blocks(title='이미지 표현 관찰실 · W05A') as app:
        gr.Markdown('# 이미지 표현 관찰실\n복원하기 · 잠재 좌표에서 만들기 · 문장으로 찾기')
        with gr.Tab('1 · AE / VAE 복원'):
            gr.Markdown('같은 테스트 이미지를 비교하세요. VAE는 평균 μ를 디코딩합니다. 모델 선택은 미리 학습한 체크포인트를 전환합니다.')
            with gr.Row():
                choice=gr.Dropdown(list(models),value='ae2',label='모델')
                index=gr.Slider(0,len(images)-1,value=0,step=1,label='테스트 이미지 번호')
            reconstruct=gr.Button('복원 비교',variant='primary')
            with gr.Row():
                original=gr.Image(label='원본',height=220,elem_classes=['digit'])
                output=gr.Image(label='복원',height=220,elem_classes=['digit'])
                residual=gr.Image(label='절대 잔차 · 고정 밝기 범위',height=220,elem_classes=['digit'])
            metrics=gr.JSON(label='실측 MSE · 잠재 좌표')
            reconstruct.click(partial(reconstruction_view,models=models,images=images),[choice,index],[original,output,residual,metrics])
        with gr.Tab('2 · 2차원 좌표 탐색'):
            gr.Markdown('VAE prior는 표준정규분포입니다. [-3,3]을 균일하게 움직이는 활동은 정규분포 샘플링과 다릅니다. AE의 좌표 척도는 다를 수 있습니다.')
            latent_model=gr.Dropdown([k for k,v in models.items() if v.latent_dim==2],value='vae1',label='2차원 모델')
            with gr.Row():
                z1=gr.Slider(-6,6,value=0,step=.1,label='z1')
                z2=gr.Slider(-6,6,value=0,step=.1,label='z2')
            decode=gr.Button('좌표 디코딩',variant='primary')
            generated=gr.Image(label='디코더 출력',height=280,elem_classes=['digit'])
            status=gr.Markdown()
            decode.click(partial(latent_view,models=models),[latent_model,z1,z2],[generated,status])
        if search_lab is not None:
            with gr.Tab('3 · CLIP 의미 검색'):
                gr.Markdown('사진 파일명으로 검색하지 않습니다. 사진 픽셀과 영어 문장을 같은 512차원 공간에서 비교합니다. 없는 대상도 top-k가 반환하므로 결과를 직접 확인하세요.')
                query=gr.Textbox(value='a cat in the snow',label='찾을 장면 · 영어 권장')
                k=gr.Slider(1,8,value=3,step=1,label='반환할 이미지 수 k')
                search=gr.Button('문장으로 사진 찾기',variant='primary')
                result=gr.Gallery(label='검색 결과 · 출처와 이용 조건은 각 캡션',columns=3,height=360)
                scores=gr.Dataframe(label='코사인 점수 · 확률이 아님',interactive=False)
                message=gr.Markdown()
                search.click(partial(search_view,lab=search_lab),[query,k],[result,scores,message])
        gr.Markdown('수업 모델은 설명용입니다. AE/VAE는 MNIST, CLIP은 별도 사전학습 모델을 사용합니다. 공유 URL은 실행 중인 런타임에서만 유지됩니다.')
    return app
