"""Observable Gradio views reusing the current W10A adapter model; no fitting."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .quantization import absmax_quantize


def observation_note(prediction: str, observation: str, limitation: str) -> str:
    """Three strings -> plain record; no correctness judgement or model call."""
    return f'예측: {prediction}\n관찰: {observation}\n한계: {limitation}'


def adapter_view(lab, text: str) -> tuple[str,str,dict]:
    """Same prompt/greedy settings, base-disabled vs current adapter outputs.

    Two actual inference calls, no model loading/training. Serialize requests
    because disable_adapter temporarily changes active routing on the SAME lab.
    """
    base=lab.generate(text,base_only=True);adapted=lab.generate(text)
    return base['text'],adapted['text'],{'input':text,'base':base,'adapted':adapted,
          'comparison':'같은 입력·greedy 설정. 가중치 갱신과 품질 평가는 별도.'}


def quantization_view(text: str, bits: int) -> tuple[pd.DataFrame,dict]:
    """Comma-separated finite numbers + bits -> toy quant table and settings.

    Does not quantize the loaded model. At most64 values, no state mutation.
    """
    values=[float(x.strip()) for x in text.split(',')]
    if not 1<=len(values)<=64: raise ValueError('숫자1..64개를 쉼표로 구분하세요.')
    r=absmax_quantize(values,int(bits))
    table=pd.DataFrame({k:r[k].tolist() for k in ('input','q','restored','error')})
    record={k:v for k,v in r.items() if not isinstance(v,np.ndarray)}
    return table,record


def build_adapter_app(lab, note_fn=None):
    """Prepared AdapterLab + optional 3-string->str callback -> Gradio Blocks.

    No downloads, training or launch here. note_fn=None uses the explicitly
    labeled default recorder. Caller launches, keeps its runtime alive and
    closes the server. Same-model inference requests share a concurrency group.
    """
    import gradio as gr
    note=observation_note if note_fn is None else note_fn
    if not callable(note): raise ValueError('note_fn은 함수여야 합니다.')
    with gr.Blocks(title='W10A · LoRA와 양자화 관측') as app:
        gr.Markdown('# W10A · 적은 가중치를 바꾸고 증거 확인하기\n현재 런타임의 실제 모델을 재사용합니다. 버튼은 학습하지 않습니다.')
        with gr.Tab('실제 LoRA 비교'):
            source=gr.Textbox(value='The park added a shaded walking path.',label='짧은 영어 원문')
            button=gr.Button('같은 입력으로 기본/어댑터 생성')
            base=gr.Textbox(label='어댑터 비활성 기본 모델');adapted=gr.Textbox(label='현재 어댑터 모델')
            record=gr.JSON(label='모델·설정·관찰 JSON')
            button.click(lambda text:adapter_view(lab,text),[source],[base,adapted,record],api_name='compare',concurrency_id='adapter',concurrency_limit=1)
        with gr.Tab('교육용 균등 격자'):
            gr.Markdown('이 탭은 작은 수치 배열 실험입니다. 실제 모델 NF4 양자화와 구별하세요.')
            values=gr.Textbox(value='-3.2,-0.2,0,0.7,3.5',label='수치 배열')
            bits=gr.Radio([4,8],value=4,label='비트 수');quant=gr.Button('정수·복원·오차 계산')
            table=gr.Dataframe(label='중간 계산');qrecord=gr.JSON(label='step·오차·실행 모드')
            quant.click(quantization_view,[values,bits],[table,qrecord],api_name='quantize')
        with gr.Tab('내 설명 함수'):
            gr.Markdown('학습자 함수 연결됨' if note_fn else '기본 기록기를 사용 중입니다. notebook의 learner_note를 작성해 다시 앱을 만드세요.')
            prediction=gr.Textbox(label='예측');observed=gr.Textbox(label='실제 관찰');limit=gr.Textbox(label='근거와 한계')
            save=gr.Button('설명 함수 실행');output=gr.Textbox(label='설명 기록')
            save.click(note,[prediction,observed,limit],output,api_name='note')
    return app
