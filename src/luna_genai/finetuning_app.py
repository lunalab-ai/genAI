"""W09B apps reuse a prepared lab; callbacks neither download nor train."""
from __future__ import annotations
from pathlib import Path
import json
import uuid
import pandas as pd
from .finetuning import FineTuningLab, instruction_record


def _export(record: dict) -> str:
    folder = Path('.build/finetuning-exports')
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'run-{uuid.uuid4().hex[:12]}.json'
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    return str(path.resolve())


def classification_view(lab: FineTuningLab, text: str):
    """Prepared classifier + nonempty str -> label, probability DataFrame, metadata, JSON path.

    Runs actual inference on the SAME object used in the notebook. Writes one
    local export file, without raw dataset rows or model weights. Example:
    label, table, state, path = classification_view(lab, 'The team won the final.').
    """
    prediction = lab.predict([text])[0]
    table = pd.DataFrame([{'class': k, 'probability': v} for k, v in prediction['scores'].items()])
    result = {'input': text, 'prediction': prediction, 'state': lab.state()}
    return prediction['label'], table, result['state'], _export(result)


def generation_view(lab: FineTuningLab, prompt: str, max_new_tokens: int,
                    temperature: float, seed: int):
    """Prepared causal lab and widget values -> continuation, actual settings, JSON path.

    Widget order MUST match prompt/length/temperature/seed. Reuses the model;
    does not pretend that formatting an instruction performs SFT.
    """
    result = lab.generate(prompt, max_new_tokens=int(max_new_tokens),
                          temperature=float(temperature), seed=int(seed))
    settings = {k: v for k, v in result.items() if k not in ('text', 'new_token_ids')}
    return result['text'], settings, _export(result)


def instruction_view(instruction: str, text: str, response: str):
    """Three strings -> formatted training example and explicit NOT-TRAINED status."""
    result = instruction_record(instruction, text, response)
    return result['serialized'], result['status']


def build_finetuning_app(lab: FineTuningLab, summary_fn=None):
    """Return Gradio Blocks without loading, fitting, launching, or sharing a model.

    lab: already prepared FineTuningLab. Optional summary_fn(prediction:str,
    observation:str, explanation:str)->str is the learner's own synthesis.
    Model callbacks are serialized. Notebook and UI use one shared model state.
    Example: app=build_finetuning_app(lab); app.launch(share=True) in Colab.
    Share URLs live only while their runtime remains running.
    """
    import gradio as gr
    with gr.Blocks(title='W09B · 언어 모델 학습 관찰실') as app:
        gr.Markdown('# 언어 모델 학습 관찰실\n입력을 바꾸기 전에 예상하고, 한 조건씩 바꾼 뒤 근거를 기록하세요.')
        with gr.Tab('모델 입력과 결과'):
            gr.Markdown(f'**현재 작업:** {"뉴스 분류" if lab.kind == "classification" else "뉴스 이어쓰기"} · '
                        f'**실제 학습 단계:** {lab.steps} · **장치:** {lab.device}')
            if lab.kind == 'classification':
                text = gr.Textbox(label='뉴스 문장 · 개인정보 없는 영어 예문', value='The team won the final game.')
                run = gr.Button('분류하고 확률 보기', variant='primary')
                label = gr.Textbox(label='예측 클래스')
                probabilities = gr.Dataframe(headers=['class', 'probability'], interactive=False)
                state = gr.JSON(label='모델과 비교 조건')
                file = gr.File(label='이번 관찰 JSON')
                run.click(lambda text: classification_view(lab, text), [text],
                          [label, probabilities, state, file], concurrency_limit=1)
            else:
                prompt = gr.Textbox(label='시작 문장 · 영어', value='The company announced')
                length = gr.Slider(8, 48, value=24, step=1, label='새 토큰 수')
                temperature = gr.Slider(0, 1.2, value=0, step=.1, label='Temperature · 0은 greedy')
                seed = gr.Number(value=42, precision=0, label='샘플링 seed')
                run = gr.Button('이어쓰기 실행', variant='primary')
                continuation = gr.Textbox(label='새로 생성한 부분')
                settings = gr.JSON(label='실행한 설정과 모델 상태')
                file = gr.File(label='이번 관찰 JSON')
                run.click(lambda p, n, t, s: generation_view(lab, p, n, t, s),
                          [prompt, length, temperature, seed], [continuation, settings, file], concurrency_limit=1)
        with gr.Tab('지시 데이터 설계'):
            gr.Markdown('지시문·입력·응답을 구성하는 활동입니다. 이 버튼은 가중치를 학습시키지 않습니다.')
            instruction = gr.Textbox(label='지시문', value='Rewrite the sentence as a short headline.')
            source = gr.Textbox(label='입력', value='A company opened a new office in the city.')
            response = gr.Textbox(label='내가 작성한 목표 응답', value='Company Opens New City Office')
            inspect = gr.Button('학습 예제 구조 보기')
            serialized = gr.Textbox(label='직렬화한 예제', lines=8)
            status = gr.Textbox(label='지금 수행한 작업')
            inspect.click(instruction_view, [instruction, source, response], [serialized, status])
        with gr.Tab('나의 실험 설명'):
            predicted = gr.Textbox(label='실행 전 예상')
            observed = gr.Textbox(label='실행 후 관찰 · 숫자와 조건')
            reason = gr.Textbox(label='설명과 한계')
            summarize = gr.Button('나의 함수로 정리')
            note = gr.Textbox(label='실험 설명', lines=5)
            if summary_fn is None:
                def summary_fn(p, o, r):
                    return f'예상: {p}\n관찰: {o}\n설명과 한계: {r}'
            summarize.click(summary_fn, [predicted, observed, reason], [note])
        gr.Markdown('출력은 사실성·정답의 보장이 아닙니다. 공유 주소는 현재 Colab 런타임 동안 유지됩니다.')
    return app
