"""RAG UI -> same prepared embeddings/index/generator, with traceable evidence."""
from __future__ import annotations
import pandas as pd
from .peft_app import observation_note


def rag_view(lab, query: str, k: int, mode: str) -> tuple[pd.DataFrame,str,str,dict]:
    """Actual RAGLab.run -> retrieval table/context/answer/full JSON.

    No retraining/download; record includes both retrieved and actually used
    document IDs. Does not decide factual correctness from an ID appearing.
    """
    r=lab.run(query,k=int(k),mode=mode)
    table=pd.DataFrame([{key:d[key] for key in ('id','score','title','text')} for d in r['retrieved']])
    return table,r['context'],r['answer'],r


def update_view(lab, text: str) -> dict:
    """Rebuild D01's index vectors; generator weights stay frozen, no fit."""
    return lab.replace_document('D01',text)


def build_rag_app(lab, note_fn=None):
    """Prepared independent RAGLab -> Gradio Blocks, no launch or new models.

    Optional three-string note_fn is the student's function. Query/update share
    one concurrency group to keep index version and used context consistent.
    """
    import gradio as gr
    note=observation_note if note_fn is None else note_fn
    if not callable(note):raise ValueError('note_fn은 함수여야 합니다.')
    with gr.Blocks(title='W10A · 검색과 생성의 증거') as app:
        gr.Markdown('# W10A · 검색 → 문맥 → 생성\n문서는 수업용 가상 규정입니다. 실제 학교 안내로 사용하지 마세요. 작은 생성기의 실패도 그대로 관찰합니다.')
        query=gr.Textbox(value='What time does the study room close on weekdays?',label='질문(영어)')
        k=gr.Slider(1,3,value=1,step=1,label='검색 k')
        mode=gr.Radio(['retrieved','none','low_rank'],value='retrieved',label='문맥 조건: 검색 / 없음 / 최하위 문서 강제')
        run=gr.Button('동일 생성기로 실행')
        table=gr.Dataframe(label='검색 ID·cosine 점수·원문');context=gr.Textbox(label='실제로 전달한 문맥')
        answer=gr.Textbox(label='실제 모델 출력');record=gr.JSON(label='실행·조건·근거 JSON')
        run.click(lambda query,k,mode:rag_view(lab,query,k,mode),[query,k,mode],[table,context,answer,record],
                  api_name='retrieve_generate',concurrency_id='rag',concurrency_limit=1)
        with gr.Accordion('문서 한 조건 변경',open=False):
            text=gr.Textbox(value='The study room closes at 20:00 on weekdays. The room is closed on Sundays.',label='D01 새 문서')
            update=gr.Button('D01 교체·재임베딩');changed=gr.JSON(label='이전/새 인덱스 버전')
            update.click(lambda text:update_view(lab,text),text,changed,api_name='update_document',concurrency_id='rag',concurrency_limit=1)
        with gr.Accordion('내 설명 함수',open=False):
            gr.Markdown('학습자 함수 연결됨' if note_fn else '기본 기록기를 사용합니다. learner_note를 작성해 연결하세요.')
            prediction=gr.Textbox(label='예측');observation=gr.Textbox(label='관찰');limitation=gr.Textbox(label='근거와 한계')
            note_button=gr.Button('설명 기록');note_output=gr.Textbox(label='나의 설명')
            note_button.click(note,[prediction,observation,limitation],note_output,api_name='note')
    return app
