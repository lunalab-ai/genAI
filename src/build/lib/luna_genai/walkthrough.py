"""W03B: inspectable toy blocks and immutable, one-token generation state.

The toy block is not Qwen. The generation state uses the real lab supplied by
the caller. Automatic generation exercise solutions are intentionally absent.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any
import numpy as np
import pandas as pd
from .transformer import (attention, layer_norm, feed_forward, prompt_ids,
                          forward_logits, choose_token, append_token, trace_row)


def block_walkthrough(causal: bool = True) -> dict[str, np.ndarray]:
    """학습하지 않은 3위치·4특징·2헤드 post-LN 블록의 중간 배열을 반환한다.

    causal=True: 각 위치의 미래 점수를 차단. False: 전체 입력 위치 허용.
    인자 외 외부 입력/다운로드/학습/전역 상태 변경 없음. 결과 dict에는 X(3,4),
    WQ/WK/WV(4,2), Q/K/V(3,2), scores/masked/A(3,3), Z/head2(3,2),
    concat/projected/residual1/U/ffn/residual2/Y(3,4), expanded(3,6)가 있다.
    head 1은 W03A의 Q/K/V를 정확히 재현한다. head 2는 별도 투영으로 계산한다.
    WO(4,4), W1(4,6), W2(6,4)는 직접 정한 상수이고 bias=0, LN gamma=1/beta=0,
    dropout 생략이다. Qwen의 RoPE/GQA/RMSNorm/SwiGLU를 구현한 것이 아니다.
    예: block_walkthrough()['A'][0] -> [1,0,0], ['Y'].shape -> (3,4).
    """
    x = np.eye(4)[:3]
    wq = np.array([[1,0],[0,1],[1,1],[0,0]], dtype=float)
    wk = wq.copy()
    wv = np.array([[1,0],[0,2],[3,1],[0,0]], dtype=float)
    q, k, v = x @ wq, x @ wk, x @ wv
    scores = q @ k.T / np.sqrt(2)
    masked = np.where(np.triu(np.ones((3,3), dtype=bool),1), -np.inf, scores) if causal else scores.copy()
    a, z = attention(q, k, v, causal=causal)
    a2, h2 = attention(x @ wq[:,::-1], k, x @ wv[:,::-1], causal=causal)
    joined = np.concatenate([z,h2], axis=1)
    wo = np.array([[1,0,.2,0],[0,1,0,.2],[.2,0,1,0],[0,.2,0,1]])
    projected = joined @ wo
    residual1 = x + projected
    u = layer_norm(residual1)
    w1 = np.array([[1,0,-1,0,1,0],[0,1,0,-1,0,1],
                   [1,0,1,0,-1,0],[0,1,0,1,0,-1]], dtype=float)
    w2 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],
                   [.5,0,-.5,0],[0,.5,0,-.5]], dtype=float)
    expanded = u @ w1
    ffn = feed_forward(u,w1,w2)
    residual2 = u + ffn
    return dict(X=x,WQ=wq,WK=wk,WV=wv,Q=q,K=k,V=v,scores=scores,masked=masked,
                A=a,Z=z,A2=a2,head2=h2,concat=joined,WO=wo,projected=projected,
                residual1=residual1,U=u,W1=w1,W2=w2,expanded=expanded,
                activated=np.maximum(0,expanded),ffn=ffn,residual2=residual2,
                Y=layer_norm(residual2))


def begin_generation(lab: Any, prompt: str, limit: int = 8) -> dict:
    """문장을 한 번 토큰화해 새로운 사용자별 생성 상태(dict)를 만든다.

    lab: 이미 준비한 LanguageModelLab. prompt: 1–1500자·384토큰 이하 비공백 문자열.
    limit: 새 토큰 수 한도, 정수 1..32(기본 8). 잘못된 값은 ValueError.
    반환: prompt(str), ids(list[int]), initial_length(int), limit(int), rows(list),
    reason(str='ready'). ids에는 입력만 있고 rows는 비어 있다. 모델 forward/학습/
    다운로드 없음. Tensor 대신 복사 가능한 정수 목록을 저장하여 gr.State에 넣는다.
    예: begin_generation(lab,'The sky is',3)['rows'] -> [].
    """
    if isinstance(limit,bool) or not isinstance(limit,(int,float,np.integer,np.floating)) or not np.isfinite(limit) or int(limit)!=limit or not 1<=limit<=32:
        raise ValueError('새 토큰 한도는 1..32 정수여야 합니다.')
    ids = prompt_ids(lab,prompt)[0].tolist()
    return dict(prompt=prompt,ids=ids,initial_length=len(ids),limit=int(limit),rows=[],reason='ready')


def step_generation(lab: Any, state: dict) -> dict:
    """실제 forward를 딱 한 번 실행하여 greedy 토큰을 추가한 새 상태를 반환한다.

    state: begin_generation 또는 이전 step_generation의 dict. None/잘못된 상태는
    ValueError. 기존 dict/ID 목록/기록은 변경하지 않는다. EOS/한도에 도달한 상태는
    복사만 반환하며 모델을 다시 호출하지 않는다. 그 외 [1,n] ID tensor → 로짓 [V]
    → argmax → [1,n+1] ID → 기록 한 행을 수행한다. 캐시·generate()·학습 없음.
    rows 열: step,id,token,model_probability,selection_probability,input_length.
    reason은 running/EOS/max_new_tokens. greedy selection_probability=1은 사실성
    확신도가 아니다. 예: s1=step_generation(lab,s0); len(s1['rows'])==1.
    """
    if not isinstance(state,dict) or not {'prompt','ids','initial_length','limit','rows','reason'} <= state.keys():
        raise ValueError('먼저 초기화 버튼을 누르세요.')
    result = deepcopy(state)
    ids = result['ids']
    if (not isinstance(ids,list) or not 1<=len(ids)<=416 or
        any(type(i) is not int or not 0<=i<lab.model.config.vocab_size for i in ids) or
        not 1<=result['initial_length']<=384 or
        len(ids)!=result['initial_length']+len(result['rows']) or
        not 1<=result['limit']<=32):
        raise ValueError('생성 상태가 올바르지 않습니다. 초기화하세요.')
    if result['reason']=='EOS':
        return result
    if len(result['rows'])>=result['limit']:
        result['reason']='max_new_tokens'
        return result
    tensor = lab.torch.tensor([ids],dtype=lab.torch.long,device=lab.device)
    choice = choose_token(forward_logits(lab,tensor),strategy='greedy')
    next_ids = append_token(lab,tensor,choice['id'])
    row = trace_row(lab,len(result['rows'])+1,choice)
    row['input_length']=len(ids)
    result['rows'].append(row)
    result['ids']=next_ids[0].tolist()
    result['reason']=('EOS' if choice['id']==lab.tokenizer.eos_token_id else
                      'max_new_tokens' if len(result['rows'])>=result['limit'] else 'running')
    return result


def generation_view(lab: Any, state: dict) -> tuple[str,pd.DataFrame,str]:
    """상태를 (입력 포함 전체 문장, 기록 표, 상태 설명)으로 바꾼다.

    lab은 준비한 모델 객체, state는 begin/step의 dict다. 전체 ids를 한 번 decode하며
    개별 token 문자열을 이어 붙이지 않는다. 표는 6열, 초기 상태는 0행이다.
    다운로드/forward/학습/입력 변경 없음. EOS는 decode에서 숨겨도 표에 남는다.
    예: generation_view(lab,begin_generation(lab,'A robot'))[1].shape == (0,6).
    """
    columns=['step','id','token','model_probability','selection_probability','input_length']
    text=lab.tokenizer.decode(state['ids'],skip_special_tokens=True)
    return text,pd.DataFrame(deepcopy(state['rows']),columns=columns),f"{state['reason']} · {len(state['rows'])}/{state['limit']} 새 토큰"


def build_step_app(lab: Any):
    """교육용 어텐션 표와 실제 토큰 누적을 가진 Gradio Blocks를 반환한다.

    lab: 준비한 실제 LanguageModelLab. 모델을 새로 로드하거나 서버를 시작하지 않는다.
    reset(prompt,limit)와 advance(prompt,limit,state) callback을 lesson_reset/lesson_advance로
    노출하여 직접 검증할 수 있다. 둘 다 (state,text,DataFrame,status)를 반환한다.
    state는 사용자 세션마다 분리한다. 입력/한도 변경 후에는 초기화를 요구한다.
    실제 오류는 gr.Error로 알리며 가짜 출력으로 대체하지 않는다. 예:
    app=build_step_app(lab); app.launch(share=False). 외부 공개는 별도 명시적 실행이다.
    """
    import gradio as gr

    def reset(prompt,limit):
        try:
            state=begin_generation(lab,prompt,limit)
            return (state,*generation_view(lab,state))
        except (ValueError,RuntimeError) as exc:
            raise gr.Error(str(exc)) from exc

    def advance(prompt,limit,state):
        try:
            if state is None:
                raise ValueError('먼저 초기화 버튼을 누르세요.')
            if prompt!=state['prompt'] or limit!=state['limit']:
                raise ValueError('문장 또는 한도가 바뀌었습니다. 초기화 버튼을 누르세요.')
            state=step_generation(lab,state)
            return (state,*generation_view(lab,state))
        except (ValueError,RuntimeError) as exc:
            raise gr.Error(str(exc)) from exc

    def inspect(causal,position):
        result=block_walkthrough(causal)
        row=int(position)-1
        table=pd.DataFrame({'참고 위치':[1,2,3],'점수':result['scores'][row],
                            '마스크 후':result['masked'][row],'참고 비중':result['A'][row],
                            'Value 1':result['V'][:,0],'Value 2':result['V'][:,1]})
        message=f"교육용 head 1 / 출력 z{row+1} = {np.round(result['Z'][row],4).tolist()} / 참고 비중 합 = {result['A'][row].sum():.4f}"
        return table,message

    with gr.Blocks(title='W03B · 블록과 생성 관찰실') as app:
        gr.Markdown('# 블록과 생성 관찰실\n교육용 행렬과 실제 Qwen 출력을 구분해서 읽습니다.')
        with gr.Tab('1. 어텐션 계산'):
            gr.Markdown('실제 모델에서 추출한 어텐션이 아닙니다. 3위치 수계산이며 미래를 가릴 때 어느 행이 바뀌는지 확인하세요.')
            causal=gr.Checkbox(value=True,label='미래 차단 (causal)')
            position=gr.Slider(1,3,value=2,step=1,label='정보를 받는 위치 (행)')
            inspect_button=gr.Button('어텐션 계산')
            weights=gr.Dataframe(value=inspect(True,2)[0],interactive=False,label='선택한 행의 계산')
            detail=gr.Textbox(value=inspect(True,2)[1],label='가중합 결과',interactive=False)
            inspect_button.click(inspect,[causal,position],[weights,detail],api_name='attention_inspect')
        with gr.Tab('2. 실제 토큰 누적'):
            gr.Markdown('① 문장/한도를 정하고 초기화 ② 다음 토큰 버튼을 반복합니다. 한도가 되면 멈춥니다. 문장을 바꾸면 다시 초기화하세요.')
            state=gr.State(None,time_to_live=3600)
            prompt=gr.Textbox(value='The sky is',label='시작 문장',lines=2)
            limit=gr.Slider(1,32,value=4,step=1,label='최대 새 토큰')
            with gr.Row():
                reset_button=gr.Button('초기화',variant='secondary')
                next_button=gr.Button('다음 토큰 1개 추가',variant='primary')
            output=gr.Textbox(label='누적 문장',lines=3,interactive=False)
            history=gr.Dataframe(headers=['step','id','token','model_probability','selection_probability','input_length'],label='생성 기록',interactive=False)
            status=gr.Textbox(label='생성 상태',interactive=False)
            reset_button.click(reset,[prompt,limit],[state,output,history,status],api_name='reset_generation')
            next_button.click(advance,[prompt,limit,state],[state,output,history,status],api_name='advance_generation')
        gr.Markdown('기본 모델의 이어쓰기 실습입니다. 사실성/챗봇 품질을 보장하지 않습니다. 개인정보를 입력하지 마세요. 공유 링크는 실행 중인 런타임에 의존합니다.')
    app.lesson_reset=reset
    app.lesson_advance=advance
    app.lesson_inspect=inspect
    return app.queue(default_concurrency_limit=1,max_size=8)
