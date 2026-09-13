"""Inspectable single-step inference and small, explicitly synthetic attention."""
from __future__ import annotations
from typing import Callable
import numpy as np
import pandas as pd
from .model import LanguageModelLab, _LOCK
from .probability import filtered_distribution


def attention(q, k, values, *, causal: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """작은 교육용 scaled attention의 (가중치, 출력)을 반환한다.

    q: (nq, dk), k: (nk, dk), values: (nk, dv)의 유한한 실수 2차원 배열.
    causal=False가 기본. True는 길이가 같은 self-attention만 허용한다.
    행별 softmax 이전 미래 점수를 -inf로 가린다. 결과는 (nq,nk), (nq,dv).
    입력 변경·학습·다운로드는 없다. 잘못된 모양/비유한 값은 ValueError.
    예: attention([[1,0]], [[1,0]], [[3,2]]) -> (array([[1.]]), array([[3.,2.]]))
    실제 Qwen의 attention kernel이 아니라 작은 수계산을 확인하는 함수다.
    """
    q,k,values=(np.asarray(a,dtype=float) for a in (q,k,values))
    if any(a.ndim != 2 or min(a.shape)==0 or not np.isfinite(a).all() for a in (q,k,values)):
        raise ValueError('Q/K/V must be nonempty finite matrices')
    if q.shape[1]!=k.shape[1] or k.shape[0]!=values.shape[0]:
        raise ValueError('Q/K feature sizes and K/V position counts must agree')
    scores=q@k.T/np.sqrt(q.shape[1])
    if causal:
        if q.shape[0]!=k.shape[0]: raise ValueError('causal toy attention requires equal lengths')
        scores=np.where(np.triu(np.ones_like(scores,dtype=bool),1),-np.inf,scores)
    weights=np.exp(scores-scores.max(axis=1,keepdims=True))
    weights/=weights.sum(axis=1,keepdims=True)
    return weights, weights@values


def toy_attention(causal: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """강의의 가상 3토큰 예 (A, Z)를 반환한다. causal 기본 True.
    Q=K=[[1,0],[0,1],[1,1]], V=[[1,0],[0,2],[3,1]]. 출력 모양 (3,3),(3,2).
    실제 모델 가중치와 무관하고 학습 상태가 없다. 예: toy_attention()[0].sum(1)는 [1,1,1].
    """
    return attention([[1,0],[0,1],[1,1]],[[1,0],[0,1],[1,1]],[[1,0],[0,2],[3,1]],causal=causal)


def layer_norm(x, eps: float = 1e-5) -> np.ndarray:
    """교육용 마지막 축 LayerNorm. gamma=1, beta=0으로 고정한다.
    x: 유한한 1차원 이상 실수 배열, eps: 양수 기본 1e-5. 같은 모양의 ndarray 반환.
    마지막 축의 평균과 모집단 분산(ddof=0)을 사용한다. 학습은 하지 않는다.
    실제 모듈의 학습되는 affine 파라미터는 생략했다. 예: layer_norm([1.,3.])는 약 [-1,1].
    """
    x=np.asarray(x,dtype=float)
    if x.ndim<1 or not x.size or not np.isfinite(x).all() or not np.isfinite(eps) or eps<=0:
        raise ValueError('finite nonempty x and positive eps required')
    return (x-x.mean(axis=-1,keepdims=True))/np.sqrt(x.var(axis=-1,keepdims=True)+eps)


def prompt_ids(lab: LanguageModelLab, prompt: str):
    """문장 하나를 [1,n] LongTensor ID로 바꾼다. lab은 이미 준비한 모델 객체.
    prompt: 공백뿐이 아닌 1–1500자, 최대 384토큰. 결과는 lab.device에 둔다.
    특수 토큰을 자동 추가하지 않고 복사된 ID를 반환한다. 다운로드/학습은 없다.
    기존 입력 검증을 재사용하며 잘못된 입력은 ValueError. 예: prompt_ids(lab,'The sky is').
    """
    return lab._inputs(prompt).input_ids.clone()


def forward_logits(lab: LanguageModelLab, ids):
    """모델 forward를 한 번 실행하고 마지막 위치 로짓 [V]를 CPU float Tensor로 반환.
    ids: lab.device의 LongTensor [1,n], 1<=n<=416. 입력 ID는 변경하지 않는다.
    use_cache=False, inference_mode로 실행한다. generate()와 학습은 호출하지 않는다.
    모델은 생성자에서 eval이어야 하며 이 함수는 가중치를 갱신하지 않는다.
    다운로드하지 않고 같은 모델을 잠금으로 직렬 사용한다. 예: forward_logits(lab,prompt_ids(lab,'A robot')).
    """
    if ids.ndim!=2 or ids.shape[0]!=1 or not 1<=ids.shape[1]<=416:
        raise ValueError('ids must have shape [1,n] with 1<=n<=416')
    if ids.dtype != lab.torch.long: raise ValueError('ids must be torch.long')
    with _LOCK, lab.torch.inference_mode():
        return lab.model(input_ids=ids,use_cache=False).logits[0,-1,:].float().cpu()


def choose_token(logits, *, strategy: str = 'greedy', temperature: float = .8,
                 top_k: int = 0, rng: np.random.Generator | None = None) -> dict:
    """로짓 [V]에서 ID와 두 확률을 반환한다. 모델/학습 상태는 변경하지 않는다.
    strategy=greedy/sample. greedy는 argmax이며 temperature/top_k를 무시한다.
    sample은 temperature>0(기본 .8), top_k=0(해제) 또는 1..V를 적용한다.
    rng: sample에 필요한 NumPy Generator. 외부에서 한 번 만들고 반복 중 재사용한다.
    반환 dict: id(int), model_probability(전체 어휘 기본 확률), selection_probability
    (실제 선택 규칙의 확률; greedy는 1.0). CPU Tensor 또는 배열 허용.
    잘못된 설정/비유한 값은 ValueError. 예: choose_token([3.,1.])['id']==0.
    """
    z=np.asarray(logits,dtype=float)
    base=filtered_distribution(z)
    if strategy=='greedy':
        picked=int(z.argmax()); selected=1.
    elif strategy=='sample':
        if not isinstance(rng,np.random.Generator): raise ValueError('sample requires a NumPy Generator')
        p=filtered_distribution(z,temperature=temperature,top_k=top_k)
        picked=int(rng.choice(len(p),p=p)); selected=float(p[picked])
    else: raise ValueError('strategy must be greedy or sample')
    return {'id':picked,'model_probability':float(base[picked]),'selection_probability':selected}


def append_token(lab: LanguageModelLab, ids, next_id: int):
    """[1,n] ID 뒤에 하나를 붙여 [1,n+1] 새 Tensor를 반환한다.
    next_id는 실제 모델 어휘 범위의 정수. 기존 ids를 변경하거나 다시 토큰화하지 않는다.
    생성의 반복/종료 판단은 호출자가 수행한다. 예: append_token(lab,ids,7).
    """
    if isinstance(next_id,bool) or not isinstance(next_id,(int,np.integer)) or not 0<=next_id<lab.model.config.vocab_size:
        raise ValueError('next_id outside vocabulary')
    return lab.torch.cat([ids,ids.new_tensor([[int(next_id)]])],dim=1)


def trace_row(lab: LanguageModelLab, step: int, choice: dict) -> dict:
    """한 선택을 표 행으로 변환. step은 1부터 세는 표시 번호, choice는 choose_token 결과.
    dict 열: step, id, token, model_probability, selection_probability.
    token은 단일 ID decode의 repr이며 문자열을 이어 붙여 최종 문장을 만들면 안 된다.
    학습/추론/다운로드 없음. 예: trace_row(lab,1,choice).
    """
    return {'step':step,'id':choice['id'],'token':repr(lab.tokenizer.decode([choice['id']])),
            'model_probability':choice['model_probability'],'selection_probability':choice['selection_probability']}


def one_token_callback(lab: LanguageModelLab) -> Callable:
    """실행 가능한 기본 1토큰 관찰 callback을 반환한다. 자동 반복 문제의 답안이 아니다.
    반환 함수 인자: prompt,strategy,temperature,top_k,seed,max_new_tokens.
    기본 모드는 max_new_tokens와 관계없이 한 단계만 수행한다고 명시한다.
    반환 (prompt 포함 전체 문자열, 1행 DataFrame, 종료/모드 설명).
    매 호출 고유 rng 사용, 같은 lab 재사용. 예: one_token_callback(lab)('A robot','greedy',.8,0,7,8).
    """
    def callback(prompt,strategy,temperature,top_k,seed,max_new_tokens):
        ids=prompt_ids(lab,prompt)
        choice=choose_token(forward_logits(lab,ids),strategy=strategy,temperature=temperature,
                            top_k=int(top_k),rng=np.random.default_rng(int(seed)))
        ids=append_token(lab,ids,choice['id'])
        eos=choice['id']==lab.tokenizer.eos_token_id
        return (lab.tokenizer.decode(ids[0],skip_special_tokens=True),pd.DataFrame([trace_row(lab,1,choice)]),
                'EOS' if eos else '수동 1토큰 모드: 자동 반복 연결 전에는 길이 설정을 적용하지 않습니다.')
    return callback


def build_transformer_app(lab: LanguageModelLab, generate_callback: Callable | None = None):
    """준비한 실제 lab으로 Gradio Blocks를 구성한다. 모델 로드/서버 시작은 하지 않는다.
    generate_callback=None이면 한 단계 forward 관찰. 자동 반복을 완성하면 함수로 전달한다.
    callback 입력 순서: prompt(str),strategy(str),temperature(float),top_k(int),seed(int),max_new_tokens(int).
    반환 순서: 전체 문장(str), 단계표(DataFrame), 종료 이유(str). 표 열은 trace_row와 같다.
    오류는 화면에 표시하고 가상 결과로 대체하지 않는다. queue 동시성 1.
    반환 Blocks에 lesson_callback을 붙여 직접 기능 검증을 지원한다.
    예: app=build_transformer_app(lab); app.launch(share=False).
    """
    import gradio as gr
    callback=generate_callback or one_token_callback(lab)
    def guarded(*args):
        try: return callback(*args)
        except (ValueError,RuntimeError,TypeError) as exc: raise gr.Error(str(exc)) from exc
    with gr.Blocks(title='W03A · 생성 과정 실험실') as app:
        gr.Markdown('# 생성 과정 실험실\n실제 Qwen2-0.5B · 입력과 설정 → forward → 선택 → 결과')
        gr.Markdown('자동 반복 연결 모드' if generate_callback else '기본: 한 번에 한 토큰. 자동 반복을 연결하면 길이 설정이 적용됩니다.')
        prompt=gr.Textbox(value='The sky is',label='시작 문장',lines=3,max_lines=5)
        strategy=gr.Radio(['greedy','sample'],value='greedy',label='선택 규칙')
        with gr.Row():
            temperature=gr.Slider(.1,2,value=.8,step=.1,label='Temperature (sample만)',min_width=280)
            top_k=gr.Slider(0,50,value=10,step=1,label='Top-k (0: 해제)',min_width=280)
        with gr.Row():
            seed=gr.Number(value=7,precision=0,label='Seed (0 이상 정수)',min_width=280)
            length=gr.Slider(1,32,value=8,step=1,label='최대 새 토큰 (자동 반복 모드)',min_width=280)
        button=gr.Button('생성 실행',variant='primary')
        result=gr.Textbox(label='문장 전체',lines=4)
        table=gr.Dataframe(headers=['step','id','token','model_probability','selection_probability'],label='선택 기록',interactive=False)
        reason=gr.Textbox(label='종료 이유 / 실행 모드')
        button.click(guarded,[prompt,strategy,temperature,top_k,seed,length],[result,table,reason],api_name='generate_step')
        gr.Markdown('확률은 사실성 확신도가 아닙니다. 개인정보를 입력하지 마세요. 공유 링크는 런타임 종료 시 중단될 수 있습니다.')
    app.lesson_callback=guarded
    return app.queue(default_concurrency_limit=1,max_size=8)


def feed_forward(x, w1, w2, b1=None, b2=None) -> np.ndarray:
    """교육용 ReLU FFN. x (...,d), w1 (d,m), w2 (m,dout), b1 (m,), b2 (dout,).
    b1/b2 기본 None은 0. 입력은 유한한 실수 배열이며 반환은 (...,dout) ndarray다.
    토큰 축은 섞지 않고 마지막 특징 축만 변환한다. 학습·다운로드·입력 변경 없음.
    원논문 FFN의 계산용이며 Qwen SwiGLU 구현이 아니다. 차원/비유한 값은 ValueError.
    예: feed_forward([[1.,-1.]],np.eye(2),np.eye(2)) -> [[1.,0.]].
    """
    x,w1,w2=(np.asarray(a,dtype=float) for a in (x,w1,w2))
    if x.ndim<1 or w1.ndim!=2 or w2.ndim!=2 or x.shape[-1]!=w1.shape[0] or w1.shape[1]!=w2.shape[0]:
        raise ValueError('FFN dimensions do not agree')
    b1=np.zeros(w1.shape[1]) if b1 is None else np.asarray(b1,dtype=float)
    b2=np.zeros(w2.shape[1]) if b2 is None else np.asarray(b2,dtype=float)
    if b1.shape!=(w1.shape[1],) or b2.shape!=(w2.shape[1],) or any(not a.size or not np.isfinite(a).all() for a in (x,w1,w2,b1,b2)):
        raise ValueError('FFN requires finite, nonempty arrays and matching bias vectors')
    return np.maximum(0,x@w1+b1)@w2+b2
