"""W07B review app: explicit toy values, no question-bank answers."""
from __future__ import annotations
from typing import Callable, Any
from .review_core import softmax_scores, filter_probs, attention_row, bigram_counts, generate_one

CORPUS = 'aba cab aba dab aba cab'


def probability_view(temperature: float, top_k: int, top_p: float,
                     summary_fn: Callable | None = None) -> tuple:
    """세 설정 → 표·설명·막대그래프. 가상 로짓[2.4,1.1,.2,-.6], 실제LLM 아님.

    top_k 정수0..4, top_p(0,1], T>0. summary_fn(result)->str는 학생 연결 지점.
    반환 (후보표 list, Markdown str, matplotlib Figure); 호출자는 Figure를 닫는다.
    """
    import matplotlib.pyplot as plt
    p = softmax_scores([2.4,1.1,.2,-.6], temperature)
    result = filter_probs(p,top_k,top_p)
    result.update(temperature=temperature, top_k=top_k, top_p=top_p)
    labels = ['가','나','다','라']
    table = [[t,round(p[i],6),round(result['after_k'][i],6),round(result['final'][i],6)] for i,t in enumerate(labels)]
    fig,ax=plt.subplots(figsize=(6.2,3.1))
    xs=list(range(4))
    ax.bar([x-.18 for x in xs],p,.36,label='Before filter',color='#94a3b8')
    ax.bar([x+.18 for x in xs],result['final'],.36,label='After filter',color='#2c72b8')
    ax.set_xticks(xs,['A','B','C','D']);ax.set_ylim(0,1.05);ax.set_ylabel('Probability');ax.legend();fig.tight_layout()
    summary=f"교육용 가상 점수입니다. 남은 후보 {len(result['kept'])}개 · 최종 확률 합 {sum(result['final']):.6f}. top-k 뒤 재정규화한 분포에 top-p를 적용합니다."
    if summary_fn is not None:summary+='\n\n'+str(summary_fn(result))
    return table,summary,fig


def attention_view(position: int, causal: bool) -> tuple:
    """0-based 위치0..2와 마스크 → Key표·출력벡터·가중치그래프.

    Q=K=[[1,0],[0,1],[1,1]], V=[[2,1],[0,4],[3,-1]]은 명시한 수계산 예다.
    (표 list, 설명 str, Figure) 반환; 학습·다운로드 없음.
    """
    import matplotlib.pyplot as plt
    qk=[[1,0],[0,1],[1,1]];values=[[2,1],[0,4],[3,-1]]
    if isinstance(position,bool) or not isinstance(position,int) or not 0<=position<3:
        raise ValueError('위치를0,1,2 중 선택하세요.')
    result=attention_row(qk[position],qk,values,position,causal)
    table=[[j,round(result['scores'][j],6),str(result['allowed'][j]),round(result['weights'][j],6),str(values[j])] for j in range(3)]
    fig,ax=plt.subplots(figsize=(6.2,2.8));ax.bar(range(3),result['weights'],color='#309c91')
    ax.set_xticks(range(3),['Key 0','Key 1','Key 2']);ax.set_ylim(0,1.05);ax.set_ylabel('Attention weight');fig.tight_layout()
    report=f"Query 위치 {position} · causal={causal} · 출력 [{result['output'][0]:.6f}, {result['output'][1]:.6f}]. 행은 참고 위치, 가중합의 출력은 특징 벡터입니다. 원점수와 마스크 허용 여부를 함께 읽으세요."
    return table,report,fig


def generation_view(context: str, seed: int, greedy: bool) -> tuple:
    """문맥·정수seed·greedy → 새문맥·전이표·모드설명. 내장 합성 말뭉치 사용.

    전이표 열은 다음문자/횟수/확률. 매 클릭1문자 추가. 실패는 명확한 오류로 전달.
    """
    result=generate_one(context,bigram_counts(CORPUS),seed,greedy)
    return result['context'],result['table'],result['mode']+' · 선택한 문자: '+repr(result['next'])


def build_review_app(summary_fn: Callable | None = None) -> Any:
    """3패널 Gradio Blocks 생성, 서버는 시작하지 않음. CPU·API키/모델다운로드 없음.

    summary_fn(result)->str를 연결하면 확률 패널에 학생 설명을 표시한다.
    예: app=build_review_app(); app.launch(share=True). 공유 URL은 런타임에 의존.
    """
    import gradio as gr
    with gr.Blocks(title='W07b · 생성 원리 복습 실험실') as app:
        gr.Markdown('# 생성 원리 복습 실험실\n작은 교육용 숫자와 문자 전이표를 사용합니다. 실제 거대언어모델이나 이미지 모델을 실행하지 않습니다. 설정 하나만 바꾸고 계산 과정을 비교하세요.')
        with gr.Tab('1 · 확률과 후보 필터'):
            gr.Markdown('가상 로짓 [2.4, 1.1, 0.2, -0.6] · 표의 가/나/다/라는 그림의 A/B/C/D와 같습니다.')
            t=gr.Slider(.2,3,value=1,step=.1,label='temperature')
            k=gr.Dropdown([0,1,2,3,4],value=0,label='top-k · 0은 해제')
            p=gr.Slider(.1,1,value=1,step=.05,label='top-p')
            go=gr.Button('확률 계산',variant='primary')
            table=gr.Dataframe(headers=['후보','T 적용 후','top-k 후','최종'],interactive=False)
            report=gr.Markdown();plot=gr.Plot()
            def view(t,k,p):return probability_view(t,k,p,summary_fn)
            go.click(view,[t,k,p],[table,report,plot],api_name='probabilities')
        with gr.Tab('2 · 어텐션 한 행'):
            gr.Markdown('Q=K=[[1,0],[0,1],[1,1]], V=[[2,1],[0,4],[3,-1]]. 0-based 위치를 선택합니다. 위치1은 두 번째 토큰입니다.')
            pos=gr.Dropdown([0,1,2],value=1,label='Query 위치')
            causal=gr.Checkbox(value=True,label='미래 차단 causal mask')
            calc=gr.Button('어텐션 계산',variant='primary')
            grid=gr.Dataframe(headers=['Key 위치','원점수','허용','가중치','Value'],interactive=False)
            info=gr.Markdown();attplot=gr.Plot()
            calc.click(attention_view,[pos,causal],[grid,info,attplot],api_name='attention')
        with gr.Tab('3 · 한 문자씩 생성'):
            gr.Markdown('말뭉치: `'+CORPUS+'`. 공백도 문자입니다. 매 클릭 마지막 문자 조건의 전이표를 다시 읽습니다. 매 호출 seed를 다시 설정하므로 긴 난수 스트림과 구별하세요.')
            context=gr.Textbox(value='a',label='현재 문맥 · 새 문자가 뒤에 붙습니다')
            seed=gr.Number(value=7,precision=0,label='seed');greedy=gr.Checkbox(value=False,label='greedy 선택')
            step=gr.Button('한 문자 추가',variant='primary')
            rows=gr.Dataframe(headers=['다음 문자','횟수','확률'],interactive=False);mode=gr.Markdown()
            step.click(generation_view,[context,seed,greedy],[context,rows,mode],api_name='next_character')
        gr.Markdown('이 앱은 원리를 관찰하는 도구입니다. 확률은 사실성 점수가 아닙니다. 공유 링크는 현재 런타임이 유지되는 동안 이용할 수 있습니다.')
    return app
