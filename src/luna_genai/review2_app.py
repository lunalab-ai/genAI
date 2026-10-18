"""W08a four-tab CPU observation app; no exercise answers or model loading."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Any

from .review2_core import (reconstruction_stats, vae_quantities, cosine_search,
                          precision_at_k, diffusion_state, recover_clean,
                          cfg_vector, ddim_step, load_review_records)

Summary = Callable[[dict], str]
TOY_MODE = "설명용 수치 계산 · 학습된 모델 실행 아님"


def _report(record: dict, summary_fn: Summary | None) -> str:
    lines = ["**"+record["mode"]+"**", "바꾼 조건: "+record["changed"],
             "관찰: "+record["observation"], "해석의 한계: "+record["limit"]]
    if summary_fn is not None:
        text = summary_fn(dict(record))
        if not isinstance(text, str):
            raise TypeError("summary_fn은 문자열을 반환해야 합니다.")
        lines += ["**내 설명 함수**", text]
    return "\n\n".join(lines)


def representation_view(beta: float = 1., epsilon_first: float = -.4,
                        summary_fn: Summary | None = None) -> tuple:
    """beta·첫 잡음값 → 픽셀표/잠재표/설명/그림(한 이미지 예시).

    beta>=0, epsilon_first는 유한값. 고정 복원 SSE+beta*KL을 다시 계산하며
    beta 변경 후 재학습한 결과가 아니다. 표의 MSE/SSE 분모와 z를 함께 읽는다.
    반환 Figure는 사용 후 close. summary_fn(record)->str는 Q60 연결 지점.
    """
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("beta는 유한한 비음수여야 합니다.")
    import matplotlib.pyplot as plt
    target, prediction = [.1,.7,.4,.9], [.2,.6,.5,.8]
    stats = reconstruction_stats(target, prediction)
    means, logs, noise = [.2,-.3], [math.log(.36),math.log(1.44)], [epsilon_first,.7]
    latent = vae_quantities(means, logs, noise)
    pixels = [[i,t,p,round(r,6),round(r*r,6)] for i,(t,p,r) in enumerate(zip(target,prediction,stats["residual"]))]
    rows = [[i,means[i],round(logs[i],6),round(latent["std"][i],6),noise[i],round(latent["z"][i],6)] for i in range(2)]
    total = stats["sse"]+beta*latent["kl"]
    record = dict(mode=TOY_MODE, changed=f"beta={beta:g}, epsilon[0]={epsilon_first:g}",
                  observation=f"고정 복원 SSE={stats['sse']:.4f}, MSE={stats['mse']:.4f}; KL={latent['kl']:.4f}; SSE+beta*KL={total:.4f}",
                  limit="고정된 평균·분산·복원값의 계산이다. beta를 움직여 네트워크를 다시 학습하지 않는다.")
    fig, axes = plt.subplots(1,2,figsize=(8,3))
    axes[0].plot(range(4),target,"o-",label="Target")
    axes[0].plot(range(4),prediction,"s--",label="Reconstruction")
    axes[0].set(xlabel="Pixel index",ylabel="Value",ylim=(0,1)); axes[0].legend()
    axes[1].bar(["SSE","KL","beta * KL"],[stats['sse'],latent['kl'],beta*latent['kl']],color=['#2374ab','#30a18e','#e5a23a'])
    axes[1].set(ylabel="One-image contribution");fig.tight_layout()
    return pixels, rows, _report(record,summary_fn), fig


def search_view(include_extra: bool = True, scale: float = 3., k: int = 2,
                summary_fn: Summary | None = None) -> tuple:
    """추가후보 여부·양수배율·k(1..4) → 순위표/설명/그림.

    질의[2,1], 후보A[1,0]/B[0,1]/C[1,1]/추가D[-1,0]는 설명용 벡터다.
    관련 ID는 검색 전 {A,C}로 고정. 후보를 추가해도 기존 cosine은 같지만
    softmax 분모는 달라진다. 실제 CLIP 임베딩/정답 확률이라고 표시하지 않는다.
    """
    import matplotlib.pyplot as plt
    if isinstance(k,bool) or not isinstance(k,int) or not 1<=k<=4:
        raise ValueError("k는 1부터4까지 정수입니다.")
    vectors = [[1,0],[0,1],[1,1]]+([[-1,0]] if include_extra else [])
    labels = list("ABCD")[:len(vectors)]
    result = cosine_search([2,1], vectors, scale)
    ordered = [labels[i] for i in result["order"]]
    rows = [[rank+1,labels[i],round(result['cosine'][i],6),round(result['relative_score'][i],6),labels[i] in {'A','C'}] for rank,i in enumerate(result['order'])]
    p = precision_at_k(ordered, {'A','C'}, k)
    record = dict(mode=TOY_MODE, changed=f"후보 {len(vectors)}개, 배율={scale:g}, k={k}",
                  observation=f"순위 {' → '.join(ordered)}, Precision@{k}={p:.4f}; 관련ID는 미리 A,C로 지정",
                  limit="상대 점수는 후보와 배율에 의존한다. 검색 순위나 큰 점수만으로 의미 일치·권리를 보증하지 않는다.")
    fig,axes=plt.subplots(1,2,figsize=(8,3))
    axes[0].bar(labels,result['cosine'],color='#2374ab');axes[0].set(ylabel='Cosine',ylim=(-1,1))
    axes[1].bar(labels,result['relative_score'],color='#30a18e');axes[1].set(ylabel='Relative score',ylim=(0,1))
    fig.tight_layout()
    return rows,_report(record,summary_fn),fig


def diffusion_view(alpha_bar: float = .49, prediction_type: str = "epsilon",
                   summary_fn: Summary | None = None) -> tuple:
    """누적alpha·예측목표 선택 → 좌표표/설명/그림. 0<alpha_bar<1.

    clean=[.3,-.7], noise=[-.2,.9]를 사용한다. 세 target은 알려진 원본/잡음에서
    만든 정답이다. 정확 target을 역변환한 오차는 모델 성능이 아닌 대수 검산.
    epsilon/sample/v_prediction 중 선택, Figure는 호출 후 close.
    """
    import matplotlib.pyplot as plt
    state = diffusion_state([.3,-.7],[-.2,.9],alpha_bar)
    if prediction_type not in {"epsilon","sample","v_prediction"}:
        raise ValueError("알 수 없는 예측 목표입니다.")
    prediction = state[prediction_type]
    recovered = recover_clean(state['noisy'],prediction,alpha_bar,prediction_type)
    rows=[[i,state['sample'][i],state['epsilon'][i],round(state['noisy'][i],6),round(prediction[i],6),round(recovered[i],6)] for i in range(2)]
    max_error=max(abs(a-b) for a,b in zip(state['sample'],recovered))
    record=dict(mode=TOY_MODE,changed=f"누적alpha={alpha_bar:g}, prediction_type={prediction_type}",
                observation=f"신호계수={state['a']:.4f}, 잡음계수={state['b']:.4f}; 정확 target의 역변환 최대오차={max_error:.3g}",
                limit="정답 target을 사용한 식 검산이다. 신경망을 학습하거나 잡음 예측 성능을 측정한 결과가 아니다.")
    fig,ax=plt.subplots(figsize=(7,3))
    for values,label in [(state['sample'],'Clean'),(state['noisy'],'Noisy'),(prediction,'Selected target')]:
        ax.plot(range(2),values,'o-',label=label)
    ax.set(xlabel='Coordinate',ylabel='Value',xticks=[0,1]);ax.legend();fig.tight_layout()
    return rows,_report(record,summary_fn),fig


def sampling_view(scale: float = 2.5, summary_fn: Summary | None = None) -> tuple:
    """CFG 배율 → CFG/독립 DDIM 수치표·설명·그림.

    CFG 두 예측은[-.2,.4],[.3,.1]. 별도 DDIM 예는 z=[.4,-.2],
    epsilon=[.15,-.25], 누적alpha .36→.81이다. eta=0/클리핑 없음.
    CFG 출력은 잡음 예측이며 이미지가 아니다. 수학식0과 실제 API 분기 구별.
    """
    import matplotlib.pyplot as plt
    base, conditional = [-.2,.4],[.3,.1]
    guided = cfg_vector(base,conditional,scale)
    step = ddim_step([.4,-.2],[.15,-.25],.36,.81)
    rows=[[i,base[i],conditional[i],round(guided[i],6),round(step['predicted_x0'][i],6),round(step['previous'][i],6)] for i in range(2)]
    record=dict(mode=TOY_MODE,changed=f"CFG 수학식 scale={scale:g}; DDIM 예시는 별도 고정값",
                observation=f"CFG 예측={guided}; DDIM 다음상태={[round(x,6) for x in step['previous']]}",
                limit="고정 예측의 대수 계산이다. Diffusers SD API는 guidance_scale<=1에서 두갈래 CFG를 생략한다. 큰 scale이나 latent norm을 화질 점수로 해석하지 않는다.")
    fig,ax=plt.subplots(figsize=(7,3))
    for vector,label,color in [(base,'Base','#94a3b8'),(conditional,'Conditional','#30a18e'),(guided,'Guided','#2374ab')]:
        ax.quiver(0,0,*vector,angles='xy',scale_units='xy',scale=1,color=color,label=label)
    extent=max(1.,*(abs(v) for v in base+conditional+guided))*1.25
    ax.set(xlim=(-extent,extent),ylim=(-extent,extent),xlabel='Prediction coordinate 0',ylabel='Prediction coordinate 1')
    ax.axhline(0,color='#d1d5db');ax.axvline(0,color='#d1d5db');ax.legend();fig.tight_layout()
    return rows,_report(record,summary_fn),fig


def recorded_view(run_key: str = "base", summary_fn: Summary | None = None) -> tuple:
    """기존 W07A run key → trace표/설정·관측설명/최종이미지 경로.

    key=base/negative-red/cfg-4/steps-10. 매 호출 manifest의 모든 해시 검증.
    현재는 CPU 기록 읽기이며 신규 생성/재학습/다운로드가 아니다.
    seed·초기latent해시·모델revision·설정을 제시하고 norm을 품질로 쓰지 않는다.
    """
    record = load_review_records()
    runs = {item['key']:item for item in record['runs']}
    if run_key not in runs:
        raise ValueError("기록에 존재하는 run key를 선택하세요.")
    run=runs[run_key]
    rows=[[s['step'],s['input_t'],s['next_t'],round(s['latent_rms_before'],6),round(s['latent_rms_after'],6)] for s in run['trace']]
    summary=dict(mode=record['current_mode'],
                 changed=f"기록선택={run_key}; CFG={run['guidance_scale']}, steps={run['steps']}, negative={run['negative_prompt']!r}",
                 observation=f"{record['created_at']} T4 기록의 {len(rows)}단계 trace와 최종이미지; seed={run['seed']}; 초기latent SHA256={run['initial_sha256']}",
                 limit="한 seed의 결과다. latent RMS는 품질 점수가 아니며 기록선택은 새 생성이나 가중치 갱신이 아니다.")
    report=_report(summary,summary_fn)+f"\n\n모델 `{record['model_id']}` · revision `{record['revision']}` · {run['scheduler']}, eta={run['eta']}. 기록 당시 mode={run['mode']!r}."
    return rows,report,str(Path(record['asset_directory'])/run['file'])


def build_review_app(summary_fn: Summary | None = None) -> Any:
    """4탭 Gradio 앱 생성, 서버 시작 없음. Q60 summary_fn(record)->str 연결.

    record의 mode/changed/observation/limit은 항상 문자열이다. 학생 함수는
    이를 문장으로 연결하며 계산·관측값을 만들지 않는다. 예:
    app=build_review_app(summary_fn=my_summary); app.launch(share=True).
    필수 경로 CPU, 유료API/모델다운로드 없음. 공유 URL은 런타임 수명에 의존.
    """
    import gradio as gr
    with gr.Blocks(title='W08a · 이미지 모델 복습 실험실') as app:
        gr.Markdown('# 이미지 모델 복습 실험실\n계산 전에 예상하고, 조건 하나를 바꾼 뒤 근거를 설명합니다. **설명용 수치 계산**과 **기존 실제 SD 기록 열람**을 구분합니다. 여기서는 모델을 학습하거나 새 이미지를 생성하지 않습니다.')
        with gr.Tab('1 · 복원과 VAE'):
            gr.Markdown('원본 [0.1,0.7,0.4,0.9], 복원 [0.2,0.6,0.5,0.8]. mu=[0.2,-0.3], variance=[0.36,1.44]. beta 변경은 고정 손실항 재계산입니다.')
            beta=gr.Slider(0,4,value=1,step=.25,label='KL 가중치 beta');eps=gr.Slider(-2,2,value=-.4,step=.1,label='epsilon[0] · epsilon[1]=0.7 고정')
            button=gr.Button('복원·VAE 계산',variant='primary')
            pixels=gr.Dataframe(headers=['픽셀','원본','복원','잔차','제곱잔차'],interactive=False)
            latent=gr.Dataframe(headers=['좌표','mu','logvar','std','epsilon','z'],interactive=False)
            report=gr.Markdown();plot=gr.Plot()
            button.click(lambda b,e:representation_view(b,e,summary_fn),[beta,eps],[pixels,latent,report,plot],api_name='representation')
        with gr.Tab('2 · 검색과 상대 점수'):
            gr.Markdown('설명용 query=[2,1]; A=[1,0], B=[0,1], C=[1,1], 추가D=[-1,0]. 관련ID는 검색 전 A,C로 정했습니다. 실제 CLIP을 실행한 벡터가 아닙니다.')
            extra=gr.Checkbox(value=True,label='후보D 추가');scale=gr.Slider(.5,8,value=3,step=.5,label='softmax 점수 배율');k=gr.Dropdown([1,2,3,4],value=2,label='Precision@k')
            button=gr.Button('검색 순위 계산',variant='primary')
            table=gr.Dataframe(headers=['순위','후보ID','cosine','상대점수','사전정의 관련성'],interactive=False);report=gr.Markdown();plot=gr.Plot()
            button.click(lambda x,s,k:search_view(x,s,k,summary_fn),[extra,scale,k],[table,report,plot],api_name='search')
        with gr.Tab('3 · 확산과 예측 목표'):
            gr.Markdown('clean=[0.3,-0.7], noise=[-0.2,0.9]. v=a·epsilon−b·x0. 원본과 잡음을 알고 만든 학습 target을 확인합니다.')
            alpha=gr.Slider(.01,.99,value=.49,step=.01,label='누적 alpha');kind=gr.Dropdown(['epsilon','sample','v_prediction'],value='epsilon',label='예측 목표')
            button=gr.Button('노이즈와 target 계산',variant='primary')
            table=gr.Dataframe(headers=['좌표','원본','잡음','noisy','선택 target','역변환 원본'],interactive=False);report=gr.Markdown();plot=gr.Plot()
            button.click(lambda a,t:diffusion_view(a,t,summary_fn),[alpha,kind],[table,report,plot],api_name='diffusion')
        with gr.Tab('4 · CFG와 샘플링 기록'):
            gr.Markdown('위: 작은 벡터 계산. 아래: 기존 W07A 실제 SD 기록. 두 실행 모드를 구분합니다. CFG 계산과 별도 DDIM 한 단계는 독립 예시입니다.')
            guidance=gr.Slider(0,5,value=2.5,step=.25,label='CFG 수학식 scale');button=gr.Button('CFG·DDIM 계산',variant='primary')
            table=gr.Dataframe(headers=['좌표','base','conditional','CFG','별도DDIM 원본추정','별도DDIM 다음상태'],interactive=False);report=gr.Markdown();plot=gr.Plot()
            button.click(lambda s:sampling_view(s,summary_fn),[guidance],[table,report,plot],api_name='sampling')
            gr.Markdown('### 기존 실제 SD 기록 읽기\n파일 해시를 확인해 저장된 결과를 엽니다. 현재 모델을 불러오거나 새 이미지를 생성하지 않습니다.')
            key=gr.Dropdown(['base','negative-red','cfg-4','steps-10'],value='base',label='기존 기록');button=gr.Button('검증된 기록 열기')
            trace=gr.Dataframe(headers=['step','input_t','next_t','latent RMS 전','latent RMS 후'],interactive=False);report=gr.Markdown();picture=gr.Image(label='기존 실제 실행의 최종 이미지',interactive=False,format='png')
            button.click(lambda k:_recorded_image_view(k,summary_fn),[key],[trace,report,picture],api_name='recorded')
        gr.Markdown('Q60의 my_summary 함수는 현재 실행 모드·바꾼 조건·관찰·한계를 연결합니다. 앱 기본 설명과 표를 근거로 본인의 해석을 덧붙이세요. 공유 주소는 현재 런타임이 유지되는 동안 사용합니다.')
    return app


def _recorded_image_view(run_key: str, summary_fn: Summary | None = None) -> tuple:
    """검증된 기록을 이미지로 복사해 Gradio의 자체 캐시에 전달한다."""
    from PIL import Image
    rows, report, image_path = recorded_view(run_key, summary_fn)
    with Image.open(image_path) as picture:
        return rows, report, picture.copy()
