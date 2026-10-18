# W08a · 이미지 모델 복습 공통 API

고정 태그 `2026-fall-w08a-r1`, 패키지0.1.14. 학생 풀이와 별도의 설명용 입력을 사용합니다. 아래 함수는 모델을 새로 학습하거나 생성하지 않습니다.

### [reconstruction_stats](https://github.com/lunalab-ai/genAI/blob/2026-fall-w08a-r1/src/luna_genai/review2_core.py#L34)

한 이미지의 같은 길이 픽셀 두 목록 → residual, SSE, MSE, pixel_count.

residual은 prediction-target, SSE는 제곱합, MSE는 픽셀 수로 나눈 값.
예: [0,1]과[.2,.8] → SSE=.08, MSE=.04. 학습·클리핑 없음.
실제 영상의 측정값인지 설명용 입력인지 호출자가 표시해야 한다.

### [vae_quantities](https://github.com/lunalab-ai/genAI/blob/2026-fall-w08a-r1/src/luna_genai/review2_core.py#L47)

잠재 좌표 d개의 평균·로그분산·고정 잡음 → variance/std/z/KL.

z=mu+exp(logvar/2)*epsilon; KL은 좌표합, 표준정규 prior 기준.
한 이미지의 설명용 계산이며 신경망·gradient·난수 추출 없음.
예: mu=[0], logvar=[0], epsilon=[.2] → z=[.2], KL=0.
epsilon은 관찰하려는 표준정규 잡음의 실현값을 직접 전달한다.

### [cosine_search](https://github.com/lunalab-ai/genAI/blob/2026-fall-w08a-r1/src/luna_genai/review2_core.py#L70)

질의 d개·후보 N×d → cosine/relative_score/order/unit_rows.

각 행을 L2 정규화; order는 높은 점수 순 인덱스이며 동률은 원래 순서.
scale>0(기본1)은 softmax 점수 배율. 상대 점수는 정답 확률이 아니다.
예: [1,0], [[1,0],[0,1]] → cosine=[1,0], order=[0,1].
0벡터/비유한값/축 불일치는 오류. CLIP 인코더를 실행하지 않는다.

### [precision_at_k](https://github.com/lunalab-ai/genAI/blob/2026-fall-w08a-r1/src/luna_genai/review2_core.py#L95)

서로 다른 검색 ID·미리 정한 관련 ID 집합·k → 상위 k의 관련 비율.

k는 양의 정수, 기본3. 결과가 k보다 적으면 빈 순위는 오답으로 센다.
예: ['A','B','C'], {'B'}, 2 → .5. 동률 정렬은 검색 단계에서 결정.
관련성 판단이나 모델 학습을 수행하지 않는다.

### [schedule_coefficients](https://github.com/lunalab-ai/genAI/blob/2026-fall-w08a-r1/src/luna_genai/review2_core.py#L109)

단계별 beta 목록 → t/beta/alpha/alpha_bar/a/b 표(0-based t).

각 beta는0보다 크고1보다 작다. alpha=1-beta, alpha_bar는 누적곱.
a/b는 신호·잡음 표준편차 계수. 예: [.1,.2] → 누적[.9,.72].
학습이나 샘플러 실행 없이 계수만 계산한다.

### [diffusion_state](https://github.com/lunalab-ai/genAI/blob/2026-fall-w08a-r1/src/luna_genai/review2_core.py#L128)

원본·고정 Gaussian 잡음 실현값·누적alpha → noisy와 세 학습 target.

0<alpha_bar<1. a=sqrt(alpha_bar), b=sqrt(1-alpha_bar), noisy=a*x0+b*eps.
v=a*eps-b*x0는 기존 W06A의 부호 관례. 학습 정답을 만드는 계산이며
학습된 모델 예측/이미지 생성이 아니다. 예: [1],[0],.25 → noisy=[.5].

### [recover_clean](https://github.com/lunalab-ai/genAI/blob/2026-fall-w08a-r1/src/luna_genai/review2_core.py#L143)

같은 길이 noisy·prediction → 원본 추정 목록, 클리핑 없음.

prediction_type=epsilon(기본)/sample/v_prediction, 0<alpha_bar<1.
정답 target이면 대수적으로 원본이 복원된다. 실제 예측 오차는 사라지지 않는다.
예: [.5],[0],.25,'epsilon' → [1.0]. 모델·난수·학습 부작용 없음.

### [cfg_vector](https://github.com/lunalab-ai/genAI/blob/2026-fall-w08a-r1/src/luna_genai/review2_core.py#L164)

두 조건의 같은 상태·시점 잡음 예측과 scale → CFG 결합 예측.

base+s*(conditional-base), 유한 scale>=0, 기본1. 두 배열 길이는 같다.
수학식의0은base/1은conditional. Diffusers API의 <=1 분기와 구별한다.
예: [0,1],[1,0],2 → [2,-1]. 이미지·품질 점수나 새 학습 결과가 아니다.

### [ddim_step](https://github.com/lunalab-ai/genAI/blob/2026-fall-w08a-r1/src/luna_genai/review2_core.py#L177)

교육용 결정론적 DDIM 한 단계 → predicted_x0와previous 배열.

0<alpha_bar<=alpha_previous<=1, eta=0, 클리핑 없음. 누적alpha를 사용.
원본 추정 후 더 낮은 잡음 시점으로 갱신하며 단순 epsilon 빼기가 아니다.
예: [.5],[0],.25,1 → predicted_x0=[1], previous=[1]. 품질 측정 없음.

### [load_review_records](https://github.com/lunalab-ai/genAI/blob/2026-fall-w08a-r1/src/luna_genai/review2_core.py#L194)

기존 W07A SD1.5 기록·25개 파일의 SHA-256 확인 → 기록 dict.

기본은 패키지 assets/w07a. directory는 같은 형식의 별도 검증용 경로.
반환에 current_mode/asset_directory를 추가하되 원파일은 수정하지 않는다.
모든 manifest 파일의 경로/해시를 검사하고 손상·누락 시 명확히 실패한다.
과거 run.mode는 생성 당시 정보다. 현재 실행은 기록 열람이며 모델·GPU·
다운로드·신규 생성 없음. 예: load_review_records()['current_mode'].

### [representation_view](https://github.com/lunalab-ai/genAI/blob/2026-fall-w08a-r1/src/luna_genai/review2_app.py#L27)

beta·첫 잡음값 → 픽셀표/잠재표/설명/그림(한 이미지 예시).

beta>=0, epsilon_first는 유한값. 고정 복원 SSE+beta*KL을 다시 계산하며
beta 변경 후 재학습한 결과가 아니다. 표의 MSE/SSE 분모와 z를 함께 읽는다.
반환 Figure는 사용 후 close. summary_fn(record)->str는 Q60 연결 지점.

### [search_view](https://github.com/lunalab-ai/genAI/blob/2026-fall-w08a-r1/src/luna_genai/review2_app.py#L57)

추가후보 여부·양수배율·k(1..4) → 순위표/설명/그림.

질의[2,1], 후보A[1,0]/B[0,1]/C[1,1]/추가D[-1,0]는 설명용 벡터다.
관련 ID는 검색 전 {A,C}로 고정. 후보를 추가해도 기존 cosine은 같지만
softmax 분모는 달라진다. 실제 CLIP 임베딩/정답 확률이라고 표시하지 않는다.

### [diffusion_view](https://github.com/lunalab-ai/genAI/blob/2026-fall-w08a-r1/src/luna_genai/review2_app.py#L84)

누적alpha·예측목표 선택 → 좌표표/설명/그림. 0<alpha_bar<1.

clean=[.3,-.7], noise=[-.2,.9]를 사용한다. 세 target은 알려진 원본/잡음에서
만든 정답이다. 정확 target을 역변환한 오차는 모델 성능이 아닌 대수 검산.
epsilon/sample/v_prediction 중 선택, Figure는 호출 후 close.

### [sampling_view](https://github.com/lunalab-ai/genAI/blob/2026-fall-w08a-r1/src/luna_genai/review2_app.py#L110)

CFG 배율 → CFG/독립 DDIM 수치표·설명·그림.

CFG 두 예측은[-.2,.4],[.3,.1]. 별도 DDIM 예는 z=[.4,-.2],
epsilon=[.15,-.25], 누적alpha .36→.81이다. eta=0/클리핑 없음.
CFG 출력은 잡음 예측이며 이미지가 아니다. 수학식0과 실제 API 분기 구별.

### [recorded_view](https://github.com/lunalab-ai/genAI/blob/2026-fall-w08a-r1/src/luna_genai/review2_app.py#L134)

기존 W07A run key → trace표/설정·관측설명/최종이미지 경로.

key=base/negative-red/cfg-4/steps-10. 매 호출 manifest의 모든 해시 검증.
현재는 CPU 기록 읽기이며 신규 생성/재학습/다운로드가 아니다.
seed·초기latent해시·모델revision·설정을 제시하고 norm을 품질로 쓰지 않는다.

### [build_review_app](https://github.com/lunalab-ai/genAI/blob/2026-fall-w08a-r1/src/luna_genai/review2_app.py#L155)

4탭 Gradio 앱 생성, 서버 시작 없음. Q60 summary_fn(record)->str 연결.

record의 mode/changed/observation/limit은 항상 문자열이다. 학생 함수는
이를 문장으로 연결하며 계산·관측값을 만들지 않는다. 예:
app=build_review_app(summary_fn=my_summary); app.launch(share=True).
필수 경로 CPU, 유료API/모델다운로드 없음. 공유 URL은 런타임 수명에 의존.
