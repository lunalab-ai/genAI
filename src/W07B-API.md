# W07B · 공통 코드 안내

## 공통 코드: 정의와 입력·출력

설치한 태그 `2026-fall-w07b`와 같은 정의 행입니다. 함수 이름을 클릭하여 구현을 바로 확인할 수 있습니다. 아래 API는 실습 편의를 위한 것이며, 연습문항은 기본 Python으로 풉니다.

### [softmax_scores](https://github.com/lunalab-ai/genAI/blob/2026-fall-w07b/src/luna_genai/review_core.py#L14)

유한 점수 N개 → 합1 확률 N개. T>0, 기본1; 입력은 수정하지 않는다.

최댓값을 뺀 뒤 T로 나누어 지수를 계산한다. 예: [0,log(2)] → [1/3,2/3].
빈 입력·비유한 값·양수가 아닌 T는 ValueError. 학습이나 추출은 하지 않는다.

### [filter_probs](https://github.com/lunalab-ai/genAI/blob/2026-fall-w07b/src/luna_genai/review_core.py#L31)

합1 비음수 확률 → {base, after_k, final, order, kept}.

top_k=0은 해제; 1..N 정수는 상위 개수. top_p는 (0,1], 기본1은 해제.
순서: top-k → 재정규화 → top-p → 재정규화. 경계를 넘는 토큰 포함.
동률은 원 인덱스 순서; final은 원 순서이며 제외 후보0. 입력 변경 없음.
예: filter_probs([.6,.3,.1],top_k=2)['final'] == [2/3,1/3,0].

### [attention_row](https://github.com/lunalab-ai/genAI/blob/2026-fall-w07b/src/luna_genai/review_core.py#L64)

Query1개·Key N×d·Value N×v → 점수/허용/가중치/출력 dict.

position은0-based Query 위치; causal=True이면 j<=position만 허용한다.
점수=q·k/sqrt(d), 허용점수에 softmax, 출력폭v의 Value 가중합.
학습·다운로드 없음. 잘못된 차원·비유한 값은 ValueError.
예: attention_row([1],[[0],[0]],[[2],[6]],1)['output'] == [4.0].

### [bigram_counts](https://github.com/lunalab-ai/genAI/blob/2026-fall-w07b/src/luna_genai/review_core.py#L92)

인접 문자 전이 횟수. 공백도 문자이며 끝→처음 연결 없음.

입력 길이2이상 문자열. 반환은 {출발문자:{다음문자:횟수}} 새 dict.
예: 'xyx' → {'x':{'y':1},'y':{'x':1}}. 파일을 읽지 않는다.

### [generate_one](https://github.com/lunalab-ai/genAI/blob/2026-fall-w07b/src/luna_genai/review_core.py#L107)

직전 문자 전이표에서1문자 선택. {context,next,table,mode} 반환.

seed는 로컬 Random에만 적용하여 전역 난수 상태를 바꾸지 않는다.
greedy 동률은 문자 정렬 순서. 매 호출 seed를 다시 설정하는 수업 실험이며,
긴 생성의 연속 난수 스트림을 모사한다고 주장하지 않는다. 미관측 문맥은 오류.

### [definition_link](https://github.com/lunalab-ai/genAI/blob/2026-fall-w07b/src/luna_genai/review_core.py#L132)

설치한 수업 함수의 실제 정의 시작 행 GitHub URL; 네트워크 요청 없음.

수업 패키지 함수만 허용한다. ref는 설치한 tag/commit과 같게 지정한다.

### [probability_view](https://github.com/lunalab-ai/genAI/blob/2026-fall-w07b/src/luna_genai/review_app.py#L9)

세 설정 → 표·설명·막대그래프. 가상 로짓[2.4,1.1,.2,-.6], 실제LLM 아님.

top_k 정수0..4, top_p(0,1], T>0. summary_fn(result)->str는 학생 연결 지점.
반환 (후보표 list, Markdown str, matplotlib Figure); 호출자는 Figure를 닫는다.

### [attention_view](https://github.com/lunalab-ai/genAI/blob/2026-fall-w07b/src/luna_genai/review_app.py#L32)

0-based 위치0..2와 마스크 → Key표·출력벡터·가중치그래프.

Q=K=[[1,0],[0,1],[1,1]], V=[[2,1],[0,4],[3,-1]]은 명시한 수계산 예다.
(표 list, 설명 str, Figure) 반환; 학습·다운로드 없음.

### [generation_view](https://github.com/lunalab-ai/genAI/blob/2026-fall-w07b/src/luna_genai/review_app.py#L50)

문맥·정수seed·greedy → 새문맥·전이표·모드설명. 내장 합성 말뭉치 사용.

전이표 열은 다음문자/횟수/확률. 매 클릭1문자 추가. 실패는 명확한 오류로 전달.

### [build_review_app](https://github.com/lunalab-ai/genAI/blob/2026-fall-w07b/src/luna_genai/review_app.py#L59)

3패널 Gradio Blocks 생성, 서버는 시작하지 않음. CPU·API키/모델다운로드 없음.

summary_fn(result)->str를 연결하면 확률 패널에 학생 설명을 표시한다.
예: app=build_review_app(); app.launch(share=True). 공유 URL은 런타임에 의존.

