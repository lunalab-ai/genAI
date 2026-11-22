# W13A · 캠퍼스 팝업 콘텐츠 제작 스튜디오

캠퍼스 행사 운영 메모와 FAQ에서 학교 게시판 공지와 학생 대상 홍보 초안을 만드는 개인 실습입니다. 운영팀에 합류한 주니어 개발자로서 데이터, 모델, 검증, 앱 기본값을 함께 결정합니다.

## 1. 학습 데이터와 운영 데이터

사진 워크숍의 운영 메모가 일시·장소·참가비를 주고 FAQ가 신청·정원을 알려 줍니다. 구조화한 목표 응답에 출처 ID를 붙입니다. Qwen2는 프롬프트를 읽고 목표 응답을 예측합니다. 손실은 목표 응답 토큰에서만 계산합니다. 가상 행사 24건은 훈련, 6건은 기본값 검증, 2건은 공통 비교용입니다. [W09B 2절](https://github.com/lunalab-ai/genAI/blob/2026-fall-w09b/course/notion/w09b/w09b-instructions.md#L17) · [W10A 손실 마스크](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/course/notion/w10a/w10a-lesson.md#L203)

## 2. Qwen2와 E5가 하는 일

E5는 질문과 문서를 384차원 벡터로 표현합니다. 길이 1로 정규화한 뒤 코사인 유사도를 내적으로 구합니다. 학생이 작성한 정렬 함수가 top-k개 문서를 실제 프롬프트로 보냅니다. 생성기는 W03B의 Qwen2-0.5B입니다. 기본 가중치를 동결하고 q와 v 투영에 FP32 LoRA를 학습해 다시 읽습니다. 수업은 4회 업데이트, 과제는 훈련 자료 24건에 대한 업데이트입니다. [W03B 8절](https://github.com/lunalab-ai/genAI/blob/2026-fall-w03b/course/notion/w03b/w03b-transformer-walkthrough.md#L174) · [W10A LoRA](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/course/notion/w10a/w10a-lesson.md#L30) · [W10A 최소 RAG](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/course/notion/w10a/w10a-lesson.md#L248) · [W10A RAG Colab](https://colab.research.google.com/github/lunalab-ai/genAI/blob/2026-fall-w10a/notebooks/student/w10a_rag.ipynb)

## 3. 비교 실험과 제품 선택

| 조건 | 실제 입력과 모델 | 비교 목적 |
|---|---|---|
| A | Qwen2 기본 · 근거 없음 | 기본 생성 능력 관찰 |
| B | 같은 Qwen2 · 문서 2개 | A와 근거의 효과 비교 |
| C | B와 같은 문서 · rank 4 학습 어댑터 | B와 학습의 효과 비교 |
| D | 문서를 2→1 또는 rank를 4→2 | 한 가지 가설을 세우고 6개 검증 사례에서 비교 |

운영 자료 기준으로 핵심 사실, 신청 미정, 항목 누락, 출처 ID를 확인합니다. 선택한 설정으로 앱을 띄우고 참가비 문서만 갱신합니다. 화면의 사실·선택 설정·문서 버전과 생성 원문을 ZIP으로 제출합니다. [학생 교안](https://github.com/lunalab-ai/genAI/releases/download/2026-fall-w13a/w13a-studio-student.pdf) · [과제 1 안내](https://github.com/lunalab-ai/genAI/blob/2026-fall-w13a/course/assignments/w13a-assignment1.md)
