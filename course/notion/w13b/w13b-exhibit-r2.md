# 다시 쓰는 전시 · 관람객용 카드 만들기

2026년 11월 25일 · 생성형 인공지능 W13B · 개인 CPU 실습

1. 체험 ZIP을 풀고 **index.html**을 여세요. 업사이클링 토트백의 운영 메모가 채워져 있습니다. 두 카드의 **모델 원문과 근거**를 엽니다. 티셔츠 예시에서 예약이 비어 있을 때의 답도 확인합니다.
2. 학생 Colab의 TODO 1–4를 완성합니다. 토큰에 학습 라벨을 붙이고, 두 rank를 각각 4회 업데이트하고, 같은 모델에서 생성 온도 두 값을 비교합니다.
3. **내 운영 메모로 카드 만들기**에서 재료와 리폼 과정을 수정하고 관람객을 고릅니다. 메모에 없는 사실은 빈칸으로 둘 수 있습니다. 비교 HTML 또는 결과 JSON을 받아 생성한 두 카드를 비교합니다.
4. 자료에서 확인되는 카드 한 장을 고르거나 보류합니다. 결정에 중요했던 수치나 문장 한 곳을 남깁니다.

| 바꾸는 곳 | 같게 두는 것 | 결과 |
|---|---|---|
| 응답 마스크 | 실제 입력 토큰과 응답 토큰 순서 | 손실에 들어가는 위치 |
| LoRA rank 2 ↔ 4 | Qwen 리비전, 학습 사례, 순서, 4회, α/r, 학습률 | 학습 가능 모수와 검증 표 |
| 생성 온도 0.35 ↔ 0.85 | 프롬프트, 어댑터, 시드, 토큰 상한 | 두 문장과 반복·형식 판정 |
| 게시 규칙 | 생성 원문과 운영 메모 | 채택 여부와 근거 |

공동 진단 사례 BAG-07에서 기존 rank 2는 사실을 맞히고 8줄을 만들었습니다. TEE-11에서 기존 rank 4의 자유문장은 같은 단어를 32번 반복했습니다. NLL, 사실, 형식, 반복을 각각 읽으세요. 4회 직접 실행과 24회 수업 실측은 화면에서 따로 고를 수 있습니다.

## 이전 강의 복습

- 토큰이 한 칸 이동하는 이유와 다음 토큰 확률을 읽습니다. [W03B Transformer](https://github.com/lunalab-ai/genAI/blob/2026-fall-w03b/course/notion/w03b/w03b-transformer-walkthrough.md#L156)
- 운영 메모와 학습 응답을 나누어 어느 토큰을 손실에 넣을지 확인합니다. [W09B 지시 학습과 응답 손실](https://github.com/lunalab-ai/genAI/blob/2026-fall-w09b/course/notion/w09b/w09b-instructions.md#L33)
- rank와 alpha가 어댑터를 어떻게 바꾸는지 읽고, 비교표에서 NLL의 분모를 확인합니다. [W10A LoRA 설정](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/course/notion/w10a/w10a-lesson.md#L99) · [W10A 모델 평가](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/course/notion/w10a/w10a-lesson.md#L220)

[학생 Colab](https://colab.research.google.com/github/lunalab-ai/genAI/blob/2026-fall-w13b-r3/notebooks/student/w13b_exhibit_r2.ipynb) · [과제 1 전용 24회·9건 Colab](https://colab.research.google.com/github/lunalab-ai/genAI/blob/2026-fall-w13b/notebooks/student/w13b_exhibit.ipynb)
