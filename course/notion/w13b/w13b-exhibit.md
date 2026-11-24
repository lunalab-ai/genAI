# 캠퍼스 리폼 전시 안내 키트

2026년 11월 25일 · 생성형 인공지능 01분반 · 개인 CPU 실습

## 먼저 앱 열기

체험용 ZIP을 풀고 **index.html**을 더블클릭합니다. 운영 메모와 **생성된 작품 소개문**을 나란히 봅니다. 아래 두 표를 열어 같은 결과의 실행 ID를 확인하세요. 기본 앱은 **수업 실측 재생**입니다. [학생 Colab](https://colab.research.google.com/github/lunalab-ai/genAI/blob/2026-fall-w13b/notebooks/student/w13b_exhibit.ipynb)

수업에서 만드는 기능은 **작품 소개문 생성**입니다. 다음 시간에는 동일한 작품 ID에 포스터를 붙입니다.

## 01 · 작품으로 나누고 응답만 학습하기

가방·운동화·티셔츠의 가상 작품입니다. 운영 메모는 Qwen이 읽는 입력이고, 첫 줄의 네 필드와 이어지는 소개 문장은 맞힐 목표입니다. **훈련 24건 → 검증 9건 → 교수자 보관 최종 평가 9건**을 서로 다른 작품으로 나눴습니다. 자료에 없는 예약의 참조값은 '확인 필요'입니다.

| 단계 | 입력 | 확인할 출력 |
|---|---|---|
| 토큰화 | 메모와 목표 응답 | 시작 토큰, 응답 경계, 길이 |
| 다음 토큰 | 앞 토큰 전체 | 정답 토큰의 확률 $p$ |
| 손실 마스크 | 프롬프트와 응답의 ID | 프롬프트 위치는 −100, 응답 위치는 정답 ID |

정답 확률이 0.5이면 $-\log p \simeq 0.693$입니다. 0.25이면 약 1.386입니다. [W03B 다음 토큰](https://github.com/lunalab-ai/genAI/blob/2026-fall-w03b/course/notion/w03b/w03b-transformer-walkthrough.md#L174)

**01의 TODO**. prefix [11, 12], response [21, 22]를 함수에 넣어 보세요. 앞의 두 위치만 손실 제외로 바꾸고 실제 학습 함수에서 그 코드가 호출되는지 확인합니다.

## 02 · LoRA rank를 하나만 바꾸기

$$W'=W+\frac{\alpha}{r}BA,\qquad \#(A,B)=r(d_{\text{in}}+d_{\text{out}}).$$

16×16 선형층에 rank 2와 4를 넣으면 각각 64개와 128개의 가중치를 학습합니다. 실제 Qwen2.5-0.5B-Instruct에서 같은 24건으로 rank 2와 4를 각각 24회 업데이트합니다. LR 0.0005, seed 42, q_proj/v_proj, 같은 프롬프트, greedy, 최대 128 생성 토큰을 고정합니다. $\alpha/r=2$를 유지합니다. [W10A LoRA](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/course/notion/w10a/w10a-lesson.md#L30) · [실제 모델](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

작품 ID, 처음·마지막 손실, 가중치 변화와 동결된 기본 모델의 해시를 확인합니다. 기본판에서 TODO가 남아 있을 때는 실제 기본 모델 한 건과 검수한 두 어댑터의 **수업 실측 재생**을 볼 수 있습니다.

## 03 · NLL과 필드 판정 비교

정답을 넣고 측정한 NLL은 **정답 토큰 손실의 합 ÷ 정답 토큰 수**입니다. 실제 생성 원문은 **종류·장소·관람료·예약**으로 나누어 운영 메모와 대조합니다. 사실이 틀리거나 형식이 무너지면 판정을 기록합니다. 3건이면 12개 필드이며, 과제의 9건이면 36개입니다. [W10A 평가](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/course/notion/w10a/w10a-lesson.md#L220)

자료 밖 값과 형식 실패가 없는 후보 중 일치 수가 큰 쪽을 적용합니다. 동률이면 작은 rank입니다. 기준을 충족하는 후보가 없으면 게시를 보류하고 오류를 추적합니다. 네 필드의 코드 판정 뒤에 자유 문장을 읽습니다.

## 04 · 학습한 설정으로 내 앱 열기

선택 → 어댑터 재로딩 → **[선택한 모델로 소개문 생성]** → 실행 ID → **[내 키트 HTML 받기]**. 브라우저의 소개문과 실행 ID를 Colab과 대조합니다. 과제에서는 검증 사례 수를 9로 바꾸고 마지막에 **[과제 ZIP 받기]**를 선택합니다.

[새 과제 1 안내](https://github.com/lunalab-ai/genAI/blob/2026-fall-w13b/course/assignments/w13b-assignment1.md)
