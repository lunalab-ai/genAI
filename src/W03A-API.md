# W03A API · 0.1.3

설치 ref는 `2026-fall-w03a`다. 아래 링크는 같은 버전의 함수 선언을 가리킨다. 기존 [0.1.2 모델 로더](https://github.com/lunalab-ai/genAI/blob/2026-fall-explained/src/luna_genai/model.py#L13)와 이전 수업 태그는 보존한다.

정의 바로가기: [attention](https://github.com/lunalab-ai/genAI/blob/2026-fall-w03a/src/luna_genai/transformer.py#L10) · [toy_attention](https://github.com/lunalab-ai/genAI/blob/2026-fall-w03a/src/luna_genai/transformer.py#L34) · [layer_norm](https://github.com/lunalab-ai/genAI/blob/2026-fall-w03a/src/luna_genai/transformer.py#L42) · [prompt_ids](https://github.com/lunalab-ai/genAI/blob/2026-fall-w03a/src/luna_genai/transformer.py#L54) · [forward_logits](https://github.com/lunalab-ai/genAI/blob/2026-fall-w03a/src/luna_genai/transformer.py#L63) · [choose_token](https://github.com/lunalab-ai/genAI/blob/2026-fall-w03a/src/luna_genai/transformer.py#L77) · [append_token](https://github.com/lunalab-ai/genAI/blob/2026-fall-w03a/src/luna_genai/transformer.py#L99) · [trace_row](https://github.com/lunalab-ai/genAI/blob/2026-fall-w03a/src/luna_genai/transformer.py#L109) · [one_token_callback](https://github.com/lunalab-ai/genAI/blob/2026-fall-w03a/src/luna_genai/transformer.py#L119) · [build_transformer_app](https://github.com/lunalab-ai/genAI/blob/2026-fall-w03a/src/luna_genai/transformer.py#L137) · [feed_forward](https://github.com/lunalab-ai/genAI/blob/2026-fall-w03a/src/luna_genai/transformer.py#L172)

| 정의·행 | 입력과 기본값 | 반환 | 부작용·한계 |
|---|---|---|---|
| attention · 10 | Q(nq,dk), K(nk,dk), V(nk,dv), causal=False | A(nq,nk), Z(nq,dv) ndarray | 합성 계산용, 학습 없음. causal은 같은 길이만 허용 |
| toy_attention · 34 | causal=True | A(3,3), Z(3,2) | 강의 가상 수치, 실제 모델 값 아님 |
| layer_norm · 42 | x, eps=1e-5 | x와 같은 모양 | 마지막 축, gamma=1,beta=0 고정 |
| prompt_ids · 54 | 준비된 lab, 문장 | [1,n] LongTensor | 1–1500자·384토큰 이하, 모델 장치에 생성 |
| forward_logits · 63 | lab, [1,n] LongTensor | [V] CPU float Tensor | 실제 forward 1회, cache=False, 학습·generate 없음 |
| choose_token · 77 | logits, strategy='greedy', temperature=.8, top_k=0, rng=None | id·두 확률 dict | sample에는 Generator 필요. greedy 설정 무시 |
| append_token · 99 | lab, ids, next_id | [1,n+1] Tensor | 입력 변경·재토큰화 없음 |
| trace_row · 109 | lab, step, choice | 표의 행 dict | 개별 decode는 표시용 repr |
| one_token_callback · 119 | lab | callback | 항상 1토큰, 자동 반복과 구분 |
| build_transformer_app · 137 | lab, generate_callback=None | Blocks | 모델 재로드·launch 없음, 큐 동시성 1 |

표 열은 `step`, `id`, `token`, `model_probability`, `selection_probability`다. 기본 모델 확률의 분모는 전체 어휘다. 선택분포 확률은 sampling의 필터 후 확률이며 greedy에서는 선택한 ID에 1이다. 사실성 확신도가 아니다.

```python
from luna_genai.transformer import toy_attention, choose_token
a,z=toy_attention()
assert a.shape==(3,3) and z.shape==(3,2)
assert choose_token([1.,3.,2.])['id']==1
```

모든 함수의 상세 계약·예외·예시는 구현 docstring에 있다. 모델이 필요한 함수는 기존 `LanguageModelLab()`를 한 번 만든 후 호출한다. 0.1.3 검토 코드는 기존 API를 제거하거나 과거 태그를 변경하지 않는다.

`feed_forward(x,w1,w2,b1=None,b2=None)`는 원논문의 ReLU FFN 수계산을 검증한다. 입력 x의 마지막 폭 d를 w1(d,m)로 늘리고, ReLU 후 w2(m,dout)로 변환하여 (...,dout) 배열을 반환한다. bias 기본값은 0이다. 학습하지 않고 같은 가중치를 모든 위치에 적용한다. Qwen의 SwiGLU 구현과 구분한다. 강의의 행렬 예에서는 [[1,-1]]을 [[0,2]]로 변환한다.
