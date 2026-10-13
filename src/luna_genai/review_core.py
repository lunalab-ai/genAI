"""Small CPU calculations for reviewing language generation, not a pretrained LLM.

All data are explicit inputs. No download, training, global random-state mutation,
or hidden answer bank. Returned lists/dicts are new objects.
"""
from __future__ import annotations
import math
import random
import inspect
from pathlib import Path
from typing import Callable, Sequence


def softmax_scores(scores: Sequence[float], temperature: float = 1.0) -> list[float]:
    """유한 점수 N개 → 합1 확률 N개. T>0, 기본1; 입력은 수정하지 않는다.

    최댓값을 뺀 뒤 T로 나누어 지수를 계산한다. 예: [0,log(2)] → [1/3,2/3].
    빈 입력·비유한 값·양수가 아닌 T는 ValueError. 학습이나 추출은 하지 않는다.
    """
    z = [float(v) for v in scores]
    if not z or not all(math.isfinite(v) for v in z):
        raise ValueError('점수는 비어 있지 않은 유한한 수 목록이어야 합니다.')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature는 양수여야 합니다. greedy는 별도 선택 규칙입니다.')
    maximum = max(z)
    weights = [math.exp((v - maximum) / temperature) for v in z]
    total = sum(weights)
    return [v / total for v in weights]


def filter_probs(probabilities: Sequence[float], top_k: int = 0,
                 top_p: float = 1.0) -> dict:
    """합1 비음수 확률 → {base, after_k, final, order, kept}.

    top_k=0은 해제; 1..N 정수는 상위 개수. top_p는 (0,1], 기본1은 해제.
    순서: top-k → 재정규화 → top-p → 재정규화. 경계를 넘는 토큰 포함.
    동률은 원 인덱스 순서; final은 원 순서이며 제외 후보0. 입력 변경 없음.
    예: filter_probs([.6,.3,.1],top_k=2)['final'] == [2/3,1/3,0].
    """
    p = [float(v) for v in probabilities]
    if not p or not all(math.isfinite(v) and v >= 0 for v in p) or not math.isclose(sum(p), 1, abs_tol=1e-9):
        raise ValueError('확률은 비음수·유한 값이며 합이1이어야 합니다.')
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 0 <= top_k <= len(p):
        raise ValueError('top_k는0부터후보수까지의정수입니다.')
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError('top_p는0보다크고1이하여야합니다.')
    order = sorted(range(len(p)), key=lambda i: (-p[i], i))
    selected = order[:top_k] if top_k else order
    mass = sum(p[i] for i in selected)
    after_k = [p[i] / mass if i in selected else 0.0 for i in range(len(p))]
    kept, cumulative = [], 0.0
    for i in selected:
        if after_k[i] == 0:
            continue
        kept.append(i)
        cumulative += after_k[i]
        if top_p < 1 and (cumulative >= top_p or math.isclose(cumulative, top_p, abs_tol=1e-12)):
            break
    mass = sum(after_k[i] for i in kept)
    final = [after_k[i] / mass if i in kept else 0.0 for i in range(len(p))]
    return dict(base=p, after_k=after_k, final=final, order=order, kept=kept)


def attention_row(query: Sequence[float], keys: Sequence[Sequence[float]],
                  values: Sequence[Sequence[float]], position: int,
                  causal: bool = True) -> dict:
    """Query1개·Key N×d·Value N×v → 점수/허용/가중치/출력 dict.

    position은0-based Query 위치; causal=True이면 j<=position만 허용한다.
    점수=q·k/sqrt(d), 허용점수에 softmax, 출력폭v의 Value 가중합.
    학습·다운로드 없음. 잘못된 차원·비유한 값은 ValueError.
    예: attention_row([1],[[0],[0]],[[2],[6]],1)['output'] == [4.0].
    """
    q = [float(x) for x in query]
    k = [[float(x) for x in row] for row in keys]
    v = [[float(x) for x in row] for row in values]
    if not q or not k or len(k) != len(v) or not v[0] or any(len(row) != len(q) for row in k) or any(len(row) != len(v[0]) for row in v):
        raise ValueError('Query·Key·Value의 행 수와 특징 차원을 확인하세요.')
    if not all(math.isfinite(x) for row in [q]+k+v for x in row):
        raise ValueError('모든 특징값은 유한해야 합니다.')
    if isinstance(position, bool) or not isinstance(position, int) or not 0 <= position < len(k):
        raise ValueError('position은0부터위치수−1까지정수입니다.')
    scores = [sum(a*b for a,b in zip(q,row))/math.sqrt(len(q)) for row in k]
    allowed = [not causal or j <= position for j in range(len(k))]
    local = softmax_scores([s for s,ok in zip(scores,allowed) if ok])
    iterator = iter(local)
    weights = [next(iterator) if ok else 0.0 for ok in allowed]
    output = [sum(weights[j]*v[j][d] for j in range(len(v))) for d in range(len(v[0]))]
    return dict(scores=scores, allowed=allowed, weights=weights, output=output)


def bigram_counts(text: str) -> dict[str, dict[str, int]]:
    """인접 문자 전이 횟수. 공백도 문자이며 끝→처음 연결 없음.

    입력 길이2이상 문자열. 반환은 {출발문자:{다음문자:횟수}} 새 dict.
    예: 'xyx' → {'x':{'y':1},'y':{'x':1}}. 파일을 읽지 않는다.
    """
    if not isinstance(text, str) or len(text) < 2:
        raise ValueError('두 문자 이상의 수업용 문자열이 필요합니다.')
    counts: dict[str, dict[str, int]] = {}
    for left, right in zip(text[:-1], text[1:]):
        row = counts.setdefault(left, {})
        row[right] = row.get(right, 0) + 1
    return counts


def generate_one(context: str, counts: dict[str, dict[str, int]],
                 seed: int = 7, greedy: bool = False) -> dict:
    """직전 문자 전이표에서1문자 선택. {context,next,table,mode} 반환.

    seed는 로컬 Random에만 적용하여 전역 난수 상태를 바꾸지 않는다.
    greedy 동률은 문자 정렬 순서. 매 호출 seed를 다시 설정하는 수업 실험이며,
    긴 생성의 연속 난수 스트림을 모사한다고 주장하지 않는다. 미관측 문맥은 오류.
    """
    if not isinstance(context, str) or not context or context[-1] not in counts:
        raise ValueError('현재 마지막 문자의 전이가 없습니다. 예시 문맥 "a"로 시작하세요.')
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError('seed는 정수입니다.')
    row = counts[context[-1]]
    if not row or not all(isinstance(k,str) and len(k)==1 and isinstance(v,int) and v>0 for k,v in row.items()):
        raise ValueError('전이표는 한 문자 후보와 양의 정수 횟수여야 합니다.')
    tokens = sorted(row)
    total = sum(row.values())
    probabilities = [row[t]/total for t in tokens]
    selected = max(range(len(tokens)), key=lambda i: probabilities[i]) if greedy else random.Random(seed).choices(range(len(tokens)), weights=probabilities, k=1)[0]
    token = tokens[selected]
    return dict(context=context+token, next=token,
                table=[[t,row[t],p] for t,p in zip(tokens,probabilities)],
                mode='교육용 문자 bigram · 매 호출 로컬 seed 설정')


def definition_link(function: Callable, ref: str = '2026-fall-w07b') -> str:
    """설치한 수업 함수의 실제 정의 시작 행 GitHub URL; 네트워크 요청 없음.

    수업 패키지 함수만 허용한다. ref는 설치한 tag/commit과 같게 지정한다.
    """
    file = Path(inspect.getsourcefile(function) or '').resolve()
    package = Path(__file__).resolve().parent
    relative = file.relative_to(package).as_posix()
    line = inspect.getsourcelines(function)[1]
    return f'https://github.com/lunalab-ai/genAI/blob/{ref}/src/luna_genai/{relative}#L{line}'
