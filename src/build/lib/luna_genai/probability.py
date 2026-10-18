"""Numerically stable educational softmax and candidate filters."""
import numpy as np
import pandas as pd

def filtered_distribution(logits, temperature: float = 1.0, top_k: int = 0,
                          top_p: float = 1.0) -> np.ndarray:
    """유한한 1차원 logits를 확률 배열로 바꾸고 후보 필터를 적용한다.
    logits: 길이 V>0 숫자 배열. temperature: 양수, 기본 1.0; 낮을수록 뾰족하다.
    top_k: 남길 상위 후보 수 0..V의 정수, 기본 0은 해제.
    top_p: 누적확률 기준 (0,1], 기본 1.0은 해제.
    softmax(logits/temperature) → top-k → 재정규화 → top-p → 재정규화 순서다.
    top-p 경계를 처음 넘는 후보까지 포함한다. 동점은 원래 인덱스 순서다.
    반환은 합이 1인 길이 V ndarray이며 제외 후보는 0이다. 입력은 변경하지 않는다.
    잘못된 모양/설정은 ValueError. temperature=0은 허용하지 않으며 greedy는 별도 규칙.
    예: filtered_distribution([3,2,1,0], top_k=2)는 앞 두 후보만 양수다."""
    z = np.asarray(logits, dtype=float)
    if z.ndim != 1 or not len(z) or not np.isfinite(z).all():
        raise ValueError("logits must be a nonempty finite 1-D array")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive (use greedy for argmax)")
    if isinstance(top_k, bool) or int(top_k) != top_k or not 0 <= top_k <= len(z):
        raise ValueError("top_k must be an integer between 0 and vocabulary size")
    if not np.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    scores = (z - z.max()) / temperature
    p = np.exp(scores)
    p /= p.sum()
    order = np.argsort(-p, kind="stable")
    if top_k:
        p[order[int(top_k):]] = 0
        p /= p.sum()
    if top_p < 1:
        count = int(np.searchsorted(np.cumsum(p[order]), top_p, side="left")) + 1
        p[order[count:]] = 0
        p /= p.sum()
    return p

def distribution_table(temperature: float = 1.0, top_k: int = 0,
                       top_p: float = 1.0) -> pd.DataFrame:
    """직접 정한 네 로짓 [3,2,1,0]의 필터 전후 확률 표를 반환한다.
    temperature=1.0, top_k=0, top_p=1.0은 filtered_distribution과 같은 뜻이다.
    반환: 후보/가상 로짓/기본 확률/설정 후 확률 열의 4행 DataFrame.
    모델 다운로드·학습·추론 없이 CPU 숫자 연산만 한다. 모델 출력이 아니다.
    예: distribution_table(temperature=0.5, top_k=2)."""
    z = np.array([3., 2., 1., 0.])
    return pd.DataFrame({"후보": ["A", "B", "C", "D"], "가상 로짓": z,
                         "기본 확률": filtered_distribution(z),
                         "설정 후 확률": filtered_distribution(z, temperature, top_k, top_p)})
