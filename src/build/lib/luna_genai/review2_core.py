"""W08a small, explicit image-model calculations; no training or model loading.

The functions use supplied numbers, not a question bank. They preserve inputs
and import only the standard library plus the existing lightweight softmax.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

from .review_core import softmax_scores, definition_link

RECORDS = Path(__file__).parent / "assets" / "w07a"
RECORD_MODE = "기존 실제 SD 실행 기록 열람 · 현재 신규 추론 아님"


def _vector(values: Sequence[float], name: str) -> list[float]:
    data = [float(v) for v in values]
    if not data or not all(math.isfinite(v) for v in data):
        raise ValueError(f"{name}: 비어 있지 않은 유한한 수 목록이 필요합니다.")
    return data


def _paired(left: Sequence[float], right: Sequence[float]) -> tuple[list, list]:
    a, b = _vector(left, "첫 배열"), _vector(right, "둘째 배열")
    if len(a) != len(b):
        raise ValueError("두 배열의 길이가 같아야 합니다.")
    return a, b


def reconstruction_stats(target: Sequence[float], prediction: Sequence[float]) -> dict:
    """한 이미지의 같은 길이 픽셀 두 목록 → residual, SSE, MSE, pixel_count.

    residual은 prediction-target, SSE는 제곱합, MSE는 픽셀 수로 나눈 값.
    예: [0,1]과[.2,.8] → SSE=.08, MSE=.04. 학습·클리핑 없음.
    실제 영상의 측정값인지 설명용 입력인지 호출자가 표시해야 한다.
    """
    x, y = _paired(target, prediction)
    residual = [p-t for t, p in zip(x, y)]
    sse = math.fsum(v*v for v in residual)
    return dict(residual=residual, sse=sse, mse=sse/len(x), pixel_count=len(x))


def vae_quantities(mu: Sequence[float], logvar: Sequence[float],
                   epsilon: Sequence[float]) -> dict:
    """잠재 좌표 d개의 평균·로그분산·고정 잡음 → variance/std/z/KL.

    z=mu+exp(logvar/2)*epsilon; KL은 좌표합, 표준정규 prior 기준.
    한 이미지의 설명용 계산이며 신경망·gradient·난수 추출 없음.
    예: mu=[0], logvar=[0], epsilon=[.2] → z=[.2], KL=0.
    epsilon은 관찰하려는 표준정규 잡음의 실현값을 직접 전달한다.
    """
    means, logs = _paired(mu, logvar)
    _, noise = _paired(means, epsilon)
    try:
        variances = [math.exp(v) for v in logs]
        std = [math.exp(v/2) for v in logs]
    except OverflowError as exc:
        raise ValueError("로그분산이 너무 커서 유한한 분산을 만들 수 없습니다.") from exc
    z = [m+s*e for m, s, e in zip(means, std, noise)]
    kl = .5*math.fsum(m*m+math.expm1(l)-l for m, l in zip(means, logs))
    if not all(math.isfinite(v) for v in variances+std+z+[kl]):
        raise ValueError("계산 결과가 유한한 범위를 벗어났습니다.")
    return dict(variance=variances, std=std, z=z, kl=kl)


def cosine_search(query: Sequence[float], candidates: Sequence[Sequence[float]],
                  scale: float = 1.0) -> dict:
    """질의 d개·후보 N×d → cosine/relative_score/order/unit_rows.

    각 행을 L2 정규화; order는 높은 점수 순 인덱스이며 동률은 원래 순서.
    scale>0(기본1)은 softmax 점수 배율. 상대 점수는 정답 확률이 아니다.
    예: [1,0], [[1,0],[0,1]] → cosine=[1,0], order=[0,1].
    0벡터/비유한값/축 불일치는 오류. CLIP 인코더를 실행하지 않는다.
    """
    q = _vector(query, "질의")
    rows = [_vector(row, "후보") for row in candidates]
    if not rows or any(len(row) != len(q) for row in rows):
        raise ValueError("후보 N개는 질의와 같은 특징 차원이어야 합니다.")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("점수 배율은 유한한 양수여야 합니다.")
    norms = [math.hypot(*row) for row in [q]+rows]
    if any(n == 0 or not math.isfinite(n) for n in norms):
        raise ValueError("코사인은 0벡터에서 정의되지 않습니다.")
    units = [[v/n for v in row] for row, n in zip([q]+rows, norms)]
    scores = [math.fsum(a*b for a, b in zip(units[0], row)) for row in units[1:]]
    return dict(cosine=scores, relative_score=softmax_scores([scale*s for s in scores]),
                order=sorted(range(len(scores)), key=lambda i: (-scores[i], i)),
                unit_rows=units[1:], unit_query=units[0])


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int = 3) -> float:
    """서로 다른 검색 ID·미리 정한 관련 ID 집합·k → 상위 k의 관련 비율.

    k는 양의 정수, 기본3. 결과가 k보다 적으면 빈 순위는 오답으로 센다.
    예: ['A','B','C'], {'B'}, 2 → .5. 동률 정렬은 검색 단계에서 결정.
    관련성 판단이나 모델 학습을 수행하지 않는다.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k는 양의 정수여야 합니다.")
    if len(set(retrieved)) != len(retrieved):
        raise ValueError("검색 ID가 중복되었습니다.")
    return sum(item in relevant for item in retrieved[:k])/k


def schedule_coefficients(betas: Sequence[float]) -> list[dict]:
    """단계별 beta 목록 → t/beta/alpha/alpha_bar/a/b 표(0-based t).

    각 beta는0보다 크고1보다 작다. alpha=1-beta, alpha_bar는 누적곱.
    a/b는 신호·잡음 표준편차 계수. 예: [.1,.2] → 누적[.9,.72].
    학습이나 샘플러 실행 없이 계수만 계산한다.
    """
    values = _vector(betas, "beta")
    if any(not 0 < b < 1 for b in values):
        raise ValueError("모든 beta는 0과 1 사이여야 합니다.")
    cumulative, rows = 1., []
    for t, beta in enumerate(values):
        alpha = 1-beta
        cumulative *= alpha
        rows.append(dict(t=t, beta=beta, alpha=alpha, alpha_bar=cumulative,
                         a=math.sqrt(cumulative), b=math.sqrt(1-cumulative)))
    return rows


def diffusion_state(clean: Sequence[float], noise: Sequence[float], alpha_bar: float) -> dict:
    """원본·고정 Gaussian 잡음 실현값·누적alpha → noisy와 세 학습 target.

    0<alpha_bar<1. a=sqrt(alpha_bar), b=sqrt(1-alpha_bar), noisy=a*x0+b*eps.
    v=a*eps-b*x0는 기존 W06A의 부호 관례. 학습 정답을 만드는 계산이며
    학습된 모델 예측/이미지 생성이 아니다. 예: [1],[0],.25 → noisy=[.5].
    """
    x, e = _paired(clean, noise)
    if not math.isfinite(alpha_bar) or not 0 < alpha_bar < 1:
        raise ValueError("누적 alpha는 0과 1 사이여야 합니다.")
    a, b = math.sqrt(alpha_bar), math.sqrt(1-alpha_bar)
    return dict(a=a, b=b, noisy=[a*u+b*v for u, v in zip(x, e)],
                epsilon=e, sample=x, v_prediction=[a*v-b*u for u, v in zip(x, e)])


def recover_clean(noisy: Sequence[float], prediction: Sequence[float],
                  alpha_bar: float, prediction_type: str = "epsilon") -> list[float]:
    """같은 길이 noisy·prediction → 원본 추정 목록, 클리핑 없음.

    prediction_type=epsilon(기본)/sample/v_prediction, 0<alpha_bar<1.
    정답 target이면 대수적으로 원본이 복원된다. 실제 예측 오차는 사라지지 않는다.
    예: [.5],[0],.25,'epsilon' → [1.0]. 모델·난수·학습 부작용 없음.
    """
    z, p = _paired(noisy, prediction)
    if not math.isfinite(alpha_bar) or not 0 < alpha_bar < 1:
        raise ValueError("누적 alpha는 0과 1 사이여야 합니다.")
    a, b = math.sqrt(alpha_bar), math.sqrt(1-alpha_bar)
    if prediction_type == "epsilon":
        return [(v-b*e)/a for v, e in zip(z, p)]
    if prediction_type == "sample":
        return p
    if prediction_type == "v_prediction":
        return [a*v-b*e for v, e in zip(z, p)]
    raise ValueError("prediction_type은 epsilon/sample/v_prediction 중 하나입니다.")


def cfg_vector(base: Sequence[float], conditional: Sequence[float], scale: float = 1.) -> list[float]:
    """두 조건의 같은 상태·시점 잡음 예측과 scale → CFG 결합 예측.

    base+s*(conditional-base), 유한 scale>=0, 기본1. 두 배열 길이는 같다.
    수학식의0은base/1은conditional. Diffusers API의 <=1 분기와 구별한다.
    예: [0,1],[1,0],2 → [2,-1]. 이미지·품질 점수나 새 학습 결과가 아니다.
    """
    u, c = _paired(base, conditional)
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("scale은 유한한 비음수여야 합니다.")
    return [v+scale*(w-v) for v, w in zip(u, c)]


def ddim_step(noisy: Sequence[float], epsilon: Sequence[float],
              alpha_bar: float, alpha_previous: float) -> dict:
    """교육용 결정론적 DDIM 한 단계 → predicted_x0와previous 배열.

    0<alpha_bar<=alpha_previous<=1, eta=0, 클리핑 없음. 누적alpha를 사용.
    원본 추정 후 더 낮은 잡음 시점으로 갱신하며 단순 epsilon 빼기가 아니다.
    예: [.5],[0],.25,1 → predicted_x0=[1], previous=[1]. 품질 측정 없음.
    """
    z, e = _paired(noisy, epsilon)
    if not all(math.isfinite(v) for v in (alpha_bar, alpha_previous)) or not 0 < alpha_bar <= alpha_previous <= 1:
        raise ValueError("0 < 현재 누적alpha <= 다음 누적alpha <= 1 이어야 합니다.")
    a, b = math.sqrt(alpha_bar), math.sqrt(1-alpha_bar)
    x0 = [(v-b*n)/a for v, n in zip(z, e)]
    previous = [math.sqrt(alpha_previous)*v+math.sqrt(1-alpha_previous)*n for v, n in zip(x0, e)]
    return dict(predicted_x0=x0, previous=previous)


def load_review_records(directory: str | Path | None = None) -> dict:
    """기존 W07A SD1.5 기록·25개 파일의 SHA-256 확인 → 기록 dict.

    기본은 패키지 assets/w07a. directory는 같은 형식의 별도 검증용 경로.
    반환에 current_mode/asset_directory를 추가하되 원파일은 수정하지 않는다.
    모든 manifest 파일의 경로/해시를 검사하고 손상·누락 시 명확히 실패한다.
    과거 run.mode는 생성 당시 정보다. 현재 실행은 기록 열람이며 모델·GPU·
    다운로드·신규 생성 없음. 예: load_review_records()['current_mode'].
    """
    root = (Path(directory) if directory is not None else RECORDS).resolve()
    record = json.loads((root / "sampling-records.json").read_text(encoding="utf-8"))
    if not isinstance(record.get("files"), dict) or not record["files"] or not record.get("runs"):
        raise ValueError("비어 있지 않은 기록 manifest와 runs가 필요합니다.")
    for name, expected in record["files"].items():
        if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
            raise ValueError("기록 파일 경로는 같은 폴더의 파일명이어야 합니다.")
        path = (root/name).resolve()
        if path.parent != root:
            raise ValueError("기록 폴더 밖의 파일은 읽지 않습니다.")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError("기존 SD 기록의 해시가 일치하지 않습니다: "+name)
    for run in record["runs"]:
        if run.get("file") not in record["files"] or not run.get("trace"):
            raise ValueError("실행 기록에 검증된 최종 이미지와 trace가 필요합니다.")
    record.update(current_mode=RECORD_MODE, asset_directory=str(root))
    return record
