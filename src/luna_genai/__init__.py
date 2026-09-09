"""Shared course APIs; importing this module does not download a model."""
from .data import load_reviews
from .probability import filtered_distribution, distribution_table
from .model import LanguageModelLab, MODEL_ID, MODEL_REVISION

__version__ = "0.1.2"

def build_app(lab=None):
    """이미 준비한 lab을 연결한 Gradio Blocks 객체를 반환한다.
    lab: LanguageModelLab 또는 None, 기본 None. 실제 lab을 주면 토큰/생성/분류
    callback이 같은 모델을 재사용한다. None이면 가상 로짓 수치 실험만 제공하고
    화면에 수치 전용 모드를 표시한다. 모델 준비·학습·서버 시작은 하지 않는다.
    호출자가 app.launch(share=True)로 서버를 실행한다. Gradio 설치가 필요하다.
    예: app=build_app(lab). 반환 객체의 close()는 실행 중 서버를 닫는다."""
    from .app import build_app as build
    return build(lab)

__all__ = ["LanguageModelLab", "MODEL_ID", "MODEL_REVISION", "load_reviews",
           "filtered_distribution", "distribution_table", "build_app"]
