"""Small course-authored data, installed with the package (no upload needed)."""
from importlib.resources import files
import json

def load_reviews(split: str = "development") -> list[dict[str, str]]:
    """패키지에 포함된 수업용 리뷰 목록을 새 객체로 읽어 반환한다.
    split: examples/development/evaluation 문자열, 기본 development.
    반환: 각 원소가 review/label 문자열을 가진 dict인 list. label은 positive/negative.
    examples만 few-shot 프롬프트에 넣고 development에서 설계를 점검한 뒤
    evaluation은 미사용 평가에 남긴다. 작은 독자적 예문이므로 성능 benchmark가 아니다.
    파일 업로드·네트워크가 필요 없고 호출자가 수정해도 다음 로딩에는 영향이 없다.
    잘못된 split은 ValueError. 예: load_reviews('evaluation')."""
    if split not in {"examples", "development", "evaluation"}:
        raise ValueError("split must be examples, development, or evaluation")
    data = json.loads(files("luna_genai").joinpath("datasets/reviews.json").read_text(encoding="utf-8"))
    return data[split]
