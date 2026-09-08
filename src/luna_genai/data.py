"""Small course-authored data, installed with the package (no upload needed)."""
from importlib.resources import files
import json

def load_reviews(split: str = "development") -> list[dict[str, str]]:
    """Return a fresh copy of examples/development/evaluation movie reviews.

    Examples enter few-shot prompts. Tune on development only; reserve evaluation.
    This tiny synthetic set is for inspecting behavior, not benchmarking.
    """
    if split not in {"examples", "development", "evaluation"}:
        raise ValueError("split must be examples, development, or evaluation")
    data = json.loads(files("luna_genai").joinpath("datasets/reviews.json").read_text(encoding="utf-8"))
    return data[split]
