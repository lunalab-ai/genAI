"""Numerically stable educational softmax and candidate filters."""
import numpy as np
import pandas as pd

def filtered_distribution(logits, temperature: float = 1.0, top_k: int = 0,
                          top_p: float = 1.0) -> np.ndarray:
    """Softmax(logits/T), then top-k, then nucleus filtering, renormalizing.

    top_k=0 and top_p=1 disable filters. Keep the token crossing the nucleus
    threshold. temperature must be positive; greedy decoding is a separate rule.
    Equal scores use original index order for the small educational array.
    """
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
    """Four invented logits for offline numeric exploration, never model output."""
    z = np.array([3., 2., 1., 0.])
    return pd.DataFrame({"후보": ["A", "B", "C", "D"], "가상 로짓": z,
                         "기본 확률": filtered_distribution(z),
                         "설정 후 확률": filtered_distribution(z, temperature, top_k, top_p)})
