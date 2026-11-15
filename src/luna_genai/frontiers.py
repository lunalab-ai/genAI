"""Small numerical experiments used to explain chapter 10."""

import numpy as np


def preference_loss(policy, reference, chosen=0, beta=.5):
    """One DPO pair, with sequence log probabilities ordered as the two responses."""
    if chosen not in (0, 1) or beta <= 0:
        raise ValueError("Choose response 0 or 1 and use a positive beta.")
    weights = np.asarray(policy) - np.asarray(reference)
    advantage = weights[chosen] - weights[1 - chosen]
    logit = beta * advantage
    return float(np.logaddexp(0, -logit)), float(logit)


def causal_mask(tokens=8, window=None):
    """Visible positions for each query, including its own position."""
    if tokens < 1 or window is not None and window < 1:
        raise ValueError("Use a positive token count and window.")
    offset = np.arange(tokens)[:, None] - np.arange(tokens)[None, :]
    return (offset >= 0) & (offset < (tokens if window is None else window))


def route_experts(logits, top_k=2):
    """Select top-k per token and normalize weights across selected experts."""
    logits = np.asarray(logits, dtype=float)
    if logits.ndim != 2 or not np.isfinite(logits).all() or not 1 <= top_k <= logits.shape[1]:
        raise ValueError("Use a finite token×expert matrix and a valid top_k.")
    selected = np.argsort(-logits, axis=1, kind="stable")[:, :top_k]
    active_logits = np.take_along_axis(logits, selected, axis=1)
    weights = np.exp(active_logits - active_logits.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    load = np.bincount(selected.ravel(), minlength=logits.shape[1])
    return selected, weights, load


def symmetric_round(values, bits=4):
    """Round to a signed grid using a single absmax scale and dequantize."""
    values = np.asarray(values, dtype=float)
    if bits not in (4, 8) or not np.isfinite(values).all():
        raise ValueError("Use 4 or 8 bits and finite values.")
    max_code = (1 << (bits - 1)) - 1
    scale = np.abs(values).max() / max_code
    if scale == 0:
        return np.zeros_like(values, dtype=int), scale, values.copy()
    codes = np.round(values / scale).astype(int)
    return codes, float(scale), codes * scale


def verify_candidate(target_prob, draft_prob, uniform):
    """One acceptance decision in distribution-preserving speculative sampling."""
    if not 0 <= target_prob <= 1 or not 0 < draft_prob <= 1 or not 0 <= uniform < 1:
        raise ValueError("Probabilities must be valid and the proposed token has q > 0.")
    acceptance = min(1.0, target_prob / draft_prob)
    return acceptance, uniform < acceptance


def filter_examples(rows, min_words=4):
    """Use a transparent toy quality filter followed by exact deduplication."""
    seen = set()
    trace = []
    for row in rows:
        key = " ".join(row.casefold().split())
        if len(key.split()) < min_words:
            reason = "too short"
        elif key in seen:
            reason = "duplicate"
        else:
            reason = "kept"
            seen.add(key)
        trace.append((row, reason))
    return trace


def image_text_scores(image_embedding, text_embeddings):
    """Cosine scores for a fixed image and each candidate caption."""
    image_embedding = np.asarray(image_embedding, dtype=float).reshape(-1)
    text_embeddings = np.asarray(text_embeddings, dtype=float)
    denominator = np.linalg.norm(text_embeddings, axis=1) * np.linalg.norm(image_embedding)
    if not np.isfinite(denominator).all() or np.any(denominator == 0):
        raise ValueError("Cosine similarity requires finite nonzero embeddings.")
    return (text_embeddings @ image_embedding) / denominator


ROUTER_EXAMPLE = np.array([[3.4, 2.1, .4, .2], [.2, 3.1, 2.2, .6],
                           [2.8, 1.1, 2.6, .8], [2.8, .4, .8, 2.4]])


def frontier_panels(chosen=0, top_k=2, context_window=3, *,
                    loss_fn=preference_loss, routing_fn=route_experts, mask_fn=causal_mask):
    from matplotlib.figure import Figure

    loss, logit = loss_fn([-1.8, -2.6], [-2.3, -2.4], int(chosen))
    _, weights, load = routing_fn(ROUTER_EXAMPLE, int(top_k))
    mask = mask_fn(8, int(context_window))
    figure = Figure(figsize=(10, 3.1), layout="constrained")
    bars, attention = figure.subplots(1, 2)
    bars.bar(list("ABCD"), load, color="#167982")
    bars.set(title="Expert load", xlabel="Expert", ylabel="Selected tokens", ylim=(0, 4.5))
    attention.imshow(mask, cmap="Blues", vmin=0, vmax=1)
    attention.set(title="Visible positions", xlabel="Key position", ylabel="Query position")
    metrics = {"DPO loss": loss, "signed logit": logit,
               "active weights": weights.tolist(), "visible cells": int(mask.sum())}
    return figure, metrics


def build_frontier_app(*, loss_fn=preference_loss, routing_fn=route_experts, mask_fn=causal_mask):
    import gradio as gr

    def explore(chosen, top_k, window):
        return frontier_panels(chosen, top_k, window, loss_fn=loss_fn,
                               routing_fn=routing_fn, mask_fn=mask_fn)

    with gr.Blocks(title="생성형 AI 사례 실험실") as app:
        gr.Markdown("# 생성형 AI 사례 실험실\n한 조건을 바꾸고 점수·라우팅·마스크를 비교합니다.")
        vote = gr.Radio([0, 1], value=0, label="선호 응답 (0 또는 1)")
        top = gr.Slider(1, 4, value=2, step=1, label="토큰마다 선택할 전문가 수")
        ctx = gr.Slider(1, 8, value=3, step=1, label="어텐션 창 길이")
        chart, metrics = gr.Plot(), gr.JSON()
        gr.Button("다시 계산", variant="primary").click(
            explore, [vote, top, ctx], [chart, metrics], api_name="explore")
    app.w12a_explore = explore
    return app
