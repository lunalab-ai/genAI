"""Thin UI calling shared model/probability methods; no model weights in the repo."""
from .probability import distribution_table

def build_app(lab=None):
    """이미 준비한 lab을 연결한 Gradio Blocks 객체를 반환한다.
    lab: LanguageModelLab 또는 None, 기본 None. 실제 lab을 주면 토큰/생성/분류
    callback이 같은 모델을 재사용한다. None이면 가상 로짓 수치 실험만 제공하고
    화면에 수치 전용 모드를 표시한다. 모델 준비·학습·서버 시작은 하지 않는다.
    호출자가 app.launch(share=True)로 서버를 실행한다. Gradio 설치가 필요하다.
    예: app=build_app(lab). 반환 객체의 close()는 실행 중 서버를 닫는다."""
    import gradio as gr
    def guarded(fn):
        def callback(*args):
            try:
                return fn(*args)
            except (ValueError, RuntimeError) as exc:
                raise gr.Error(str(exc)) from exc
        return callback

    with gr.Blocks(title="언어 모델 실험실", analytics_enabled=False, fill_width=True) as app:
        gr.Markdown("# 언어 모델 실험실\n입력을 바꾸고, 출력을 관찰하고, 한 문장으로 해석하세요.")
        gr.Markdown("실제 Qwen2 기본 모델 · 같은 모델 재사용" if lab else
                    "**수치 실험 전용 모드** · 모델을 연결하지 않았습니다. 아래 확률은 직접 정한 로짓의 계산 결과입니다.")
        if lab is not None:
            with gr.Tab("토큰과 다음 후보"):
                with gr.Row():
                    with gr.Column(min_width=260):
                        text = gr.Textbox(value="A small robot opened the door and", label="문장", lines=3)
                        inspect_btn = gr.Button("토큰·후보 관찰", variant="primary")
                    with gr.Column(min_width=260):
                        restored = gr.Textbox(label="전체 ID를 decode한 문장", interactive=False)
                        tokens = gr.Dataframe(label="토큰 표", interactive=False, wrap=True)
                        candidates = gr.Dataframe(label="다음 후보 · 전체 어휘 확률", interactive=False, wrap=True)
                def inspect_text(value):
                    table = lab.token_table(value)
                    return lab.tokenizer.decode(table["ID"].tolist()), table, lab.next_tokens(value)
                inspect_btn.click(guarded(inspect_text), text, [restored, tokens, candidates], api_name="inspect_text")
            with gr.Tab("생성 비교"):
                with gr.Row():
                    with gr.Column(min_width=260):
                        prompt = gr.Textbox(value="A small robot opened the door and", label="시작 문장", lines=3)
                        strategy = gr.Radio(["greedy", "sample", "beam"], value="greedy", label="생성 방식")
                        temp = gr.Slider(0.2, 2, value=0.8, step=0.1, label="temperature · sample에 적용")
                        k = gr.Slider(0, 50, value=0, step=1, label="top-k · 0이면 필터 해제")
                        p = gr.Slider(0.1, 1, value=1, step=0.05, label="top-p · 1이면 필터 해제")
                        seed = gr.Number(value=7, precision=0, label="seed")
                        length = gr.Slider(1, 64, value=24, step=1, label="최대 새 토큰 수")
                        generate_btn = gr.Button("이어 쓰기", variant="primary")
                    with gr.Column(min_width=260):
                        gr.Markdown("한 번에 한 설정만 바꾸세요. greedy/beam은 샘플링 설정을 사용하지 않습니다. 기본 모델은 반복하거나 엉뚱한 문장을 생성할 수 있습니다.")
                        continuation = gr.Textbox(label="새로 생성한 부분", lines=8, interactive=False)
                generate_btn.click(guarded(lab.generate), [prompt, strategy, temp, k, p, seed, length], continuation, api_name="generate_text")
            with gr.Tab("제로샷·퓨샷 분류"):
                review = gr.Textbox(value="Wonderful acting made this a lovely evening.", label="영화 리뷰", lines=3)
                few = gr.Checkbox(label="고정 예시 2개 추가 (2-shot)", value=False)
                classify_btn = gr.Button("레이블 비교", variant="primary")
                prediction = gr.Textbox(label="더 높은 점수의 레이블", interactive=False)
                scores = gr.Dataframe(label="전체 확률과 두 후보 내 비율", interactive=False, wrap=True)
                gr.Markdown("두 후보 내 비율은 정답일 확률이 아닙니다. 프롬프트·레이블·모델에 따라 달라집니다.")
                used_prompt = gr.Textbox(label="실제로 모델에 넣은 프롬프트", lines=7, interactive=False)
                def classify_review(value, use_examples):
                    result = lab.classify(value, use_examples)
                    return result["prediction"], result["scores"], result["prompt"]
                classify_btn.click(guarded(classify_review), [review, few], [prediction, scores, used_prompt], api_name="classify_review")
        with gr.Tab("수치 실험"):
            gr.Markdown("실제 모델 출력과 구분되는 **가상 로짓 [3, 2, 1, 0]**입니다. 필터 후에는 남긴 확률을 다시 합 1로 맞춥니다.")
            with gr.Row():
                t = gr.Slider(0.2, 2, value=1, step=0.1, label="수치 실험 temperature")
                n = gr.Slider(0, 4, value=0, step=1, label="수치 실험 top-k")
                q = gr.Slider(0.1, 1, value=1, step=0.05, label="수치 실험 top-p")
            calculate_btn = gr.Button("확률 계산")
            table = gr.Dataframe(value=distribution_table(), interactive=False, label="확률 분포", wrap=True)
            calculate_btn.click(guarded(distribution_table), [t, n, q], table, api_name="numeric_distribution")
        gr.Markdown("수업용 짧은 예문으로 실험하세요. Colab 공유 주소는 런타임을 종료하면 사용할 수 없습니다.")
    return app.queue(default_concurrency_limit=1, max_size=12)
