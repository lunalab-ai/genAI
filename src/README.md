# 공통 실습 패키지 · luna-genai

W02B에서 시작하는 누적 코드입니다. 노트북과 웹 앱이 같은 모델 인스턴스와 함수를 사용합니다. 예제 데이터도 패키지에 포함되어 별도 업로드가 필요 없습니다.

```bash
pip install "git+https://github.com/lunalab-ai/genAI.git@genai-lab-v0.1.0#subdirectory=src"
python -m luna_genai
```

```python
from luna_genai import LanguageModelLab, load_reviews, build_app
lab = LanguageModelLab()  # 한 번만 로드
print(lab.next_tokens("The movie was"))
print(lab.classify(load_reviews()[0]["review"])["scores"])
app = build_app(lab)
app.launch()  # Colab에서는 share=True
```

Python 3.10–3.13, torch 2.8.0, transformers 5.15.1, gradio 6.26.0. CPU 기본 경로를 검증했으며 CUDA는 사용 가능한 경우 선택됩니다. 첫 실행은 공개 Qwen 모델 약 1 GB를 Hugging Face 캐시에 다운로드합니다. 키는 필요 없습니다. 실패 원인을 출력하며 가상 모델로 대체하지 않습니다. `python -m luna_genai --numeric-only`는 모델 없이 가상 로짓의 확률만 탐색합니다.

| API | 용도 |
|---|---|
| `load_reviews(split)` | 자체 작성 예시/개발/평가 리뷰의 새 복사본 |
| `token_table(text)` | ID, 내부 토큰 표기, 개별 decode |
| `last_logits(text)` | 마지막 위치의 전체 어휘 로짓 |
| `next_tokens(text, n)` | 전체 어휘로 정규화한 상위 n 후보 |
| `generate(text, ...)` | 시작 문장을 제외한 생성 부분 |
| `classify(review, few_shot)` | 두 후보 점수·실제 프롬프트 |
| `filtered_distribution(logits, T, k, p)` | 별도의 작은 수치 실험 |
| `build_app(lab)` | 같은 lab을 재사용하는 Gradio Blocks |

Qwen은 instruction-tuned 챗봇이 아닌 기본 모델입니다. 두 레이블 내 비율은 정답 확신도가 아니며, 소형 합성 데이터는 성능 벤치마크가 아닙니다. 모델 출처: [Qwen2-0.5B](https://huggingface.co/Qwen/Qwen2-0.5B), revision은 `MODEL_REVISION`에 고정합니다. 새 차시에서는 이 API와 기존 tag를 보존하며 기능을 추가합니다.
