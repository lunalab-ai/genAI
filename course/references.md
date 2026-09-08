# 전체 참고자료

[강의 홈](../README.md)

## 생성형 인공지능과 생성 미디어 입문

[강의 원문](../course/notion/week01/week01_generative_ai_intro.md)

- 주교재: 오마르 산세비에로 외, 장윤경 역, 『핸즈온 생성형 AI』, 1장 “생성 미디어 입문”, 한빛미디어, 2025.
- 교재 제공 실습: `01_introduction.ipynb` — 이미지 생성, 텍스트 분류·생성, MusicGen 오디오 생성의 입문 예시.
- 강의계획서: 「데이터사이언스를 위한 생성형인공지능」, 2026학년도 2학기.
- 교재 코드 저장소: [yk-genai/genaibook](https://github.com/yk-genai/genaibook)
- 강의 실습 저장소: [lunalab-ai/genAI](https://github.com/lunalab-ai/genAI)

> 이 페이지의 도식은 강의용으로 새로 제작한 설명 그림이며, 스캔 교재의 페이지 이미지를 포함하지 않습니다.

https://github.com/lunalab-ai/genAI

https://github.com/yk-genai/genaibook

## Colab 실습 환경 설정 및 생성형 AI 데모 실습

[강의 원문](../course/notion/w02a/w02a-colab-generative-demo.md)

- 기존 1주차 강의·실습의 Colab 설정, seed, 샘플링, 문자 생성, 이미지·오디오 기준선을 재구성했습니다. 따라 하기 절차와 비교 과제, 작은 예시 코드는 이번 차시용으로 새로 작성했습니다.
- [Google Colab 공식 FAQ](https://research.google.com/colaboratory/faq.html): 노트북 저장·공유와 런타임의 차이 확인.
- [DistilGPT2 모델 카드](https://huggingface.co/distilbert/distilgpt2): 선택 시연 모델의 용도·언어·한계.
- [Transformers pipeline 안내](https://huggingface.co/docs/transformers/main/en/pipeline_tutorial): 선택 시연 API 참고.
- 주교재: 『핸즈온 생성형 AI』, 1장 생성 미디어 입문. 이번 차시에서는 추가 교재 원본을 사용하지 않았습니다.

https://huggingface.co/distilbert/distilgpt2

https://huggingface.co/docs/transformers/main/en/pipeline_tutorial

https://research.google.com/colaboratory/faq.html

## 언어 모델의 활용 사례

[강의 원문](../course/notion/w02b/w02b-language-model-applications.md)

- 주교재 『핸즈온 생성형 AI』 2.1 언어 모델의 활용 사례, 인쇄 p.44–67, 그림 2-1~2-8. 원문의 토큰화·확률 예측·생성·제로샷·퓨샷 용어와 흐름을 따른다. 도식 1~5는 개념을 참고해 새 예문·수치·배치로 독립 제작했고 도식 6과 리뷰·앱은 수업용 추가 설계이다.
- [Qwen2-0.5B](https://huggingface.co/Qwen/Qwen2-0.5B), [토큰화 문서](https://huggingface.co/docs/transformers/main/en/tokenizer_summary), [생성 문서](https://huggingface.co/docs/transformers/main/en/generation_strategies): 모델 선택과 API 설명.
- [Holtzman 외, The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751), [Brown 외, Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165): 초록의 nucleus sampling 동기와 가중치 업데이트 없는 few-shot 개념 참고. 이 논문들의 실험 결과를 이번 소형 모델의 성능 보장으로 사용하지 않는다.
- [Transformers 5.15.1 Qwen2 API](https://huggingface.co/docs/transformers/v5.15.1/en/model_doc/qwen2), [Gradio Blocks](https://github.com/gradio-app/gradio/blob/main/guides/03_building-with-blocks/01_blocks-and-event-listeners.md): 설치 버전·이벤트 연결 확인. 외부 자료 확인일 2026-09-08.

https://arxiv.org/abs/1904.09751

https://arxiv.org/abs/2005.14165

https://colab.research.google.com/github/lunalab-ai/genAI/blob/main/notebooks/student/w02b_language_model_applications.ipynb

https://github.com/gradio-app/gradio/blob/main/guides/03_building-with-blocks/01_blocks-and-event-listeners.md

https://github.com/lunalab-ai/genAI/blob/main/course/handouts/w02b-quiz.md

https://github.com/lunalab-ai/genAI/tree/main/src

https://huggingface.co/Qwen/Qwen2-0.5B

https://huggingface.co/docs/transformers/main/en/generation_strategies

https://huggingface.co/docs/transformers/main/en/tokenizer_summary

https://huggingface.co/docs/transformers/v5.15.1/en/model_doc/qwen2

[강의 원문](../course/notion/w02b/w02b-experiment-log.md)
