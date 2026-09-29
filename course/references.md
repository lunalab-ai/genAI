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

https://colab.research.google.com/drive/1zd4AEMV3FW83CCOLKOQcDv5C8CgZyg5-?usp=sharing

https://colab.research.google.com/github/lunalab-ai/genAI/blob/2026-fall-explained/notebooks/student/week01_generative_ai_intro.ipynb

https://colab.research.google.com/github/lunalab-ai/genAI/blob/main/notebooks/student/week01_generative_ai_intro.ipynb

https://github.com/lunalab-ai/genAI

https://github.com/lunalab-ai/genAI/blob/2026-fall-explained/src/API.md

https://github.com/lunalab-ai/genAI/blob/2026-fall-explained/src/luna_genai/pretrained_demo.py#L12

https://github.com/lunalab-ai/genAI/blob/2026-fall-explained/src/luna_genai/pretrained_demo.py#L37

https://github.com/lunalab-ai/genAI/blob/2026-fall-explained/src/luna_genai/runtime.py#L15

https://github.com/yk-genai/genaibook

## Colab 실습 환경 설정 및 생성형 AI 데모 실습

[강의 원문](../course/notion/w02a/w02a-colab-generative-demo.md)

- 기존 1주차 강의·실습의 Colab 설정, seed, 샘플링, 문자 생성, 이미지·오디오 기준선을 재구성했습니다. 따라 하기 절차와 비교 과제, 작은 예시 코드는 이번 차시용으로 새로 작성했습니다.
- [Google Colab 공식 FAQ](https://research.google.com/colaboratory/faq.html): 노트북 저장·공유와 런타임의 차이 확인.
- [DistilGPT2 모델 카드](https://huggingface.co/distilbert/distilgpt2): 선택 시연 모델의 용도·언어·한계.
- [Transformers pipeline 안내](https://huggingface.co/docs/transformers/main/en/pipeline_tutorial): 선택 시연 API 참고.
- 주교재: 『핸즈온 생성형 AI』, 1장 생성 미디어 입문. 이번 차시에서는 추가 교재 원본을 사용하지 않았습니다.

https://github.com/lunalab-ai/genAI/blob/2026-fall-explained/src/API.md

https://github.com/lunalab-ai/genAI/blob/2026-fall-explained/src/luna_genai/pretrained_demo.py#L12

https://github.com/lunalab-ai/genAI/blob/2026-fall-explained/src/luna_genai/pretrained_demo.py#L37

https://github.com/lunalab-ai/genAI/blob/2026-fall-explained/src/luna_genai/runtime.py#L15

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

https://github.com/lunalab-ai/genAI/blob/2026-fall-explained/src/API.md

https://github.com/lunalab-ai/genAI/blob/2026-fall-explained/src/luna_genai/__init__.py#L8

https://github.com/lunalab-ai/genAI/blob/2026-fall-explained/src/luna_genai/data.py#L5

https://github.com/lunalab-ai/genAI/blob/2026-fall-explained/src/luna_genai/model.py#L106

https://github.com/lunalab-ai/genAI/blob/2026-fall-explained/src/luna_genai/model.py#L13

https://github.com/lunalab-ai/genAI/blob/2026-fall-explained/src/luna_genai/model.py#L158

https://github.com/lunalab-ai/genAI/blob/2026-fall-explained/src/luna_genai/pretrained_demo.py#L12

https://github.com/lunalab-ai/genAI/blob/main/course/handouts/w02b-quiz.md

https://github.com/lunalab-ai/genAI/tree/main/src

https://huggingface.co/Qwen/Qwen2-0.5B

https://huggingface.co/docs/transformers/main/en/generation_strategies

https://huggingface.co/docs/transformers/main/en/tokenizer_summary

https://huggingface.co/docs/transformers/v5.15.1/en/model_doc/qwen2

[강의 원문](../course/notion/w02b/w02b-experiment-log.md)

## 트랜스포머 블록과 사전학습: 모델 계보에서 텍스트 생성 웹앱까지

[강의 원문](../course/notion/w03a/w03a-transformers.md)

https://aclanthology.org/N19-1423/

https://arxiv.org/html/1706.03762v7

https://colab.research.google.com/github/lunalab-ai/genAI/blob/2026-fall-w03a/notebooks/student/w03a_transformers.ipynb

https://github.com/lunalab-ai/genAI/blob/2026-fall-w03a/src/W03A-API.md

https://huggingface.co/Qwen/Qwen2-0.5B

https://huggingface.co/Qwen/Qwen2-0.5B/blob/91d2aff3f957f99e4c74c962f2f408dcc88a18d8/config.json

https://huggingface.co/docs/transformers/main/en/cache_explanation

https://huggingface.co/docs/transformers/v5.15.1/en/model_doc/qwen2

https://huggingface.co/learn/llm-course/chapter1/6

https://jmlr.org/papers/v21/20-074.html

https://www.gradio.app/guides/blocks-and-event-listeners

[강의 원문](../course/notion/w03a/w03a-experiment-log.md)

## 트랜스포머 구조 다시 보기: 블록 해부와 단계별 텍스트 생성 실습

[강의 원문](../course/notion/w03b/w03b-transformer-walkthrough.md)

https://arxiv.org/html/1706.03762v7

https://arxiv.org/html/1706.03762v7/Figures/ModalNet-19.png

https://arxiv.org/html/1706.03762v7/Figures/ModalNet-20.png

https://arxiv.org/html/1706.03762v7/Figures/ModalNet-21.png

https://colab.research.google.com/github/lunalab-ai/genAI/blob/2026-fall-w03b/notebooks/student/w03b_transformer_walkthrough.ipynb

https://github.com/lunalab-ai/genAI/blob/main/course/handouts/w03b-quiz.md

https://github.com/lunalab-ai/genAI/blob/main/course/notion/w03a/w03a-transformers.md

https://huggingface.co/Qwen/Qwen2-0.5B/tree/91d2aff3f957f99e4c74c962f2f408dcc88a18d8

https://huggingface.co/learn/llm-course/chapter1/6

[강의 원문](../course/notion/w03b/w03b-workbook.md)

## 오토인코더: 압축과 복원으로 배우는 잠재 표현

[강의 원문](../course/notion/w04a/w04a-autoencoders.md)

주교재의3.1 설명·용어·구조를 읽고 독립 작성했다. 교재 스캔이나 출판사 코드를 공개 자료에 포함하지 않았다. 수업 그림은 새 도식 및 실제 수업 구현의 계산 결과다. 실행 설정은seed1337, train6,000/validation1,000/test1,000, Adam학습률0.001, batch128,5epoch, CPU2threads다. 처음 관측한 데이터 다운로드와 두 모델 학습·평가 전체는 해당 PC에서약15.6초였으며 학생 환경의 시간 보장이 아니다.

- [TensorFlow: Intro to Autoencoders](https://www.tensorflow.org/tutorials/generative/autoencoder) — 기본 구조·정규화·복원의 개념 참고.
- [PyTorch: MSELoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.modules.loss.MSELoss.html) — 전체 원소 평균 정의. 온라인 문서는2.14, 실제 수업 검증 런타임은2.8이며 사용한 핵심 API를 실제 실행으로 확인한다.
- [PyTorch: ConvTranspose2d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.modules.conv.ConvTranspose2d.html) — 출력 shape와 역함수에 대한 주의.
- [torchvision MNIST 구현](https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py) — 공개 미러와 파일별 체크섬. 실습은 torchvision을 import하지 않는 독립 로더다.
- [MNIST 데이터 카드](https://huggingface.co/datasets/ylecun/mnist) — 이미지 크기·원래 split·데이터 설명.
- [Gradio: Blocks and Event Listeners](https://www.gradio.app/guides/blocks-and-event-listeners) — 화면 구성과 이벤트의 입력·출력 연결.

자료 확인일:2026-09-20. 별도 성적 반영 과제나 제출 기한은 없다. 실험 기록과 점검 질문을 복습에 활용한다.

https://colab.research.google.com/github/lunalab-ai/genAI/blob/2026-fall-w04a/notebooks/student/w04a_autoencoders.ipynb

https://docs.pytorch.org/docs/2.14/generated/torch.nn.modules.conv.ConvTranspose2d.html

https://docs.pytorch.org/docs/2.14/generated/torch.nn.modules.loss.MSELoss.html

https://github.com/lunalab-ai/genAI/blob/2026-fall-w04a/src/W04A-API.md

https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py

https://huggingface.co/datasets/ylecun/mnist

https://www.gradio.app/guides/blocks-and-event-listeners

https://www.tensorflow.org/tutorials/generative/autoencoder

[강의 원문](../course/notion/w04a/w04a-workbook.md)

https://colab.research.google.com/github/lunalab-ai/genAI/blob/2026-fall-w04a/notebooks/student/w04a_autoencoders.ipynb

## 이미지 표현의 세 가지 목적: 오토인코더·VAE·CLIP과 의미 기반 이미지 검색

[강의 원문](../course/notion/w05a/w05a-image-representations.md)

주교재의 용어·그림·전개를 바탕으로 수업 예제와 설명을 독립 작성했다. 원본 그림의 출처는 각 캡션에 표시했다. 교재 전체 스캔과 출판사 예제 원본은 학생 저장소에 포함하지 않는다. 실측은 Python 3.12.10, PyTorch 2.8 CPU, 위에 명시한 분할과 설정에서 얻었다. 실제 실행의 상세 조건은 [모델 기록](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/assets/w05a/models.json)과 [공통 API](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/W05A-API.md)에 제공한다.

- [Kingma & Welling, Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114): 변분 하한·재매개변수화의 이론적 근거.
- [Radford et al., Learning Transferable Visual Models From Natural Language Supervision](https://proceedings.mlr.press/v139/radford21a.html): CLIP의 이미지–텍스트 대조 학습과 제로샷 전이.
- [OpenAI CLIP 공식 구현](https://github.com/openai/CLIP): 인코딩·정규화·유사도 계산의 원 구현.
- [Transformers 5.15.1 CLIP 문서](https://huggingface.co/docs/transformers/v5.15.1/en/model_doc/clip): 현재 실습 버전의 반환 형식과 전처리.
- [ViT-B/32 모델 카드](https://huggingface.co/openai/clip-vit-base-patch32): 모델의 사용 범위와 언어·평가 한계.
- [PyTorch 2.8 MSELoss](https://docs.pytorch.org/docs/2.8/generated/torch.nn.MSELoss.html): reduction과 평균 정의.
- [Gradio Blocks와 이벤트](https://www.gradio.app/guides/blocks-and-event-listeners): 입력·callback·출력 연결.

이 수업에서는 별도 성적 반영 과제나 제출 기한을 추가하지 않는다. 기록지와 점검 질문으로 관찰과 설명을 연습한다.

https://arxiv.org/abs/1312.6114

https://colab.research.google.com/github/lunalab-ai/genAI/blob/2026-fall-w05a-v2/notebooks/student/w05a_ae_vae.ipynb

https://colab.research.google.com/github/lunalab-ai/genAI/blob/2026-fall-w05a-v2/notebooks/student/w05a_clip_search.ipynb

https://docs.pytorch.org/docs/2.8/generated/torch.nn.MSELoss.html

https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/GALLERY-CREDITS.md

https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/W05A-API.md

https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/assets/w05a/models.json

https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/embedding_app.py#L12

https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/embedding_app.py#L28

https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/embedding_app.py#L42

https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/representation.py#L115

https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/representation.py#L190

https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/representation.py#L21

https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/representation.py#L249

https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/representation.py#L42

https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/representation.py#L46

https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/representation.py#L77

https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/representation.py#L97

https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/semantic_search.py#L106

https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/semantic_search.py#L125

https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/semantic_search.py#L137

https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/semantic_search.py#L159

https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/semantic_search.py#L176

https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/semantic_search.py#L23

https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/semantic_search.py#L36

https://github.com/lunalab-ai/genAI/blob/2026-fall-w05a-v2/src/luna_genai/semantic_search.py#L65

https://github.com/openai/CLIP

https://huggingface.co/docs/transformers/v5.15.1/en/model_doc/clip

https://huggingface.co/openai/clip-vit-base-patch32

https://proceedings.mlr.press/v139/radford21a.html

https://www.gradio.app/guides/blocks-and-event-listeners

[강의 원문](../course/notion/w05a/w05a-workbook.md)

## CLIP 의미 검색과 확산 모델의 원리·학습

[강의 원문](../course/notion/w05b/w05b-clip-search.md)

- 주교재 『핸즈온 생성형 AI』 3.3 CLIP, 3.5 의미 기반 이미지 검색. 교재 그림의 필요한 부분은 출처를 표시해 직접 사용했다.
- [Radford et al., Learning Transferable Visual Models From Natural Language Supervision](https://proceedings.mlr.press/v139/radford21a.html): 이미지–문장 대조 사전학습의 원논문.
- [Hugging Face CLIPModel 공식 문서](https://huggingface.co/docs/transformers/model_doc/clip): 모델 입력·출력·투영 API.
- [수업 API와 실제 소스 선언 줄](https://github.com/lunalab-ai/genAI/blob/2026-fall-w05b/src/W05B-API.md): 설치와 같은 고정 태그.
- 수계산·40장 후보 실험·후보 의존성·실패 분석·웹 앱은 수업에서 추가한 설명이다. 2026-09-29 실제 CPU 추론값이며 모델이나 후보를 바꾸면 결과도 달라질 수 있다.

https://colab.research.google.com/github/lunalab-ai/genAI/blob/2026-fall-w05b/notebooks/student/w05b_clip_search.ipynb

https://github.com/lunalab-ai/genAI/blob/2026-fall-w05b/src/GALLERY-CREDITS.md

https://github.com/lunalab-ai/genAI/blob/2026-fall-w05b/src/W05B-API.md

https://huggingface.co/docs/transformers/model_doc/clip

https://proceedings.mlr.press/v139/radford21a.html

[강의 원문](../course/notion/w05b/w05b-diffusion.md)

- 주교재 『핸즈온 생성형 AI』 4.1 반복 정제, 4.2.1–4.2.6 데이터·노이즈·모델·학습·샘플링·평가, 156–172쪽 중4.3 시작 전까지. 그림4-1·4-3은 선별한 원본을 직접 사용했다.
- [Ho, Jain & Abbeel, Denoising Diffusion Probabilistic Models](https://arxiv.org/html/2006.11239v2): 전방 Gaussian 과정, 직접 노이즈 식, epsilon 예측과 역과정의 수학적 보강.
- [DDPM 저자 프로젝트](https://hojonathanho.github.io/diffusion/): 학습·샘플링 알고리즘과 반복 생성의 맥락.
- [Diffusers DDPMScheduler 공식 문서](https://huggingface.co/docs/diffusers/api/schedulers/ddpm): add_noise, step, pred_original_sample, prev_sample.
- 작은 MNIST U-Net·실제 학습 기록·평가 분리·웹 앱은 수업용 추가 구현이다. 교재의 큰 나비 모델과 데이터·크기·학습량이 다름을 명시했다.

https://arxiv.org/html/2006.11239v2

https://colab.research.google.com/github/lunalab-ai/genAI/blob/2026-fall-w05b/notebooks/student/w05b_diffusion.ipynb

https://github.com/lunalab-ai/genAI/blob/2026-fall-w05b/src/W05B-API.md

https://hojonathanho.github.io/diffusion/

https://huggingface.co/docs/diffusers/api/schedulers/ddpm

[강의 원문](../course/notion/w05b/w05b-workbook.md)

https://colab.research.google.com/github/lunalab-ai/genAI/blob/2026-fall-w05b/notebooks/student/w05b_clip_search.ipynb

https://colab.research.google.com/github/lunalab-ai/genAI/blob/2026-fall-w05b/notebooks/student/w05b_diffusion.ipynb
