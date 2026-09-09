# 수업 공통 코드 사용 안내

입력·기본값·출력·상태 설명은 구현과 대조한 docstring에서 가져옵니다. 정의 링크는 같은 공개 버전의 실제 선언 줄을 가리킵니다.

## build_app

[src/luna_genai/__init__.py · definition](https://github.com/lunalab-ai/genAI/blob/2026-fall-explained/src/luna_genai/__init__.py#L8)

```python
build_app(lab=None)
```

이미 준비한 lab을 연결한 Gradio Blocks 객체를 반환한다.
lab: LanguageModelLab 또는 None, 기본 None. 실제 lab을 주면 토큰/생성/분류
callback이 같은 모델을 재사용한다. None이면 가상 로짓 수치 실험만 제공하고
화면에 수치 전용 모드를 표시한다. 모델 준비·학습·서버 시작은 하지 않는다.
호출자가 app.launch(share=True)로 서버를 실행한다. Gradio 설치가 필요하다.
예: app=build_app(lab). 반환 객체의 close()는 실행 중 서버를 닫는다.

## main

[src/luna_genai/__main__.py · definition](https://github.com/lunalab-ai/genAI/blob/2026-fall-explained/src/luna_genai/__main__.py#L4)

```python
main()
```

Documentation review required.

## build_app

[src/luna_genai/app.py · definition](https://github.com/lunalab-ai/genAI/blob/2026-fall-explained/src/luna_genai/app.py#L4)

```python
build_app(lab=None)
```

이미 준비한 lab을 연결한 Gradio Blocks 객체를 반환한다.
lab: LanguageModelLab 또는 None, 기본 None. 실제 lab을 주면 토큰/생성/분류
callback이 같은 모델을 재사용한다. None이면 가상 로짓 수치 실험만 제공하고
화면에 수치 전용 모드를 표시한다. 모델 준비·학습·서버 시작은 하지 않는다.
호출자가 app.launch(share=True)로 서버를 실행한다. Gradio 설치가 필요하다.
예: app=build_app(lab). 반환 객체의 close()는 실행 중 서버를 닫는다.

## load_reviews

[src/luna_genai/data.py · definition](https://github.com/lunalab-ai/genAI/blob/2026-fall-explained/src/luna_genai/data.py#L5)

```python
load_reviews(split: str='development')
```

패키지에 포함된 수업용 리뷰 목록을 새 객체로 읽어 반환한다.
split: examples/development/evaluation 문자열, 기본 development.
반환: 각 원소가 review/label 문자열을 가진 dict인 list. label은 positive/negative.
examples만 few-shot 프롬프트에 넣고 development에서 설계를 점검한 뒤
evaluation은 미사용 평가에 남긴다. 작은 독자적 예문이므로 성능 benchmark가 아니다.
파일 업로드·네트워크가 필요 없고 호출자가 수정해도 다음 로딩에는 영향이 없다.
잘못된 split은 ValueError. 예: load_reviews('evaluation').

## configure_hub

[src/luna_genai/download.py · definition](https://github.com/lunalab-ai/genAI/blob/2026-fall-explained/src/luna_genai/download.py#L11)

```python
configure_hub()
```

입력 없이 Hugging Face 요청 제한 환경변수를 설정하고 None을 반환한다.
기존 값이 없을 때 HF_HUB_ETAG_TIMEOUT=30, HF_HUB_DOWNLOAD_TIMEOUT=120을
넣으며 사용자가 설정한 값은 유지한다. Hub가 상수를 읽기 전에 호출한다.
다운로드나 모델 로딩은 하지 않는다. 이미 읽힌 설정 변경은 새 프로세스가 필요하다.

## model_directory

[src/luna_genai/download.py · definition](https://github.com/lunalab-ai/genAI/blob/2026-fall-explained/src/luna_genai/download.py#L20)

```python
model_directory(model_id: str, revision: str, *, local_files_only: bool=False, cache_dir: str | Path | None=None)
```

고정 revision의 필수 모델 파일을 준비하고 캐시 디렉터리 Path를 반환한다.
model_id: Hub 저장소 이름. revision: 모델의 고정 커밋 문자열.
local_files_only: 기본 False; True면 다운로드 없이 완전한 캐시만 허용한다.
cache_dir: 문자열/Path/None, 기본 None은 Hub 기본 캐시다.
필수 7개 파일마다 먼저 캐시를 확인하고 없으면 내려받는다. 파일별 실패는
2초 후 한 번 재시도하며 이미 받은 파일은 유지한다. 부분 캐시는 성공이 아니다.
오프라인 누락/다운로드 실패는 복구 안내 RuntimeError. 폴더·파일 쓰기가 발생한다.
예: model_directory(MODEL_ID, MODEL_REVISION, local_files_only=True).

## LanguageModelLab

[src/luna_genai/model.py · definition](https://github.com/lunalab-ai/genAI/blob/2026-fall-explained/src/luna_genai/model.py#L13)

```python
LanguageModelLab
```

고정 Qwen2 기본 언어 모델을 한 번 준비해 여러 실험에서 재사용하는 클래스.
device: auto/cpu/cuda, 기본 auto. local_files_only: 기본 False인 오프라인 여부.
생성자에서 다운로드와 모델 로딩이 발생한다. 최초 약 1 GB 파일을 준비하며
후속 실행은 고정 revision 캐시를 사용한다. CPU는 float32, CUDA는 float16이다.
학습하는 클래스가 아니며 사전학습된 모델을 eval 모드로 추론한다. 가상 출력으로
조용히 대체하지 않는다. 저장 상태 tokenizer/model/device/label_ids를 메소드가 재사용한다.
예: lab=LanguageModelLab(device='cpu'); lab.next_tokens('A small robot').

## LanguageModelLab.__init__

[src/luna_genai/model.py · definition](https://github.com/lunalab-ai/genAI/blob/2026-fall-explained/src/luna_genai/model.py#L21)

```python
__init__(self, device: str='auto', local_files_only: bool=False)
```

device/local_files_only 설정으로 실제 모델을 준비한다. 반환은 None이다.
device='auto'이면 사용 가능한 CUDA를 고르고 없으면 CPU, 직접 cpu/cuda도 가능하다.
local_files_only=False가 기본이며 True일 때 필요한 캐시 누락은 오류다.
Qwen import → 고정 파일 준비 → tokenizer/모델 로딩을 분리해 실패 원인을 안내한다.
torch CPU 스레드는 최대 4로 설정한다. 라벨 앞 공백을 포함한 positive/negative가
각각 한 토큰인지 검사하고 label_ids에 저장한다. 잘못된 설정은 ValueError,
라이브러리/다운로드/로딩 문제는 단계별 RuntimeError다.

## LanguageModelLab.token_table

[src/luna_genai/model.py · definition](https://github.com/lunalab-ai/genAI/blob/2026-fall-explained/src/luna_genai/model.py#L71)

```python
token_table(self, text: str)
```

text의 토큰 구조를 DataFrame으로 반환한다.
text: 비어 있지 않은 문자열, 최대 1500자이면서 최대 384토큰. 위반은 ValueError.
출력 열 위치/ID/내부 표기/개별 decode (repr), 행 수는 입력 토큰 수다.
특수 토큰을 자동 추가하지 않는다. 조각 단독 decode는 대체 문자를 낼 수 있어
전체 ID의 decode와 조각 문자열 이어 붙이기는 다르다. 모델 학습 상태는 바꾸지 않는다.
예: lab.token_table('안녕하세요').

## LanguageModelLab.last_logits

[src/luna_genai/model.py · definition](https://github.com/lunalab-ai/genAI/blob/2026-fall-explained/src/luna_genai/model.py#L83)

```python
last_logits(self, text: str)
```

text 다음 위치의 전체 어휘 로짓을 CPU float Tensor로 반환한다.
text: 1~1500자, 최대 384토큰 문자열. 출력 모양은 (어휘 수,)이고 확률이 아니다.
모델의 마지막 입력 위치에서 모든 후보 점수를 꺼낸다. gradient를 저장하지 않는
추론이며 모델 가중치를 갱신하지 않는다. 예: lab.last_logits('The sky is').

## LanguageModelLab.next_tokens

[src/luna_genai/model.py · definition](https://github.com/lunalab-ai/genAI/blob/2026-fall-explained/src/luna_genai/model.py#L91)

```python
next_tokens(self, text: str, n: int=8)
```

text 다음 토큰 후보 상위 n개와 전체 어휘 확률을 표로 반환한다.
text: 공통 입력 제한인 1~1500자·최대 384토큰. n: 1~20 정수, 기본 8.
출력 열 ID/후보 (repr)/로짓/전체 어휘 확률, n행이다. 확률은 표시된 n개가
아닌 전체 어휘에서 softmax하므로 표의 합은 보통 1보다 작다. 모델은 재학습하지
않는다. 잘못된 입력/n은 ValueError. 예: lab.next_tokens('Once upon a', n=5).

## LanguageModelLab.generate

[src/luna_genai/model.py · definition](https://github.com/lunalab-ai/genAI/blob/2026-fall-explained/src/luna_genai/model.py#L106)

```python
generate(self, text: str, strategy: str='greedy', temperature: float=0.8, top_k: int=0, top_p: float=1.0, seed: int=7, max_new_tokens: int=24)
```

text 뒤의 새 문자열만 반환한다. 원래 프롬프트는 반환 문자열에서 제외한다.
text: 비어 있지 않은 최대 1500자·384토큰 문자열.
strategy: greedy/sample/beam, 기본 greedy. temperature: 양수, 기본 0.8.
top_k: 어휘 수 이하의 비음수 정수, 기본 0은 해제. top_p: (0,1], 기본 1은 해제.
seed: 0..2**32-1 정수, 기본 7. max_new_tokens: 1~64 정수, 기본 24.
temperature/top_k/top_p는 sample에서만 생성에 적용한다. greedy는 매번 argmax,
beam은 후보 경로 3개와 length_penalty=0의 누적 로그 점수를 사용한다.
종료 토큰으로 최대 길이 전에 끝날 수 있다. 동일 소프트웨어·장치의 seed 재현성을
다루며 모든 하드웨어의 동일 출력을 보장하지 않는다. RNG 상태는 호출 후 복구한다.
잘못된 설정은 ValueError. 모델 가중치는 변경하지 않는다.
예: lab.generate('A robot', strategy='sample', temperature=0.7, seed=7).

## LanguageModelLab.classification_prompt

[src/luna_genai/model.py · definition](https://github.com/lunalab-ai/genAI/blob/2026-fall-explained/src/luna_genai/model.py#L144)

```python
classification_prompt(self, review: str, few_shot: bool=False)
```

review를 두 레이블 분류용 프롬프트 문자열로 변환한다.
review: 비어 있지 않은 1~600자 문자열. few_shot: 기본 False; True면 examples
분할의 고정 예문 두 개만 앞에 넣는다. development/evaluation 정답은 넣지 않는다.
출력은 Sentiment:로 끝나며 다음 토큰을 채점할 준비가 된다. 추론/학습은 하지 않는다.
잘못된 리뷰는 ValueError. 예: lab.classification_prompt('I loved it.', True).

## LanguageModelLab.classify

[src/luna_genai/model.py · definition](https://github.com/lunalab-ai/genAI/blob/2026-fall-explained/src/luna_genai/model.py#L158)

```python
classify(self, review: str, few_shot: bool=False)
```

review의 다음 토큰으로 positive/negative를 비교한 dict를 반환한다.
review: 1~600자 문자열, 최종 프롬프트도 384토큰 제한을 지켜야 한다.
few_shot: 기본 False, True면 고정 예문 두 개를 추가한다. 모델 학습은 없다.
반환 키 prediction(선택 문자열), scores(레이블/ID/전체 어휘 확률/두 레이블 내 비율 표),
label_mass(전체 어휘 중 두 레이블 확률의 합), prompt(실제 사용 문자열).
두 레이블 내 비율은 그 두 토큰으로 제한해 재정규화한 값이며 보정된 정답 확신도가
아니다. label_mass가 작아도 두 비율의 합은 1이다. 예: lab.classify('A dull film.').

## PretrainedTextDemo

[src/luna_genai/pretrained_demo.py · definition](https://github.com/lunalab-ai/genAI/blob/2026-fall-explained/src/luna_genai/pretrained_demo.py#L12)

```python
PretrainedTextDemo
```

고정 DistilGPT2를 CPU에서 한 번 로드하는 선택 시연 객체.

local_files_only: 기본 False. True이면 완전한 기존 캐시만 사용한다.
생성자에서 다운로드·모델 로딩이 발생하며 학습은 하지 않는다.
먼저 python -m luna_genai.runtime --repair-optional로 설치를 검사한다.
예: demo = PretrainedTextDemo(); demo.generate('Data science helps us').

## PretrainedTextDemo.__init__

[src/luna_genai/pretrained_demo.py · definition](https://github.com/lunalab-ai/genAI/blob/2026-fall-explained/src/luna_genai/pretrained_demo.py#L20)

```python
__init__(self, local_files_only: bool=False)
```

local_files_only 설정으로 tokenizer/model을 준비한다. 반환 None.

텍스트 모델 클래스를 직접 로드하며 이미지/오디오 pipeline을 구성하지
않는다. import 문제와 모델 다운로드 문제를 서로 다른 오류로 안내한다.

## PretrainedTextDemo.generate

[src/luna_genai/pretrained_demo.py · definition](https://github.com/lunalab-ai/genAI/blob/2026-fall-explained/src/luna_genai/pretrained_demo.py#L37)

```python
generate(self, prompt: str, *, max_new_tokens: int=40, temperature: float=0.8, top_p: float=1.0, seed: int=42, num_return_sequences: int=1)
```

prompt를 포함한 영어 이어쓰기 문자열 목록을 반환한다.

prompt: 비어 있지 않은 최대 500자 문자열. max_new_tokens: 1~64, 기본40.
temperature: 양수 기본0.8. top_p: (0,1], 기본1. seed: 비음수 정수 기본42.
num_return_sequences: 1~3 정수 기본1. 모두 sampling을 사용한다.
입력은 최대256토큰이며 결과 길이는 종료 토큰으로 더 짧을 수 있다.
반환 list[str] 길이는 num_return_sequences다. 잘못된 설정은 ValueError.
같은 환경에서 seed 비교를 하며 가중치를 갱신하지 않는다.

## filtered_distribution

[src/luna_genai/probability.py · definition](https://github.com/lunalab-ai/genAI/blob/2026-fall-explained/src/luna_genai/probability.py#L5)

```python
filtered_distribution(logits, temperature: float=1.0, top_k: int=0, top_p: float=1.0)
```

유한한 1차원 logits를 확률 배열로 바꾸고 후보 필터를 적용한다.
logits: 길이 V>0 숫자 배열. temperature: 양수, 기본 1.0; 낮을수록 뾰족하다.
top_k: 남길 상위 후보 수 0..V의 정수, 기본 0은 해제.
top_p: 누적확률 기준 (0,1], 기본 1.0은 해제.
softmax(logits/temperature) → top-k → 재정규화 → top-p → 재정규화 순서다.
top-p 경계를 처음 넘는 후보까지 포함한다. 동점은 원래 인덱스 순서다.
반환은 합이 1인 길이 V ndarray이며 제외 후보는 0이다. 입력은 변경하지 않는다.
잘못된 모양/설정은 ValueError. temperature=0은 허용하지 않으며 greedy는 별도 규칙.
예: filtered_distribution([3,2,1,0], top_k=2)는 앞 두 후보만 양수다.

## distribution_table

[src/luna_genai/probability.py · definition](https://github.com/lunalab-ai/genAI/blob/2026-fall-explained/src/luna_genai/probability.py#L38)

```python
distribution_table(temperature: float=1.0, top_k: int=0, top_p: float=1.0)
```

직접 정한 네 로짓 [3,2,1,0]의 필터 전후 확률 표를 반환한다.
temperature=1.0, top_k=0, top_p=1.0은 filtered_distribution과 같은 뜻이다.
반환: 후보/가상 로짓/기본 확률/설정 후 확률 열의 4행 DataFrame.
모델 다운로드·학습·추론 없이 CPU 숫자 연산만 한다. 모델 출력이 아니다.
예: distribution_table(temperature=0.5, top_k=2).

## prepare_text_runtime

[src/luna_genai/runtime.py · definition](https://github.com/lunalab-ai/genAI/blob/2026-fall-explained/src/luna_genai/runtime.py#L15)

```python
prepare_text_runtime(*, repair_optional: bool=False)
```

새 자식 Python에서 텍스트 모델 import를 검사하고 제거한 패키지 이름을 반환한다.
repair_optional: 기본 False. True인 경우에만 설치되어 있지만 import 실패한
torchvision/torchaudio/torchcodec를 현재 환경의 pip로 제거한다. 정상 선택 패키지와
torch는 유지한다. 반환 list[str]는 실제 제거 이름이며 정상 환경은 빈 목록이다.
각 import 검사 제한은 90초다. 최종 Qwen2ForCausalLM/AutoTokenizer/torch import가
실패하면 RuntimeError; 시간 초과와 pip 실패도 예외로 전달된다. 모델은 다운로드하지
않는다. 이미 해당 패키지를 불러온 notebook은 복구 후 런타임 재시작이 필요하다.
명령: python -m luna_genai.runtime --repair-optional.
