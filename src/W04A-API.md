# W04A 공통 API · luna-genai 0.1.5

고정 태그: `2026-fall-w04a`. PyTorch CPU에서 MNIST를 실제 학습한다. 기존 언어 모델 API는 그대로 유지한다.

## 데이터와 기본 모델

`load_mnist()`가 반환하는 `MNISTSplit`은 float32 0–1 이미지 세 집합과 관찰용 test_labels, split 원본 인덱스, 다운로드 목록을 담는다. 원 train 60000 중 train6000/validation1000을 비중복 선택하고, 별도 원 test10000 중1000을 고정 선택한다. 기본 seed1337이다.

캐시는 `~/.cache/luna-genai/mnist` 또는 `LUNA_MNIST_CACHE`를 사용한다. 최초4파일 약11.6MB, 체크섬 검증,60초 네트워크 제한, 원자적 저장. `offline=True`는 캐시 누락/손상 시 오류를 낸다.

`ConvAutoencoder(16)`은 작은 모델을 초기화한다. `encode(x)`는 `[B,d]`, `decode(z)`는 `[B,1,28,28]`, `model(x)`는 전체 복원이다. 생성자 seed는 torch 난수 상태를 바꾼다.

## 정의 바로가기

- [MNISTSplit 정의](https://github.com/lunalab-ai/genAI/blob/2026-fall-w04a/src/luna_genai/autoencoder.py#L37) — 입력·반환·기본 인자·부작용은 docstring을 함께 읽는다.

- [load_mnist 정의](https://github.com/lunalab-ai/genAI/blob/2026-fall-w04a/src/luna_genai/autoencoder.py#L92) — 입력·반환·기본 인자·부작용은 docstring을 함께 읽는다.

- [ConvAutoencoder 정의](https://github.com/lunalab-ai/genAI/blob/2026-fall-w04a/src/luna_genai/autoencoder.py#L132) — 입력·반환·기본 인자·부작용은 docstring을 함께 읽는다.

- [reconstruct 정의](https://github.com/lunalab-ai/genAI/blob/2026-fall-w04a/src/luna_genai/autoencoder.py#L173) — 입력·반환·기본 인자·부작용은 docstring을 함께 읽는다.

- [reconstruction_mse 정의](https://github.com/lunalab-ai/genAI/blob/2026-fall-w04a/src/luna_genai/autoencoder.py#L187) — 입력·반환·기본 인자·부작용은 docstring을 함께 읽는다.

- [fit_autoencoder 정의](https://github.com/lunalab-ai/genAI/blob/2026-fall-w04a/src/luna_genai/autoencoder.py#L192) — 입력·반환·기본 인자·부작용은 docstring을 함께 읽는다.

- [comparison_rows 정의](https://github.com/lunalab-ai/genAI/blob/2026-fall-w04a/src/luna_genai/autoencoder.py#L231) — 입력·반환·기본 인자·부작용은 docstring을 함께 읽는다.

- [reconstruction_figure 정의](https://github.com/lunalab-ai/genAI/blob/2026-fall-w04a/src/luna_genai/autoencoder.py#L244) — 입력·반환·기본 인자·부작용은 docstring을 함께 읽는다.

- [latent_figure 정의](https://github.com/lunalab-ai/genAI/blob/2026-fall-w04a/src/luna_genai/autoencoder.py#L267) — 입력·반환·기본 인자·부작용은 docstring을 함께 읽는다.

- [reconstruction_callback 정의](https://github.com/lunalab-ai/genAI/blob/2026-fall-w04a/src/luna_genai/autoencoder.py#L288) — 입력·반환·기본 인자·부작용은 docstring을 함께 읽는다.

- [build_autoencoder_app 정의](https://github.com/lunalab-ai/genAI/blob/2026-fall-w04a/src/luna_genai/autoencoder.py#L304) — 입력·반환·기본 인자·부작용은 docstring을 함께 읽는다.

## 학습과 표시

`fit_autoencoder(model, train, validation)`은 기본5epoch, batch128, Adam학습률0.001,seed1337,CPU2threads로 모델을 갱신한다. 라벨과 test를 받지 않는다. epoch별 train_mse/val_mse/seconds 목록을 반환한다. 다시 호출하면 이어서 학습하며 process의 torch thread 수를 바꾼다.

`reconstruct(model, images, batch_size=256)`는 eval/inference_mode의 복원 텐서, `reconstruction_mse`는 전체 픽셀 평균 float를 반환한다. 입력은 비어 있지 않은CPU float32 `[N,1,28,28]`,0–1이다.

`comparison_rows(models,data)`는 train평균 기준선과 모델별 test오차/모수수 목록을 반환한다. `reconstruction_figure(...,count=8)`와 `latent_figure(model,images,labels)`는 Figure를 반환하며 파일을 쓰지 않는다. 후자는2차원 모델만 받는다. 호출자가 display/savefig/close한다.

`reconstruction_callback(index,latent_dim,models,images)`는 uint8원본/uint8복원/설명문자열을 반환한다. 잘못된번호와차원은오류다. `build_autoencoder_app`은 `(Blocks,callback)`만 만들며 다운로드·학습·서버시작을하지않는다.

## 작은 사용 예

```python
from luna_genai.autoencoder import load_mnist, ConvAutoencoder, fit_autoencoder, reconstruction_mse
data = load_mnist()
model = ConvAutoencoder(16)
history = fit_autoencoder(model, data.train, data.validation)
print(reconstruction_mse(model, data.test))
```

새 좌표 탐색 UI는 노트북 활동에서 조립한다. 공개 공통 모듈에 그 활동의 완성 연결 코드를 숨겨 두지 않는다.


웹 앱 시작은 `app.launch(share=False, css=MNIST_CSS)`를 사용한다. MNIST_CSS는28×28 배열을 화면에서224×224로 확대하며 원본 데이터와 손실을 바꾸지 않는다.
