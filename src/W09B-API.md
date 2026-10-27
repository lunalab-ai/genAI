# W09B · 실습 API와 실제 정의

노트북과 앱은 같은 `FineTuningLab` 객체를 사용합니다. 아래 링크는 설치 기준 tag와 같은 구현의 정의 줄입니다. 입력·반환·다운로드·가중치 변경을 확인한 뒤 호출하세요.

```python
from luna_genai.finetuning import load_news, FineTuningLab
news = load_news()
lab = FineTuningLab('classification', device='cpu', seed=42, max_length=64)
state = lab.state()
```

이 예제의 생성자는 모델을 준비하고 `fit`을 호출해야 실제로 가중치를 갱신합니다. 학습/데이터 원문과 모델 체크포인트는 이 공개 패키지에 포함하지 않습니다.

## text_key

[실제 정의 · finetuning.py:34](https://github.com/lunalab-ai/genAI/blob/2026-fall-w09b/src/luna_genai/finetuning.py#L34)

Hash case-folded, whitespace-normalized text; detects exact normalized duplicates.

Input: nonempty str. Output: SHA-256 hex str. No network or mutation.
This does not detect paraphrases or prove semantic independence.

## partition_news

[실제 정의 · finetuning.py:59](https://github.com/lunalab-ai/genAI/blob/2026-fall-w09b/src/luna_genai/finetuning.py#L59)

Deterministically split supplied rows, keeping normalized texts disjoint.

Rows have text:str and label:int in 0..3. Returns NEW train/validation/test
row dictionaries with source IDs. Test comes only from supplied test rows;
all official-test text keys are excluded from train/validation. Every split
is class-balanced. Too few unique rows raises, never silently downsamples.
Example: partition_news(train, test, train_per_class=8, validation_per_class=4).

## load_news

[실제 정의 · finetuning.py:104](https://github.com/lunalab-ai/genAI/blob/2026-fall-w09b/src/luna_genai/finetuning.py#L104)

Automatically download/cache pinned AG News and return balanced split lists.

First call needs Internet (~20 MB compressed source); cache is reused.
Output keys: train, validation, test (list[dict]), provenance (dict).
train/validation derive from official train; test is held out. No Hub login,
upload, synthetic fallback, or local teacher file is required. Defaults yield
256/128/128 rows. Dataset originals are not redistributed with this package.

## classification_metrics

[실제 정의 · finetuning.py:127](https://github.com/lunalab-ai/genAI/blob/2026-fall-w09b/src/luna_genai/finetuning.py#L127)

Accuracy, per-class P/R/F1, macro/weighted F1 and count confusion matrix.

Two equal-length NONEMPTY 1-D integer label sequences, IDs 0..len(names)-1.
Confusion matrix rows=actual, columns=predicted. Zero-support/zero-prediction
fractions are defined as zero. Returns ordinary Python values; no fitting.
Example: classification_metrics([0,1],[0,0], names=('A','B')).

## cross_entropy

[실제 정의 · finetuning.py:157](https://github.com/lunalab-ai/genAI/blob/2026-fall-w09b/src/luna_genai/finetuning.py#L157)

Stable mean multiclass negative log likelihood for small hand-checks.

logits: finite [B,C] real values; labels: [B] integer IDs. No torch/model.
Example: cross_entropy(np.log([[.8,.2]]), [0]) == -log(.8).

## causal_batch

[실제 정의 · finetuning.py:172](https://github.com/lunalab-ai/genAI/blob/2026-fall-w09b/src/luna_genai/finetuning.py#L172)

Build causal-LM tensors [B,L]; keep real EOS even when EOS == PAD.

Append ONE EOS within max_length, pad dynamically, copy input_ids to labels,
mask only attention_mask==0 as -100. The MODEL performs the one-token loss
alignment; this function does NOT shift labels. Mutates no model or text.
Requires a tokenizer with eos_token_id and pad_token_id set.

## target_table

[실제 정의 · finetuning.py:196](https://github.com/lunalab-ai/genAI/blob/2026-fall-w09b/src/luna_genai/finetuning.py#L196)

Show the REAL next-token alignment for one causal batch row.

Returns columns position, input_id/token, next_target_id/token, loss_kept.
Last position has no following target. Does not shift the supplied tensors.

## instruction_record

[실제 정의 · finetuning.py:213](https://github.com/lunalab-ai/genAI/blob/2026-fall-w09b/src/luna_genai/finetuning.py#L213)

Validate and serialize ONE user-authored instruction/input/response record.

Nonempty strings -> dict with those three fields, serialized and status.
Formatting does not train a model, verify its factual label, or prove SFT
quality. The caller supplies the response; no hidden answer is generated.

## FineTuningLab

[실제 정의 · finetuning.py:227](https://github.com/lunalab-ai/genAI/blob/2026-fall-w09b/src/luna_genai/finetuning.py#L227)

A stateful real pretrained model for a small, observable fine-tuning run.

kind='classification' -> DistilBERT 4-class head; 'causal' -> SmolLM-135M.
Defaults: CPU, seed42, max_length64. Initial model download ~269/541 MB;
pinned snapshots and cached files are reused. fit changes weights; predict,
evaluate and generate do not. No login, API call inference or Hub upload.
Example: lab=FineTuningLab('classification'); lab.fit(news['train'],steps=8).

## FineTuningLab.__init__

[실제 정의 · finetuning.py:236](https://github.com/lunalab-ai/genAI/blob/2026-fall-w09b/src/luna_genai/finetuning.py#L236)



## FineTuningLab.batch

[실제 정의 · finetuning.py:276](https://github.com/lunalab-ai/genAI/blob/2026-fall-w09b/src/luna_genai/finetuning.py#L276)

New device tensors; classification labels[B], causal labels[B,L].

Each row has nonempty text; classification additionally has label 0..3.
Tokenization truncates at self.max_length and pads only this batch.
No weight update. Example: batch=lab.batch(news['train'][:2]).

## FineTuningLab.state

[실제 정의 · finetuning.py:300](https://github.com/lunalab-ai/genAI/blob/2026-fall-w09b/src/luna_genai/finetuning.py#L300)

Return reproducibility metadata and parameter counts; never weights/texts.

## FineTuningLab.set_encoder_trainable

[실제 정의 · finetuning.py:308](https://github.com/lunalab-ai/genAI/blob/2026-fall-w09b/src/luna_genai/finetuning.py#L308)

Classification only: change encoder requires_grad, leaving head trainable.

This configures FUTURE training, without updating weights or proving a
performance effect. Return current state/counts. Reset True to restore
full fine-tuning. Calling fit starts a new optimizer for this setting.

## FineTuningLab.fit

[실제 정의 · finetuning.py:321](https://github.com/lunalab-ai/genAI/blob/2026-fall-w09b/src/luna_genai/finetuning.py#L321)

Run REAL AdamW updates; return step loss/gradient and measured changes.

Positive steps<=128, batch_size<=16, lr in (0,.01]. Samples use a fixed
seed; each epoch reshuffles. All current requires_grad parameters train.
Fixed lr, weight_decay=.01, clipping=1.0; no scheduler in this minimum
experiment. Every call creates a new optimizer and CONTINUES current
weights. Compare settings fairly by creating fresh same-seed labs.

## FineTuningLab.predict

[실제 정의 · finetuning.py:382](https://github.com/lunalab-ai/genAI/blob/2026-fall-w09b/src/luna_genai/finetuning.py#L382)

Classify strings -> label/id and four probabilities; no weight change.

Inference is batched by at most eight examples. Scores are softmax model
outputs, not a calibrated guarantee of correctness. No network on call.

## FineTuningLab.evaluate

[실제 정의 · finetuning.py:404](https://github.com/lunalab-ai/genAI/blob/2026-fall-w09b/src/luna_genai/finetuning.py#L404)

Held-out evaluation; caller chooses validation vs final test explicitly.

Classification -> accuracy/F1/confusion. Causal -> token-weighted NLL,
exp(NLL) perplexity and the actual count of nonignored next-token targets.
Does not update weights or automatically tune hyperparameters.

## FineTuningLab.generate

[실제 정의 · finetuning.py:432](https://github.com/lunalab-ai/genAI/blob/2026-fall-w09b/src/luna_genai/finetuning.py#L432)

Actual causal-LM continuation plus settings; does not train or use a chat template.

temperature=0 -> greedy; (0,2] -> sampling with top_p=.9. Max tokens1..96.
RNG state is restored afterwards. Output text excludes the prompt. Empty
decoded continuation raises rather than reporting a fake successful run.

## FineTuningLab.save

[실제 정의 · finetuning.py:464](https://github.com/lunalab-ai/genAI/blob/2026-fall-w09b/src/luna_genai/finetuning.py#L464)

Write model/tokenizer/run metadata locally, NEVER upload to the Hub.

Use a new/owned directory outside public course paths. AG-News-derived
checkpoint redistribution conditions are not asserted by this helper.

## FineTuningLab.load

[실제 정의 · finetuning.py:478](https://github.com/lunalab-ai/genAI/blob/2026-fall-w09b/src/luna_genai/finetuning.py#L478)

Reload a LOCAL saved lab; no new base download and no Hub upload.

## classification_view

[실제 정의 · finetuning_app.py:18](https://github.com/lunalab-ai/genAI/blob/2026-fall-w09b/src/luna_genai/finetuning_app.py#L18)

Prepared classifier + nonempty str -> label, probability DataFrame, metadata, JSON path.

Runs actual inference on the SAME object used in the notebook. Writes one
local export file, without raw dataset rows or model weights. Example:
label, table, state, path = classification_view(lab, 'The team won the final.').

## generation_view

[실제 정의 · finetuning_app.py:31](https://github.com/lunalab-ai/genAI/blob/2026-fall-w09b/src/luna_genai/finetuning_app.py#L31)

Prepared causal lab and widget values -> continuation, actual settings, JSON path.

Widget order MUST match prompt/length/temperature/seed. Reuses the model;
does not pretend that formatting an instruction performs SFT.

## instruction_view

[실제 정의 · finetuning_app.py:44](https://github.com/lunalab-ai/genAI/blob/2026-fall-w09b/src/luna_genai/finetuning_app.py#L44)

Three strings -> formatted training example and explicit NOT-TRAINED status.

## build_finetuning_app

[실제 정의 · finetuning_app.py:50](https://github.com/lunalab-ai/genAI/blob/2026-fall-w09b/src/luna_genai/finetuning_app.py#L50)

Return Gradio Blocks without loading, fitting, launching, or sharing a model.

lab: already prepared FineTuningLab. Optional summary_fn(prediction:str,
observation:str, explanation:str)->str is the learner's own synthesis.
Model callbacks are serialized. Notebook and UI use one shared model state.
Example: app=build_finetuning_app(lab); app.launch(share=True) in Colab.
Share URLs live only while their runtime remains running.
