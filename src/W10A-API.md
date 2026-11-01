# W10A · 공통 API와 실제 정의

고정 패키지0.1.16. importing은 모델을 다운로드하지 않습니다. 아래 생성/학습 객체의 준비 단계에서만 공식 고정revision 모델을 내려받고 완전한 캐시를 재사용합니다. 영어모델/자체가상데이터의 한계를 유지하며 API가 생성답의 사실성을 자동 보장하지 않습니다.

| 모듈 | 입력·반환의 핵심 | 상태 변화/단위/부작용 |
|---|---|---|
| quantization | 유한 수치배열→q/복원/오차, 모수수·bits→bytes/GB/GiB | 모델/네트워크 없음. q는int64관찰배열, NF4/실제압축 아님 |
| peft_lab | 원문/목표헤드라인→[B,L]labels, AdapterLab→실제학습/생성/복원검사 | 생성시모델다운로드, fit은어댑터갱신·로컬Trainer기록, save는adapter파일 |
| text_evaluation | 평균NLL·유효토큰수→NLL/PPL, 후보/참조→교육용겹침 | 순수계산. 토크나이저/단일참조/분모계약 유지 |
| rag | 문자열→실제[문서수,384]임베딩, 질문/k→ID·cosine·문맥·실제답 | 준비시2모델다운로드. 문서교체는재임베딩/인덱스변경, 생성기학습0 |
| peft_app | 같은AdapterLab+입력→기본/어댑터출력·JSON | 다운로드/학습 없음. 일시adapter전환이있으므로요청직렬화 |
| rag_app | 같은RAGLab+질문/k/모드→검색표/문맥/답/JSON | 같은인덱스/생성기 사용. 문서갱신과검색요청직렬화 |


모델 고정revision: base1d461723eec654e65efdc40cf49301c89c0c92f4 / Instructb5f4b394abe60fe0e4d666139e57bfce520302a9 / MiniLM1110a243fdf4706b3f48f1d95db1a4f5529b4d41. 출력토큰 수는max_new_tokens, 입력예산은max_input_tokens이며 문서 수k와 다른 단위입니다. 실제 함수·메소드별 계약과 기본값은 다음 정의/설명을 확인하세요.

## quantization

### [absmax_quantize](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/src/luna_genai/quantization.py#L6)

Finite nonempty array -> symmetric signed integers, reconstruction/errors.

bits: integer 2..8. qmax=2**(bits-1)-1, step=max(abs(x))/qmax.
Rounding uses NumPy ties-to-even, then clipping. Zero array uses step=1.
Returns new arrays; no mutation, model loading or fitted codebook.
Example: absmax_quantize([-3.5, 0, 3.5], 4)['q'] -> [-7,0,7].
Metadata/packing is not included: q is an int64 observation array, NOT an
actually packed 4-bit storage file. step has the same units as values.

### [weight_storage](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/src/luna_genai/quantization.py#L31)

Parameter count, bits/weight -> theoretical WEIGHT-only bytes/GB/GiB.

Does not measure process RAM or include scales, nonquantized layers,
gradients, optimizer, activations, KV cache. No allocation or model load.
Example: weight_storage(70_000_000_000,16)['GB'] == 140.

### [lora_dimensions](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/src/luna_genai/quantization.py#L45)

One dense layer W[d,k], A[r,k], B[d,r] -> shapes/counts, no fitting.

Biases are excluded. rank must be 1..min(d,k). This compares one chosen
layer only; whole-model reduction also depends on selected target_modules.

## text_evaluation

### [perplexity_from_losses](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/src/luna_genai/text_evaluation.py#L8)

Batch mean NLLs and valid NEXT-token counts -> weighted NLL and exp(NLL).

Nonempty equal-length sequences; losses finite >=0, counts positive ints.
Padding/prompt ignored by the caller must not enter counts. Comparable PPL
also needs the same tokenizer, documents, context and loss-mask convention.

### [overlap_metrics](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/src/luna_genai/text_evaluation.py#L33)

Single-reference educational BLEU-N (no smoothing), ROUGE-1 and L.

Lowercase word/punctuation tokenization, max_order 1..4; NOT an official
benchmark pipeline. BLEU uses clipped precision, geometric mean and BP;
any missing order scores zero. ROUGE-L uses dynamic-programming LCS.
Returns intermediate counts so denominators can be inspected. No fitting.

## peft_lab

### [pinned_model_directory](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/src/luna_genai/peft_lab.py#L24)

Download ONLY declared files at an immutable revision; reuse full cache.

No authentication/upload. Sets existing course Hub timeouts. Missing local
files trigger an official download (two workers); failures raise a clear
error, never a fake model. First call writes HF cache; subsequent complete
calls need no online metadata. embedding=True selects BERT vocab.txt.

### [load_headline_data](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/src/luna_genai/peft_lab.py#L51)

Packaged 24 self-authored fictional source documents -> split row lists.

No external data download. 16/4/4 source-document split precedes formatting;
no source_id or normalized input crosses splits. Returns a new dictionary.
English input/response; Korean teaching explanations are in the notebook.

### [headline_prompt](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/src/luna_genai/peft_lab.py#L68)

Nonempty English source -> custom BASE prompt, not a native chat template.

### [response_batch](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/src/luna_genai/peft_lab.py#L74)

Build [B,L] response-only labels for an inspectable causal-LM batch.

Uses separate prompt/response tokenization to make the boundary explicit.
The same pretokenized rows go to SFTTrainer (skip_prepare_dataset=True).
Prompt and padding labels=-100, real response EOS retained, no manual
shift: the model loss aligns once. Truncating away every response raises.
Does not change model/tokenizer. Returns fresh CPU tensors.

### [parameter_digest](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/src/luna_genai/peft_lab.py#L101)

SHA-256 of ALL selected named parameter bytes, no persistent copies.

adapters=False selects every non-LoRA parameter, including frozen norms;
True selects lora_A/B. Quantization metadata buffers are not parameters
and are outside this digest. Temporary per-tensor CPU byte views may copy.
No model update, file save or network request.

### [AdapterLab](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/src/luna_genai/peft_lab.py#L118)

Pinned 135M base + q_proj/v_proj LoRA, optionally real NF4 base storage.

Constructor downloads/loads one model and mutates its trainability.
device: 'cpu' or available 'cuda'; rank4/alpha8/dropout0 defaults. quantized
False is the required float CPU path. True is a separately validated NF4
path; CPU computes float32, supported CUDA chooses BF16 else FP16. No
device_map='auto', Hub uploads, hidden fallback or automatic long training.

#### [AdapterLab.__init__](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/src/luna_genai/peft_lab.py#L127)

```python
__init__(self, *, device: str='cpu', rank: int=4, alpha: int=8, max_length: int=64, seed: int=42, quantized: bool=False)
```

초기화 계약은 클래스 설명을 확인하세요.

#### [AdapterLab.state](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/src/luna_genai/peft_lab.py#L161)

```python
state(self)
```

Return actual counts/dtypes/module configuration; no update/download.

#### [AdapterLab.batch](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/src/luna_genai/peft_lab.py#L176)

```python
batch(self, rows: list[dict])
```

Response-only tensors [B,L] moved to this device; no parameter update.

#### [AdapterLab.evaluate](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/src/luna_genai/peft_lab.py#L180)

```python
evaluate(self, rows: list[dict])
```

Token-weighted response NLL/PPL on caller-selected held-out rows.

#### [AdapterLab.generate](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/src/luna_genai/peft_lab.py#L191)

```python
generate(self, text: str, *, base_only: bool=False, max_new_tokens: int=16)
```

Custom prompt -> actual greedy continuation, reusing the loaded model.

base_only temporarily disables the adapter. It neither loads a second
model nor trains. Calls must be serialized when toggling adapters.

#### [AdapterLab.fit](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/src/luna_genai/peft_lab.py#L209)

```python
fit(self, rows: list[dict], *, steps: int=4, learning_rate: float=0.0005, output_dir: str | Path='.build/w10a-sft')
```

REAL TRL SFTTrainer, response labels=-100 on prompt/PAD, AdamW steps.

steps1..16, batch1, packing=False, no checkpoints/report/Hub. Continues
current adapter with a new optimizer per call. Records first A/B grads,
ALL non-LoRA parameter hashes and adapter changes. Writes trainer state
only under supplied local output_dir. No quality-improvement assertion.

#### [AdapterLab.save_reload_check](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/src/luna_genai/peft_lab.py#L273)

```python
save_reload_check(self, directory: str | Path, row: dict)
```

Save adapter only and compare eval logits after SAME-revision reload.

Creates an adapter directory and a temporary second base model (~RAM).
Float path only; quantized reload/merge needs separate validation.
After comparison discards the second model. No Hub uploads.

#### [AdapterLab.merge_check](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/src/luna_genai/peft_lab.py#L304)

```python
merge_check(self, row: dict)
```

Float ONLY: save/reload a temporary adapter, merge the second model.

This model stays untouched (unmerging can add floating roundoff). Uses
a temporary adapter directory plus a second base model, then removes
temporary files through TemporaryDirectory. Never persists base weights.

## rag

### [load_documents](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/src/luna_genai/rag.py#L15)

Fresh 8 self-authored fictional English lab docs, packaged in the wheel.

id/title/text/ko strings. CC0 teaching data, NOT actual university policies.
No Internet or teacher file path. Model weights download separately.

### [TextEmbedder](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/src/luna_genai/rag.py#L24)

Pinned all-MiniLM-L6-v2, mask-aware mean pool + unit normalization.

Constructor downloads ~91MB weights and loads frozen model. encode(strings)
returns new finite float32 [N,384] unit vectors. CPU, max_length128 default;
truncation is explicit. No training, lexical fallback or remote inference.

#### [TextEmbedder.__init__](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/src/luna_genai/rag.py#L31)

```python
__init__(self, *, max_length: int=128)
```

초기화 계약은 클래스 설명을 확인하세요.

#### [TextEmbedder.encode](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/src/luna_genai/rag.py#L40)

```python
encode(self, texts: list[str])
```

English strings -> [N,384] normalized dense vectors; no cache/update.

### [DenseIndex](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/src/luna_genai/rag.py#L58)

Small exact cosine index; injected encode callable enables toy tests.

documents: nonempty unique id/title/text dictionaries. Constructor embeds
all docs. search returns NEW id/title/text/score rows, not generated answers.
replace_document rebuilds embeddings and content hash; no stale vector reuse.
documents property returns copies; internal vectors are read-only.

#### [DenseIndex.__init__](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/src/luna_genai/rag.py#L66)

```python
__init__(self, documents: list[dict], encode)
```

초기화 계약은 클래스 설명을 확인하세요.

#### [DenseIndex.documents](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/src/luna_genai/rag.py#L84)

```python
documents(self)
```

Copies: editing the returned list never silently mutates the index.

#### [DenseIndex.search](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/src/luna_genai/rag.py#L88)

```python
search(self, query: str, k: int=2)
```

Query text,k1..N -> score-descending exact normalized cosine rows.

score is similarity, not probability/confidence of answer correctness.
Stable document order breaks exact ties. No generation or weight update.

#### [DenseIndex.replace_document](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/src/luna_genai/rag.py#L103)

```python
replace_document(self, document_id: str, text: str)
```

Replace ONE document, re-embed all, return new content revision.

Unknown ID or empty text raises before changing state. Original caller
lists remain unchanged. No generator training/parameter update.

### [RAGLab](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/src/luna_genai/rag.py#L118)

Independent B runtime: real dense index and frozen Instruct generator.

Constructor prepares both models; does not need A adapters/files. CPU only
by default. English model/docs with Korean explanations. No train method.
A 135M generator can ignore good evidence; output failures stay observable.

#### [RAGLab.__init__](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/src/luna_genai/rag.py#L125)

```python
__init__(self, *, documents: list[dict] | None=None, seed: int=42, max_input_tokens: int=384)
```

초기화 계약은 클래스 설명을 확인하세요.

#### [RAGLab.state](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/src/luna_genai/rag.py#L140)

```python
state(self)
```

Actual immutable revisions/index state, no model update/download.

#### [RAGLab.run](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/src/luna_genai/rag.py#L153)

```python
run(self, query: str, *, k: int=2, mode: str='retrieved', max_new_tokens: int=40)
```

Query -> real retrieval, budgeted chat context, actual greedy answer.

mode: retrieved / none / low_rank (forced lowest-score doc, relevance
still judged by humans). k1..N; output cap1..64. If context exceeds
token budget, drop lowest-ranked whole documents; never silently cut
the question/control tokens. Records all retrieved vs actually used IDs.
No retraining/download on calls; serialize calls with document updates.

#### [RAGLab.replace_document](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/src/luna_genai/rag.py#L186)

```python
replace_document(self, document_id: str, text: str)
```

Re-embed changed document set; returns index revisions, trains no LLM.

## peft_app

### [observation_note](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/src/luna_genai/peft_app.py#L8)

Three strings -> plain record; no correctness judgement or model call.

### [adapter_view](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/src/luna_genai/peft_app.py#L13)

Same prompt/greedy settings, base-disabled vs current adapter outputs.

Two actual inference calls, no model loading/training. Serialize requests
because disable_adapter temporarily changes active routing on the SAME lab.

### [quantization_view](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/src/luna_genai/peft_app.py#L24)

Comma-separated finite numbers + bits -> toy quant table and settings.

Does not quantize the loaded model. At most64 values, no state mutation.

### [build_adapter_app](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/src/luna_genai/peft_app.py#L37)

Prepared AdapterLab + optional 3-string->str callback -> Gradio Blocks.

No downloads, training or launch here. note_fn=None uses the explicitly
labeled default recorder. Caller launches, keeps its runtime alive and
closes the server. Same-model inference requests share a concurrency group.

## rag_app

### [rag_view](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/src/luna_genai/rag_app.py#L7)

Actual RAGLab.run -> retrieval table/context/answer/full JSON.

No retraining/download; record includes both retrieved and actually used
document IDs. Does not decide factual correctness from an ID appearing.

### [update_view](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/src/luna_genai/rag_app.py#L18)

Rebuild D01's index vectors; generator weights stay frozen, no fit.

### [build_rag_app](https://github.com/lunalab-ai/genAI/blob/2026-fall-w10a/src/luna_genai/rag_app.py#L23)

Prepared independent RAGLab -> Gradio Blocks, no launch or new models.

Optional three-string note_fn is the student's function. Query/update share
one concurrency group to keep index version and used context consistent.
