"""W09B: observable, small REAL fine-tuning experiments, without Hub uploads.

Data remains in the user's Hugging Face cache. Importing downloads nothing.
Full fine-tuning is the default; a short run is not a quality benchmark.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import hashlib
import json
import math
import random
import re
import time
from typing import Any, Literal

import numpy as np
import pandas as pd

DATA_ID = "fancyzhx/ag_news"
DATA_REVISION = "eb185aade064a813bc0b7f42de02595523103ca4"
LABELS = ("World", "Sports", "Business", "Sci/Tech")
MODELS = {
    "classification": ("distilbert/distilbert-base-uncased", "12040accade4e8a0f71eabdb258fecc2e7e948be"),
    "causal": ("HuggingFaceTB/SmolLM-135M", "1d461723eec654e65efdc40cf49301c89c0c92f4"),
}
MODEL_FILES = {
    "classification": ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.txt"),
    "causal": ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json", "merges.txt"),
}


def text_key(text: str) -> str:
    """Hash case-folded, whitespace-normalized text; detects exact normalized duplicates.

    Input: nonempty str. Output: SHA-256 hex str. No network or mutation.
    This does not detect paraphrases or prove semantic independence.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("텍스트는 비어 있지 않은 문자열이어야 합니다.")
    return hashlib.sha256(re.sub(r"\s+", " ", text.strip()).casefold().encode()).hexdigest()


@lru_cache(maxsize=1)
def _raw_news():
    from .download import configure_hub
    configure_hub()
    from datasets import load_dataset, DownloadConfig
    data = load_dataset(DATA_ID, revision=DATA_REVISION, token=False,
                        download_config=DownloadConfig(max_retries=1))
    if (len(data["train"]), len(data["test"])) != (120000, 7600):
        raise ValueError("AG News 원본 행 수가 예상과 다릅니다.")
    if tuple(data["train"].features["label"].names) != LABELS:
        raise ValueError("AG News 라벨 순서를 확인하세요.")
    return data


def partition_news(train: list[dict], test: list[dict], *, train_per_class: int = 64,
                   validation_per_class: int = 32, test_per_class: int = 32,
                   seed: int = 42) -> dict:
    """Deterministically split supplied rows, keeping normalized texts disjoint.

    Rows have text:str and label:int in 0..3. Returns NEW train/validation/test
    row dictionaries with source IDs. Test comes only from supplied test rows;
    all official-test text keys are excluded from train/validation. Every split
    is class-balanced. Too few unique rows raises, never silently downsamples.
    Example: partition_news(train, test, train_per_class=8, validation_per_class=4).
    """
    sizes = (train_per_class, validation_per_class, test_per_class)
    if any(type(n) is not int or n < 1 for n in sizes):
        raise ValueError("각 클래스의 표본 수는 양의 정수여야 합니다.")
    rng = random.Random(seed)
    official_test = {text_key(r["text"]) for r in test}
    used: set[str] = set()
    pools = {"train": [[] for _ in LABELS], "test": [[] for _ in LABELS]}
    # Process test FIRST, preventing official-test documents entering training.
    for split, rows in (("test", test), ("train", train)):
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        for i in indices:
            row = rows[i]
            label = row["label"]
            if type(label) is not int or label not in range(4):
                raise ValueError("라벨은 0..3의 정수여야 합니다.")
            key = text_key(row["text"])
            if key in used or (split == "train" and key in official_test):
                continue
            used.add(key)
            pools[split][label].append({"id": f"{split}:{i}", "text": row["text"], "label": label})
    result = {"train": [], "validation": [], "test": []}
    for label in range(4):
        rows, final = pools["train"][label], pools["test"][label]
        if len(rows) < train_per_class + validation_per_class or len(final) < test_per_class:
            raise ValueError(f"클래스 {label}: 중복 제거 후 표본이 부족합니다.")
        result["train"].extend(rows[:train_per_class])
        result["validation"].extend(rows[train_per_class:train_per_class + validation_per_class])
        result["test"].extend(final[:test_per_class])
    for split in result.values():
        rng.shuffle(split)
    return result


def load_news(*, train_per_class: int = 64, validation_per_class: int = 32,
              test_per_class: int = 32, seed: int = 42) -> dict:
    """Automatically download/cache pinned AG News and return balanced split lists.

    First call needs Internet (~20 MB compressed source); cache is reused.
    Output keys: train, validation, test (list[dict]), provenance (dict).
    train/validation derive from official train; test is held out. No Hub login,
    upload, synthetic fallback, or local teacher file is required. Defaults yield
    256/128/128 rows. Dataset originals are not redistributed with this package.
    """
    raw = _raw_news()
    result = partition_news(list(raw["train"]), list(raw["test"]),
                            train_per_class=train_per_class,
                            validation_per_class=validation_per_class,
                            test_per_class=test_per_class, seed=seed)
    result["provenance"] = {"dataset": DATA_ID, "revision": DATA_REVISION,
                            "mode": "실제 AG News · 공식 다운로드/캐시",
                            "seed": seed, "classes": list(LABELS),
                            "deduplication": "casefold + whitespace; not paraphrase detection",
                            "rows": {k: len(v) for k, v in result.items()}}
    return result


def classification_metrics(actual, predicted, names=LABELS) -> dict:
    """Accuracy, per-class P/R/F1, macro/weighted F1 and count confusion matrix.

    Two equal-length NONEMPTY 1-D integer label sequences, IDs 0..len(names)-1.
    Confusion matrix rows=actual, columns=predicted. Zero-support/zero-prediction
    fractions are defined as zero. Returns ordinary Python values; no fitting.
    Example: classification_metrics([0,1],[0,0], names=('A','B')).
    """
    y, p = np.asarray(actual), np.asarray(predicted)
    n = len(names)
    if y.ndim != 1 or p.shape != y.shape or not y.size:
        raise ValueError("정답과 예측은 길이가 같은 비어 있지 않은 1차원 배열이어야 합니다.")
    if not (np.issubdtype(y.dtype, np.integer) and np.issubdtype(p.dtype, np.integer)):
        raise ValueError("라벨 ID는 정수여야 합니다.")
    if min(y.min(), p.min()) < 0 or max(y.max(), p.max()) >= n:
        raise ValueError("라벨 ID가 클래스 범위를 벗어났습니다.")
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (y, p), 1)
    support, called, tp = cm.sum(1), cm.sum(0), cm.diagonal()
    precision = np.divide(tp, called, out=np.zeros(n), where=called > 0)
    recall = np.divide(tp, support, out=np.zeros(n), where=support > 0)
    f1 = np.divide(2 * tp, support + called, out=np.zeros(n), where=(support + called) > 0)
    return {"accuracy": float(tp.sum() / y.size), "macro_f1": float(f1.mean()),
            "weighted_f1": float(np.dot(support, f1) / y.size),
            "confusion": cm.tolist(), "count": int(y.size),
            "per_class": [{"label": name, "support": int(support[i]),
                           "precision": float(precision[i]), "recall": float(recall[i]),
                           "f1": float(f1[i])} for i, name in enumerate(names)]}


def cross_entropy(logits, labels) -> float:
    """Stable mean multiclass negative log likelihood for small hand-checks.

    logits: finite [B,C] real values; labels: [B] integer IDs. No torch/model.
    Example: cross_entropy(np.log([[.8,.2]]), [0]) == -log(.8).
    """
    z, y = np.asarray(logits, dtype=float), np.asarray(labels)
    if z.ndim != 2 or not z.shape[0] or y.shape != (z.shape[0],) or not np.isfinite(z).all():
        raise ValueError("유한 logits[B,C]와 labels[B]가 필요합니다.")
    if not np.issubdtype(y.dtype, np.integer) or np.any((y < 0) | (y >= z.shape[1])):
        raise ValueError("올바른 정수 라벨 ID가 필요합니다.")
    shifted = z - z.max(axis=1, keepdims=True)
    return float((np.log(np.exp(shifted).sum(1)) - shifted[np.arange(len(y)), y]).mean())


def causal_batch(tokenizer, texts: list[str], *, max_length: int = 64) -> dict:
    """Build causal-LM tensors [B,L]; keep real EOS even when EOS == PAD.

    Append ONE EOS within max_length, pad dynamically, copy input_ids to labels,
    mask only attention_mask==0 as -100. The MODEL performs the one-token loss
    alignment; this function does NOT shift labels. Mutates no model or text.
    Requires a tokenizer with eos_token_id and pad_token_id set.
    """
    import torch
    if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ValueError("비어 있지 않은 텍스트 목록이 필요합니다.")
    if not 4 <= max_length <= 512 or tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError("길이 4..512, EOS와 PAD가 있는 토크나이저가 필요합니다.")
    encoded = tokenizer(texts, truncation=True, max_length=max_length - 1, add_special_tokens=False)
    rows = [{"input_ids": ids + [tokenizer.eos_token_id], "attention_mask": [1] * (len(ids) + 1)}
            for ids in encoded["input_ids"]]
    batch = tokenizer.pad(rows, padding=True, return_tensors="pt")
    batch["labels"] = batch["input_ids"].clone()
    batch["labels"][batch["attention_mask"] == 0] = -100
    if not torch.any(batch["labels"][:, 1:] != -100):
        raise ValueError("예측할 다음 토큰이 없습니다.")
    return dict(batch)


def target_table(batch: dict, tokenizer, row: int = 0) -> pd.DataFrame:
    """Show the REAL next-token alignment for one causal batch row.

    Returns columns position, input_id/token, next_target_id/token, loss_kept.
    Last position has no following target. Does not shift the supplied tensors.
    """
    ids = batch["input_ids"][row].detach().cpu().tolist()
    labels = batch["labels"][row].detach().cpu().tolist()
    return pd.DataFrame([{"position": i, "input_id": v,
                          "input_token": tokenizer.convert_ids_to_tokens(v),
                          "next_target_id": labels[i + 1] if i + 1 < len(ids) else None,
                          "next_target_token": tokenizer.convert_ids_to_tokens(labels[i + 1])
                          if i + 1 < len(ids) and labels[i + 1] >= 0 else "—",
                          "loss_kept": i + 1 < len(ids) and labels[i + 1] != -100}
                         for i, v in enumerate(ids)])


def instruction_record(instruction: str, text: str, response: str) -> dict:
    """Validate and serialize ONE user-authored instruction/input/response record.

    Nonempty strings -> dict with those three fields, serialized and status.
    Formatting does not train a model, verify its factual label, or prove SFT
    quality. The caller supplies the response; no hidden answer is generated.
    """
    if any(not isinstance(x, str) or not x.strip() for x in (instruction, text, response)):
        raise ValueError("지시문·입력·응답을 모두 작성하세요.")
    return {"instruction": instruction.strip(), "input": text.strip(), "response": response.strip(),
            "serialized": f"### Instruction\n{instruction.strip()}\n### Input\n{text.strip()}\n### Response\n{response.strip()}",
            "status": "데이터 형식화만 수행 · SFT 학습하지 않음"}


class FineTuningLab:
    """A stateful real pretrained model for a small, observable fine-tuning run.

    kind='classification' -> DistilBERT 4-class head; 'causal' -> SmolLM-135M.
    Defaults: CPU, seed42, max_length64. Initial model download ~269/541 MB;
    pinned snapshots and cached files are reused. fit changes weights; predict,
    evaluate and generate do not. No login, API call inference or Hub upload.
    Example: lab=FineTuningLab('classification'); lab.fit(news['train'],steps=8).
    """
    def __init__(self, kind: Literal["classification", "causal"] = "classification", *,
                 device: str = "cpu", seed: int = 42, max_length: int = 64,
                 model_path: str | Path | None = None):
        import torch
        from .download import configure_hub
        configure_hub()
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
        if kind not in MODELS or not 4 <= max_length <= 512:
            raise ValueError("kind와 max_length(4..512)를 확인하세요.")
        if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
            raise ValueError("선택한 장치를 사용할 수 없습니다. CPU 또는 할당된 CUDA를 명시하세요.")
        self.kind, self.device, self.seed, self.max_length = kind, device, seed, max_length
        self.model_id, self.revision = MODELS[kind]
        self.steps = 0
        torch.manual_seed(seed)
        if model_path is None:
            # A complete pinned cache can be used without an online metadata request.
            try:
                local = snapshot_download(self.model_id, revision=self.revision,
                                          allow_patterns=MODEL_FILES[kind], local_files_only=True, token=False)
                if not all((Path(local) / f).is_file() for f in MODEL_FILES[kind]):
                    raise FileNotFoundError("incomplete snapshot")
            except (OSError, FileNotFoundError):
                local = snapshot_download(self.model_id, revision=self.revision,
                                          allow_patterns=MODEL_FILES[kind], token=False, max_workers=2)
        else:
            local = str(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(local, local_files_only=True)
        if kind == "causal":
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(local, local_files_only=True)
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                local, num_labels=4, id2label=dict(enumerate(LABELS)),
                label2id={name: i for i, name in enumerate(LABELS)}, local_files_only=True)
        self.model.to(device).eval()
        self.history: list[dict] = []

    def batch(self, rows: list[dict]) -> dict:
        """New device tensors; classification labels[B], causal labels[B,L].

        Each row has nonempty text; classification additionally has label 0..3.
        Tokenization truncates at self.max_length and pads only this batch.
        No weight update. Example: batch=lab.batch(news['train'][:2]).
        """
        import torch
        if not rows:
            raise ValueError("배치가 비었습니다.")
        texts = [r["text"] for r in rows]
        if any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("비어 있지 않은 text가 필요합니다.")
        if self.kind == "causal":
            encoded = causal_batch(self.tokenizer, texts, max_length=self.max_length)
        else:
            labels = [r["label"] for r in rows]
            if any(type(x) is not int or x not in range(4) for x in labels):
                raise ValueError("분류 label은 0..3 정수여야 합니다.")
            encoded = dict(self.tokenizer(texts, padding=True, truncation=True,
                                          max_length=self.max_length, return_tensors="pt"))
            encoded["labels"] = torch.tensor(labels, dtype=torch.long)
        return {k: v.to(self.device) for k, v in encoded.items()}

    def state(self) -> dict:
        """Return reproducibility metadata and parameter counts; never weights/texts."""
        return {"kind": self.kind, "model": self.model_id, "revision": self.revision,
                "device": self.device, "seed": self.seed, "max_length": self.max_length,
                "steps": self.steps, "parameters": sum(p.numel() for p in self.model.parameters()),
                "trainable": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                "mode": "현재 런타임의 실제 모델"}

    def set_encoder_trainable(self, trainable: bool) -> dict:
        """Classification only: change encoder requires_grad, leaving head trainable.

        This configures FUTURE training, without updating weights or proving a
        performance effect. Return current state/counts. Reset True to restore
        full fine-tuning. Calling fit starts a new optimizer for this setting.
        """
        if self.kind != "classification" or type(trainable) is not bool:
            raise ValueError("분류 encoder의 True/False 설정에만 사용합니다.")
        for parameter in self.model.distilbert.parameters():
            parameter.requires_grad_(trainable)
        return self.state()

    def fit(self, rows: list[dict], *, steps: int = 8, batch_size: int = 2,
            learning_rate: float = 5e-5) -> dict:
        """Run REAL AdamW updates; return step loss/gradient and measured changes.

        Positive steps<=128, batch_size<=16, lr in (0,.01]. Samples use a fixed
        seed; each epoch reshuffles. All current requires_grad parameters train.
        Fixed lr, weight_decay=.01, clipping=1.0; no scheduler in this minimum
        experiment. Every call creates a new optimizer and CONTINUES current
        weights. Compare settings fairly by creating fresh same-seed labs.
        """
        import torch
        if (type(steps) is not int or not 1 <= steps <= 128 or type(batch_size) is not int
                or not 1 <= batch_size <= 16 or not math.isfinite(learning_rate) or not 0 < learning_rate <= .01):
            raise ValueError("steps(1..128), batch_size(1..16), learning_rate(0,.01]을 확인하세요.")
        if not rows:
            raise ValueError("학습 데이터가 비었습니다.")
        torch.manual_seed(self.seed)
        rng = random.Random(self.seed)
        parameters = {n: p for n, p in self.model.named_parameters()}
        probe_names = (["distilbert.transformer.layer.0.attention.q_lin.weight", "classifier.weight"]
                       if self.kind == "classification" else ["model.layers.0.self_attn.q_proj.weight"])
        before = {n: parameters[n].detach().flatten()[:1024].cpu().clone() for n in probe_names}
        optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],
                                      lr=learning_rate, weight_decay=.01)
        start = time.perf_counter()
        log: list[dict] = []
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        cursor = 0
        self.model.train()
        try:
            for step in range(steps):
                if cursor >= len(indices):
                    rng.shuffle(indices)
                    cursor = 0
                selected = indices[cursor:cursor + batch_size]
                cursor += len(selected)
                batch = self.batch([rows[i] for i in selected])
                optimizer.zero_grad(set_to_none=True)
                output = self.model(**batch)
                if not torch.isfinite(output.loss):
                    raise ValueError("손실이 유한하지 않습니다. 학습 성공으로 처리하지 않습니다.")
                output.loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                if not torch.isfinite(gradient):
                    raise ValueError("기울기가 유한하지 않습니다.")
                optimizer.step()
                self.steps += 1
                log.append({"step": self.steps, "loss": float(output.loss.detach().cpu()),
                            "gradient_norm_before_clip": float(gradient.detach().cpu()),
                            "batch_size": len(selected), "learning_rate": learning_rate})
        finally:
            optimizer.zero_grad(set_to_none=True)
            self.model.eval()
        changes = {n: float((parameters[n].detach().flatten()[:1024].cpu() - v).norm()) for n, v in before.items()}
        result = {"history": log, "parameter_probe_l2_changes": changes,
                  "probe_scope": "first 1024 elements of named parameters; not full weight norm",
                  "seconds": time.perf_counter() - start, "state": self.state()}
        self.history.extend(log)
        return result

    def predict(self, texts: list[str]) -> list[dict]:
        """Classify strings -> label/id and four probabilities; no weight change.

        Inference is batched by at most eight examples. Scores are softmax model
        outputs, not a calibrated guarantee of correctness. No network on call.
        """
        import torch
        if self.kind != "classification" or not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("분류 모델과 비어 있지 않은 텍스트 목록이 필요합니다.")
        results = []
        self.model.eval()
        with torch.inference_mode():
            for offset in range(0, len(texts), 8):
                encoded = self.tokenizer(texts[offset:offset + 8], padding=True, truncation=True,
                                          max_length=self.max_length, return_tensors="pt").to(self.device)
                probabilities = self.model(**encoded).logits.float().softmax(-1).cpu().numpy()
                for row in probabilities:
                    k = int(row.argmax())
                    results.append({"label": LABELS[k], "label_id": k,
                                    "scores": {name: float(row[i]) for i, name in enumerate(LABELS)}})
        return results

    def evaluate(self, rows: list[dict], *, batch_size: int = 4) -> dict:
        """Held-out evaluation; caller chooses validation vs final test explicitly.

        Classification -> accuracy/F1/confusion. Causal -> token-weighted NLL,
        exp(NLL) perplexity and the actual count of nonignored next-token targets.
        Does not update weights or automatically tune hyperparameters.
        """
        import torch
        if not rows or type(batch_size) is not int or not 1 <= batch_size <= 16:
            raise ValueError("평가 데이터와 배치 크기를 확인하세요.")
        if self.kind == "classification":
            predictions = self.predict([r["text"] for r in rows])
            return classification_metrics([r["label"] for r in rows], [r["label_id"] for r in predictions])
        self.model.eval()
        total, count = 0., 0
        with torch.inference_mode():
            for start in range(0, len(rows), batch_size):
                batch = self.batch(rows[start:start + batch_size])
                n = int((batch["labels"][:, 1:] != -100).sum().cpu())
                loss = float(self.model(**batch).loss.cpu())
                if not math.isfinite(loss) or not n:
                    raise ValueError("유효한 다음 토큰과 유한 평가 손실이 필요합니다.")
                total += loss * n
                count += n
        nll = total / count
        return {"nll": nll, "perplexity": math.exp(nll) if nll < 700 else float("inf"),
                "target_tokens": count, "documents": len(rows), "max_length": self.max_length}

    def generate(self, prompt: str, *, max_new_tokens: int = 24, temperature: float = 0., seed: int = 42) -> dict:
        """Actual causal-LM continuation plus settings; does not train or use a chat template.

        temperature=0 -> greedy; (0,2] -> sampling with top_p=.9. Max tokens1..96.
        RNG state is restored afterwards. Output text excludes the prompt. Empty
        decoded continuation raises rather than reporting a fake successful run.
        """
        import torch
        if self.kind != "causal" or not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("생성 모델과 비어 있지 않은 영어 프롬프트가 필요합니다.")
        if not 1 <= max_new_tokens <= 96 or not 0 <= temperature <= 2:
            raise ValueError("생성 길이1..96, temperature0..2를 사용하세요.")
        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                  max_length=self.max_length).to(self.device)
        options = {"max_new_tokens": int(max_new_tokens), "do_sample": temperature > 0,
                   "pad_token_id": self.tokenizer.pad_token_id}
        if temperature > 0:
            options.update(temperature=float(temperature), top_p=.9)
        self.model.eval()
        start = time.perf_counter()
        devices = [torch.cuda.current_device()] if self.device == "cuda" else []
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            torch.manual_seed(int(seed))
            ids = self.model.generate(**encoded, **options)
        new = ids[0, encoded["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new, skip_special_tokens=True)
        if not text.strip():
            raise ValueError("생성된 표시 가능 텍스트가 비었습니다. 프롬프트·출력 토큰을 확인하세요.")
        return {"text": text, "new_token_ids": new.cpu().tolist(), "prompt": prompt,
                "temperature": temperature, "seed": int(seed), "max_new_tokens": int(max_new_tokens),
                "seconds": time.perf_counter() - start, "state": self.state()}

    def save(self, directory: str | Path) -> Path:
        """Write model/tokenizer/run metadata locally, NEVER upload to the Hub.

        Use a new/owned directory outside public course paths. AG-News-derived
        checkpoint redistribution conditions are not asserted by this helper.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)
        (directory / "lesson_run.json").write_text(json.dumps(self.state(), ensure_ascii=False, indent=2), encoding="utf-8")
        return directory

    @classmethod
    def load(cls, directory: str | Path, *, device: str = "cpu") -> "FineTuningLab":
        """Reload a LOCAL saved lab; no new base download and no Hub upload."""
        directory = Path(directory)
        state = json.loads((directory / "lesson_run.json").read_text(encoding="utf-8"))
        lab = cls(state["kind"], device=device, seed=state["seed"], max_length=state["max_length"], model_path=directory)
        lab.steps = state["steps"]
        return lab
