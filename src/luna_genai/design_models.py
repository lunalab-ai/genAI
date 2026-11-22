"""Actual frozen E5 retrieval and Qwen2/PEFT CPU execution for W13A."""
from __future__ import annotations

from contextlib import nullcontext
from hashlib import sha256
from pathlib import Path
from time import perf_counter

from .event_design import (
    EMBED_ID, EMBED_REVISION, MODEL_ID, MODEL_REVISION, SETTINGS, digest,
    document_version, format_sources, prompt, response_mask, target,
    training_documents, training_spec,
)


class EvidenceEncoder:
    def __init__(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(min(4, torch.get_num_threads()))
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(EMBED_ID, revision=EMBED_REVISION)
        self.model = AutoModel.from_pretrained(EMBED_ID, revision=EMBED_REVISION).eval()
        self.model.requires_grad_(False)
        self._cache = {}

    def encode(self, texts):
        import torch.nn.functional as functional

        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        with self.torch.inference_mode():
            hidden = self.model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            vectors = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            return functional.normalize(vectors, p=2, dim=1)

    def trace(self, event):
        query = f"query: {event['name']} 행사의 일시, 장소, 참가비, 신청 방법, 정원 안내"
        passages = ["passage: " + format_sources([doc]) for doc in event["docs"]]
        vectors = self.encode([query, *passages])
        scores = (vectors[1:] @ vectors[0]).tolist()
        rows = sorted(
            [{"id": doc["id"], "score": round(score, 6)} for doc, score in zip(event["docs"], scores)],
            key=lambda item: (-item["score"], item["id"]),
        )
        return {
            "event": event["id"], "version": document_version(event), "scores": rows,
            "query": vectors[0].tolist(), "documents": {
                doc["id"]: vec.tolist() for doc, vec in zip(event["docs"], vectors[1:])
            },
        }

    def ranking(self, event):
        key = document_version(event)
        if key not in self._cache:
            self._cache[key] = self.trace(event)["scores"]
        return self._cache[key]


def frozen_hash(model):
    checksum = sha256()
    for name, param in model.named_parameters():
        if "lora_" not in name:
            array = param.detach().cpu().contiguous().numpy()
            checksum.update(memoryview(array).cast("B"))
    return checksum.hexdigest()


class QwenDesign:
    def __init__(self, threads=2):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(max(1, threads))
        self.torch = torch
        start = perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, dtype=torch.float32,
        ).eval()
        self.training = {}
        self.adapter_paths = {}
        self.installed = set()
        self.load_seconds = round(perf_counter() - start, 3)

    def tokens(self, event):
        return {
            "event": event["id"],
            "prefix": self.tokenizer.encode(prompt(event, training_documents(event)), add_special_tokens=False),
            "response": self.tokenizer.encode(target(event), add_special_tokens=False) + [self.tokenizer.eos_token_id],
        }

    def fit(self, rows, directory, *, rank=4, steps=4, mask=response_mask):
        from peft import LoraConfig, get_peft_model

        spec = training_spec(rank, steps)
        if len(rows) < steps or any(row["split"] != "train" for row in rows):
            raise ValueError("훈련 분할의 사례만 학습에 쓰세요.")
        torch = self.torch
        examples = [self.tokens(row) for row in rows[:steps]]
        labels = [mask(x["prefix"], x["response"]) for x in examples]
        if any(value != response_mask(x["prefix"], x["response"]) for x, value in zip(examples, labels)):
            raise ValueError("프롬프트에 손실을 적용했거나 정답 라벨이 빠졌습니다.")
        if any(len(value) > spec["max_length"] for value in labels):
            raise ValueError("입력과 정답이 훈련 토큰 한도를 넘었습니다.")
        if self.installed:
            self.model = self.model.unload()
            self.installed.clear()
        torch.manual_seed(spec["seed"])
        adapter = get_peft_model(self.model, LoraConfig(
            r=rank, lora_alpha=2 * rank, lora_dropout=0, target_modules=spec["target_modules"],
            bias="none", task_type="CAUSAL_LM",
        ))
        trainable = [param for param in adapter.parameters() if param.requires_grad]
        original = [param.detach().clone() for param in trainable]
        base_before = frozen_hash(adapter)
        optimizer = torch.optim.AdamW(trainable, lr=spec["learning_rate"], weight_decay=0)
        adapter.train()
        history = []
        start = perf_counter()
        for step, (example, label) in enumerate(zip(examples, labels), 1):
            input_ids = torch.tensor([example["prefix"] + example["response"]])
            optimizer.zero_grad(set_to_none=True)
            loss = adapter(input_ids=input_ids, labels=torch.tensor([label]), use_cache=False).loss
            if not torch.isfinite(loss):
                raise ValueError("손실이 유한하지 않습니다.")
            loss.backward()
            optimizer.step()
            history.append({"step": step, "event": example["event"], "loss": round(loss.item(), 6),
                            "prompt_tokens": len(example["prefix"]), "response_tokens": len(example["response"])})
        training_seconds = round(perf_counter() - start, 3)
        base_after = frozen_hash(adapter)
        changed = sum(not torch.equal(old, new.detach()) for old, new in zip(original, trainable))
        if base_before != base_after or not changed:
            raise RuntimeError("기본 가중치 또는 어댑터 변화 검사가 실패했습니다.")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        adapter.save_pretrained(directory, safe_serialization=True)
        sha = sha256((directory / "adapter_model.safetensors").read_bytes()).hexdigest()
        trained = {
            "spec": spec, "model": MODEL_ID, "revision": MODEL_REVISION,
            "base_before": base_before, "base_after": base_after,
            "changed_tensors": changed, "trainable": sum(p.numel() for p in trainable),
            "adapter_sha256": sha, "updates": history, "training_seconds": training_seconds,
            "threads": torch.get_num_threads(), "base_load_seconds": self.load_seconds,
        }
        self.model = adapter.unload().eval()
        self.register_adapter(rank, steps, directory, trained)
        return trained

    def register_adapter(self, rank, steps, directory, record):
        key = (rank, steps)
        directory = Path(directory)
        actual = sha256((directory / "adapter_model.safetensors").read_bytes()).hexdigest()
        if actual != record["adapter_sha256"] or record["spec"] != training_spec(rank, steps):
            raise ValueError("저장한 어댑터와 학습 기록이 일치하지 않습니다.")
        if key in self.installed:
            self.model.delete_adapter(self.name(*key))
            self.installed.remove(key)
        self.adapter_paths[key] = directory
        self.training[key] = record

    @staticmethod
    def name(rank, steps):
        return f"rank{rank}_steps{steps}"

    def activate(self, rank, steps):
        from peft import PeftModel

        key = (rank, steps)
        name = self.name(*key)
        if key not in self.adapter_paths:
            raise ValueError("선택한 rank와 업데이트에 맞는 어댑터를 먼저 학습하세요.")
        if key not in self.installed:
            directory = self.adapter_paths[key]
            if self.installed:
                self.model.load_adapter(directory, adapter_name=name, is_trainable=False)
            else:
                self.model = PeftModel.from_pretrained(self.model, directory, adapter_name=name, is_trainable=False)
            self.installed.add(key)
        self.model.set_adapter(name)
        self.model.eval()
        return self.training[key]["adapter_sha256"]

    def first_token(self, example, rank=0, steps=4):
        from math import log

        torch = self.torch
        if rank:
            self.activate(rank, steps)
        context = self.model.disable_adapter() if self.installed and not rank else nullcontext()
        target_id = example["response"][0]
        with context, torch.inference_mode():
            logits = self.model(input_ids=torch.tensor([example["prefix"]]), use_cache=False).logits[0, -1]
            probabilities = torch.softmax(logits.float(), dim=-1)
        probability = float(probabilities[target_id])
        return {
            "target_id": target_id, "target_text": self.tokenizer.decode([target_id]),
            "probability": round(probability, 9), "loss": round(-log(probability), 6),
            "preferred_id": int(probabilities.argmax()),
        }

    def generate(self, text, rank, steps):
        return self.generate_batch([text], rank, steps)[0]

    def generate_batch(self, texts, rank, steps):
        from transformers import GenerationConfig

        torch = self.torch
        adapter = self.activate(rank, steps) if rank else None
        frozen = self.model.disable_adapter() if self.installed and not rank else nullcontext()
        input_ids = [self.tokenizer.encode(text, add_special_tokens=False) for text in texts]
        width = max(map(len, input_ids))
        eos = self.tokenizer.eos_token_id
        padded = [[eos] * (width - len(ids)) + ids for ids in input_ids]
        masks = [[0] * (width - len(ids)) + [1] * len(ids) for ids in input_ids]
        cfg = GenerationConfig(**SETTINGS, pad_token_id=eos, eos_token_id=eos, use_cache=True)
        with frozen, torch.inference_mode():
            output = self.model.generate(
                input_ids=torch.tensor(padded), attention_mask=torch.tensor(masks), generation_config=cfg,
            )
        results = []
        for text, input_row, output_row in zip(texts, input_ids, output):
            ids = output_row[width:].tolist()
            if eos in ids:
                ids = ids[:ids.index(eos)]
            results.append({
                "raw": self.tokenizer.decode(ids, skip_special_tokens=True),
                "input_tokens": len(input_row), "output_tokens": len(ids),
                "prompt_sha256": digest(text), "adapter_sha256": adapter,
                "adapter_reloaded": bool(rank),
            })
        return results
