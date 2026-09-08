"""One cached model instance shared by notebook experiments and the web app."""
from __future__ import annotations
import os
from threading import RLock
import pandas as pd
from .data import load_reviews
from .download import model_directory, configure_hub

MODEL_ID = "Qwen/Qwen2-0.5B"
MODEL_REVISION = "91d2aff3f957f99e4c74c962f2f408dcc88a18d8"
_LOCK = RLock()

class LanguageModelLab:
    """Inspect a base causal LM. Construct once, then reuse across activities.

    Downloads approximately 1 GB on the first run; reads the pinned HF cache
    first on later runs. Float32 CPU is the reference path; CUDA is optional.
    Never silently replaces the model with synthetic predictions.
    """
    def __init__(self, device: str = "auto", local_files_only: bool = False):
        configure_hub()
        try:
            import torch
            from transformers import Qwen2ForCausalLM, AutoTokenizer
        except (ImportError, RuntimeError, OSError) as exc:
            raise RuntimeError("언어 모델 라이브러리 불러오기 실패: 첫 설치 셀을 다시 실행하세요. "
                               "이미 오류가 난 Colab은 런타임을 삭제하고 새로 연결하세요. "
                               f"다운로드 문제가 아닙니다 ({type(exc).__name__}).") from None
        self.torch = torch
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        self.device = device
        dtype = torch.float16 if device == "cuda" else torch.float32
        directory = model_directory(MODEL_ID, MODEL_REVISION, local_files_only=local_files_only)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
            self.model = Qwen2ForCausalLM.from_pretrained(str(directory), local_files_only=True, dtype=dtype)
            self.model.to(device).eval()
        except (ImportError, OSError, RuntimeError) as exc:
            raise RuntimeError("모델 파일을 확보했지만 로드에 실패했습니다. "
                               "첫 설치 셀과 CPU 메모리 약 4 GB 이상의 여유를 확인하세요. "
                               f"오류 종류: {type(exc).__name__}.") from None
        self.label_ids = {}
        for label in ("positive", "negative"):
            ids = self.tokenizer.encode(" " + label, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError("This scoring lesson requires each space-prefixed label to be one token")
            self.label_ids[label] = ids[0]

    def _inputs(self, text: str):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("비어 있지 않은 문장을 입력하세요.")
        if len(text) > 1500:
            raise ValueError("입력은 1500자 이하로 줄이세요.")
        batch = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        if batch.input_ids.shape[1] > 384:
            raise ValueError("입력은 384토큰 이하로 줄이세요.")
        return batch.to(self.device)

    def token_table(self, text: str) -> pd.DataFrame:
        """Show IDs, internal token strings, and single-ID decoding (repr).

        Byte fragments may decode to a replacement character in isolation.
        decode(all IDs), rather than joining per-ID decodes, restores the text.
        """
        ids = self._inputs(text).input_ids[0].tolist()
        return pd.DataFrame({"위치": range(len(ids)), "ID": ids,
                             "내부 표기": self.tokenizer.convert_ids_to_tokens(ids),
                             "개별 decode (repr)": [repr(self.tokenizer.decode([i])) for i in ids]})

    def last_logits(self, text: str):
        """Return a CPU float tensor of vocabulary scores for the next position."""
        with _LOCK, self.torch.inference_mode():
            return self.model(**self._inputs(text)).logits[0, -1, :].float().cpu()

    def next_tokens(self, text: str, n: int = 8) -> pd.DataFrame:
        """Top n candidates with probabilities normalized over the ENTIRE vocabulary."""
        if int(n) != n or not 1 <= n <= 20:
            raise ValueError("n must be an integer in 1..20")
        z = self.last_logits(text)
        p = self.torch.softmax(z.double(), dim=-1)
        values, ids = self.torch.topk(p, int(n))
        return pd.DataFrame({"ID": ids.tolist(),
                             "후보 (repr)": [repr(self.tokenizer.decode([i])) for i in ids.tolist()],
                             "로짓": z[ids].tolist(), "전체 어휘 확률": values.tolist()})

    def generate(self, text: str, strategy: str = "greedy", temperature: float = 0.8,
                 top_k: int = 0, top_p: float = 1.0, seed: int = 7,
                 max_new_tokens: int = 24) -> str:
        """Return only the continuation. Sampling controls apply to sample only.

        Beam uses three beams with length_penalty=0 (raw accumulated log score).
        Seeds repeat within the same software/device, not across all hardware.
        """
        import math
        if strategy not in {"greedy", "sample", "beam"}:
            raise ValueError("strategy must be greedy, sample, or beam")
        if int(max_new_tokens) != max_new_tokens or not 1 <= max_new_tokens <= 64:
            raise ValueError("max_new_tokens must be an integer in 1..64")
        if int(seed) != seed or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in 0..2**32-1")
        if not math.isfinite(temperature) or temperature <= 0 or not 0 < top_p <= 1:
            raise ValueError("temperature > 0, 0 < top_p <= 1")
        if int(top_k) != top_k or not 0 <= top_k <= self.model.config.vocab_size:
            raise ValueError("invalid top_k")
        batch = self._inputs(text)
        settings = dict(max_new_tokens=int(max_new_tokens), do_sample=strategy == "sample",
                        pad_token_id=self.tokenizer.eos_token_id)
        if strategy == "sample":
            settings.update(temperature=temperature, top_k=int(top_k), top_p=top_p)
        elif strategy == "beam":
            settings.update(num_beams=3, length_penalty=0.0)
        devices = list(range(self.torch.cuda.device_count())) if self.device == "cuda" else []
        with _LOCK, self.torch.inference_mode(), self.torch.random.fork_rng(devices=devices):
            self.torch.manual_seed(int(seed))
            ids = self.model.generate(**batch, **settings)[0, batch.input_ids.shape[1]:]
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def classification_prompt(self, review: str, few_shot: bool = False) -> str:
        """Two fixed examples only; development/evaluation reviews never enter examples."""
        if not isinstance(review, str) or not review.strip() or len(review) > 600:
            raise ValueError("리뷰는 1~600자로 입력하세요.")
        prompt = "Classify the movie review as positive or negative.\n"
        if few_shot:
            for row in load_reviews("examples"):
                prompt += f"Review: {row['review']}\nSentiment: {row['label']}\n\n"
        return prompt + f"Review: {review}\nSentiment:"

    def classify(self, review: str, few_shot: bool = False) -> dict:
        """Compare two next-token labels; conditional score is NOT calibrated confidence."""
        prompt = self.classification_prompt(review, few_shot)
        z = self.last_logits(prompt)
        ids = list(self.label_ids.values())
        raw = self.torch.softmax(z.double(), dim=-1)[ids]
        choice = self.torch.softmax(z[ids].double(), dim=-1)
        labels = list(self.label_ids)
        table = pd.DataFrame({"레이블": labels, "ID": ids,
                              "전체 어휘 확률": raw.tolist(), "두 레이블 내 비율": choice.tolist()})
        return {"prediction": labels[int(choice.argmax())], "scores": table,
                "label_mass": float(raw.sum()), "prompt": prompt}
