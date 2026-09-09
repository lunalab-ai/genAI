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
    """고정 Qwen2 기본 언어 모델을 한 번 준비해 여러 실험에서 재사용하는 클래스.
    device: auto/cpu/cuda, 기본 auto. local_files_only: 기본 False인 오프라인 여부.
    생성자에서 다운로드와 모델 로딩이 발생한다. 최초 약 1 GB 파일을 준비하며
    후속 실행은 고정 revision 캐시를 사용한다. CPU는 float32, CUDA는 float16이다.
    학습하는 클래스가 아니며 사전학습된 모델을 eval 모드로 추론한다. 가상 출력으로
    조용히 대체하지 않는다. 저장 상태 tokenizer/model/device/label_ids를 메소드가 재사용한다.
    예: lab=LanguageModelLab(device='cpu'); lab.next_tokens('A small robot')."""
    def __init__(self, device: str = "auto", local_files_only: bool = False):
        """device/local_files_only 설정으로 실제 모델을 준비한다. 반환은 None이다.
        device='auto'이면 사용 가능한 CUDA를 고르고 없으면 CPU, 직접 cpu/cuda도 가능하다.
        local_files_only=False가 기본이며 True일 때 필요한 캐시 누락은 오류다.
        Qwen import → 고정 파일 준비 → tokenizer/모델 로딩을 분리해 실패 원인을 안내한다.
        torch CPU 스레드는 최대 4로 설정한다. 라벨 앞 공백을 포함한 positive/negative가
        각각 한 토큰인지 검사하고 label_ids에 저장한다. 잘못된 설정은 ValueError,
        라이브러리/다운로드/로딩 문제는 단계별 RuntimeError다."""
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
        """text의 토큰 구조를 DataFrame으로 반환한다.
        text: 비어 있지 않은 문자열, 최대 1500자이면서 최대 384토큰. 위반은 ValueError.
        출력 열 위치/ID/내부 표기/개별 decode (repr), 행 수는 입력 토큰 수다.
        특수 토큰을 자동 추가하지 않는다. 조각 단독 decode는 대체 문자를 낼 수 있어
        전체 ID의 decode와 조각 문자열 이어 붙이기는 다르다. 모델 학습 상태는 바꾸지 않는다.
        예: lab.token_table('안녕하세요')."""
        ids = self._inputs(text).input_ids[0].tolist()
        return pd.DataFrame({"위치": range(len(ids)), "ID": ids,
                             "내부 표기": self.tokenizer.convert_ids_to_tokens(ids),
                             "개별 decode (repr)": [repr(self.tokenizer.decode([i])) for i in ids]})

    def last_logits(self, text: str):
        """text 다음 위치의 전체 어휘 로짓을 CPU float Tensor로 반환한다.
        text: 1~1500자, 최대 384토큰 문자열. 출력 모양은 (어휘 수,)이고 확률이 아니다.
        모델의 마지막 입력 위치에서 모든 후보 점수를 꺼낸다. gradient를 저장하지 않는
        추론이며 모델 가중치를 갱신하지 않는다. 예: lab.last_logits('The sky is')."""
        with _LOCK, self.torch.inference_mode():
            return self.model(**self._inputs(text)).logits[0, -1, :].float().cpu()

    def next_tokens(self, text: str, n: int = 8) -> pd.DataFrame:
        """text 다음 토큰 후보 상위 n개와 전체 어휘 확률을 표로 반환한다.
        text: 공통 입력 제한인 1~1500자·최대 384토큰. n: 1~20 정수, 기본 8.
        출력 열 ID/후보 (repr)/로짓/전체 어휘 확률, n행이다. 확률은 표시된 n개가
        아닌 전체 어휘에서 softmax하므로 표의 합은 보통 1보다 작다. 모델은 재학습하지
        않는다. 잘못된 입력/n은 ValueError. 예: lab.next_tokens('Once upon a', n=5)."""
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
        """text 뒤의 새 문자열만 반환한다. 원래 프롬프트는 반환 문자열에서 제외한다.
        text: 비어 있지 않은 최대 1500자·384토큰 문자열.
        strategy: greedy/sample/beam, 기본 greedy. temperature: 양수, 기본 0.8.
        top_k: 어휘 수 이하의 비음수 정수, 기본 0은 해제. top_p: (0,1], 기본 1은 해제.
        seed: 0..2**32-1 정수, 기본 7. max_new_tokens: 1~64 정수, 기본 24.
        temperature/top_k/top_p는 sample에서만 생성에 적용한다. greedy는 매번 argmax,
        beam은 후보 경로 3개와 length_penalty=0의 누적 로그 점수를 사용한다.
        종료 토큰으로 최대 길이 전에 끝날 수 있다. 동일 소프트웨어·장치의 seed 재현성을
        다루며 모든 하드웨어의 동일 출력을 보장하지 않는다. RNG 상태는 호출 후 복구한다.
        잘못된 설정은 ValueError. 모델 가중치는 변경하지 않는다.
        예: lab.generate('A robot', strategy='sample', temperature=0.7, seed=7)."""
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
        """review를 두 레이블 분류용 프롬프트 문자열로 변환한다.
        review: 비어 있지 않은 1~600자 문자열. few_shot: 기본 False; True면 examples
        분할의 고정 예문 두 개만 앞에 넣는다. development/evaluation 정답은 넣지 않는다.
        출력은 Sentiment:로 끝나며 다음 토큰을 채점할 준비가 된다. 추론/학습은 하지 않는다.
        잘못된 리뷰는 ValueError. 예: lab.classification_prompt('I loved it.', True)."""
        if not isinstance(review, str) or not review.strip() or len(review) > 600:
            raise ValueError("리뷰는 1~600자로 입력하세요.")
        prompt = "Classify the movie review as positive or negative.\n"
        if few_shot:
            for row in load_reviews("examples"):
                prompt += f"Review: {row['review']}\nSentiment: {row['label']}\n\n"
        return prompt + f"Review: {review}\nSentiment:"

    def classify(self, review: str, few_shot: bool = False) -> dict:
        """review의 다음 토큰으로 positive/negative를 비교한 dict를 반환한다.
        review: 1~600자 문자열, 최종 프롬프트도 384토큰 제한을 지켜야 한다.
        few_shot: 기본 False, True면 고정 예문 두 개를 추가한다. 모델 학습은 없다.
        반환 키 prediction(선택 문자열), scores(레이블/ID/전체 어휘 확률/두 레이블 내 비율 표),
        label_mass(전체 어휘 중 두 레이블 확률의 합), prompt(실제 사용 문자열).
        두 레이블 내 비율은 그 두 토큰으로 제한해 재정규화한 값이며 보정된 정답 확신도가
        아니다. label_mass가 작아도 두 비율의 합은 1이다. 예: lab.classify('A dull film.')."""
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
