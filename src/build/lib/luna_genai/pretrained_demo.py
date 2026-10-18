"""W01/W02A optional English continuation, with the shared text-runtime repair path.

Model/source: https://huggingface.co/distilbert/distilgpt2 (Apache-2.0).
Direct causal-model loading follows the model card; this wrapper is course code.
"""
from .download import configure_hub, model_directory

MODEL_ID = "distilbert/distilgpt2"
MODEL_REVISION = "2290a62682d06624634c1f46a6ad5be0f47f38aa"


class PretrainedTextDemo:
    """고정 DistilGPT2를 CPU에서 한 번 로드하는 선택 시연 객체.

    local_files_only: 기본 False. True이면 완전한 기존 캐시만 사용한다.
    생성자에서 다운로드·모델 로딩이 발생하며 학습은 하지 않는다.
    먼저 python -m luna_genai.runtime --repair-optional로 설치를 검사한다.
    예: demo = PretrainedTextDemo(); demo.generate('Data science helps us').
    """
    def __init__(self, local_files_only: bool = False):
        """local_files_only 설정으로 tokenizer/model을 준비한다. 반환 None.

        텍스트 모델 클래스를 직접 로드하며 이미지/오디오 pipeline을 구성하지
        않는다. import 문제와 모델 다운로드 문제를 서로 다른 오류로 안내한다.
        """
        configure_hub()
        try:
            import torch
            from transformers import GPT2LMHeadModel, AutoTokenizer
        except (ImportError, RuntimeError, OSError) as exc:
            raise RuntimeError("선택 모델 import 실패: 첫 설치 셀의 텍스트 환경 검사를 실행하고 런타임을 재시작하세요.") from exc
        self.torch = torch
        directory = model_directory(MODEL_ID, MODEL_REVISION, local_files_only=local_files_only)
        self.tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
        self.model = GPT2LMHeadModel.from_pretrained(str(directory), local_files_only=True, dtype=torch.float32).eval()

    def generate(self, prompt: str, *, max_new_tokens: int = 40, temperature: float = 0.8,
                 top_p: float = 1.0, seed: int = 42, num_return_sequences: int = 1) -> list[str]:
        """prompt를 포함한 영어 이어쓰기 문자열 목록을 반환한다.

        prompt: 비어 있지 않은 최대 500자 문자열. max_new_tokens: 1~64, 기본40.
        temperature: 양수 기본0.8. top_p: (0,1], 기본1. seed: 비음수 정수 기본42.
        num_return_sequences: 1~3 정수 기본1. 모두 sampling을 사용한다.
        입력은 최대256토큰이며 결과 길이는 종료 토큰으로 더 짧을 수 있다.
        반환 list[str] 길이는 num_return_sequences다. 잘못된 설정은 ValueError.
        같은 환경에서 seed 비교를 하며 가중치를 갱신하지 않는다.
        """
        import math
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 500:
            raise ValueError("prompt must contain 1..500 characters")
        if not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= 64:
            raise ValueError("max_new_tokens must be an integer in 1..64")
        if not isinstance(num_return_sequences, int) or not 1 <= num_return_sequences <= 3:
            raise ValueError("num_return_sequences must be in 1..3")
        if not math.isfinite(temperature) or temperature <= 0 or not 0 < top_p <= 1:
            raise ValueError("temperature > 0 and 0 < top_p <= 1 required")
        if not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in 0..2**32-1")
        batch = self.tokenizer(prompt, return_tensors="pt")
        if batch.input_ids.shape[1] > 256:
            raise ValueError("Shorten prompt to at most 256 tokens")
        with self.torch.inference_mode(), self.torch.random.fork_rng(devices=[]):
            self.torch.manual_seed(seed)
            output = self.model.generate(**batch, max_new_tokens=max_new_tokens,
                do_sample=True, temperature=temperature, top_k=0, top_p=top_p,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)
