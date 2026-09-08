"""Resolve the exact model files before Transformers loads a local directory."""
from __future__ import annotations
import os
from pathlib import Path
import time

MODEL_FILES = ("config.json", "generation_config.json", "tokenizer_config.json",
               "tokenizer.json", "vocab.json", "merges.txt", "model.safetensors")


def configure_hub() -> None:
    """Set timeouts before Transformers imports the Hub's environment constants."""
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")


def model_directory(model_id: str, revision: str, *, local_files_only: bool = False,
                    cache_dir: str | Path | None = None) -> Path:
    """Reuse pinned cached files; fetch missing files with one bounded retry.

    Offline requests never download. A partial cache is not a complete model.
    Errors here are download errors, never model-import/optional-library errors.
    """
    configure_hub()
    import httpx
    from huggingface_hub import hf_hub_download, constants
    from huggingface_hub.errors import LocalEntryNotFoundError
    offline = local_files_only or constants.HF_HUB_OFFLINE
    directory = None
    for filename in MODEL_FILES:
        kwargs = dict(repo_id=model_id, filename=filename, revision=revision, cache_dir=cache_dir)
        try:
            path = hf_hub_download(**kwargs, local_files_only=True)
        except LocalEntryNotFoundError:
            if offline:
                raise RuntimeError(f"오프라인 캐시에 {filename} 파일이 없습니다. "
                                   "인터넷 연결 상태에서 모델을 한 번 준비하세요. "
                                   "HF_HUB_OFFLINE 설정을 바꿨다면 런타임을 다시 시작하세요.") from None
            for attempt in range(2):
                try:
                    path = hf_hub_download(**kwargs, local_files_only=False, etag_timeout=30)
                    break
                except (OSError, httpx.HTTPError) as exc:
                    if attempt == 1:
                        raise RuntimeError(f"Hugging Face 다운로드 실패 ({filename}, {type(exc).__name__}). "
                                           "huggingface.co 연결과 디스크 여유를 확인한 뒤 이 셀을 다시 실행하세요. "
                                           "받은 파일은 캐시에 남습니다.") from None
                    time.sleep(2)
        directory = Path(path).parent
    return directory
