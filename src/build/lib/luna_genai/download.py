"""Resolve the exact model files before Transformers loads a local directory."""
from __future__ import annotations
import os
from pathlib import Path
import time

MODEL_FILES = ("config.json", "generation_config.json", "tokenizer_config.json",
               "tokenizer.json", "vocab.json", "merges.txt", "model.safetensors")


def configure_hub() -> None:
    """입력 없이 Hugging Face 요청 제한 환경변수를 설정하고 None을 반환한다.
    기존 값이 없을 때 HF_HUB_ETAG_TIMEOUT=30, HF_HUB_DOWNLOAD_TIMEOUT=120을
    넣으며 사용자가 설정한 값은 유지한다. Hub가 상수를 읽기 전에 호출한다.
    다운로드나 모델 로딩은 하지 않는다. 이미 읽힌 설정 변경은 새 프로세스가 필요하다."""
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")


def model_directory(model_id: str, revision: str, *, local_files_only: bool = False,
                    cache_dir: str | Path | None = None) -> Path:
    """고정 revision의 필수 모델 파일을 준비하고 캐시 디렉터리 Path를 반환한다.
    model_id: Hub 저장소 이름. revision: 모델의 고정 커밋 문자열.
    local_files_only: 기본 False; True면 다운로드 없이 완전한 캐시만 허용한다.
    cache_dir: 문자열/Path/None, 기본 None은 Hub 기본 캐시다.
    필수 7개 파일마다 먼저 캐시를 확인하고 없으면 내려받는다. 파일별 실패는
    2초 후 한 번 재시도하며 이미 받은 파일은 유지한다. 부분 캐시는 성공이 아니다.
    오프라인 누락/다운로드 실패는 복구 안내 RuntimeError. 폴더·파일 쓰기가 발생한다.
    예: model_directory(MODEL_ID, MODEL_REVISION, local_files_only=True)."""
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
