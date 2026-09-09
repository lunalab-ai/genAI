"""Check text-model imports in a fresh process, before downloading weights.

Usage: python -m luna_genai.runtime --repair-optional
Only broken optional vision/audio packages are removed when repair is requested.
PyTorch and working optional packages are retained. Restart an already-used
notebook kernel after repairing its installed packages.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import subprocess
import sys


def prepare_text_runtime(*, repair_optional: bool = False) -> list[str]:
    """새 자식 Python에서 텍스트 모델 import를 검사하고 제거한 패키지 이름을 반환한다.
    repair_optional: 기본 False. True인 경우에만 설치되어 있지만 import 실패한
    torchvision/torchaudio/torchcodec를 현재 환경의 pip로 제거한다. 정상 선택 패키지와
    torch는 유지한다. 반환 list[str]는 실제 제거 이름이며 정상 환경은 빈 목록이다.
    각 import 검사 제한은 90초다. 최종 Qwen2ForCausalLM/AutoTokenizer/torch import가
    실패하면 RuntimeError; 시간 초과와 pip 실패도 예외로 전달된다. 모델은 다운로드하지
    않는다. 이미 해당 패키지를 불러온 notebook은 복구 후 런타임 재시작이 필요하다.
    명령: python -m luna_genai.runtime --repair-optional."""
    removed = []
    for name in ("torchvision", "torchaudio", "torchcodec"):
        try:
            importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        probe = subprocess.run([sys.executable, "-c", f"import {name}"],
                               capture_output=True, text=True, errors="replace", timeout=90)
        if probe.returncode == 0:
            continue
        if not repair_optional:
            raise RuntimeError(f"{name} import failed. Run the notebook setup cell in a fresh runtime.")
        print(f"Text-only lab: removing incompatible optional package {name}.", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", name], check=True)
        removed.append(name)
    probe = subprocess.run([sys.executable, "-c",
                            "from transformers import Qwen2ForCausalLM, AutoTokenizer; "
                            "import torch; print('Text model imports OK; torch', torch.__version__)"],
                           capture_output=True, text=True, errors="replace", timeout=90)
    if probe.returncode:
        raise RuntimeError("Text model import failed before download. Restart the runtime and rerun setup.\n"
                           + probe.stderr[-2500:])
    print(probe.stdout.strip(), flush=True)
    return removed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-optional", action="store_true")
    args = parser.parse_args()
    prepare_text_runtime(repair_optional=args.repair_optional)
