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
    """Verify optional binary extensions and Qwen imports; return removed names."""
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
