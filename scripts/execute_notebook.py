"""Execute a reviewed notebook in this Python environment; keep outputs private."""

import argparse
import json
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("notebook", type=Path)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = a.notebook.resolve()
    path.relative_to(root)
    nb = nbformat.read(path, as_version=4)
    km = KernelManager(kernel_name="python3")
    km.kernel_spec.argv[0] = sys.executable
    client = NotebookClient(
        nb, km=km, timeout=180, allow_errors=False, resources={"metadata": {"path": str(root)}}
    )
    client.execute()
    dest = root / ".build/executed" / path.relative_to(root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, dest)
    print(
        json.dumps(
            {
                "notebook": str(path.relative_to(root)),
                "python": sys.executable,
                "cells": len(nb.cells),
                "error_outputs": sum(
                    o.output_type == "error" for c in nb.cells for o in c.get("outputs", [])
                ),
                "executed_copy": str(dest),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
