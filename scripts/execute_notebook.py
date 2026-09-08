"""Execute a reviewed notebook in this Python environment; keep outputs private."""

import argparse
import json
import sys
import platform
from importlib.metadata import distributions
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("notebook", type=Path)
    p.add_argument("--require-checks", action="store_true")
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = a.notebook.resolve()
    path.relative_to(root)
    nb = nbformat.read(path, as_version=4)
    checks = [c for c in nb.cells if c.cell_type == 'code' and
              'runtime-check' in c.metadata.get('tags', []) and c.source.strip()]
    if a.require_checks and not checks:
        raise ValueError('Missing executable runtime-check cells')
    if any('skip-execution' in c.metadata.get('tags', []) for c in nb.cells):
        raise ValueError('Validation cannot skip cells')
    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    km = KernelManager(kernel_name="python3")
    km.kernel_spec.argv[0] = sys.executable
    client = NotebookClient(
        nb, km=km, timeout=600, allow_errors=False, force_raise_errors=True,
        resources={"metadata": {"path": str(root)}}
    )
    dest = root / ".build/executed" / path.relative_to(root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    status = 'failed'
    try:
        client.execute()
        if any(c.cell_type == 'code' and c.source.strip() and c.execution_count is None for c in nb.cells):
            raise ValueError('Some code cells were not executed')
        status = 'passed'
    finally:
        nbformat.write(nb, dest)
        report = {'status': status, 'python_version': platform.python_version(),
                  'platform': platform.platform(), 'runtime_checks': len(checks),
                  'packages': {d.metadata['Name']: d.version for d in distributions() if d.metadata['Name']}}
        dest.with_suffix('.runtime.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
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
