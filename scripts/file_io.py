"""Write complete files before replacement; keep Windows target ACLs intact."""

from __future__ import annotations

import os
from pathlib import Path
import stat
import tempfile


def _replace(source: Path, target: Path) -> None:
    if os.name != "nt" or not target.exists():
        os.replace(source, target)
        return
    import ctypes
    from ctypes import wintypes

    replace = ctypes.WinDLL("kernel32", use_last_error=True).ReplaceFileW
    replace.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.LPCWSTR,
                        wintypes.DWORD, wintypes.LPVOID, wintypes.LPVOID]
    replace.restype = wintypes.BOOL
    # The backup also preserves the original on the documented partial failures.
    backup = source.with_suffix(source.suffix + ".previous")
    if not replace(str(target.resolve()), str(source.resolve()),
                   str(backup.resolve()), 0, None, None):
        error = ctypes.WinError(ctypes.get_last_error())
        if not target.exists() and backup.exists():
            os.replace(backup, target)
        # If recovery was not needed, retain any backup for inspection.
        raise error
    backup.unlink(missing_ok=True)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else None
    fd, name = tempfile.mkstemp(prefix=".write-", suffix=".tmp", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if mode is not None and os.name != "nt":
            temp.chmod(mode)
        _replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))
