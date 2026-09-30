"""Shared readers for the released data files."""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import IO


def jsonl_path(path: Path) -> Path:
    """Return the file to read: the .jsonl, or the .jsonl.gz shipped in its place."""

    if path.exists():
        return path
    packed = path.with_name(path.name + ".gz")
    if packed.exists():
        return packed
    raise FileNotFoundError(path)


def open_jsonl(path: Path) -> IO[str]:
    resolved = jsonl_path(path)
    if resolved.suffix == ".gz":
        return gzip.open(resolved, "rt", encoding="utf-8")
    return resolved.open(encoding="utf-8")
