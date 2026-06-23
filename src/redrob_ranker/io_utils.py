from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterator


def _open_text(path: str | Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def iter_candidates(path: str | Path) -> Iterator[dict]:
    """Yield one parsed candidate dict per non-blank line of the JSONL file.

    Malformed lines are skipped with a printed warning rather than crashing
    the whole run -- a single corrupt row should never take down a ranking
    job over 100,000 candidates.
    """
    with _open_text(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[io_utils] WARNING: skipping malformed line {line_no}: {exc}")


def count_candidates(path: str | Path) -> int:
    """Cheap line count, used only for progress reporting / sanity checks."""
    n = 0
    with _open_text(path) as f:
        for line in f:
            if line.strip():
                n += 1
    return n
