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
    """Yield one parsed candidate dict per candidate in the file.

    Supports two formats, auto-detected from the first non-whitespace byte:
    - JSON array  (first char == '[') -- parsed in one shot, e.g. sample_candidates.json
    - JSONL       (anything else)     -- streamed line-by-line for memory efficiency

    Malformed JSONL lines are skipped with a printed warning rather than
    crashing the whole run -- a single corrupt row should never take down a
    ranking job over 100,000 candidates.
    """
    with _open_text(path) as f:
        # Peek at the first non-whitespace character to detect format.
        first_char = ""
        while True:
            ch = f.read(1)
            if ch == "":
                return  # empty file
            if ch.strip():
                first_char = ch
                break
        f.seek(0)

        if first_char == "[":
            # JSON array -- parse whole file (typically small sample files)
            try:
                for record in json.load(f):
                    if isinstance(record, dict):
                        yield record
            except json.JSONDecodeError as exc:
                print(f"[io_utils] ERROR: could not parse JSON array from {path}: {exc}")
        else:
            # JSONL -- stream line by line (memory-efficient for 100K pool)
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        print(f"[io_utils] WARNING: skipping non-object line {line_no} (got {type(record).__name__})")
                        continue
                    yield record
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