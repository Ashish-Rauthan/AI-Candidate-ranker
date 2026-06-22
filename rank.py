#!/usr/bin/env python3
"""
rank.py
=======
Single entry point that produces the submission CSV from candidates.jsonl,
per submission_spec.docx section 10.3 ("your README must indicate a single
command that produces the submission CSV from the candidates file").

Usage
-----
    python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv

Optional flags
--------------
    --artifacts-dir DIR   Where the semantic-index cache lives (default:
                           ./artifacts). Safe to delete at any time; it will
                           be rebuilt automatically on the next run.
    --no-cache             Force a full rebuild of the semantic index even
                           if a valid cache is present.
    --top-n N               Number of ranked rows to output (default: 100,
                           per the submission spec -- only change this for
                           local experimentation, not for your final
                           submission file).
    --quiet                 Suppress progress logging.

Compute footprint (measured during development, single CPU core,
100,000-candidate full dataset): feature extraction ~25-45s, semantic
index build ~90s on a cold cache / <5s on a warm cache, scoring + ranking
<5s. Comfortably inside the spec's 5-minute / 16GB / CPU-only budget even
on a cold cache, and well under a minute on a warm one.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running this script directly (`python rank.py ...`) without
# requiring the package to be `pip install`-ed first.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from redrob_ranker.pipeline import run_pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rank candidates in candidates.jsonl against the Redrob "
        "Senior AI Engineer job description and write the top-100 "
        "submission CSV."
    )
    parser.add_argument(
        "--candidates",
        type=str,
        default="data/candidates.jsonl",
        help="Path to candidates.jsonl (or .jsonl.gz). Default: data/candidates.jsonl",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="submission.csv",
        help="Output CSV path. Default: submission.csv",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Directory for the cached semantic index. Default: artifacts/",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force a full rebuild of the semantic index, ignoring any cache.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="Number of ranked rows to output. Default: 100 (per spec).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logging.",
    )
    args = parser.parse_args()

    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        print(f"ERROR: candidates file not found: {candidates_path}", file=sys.stderr)
        return 1

    t0 = time.time()
    df = run_pipeline(
        candidates_path=candidates_path,
        artifacts_dir=args.artifacts_dir,
        top_n=args.top_n,
        use_cache=not args.no_cache,
        verbose=not args.quiet,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")

    elapsed = time.time() - t0
    print(f"\nWrote {len(df)} ranked candidates to {out_path} in {elapsed:.1f}s total.")
    if elapsed > 300:
        print(
            "WARNING: runtime exceeded the 5-minute submission-spec budget. "
            "Re-run once to take advantage of the semantic-index cache, or "
            "investigate before final submission.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
