from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from redrob_ranker import config  # noqa: E402
from redrob_ranker.pipeline import run_pipeline  # noqa: E402

st.set_page_config(page_title="Redrob Candidate Ranker", page_icon="🎯", layout="wide")

st.title("🎯 Redrob Candidate Ranker — Sandbox Demo")
st.caption(
    "Hybrid rule-based + TF-IDF/SVD semantic ranker built for the Redrob "
    "Intelligent Candidate Discovery & Ranking Challenge. This demo runs the "
    "exact same `redrob_ranker` pipeline used to produce the full 100,000-"
    "candidate submission, on a small sample so it's fast to try out."
)

with st.expander("ℹ️ How this works", expanded=False):
    st.markdown(
        """
This system scores every candidate against the Redrob Senior AI Engineer
JD across seven JD-grounded components -- **title tier, must-have skills
(with a trust score that discounts thin/unverified claims), semantic
similarity to the JD, experience-band fit, location, notice period, and
education** -- then applies three multiplicative adjustments:

1. **JD disqualifier penalties** (consulting-only career, recent
   LLM-wrapper-only pivot, CV/speech-only without NLP/IR overlap,
   closed-source-only with no external validation, title-chasing
   job-hops, pure-research-only).
2. **Honeypot detection** -- internally-impossible profiles (e.g. stated
   years of experience that doesn't match career-history duration,
   "expert" skills claimed with near-zero time spent) are crushed toward
   zero, regardless of how good they look on the surface.
3. **Behavioral availability multiplier** -- built from the 23
   `redrob_signals` fields (recency, recruiter response rate, interview
   completion, verification, notice period), bounded to
   `[{floor}, {ceiling}]` so it meaningfully re-ranks candidates without
   ever letting availability alone outrank genuine skills mismatch.

Full source: `src/redrob_ranker/`. Full methodology: `docs/`.
        """.format(
            floor=config.BEHAVIORAL_MULT_FLOOR, ceiling=config.BEHAVIORAL_MULT_CEILING
        )
    )

# ---------------------------------------------------------------------------
# Input: pre-loaded public sample, or an uploaded small candidate file
# ---------------------------------------------------------------------------
st.subheader("1. Candidate pool")

SAMPLE_PATH = ROOT / "data" / "sample_candidates.json"

source = st.radio(
    "Choose a candidate source",
    ["Use the bundled 50-candidate public sample", "Upload my own (.json or .jsonl, ≤100 candidates)"],
    horizontal=True,
)

candidates: list[dict] | None = None

if source.startswith("Use the bundled"):
    with open(SAMPLE_PATH, encoding="utf-8") as f:
        candidates = json.load(f)
    st.success(f"Loaded {len(candidates)} candidates from the bundled public sample.")
else:
    uploaded = st.file_uploader("Upload a candidate file", type=["json", "jsonl"])
    if uploaded is not None:
        raw = uploaded.read().decode("utf-8")
        try:
            if uploaded.name.endswith(".jsonl"):
                candidates = [json.loads(line) for line in raw.splitlines() if line.strip()]
            else:
                parsed = json.loads(raw)
                candidates = parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError as exc:
            st.error(f"Could not parse uploaded file: {exc}")
            candidates = None

        if candidates is not None:
            if len(candidates) > 100:
                st.warning(
                    f"Uploaded file has {len(candidates)} candidates; the sandbox "
                    "is meant for small samples (≤100). Using the first 100."
                )
                candidates = candidates[:100]
            st.success(f"Loaded {len(candidates)} candidates from your upload.")

# ---------------------------------------------------------------------------
# JD text (editable, defaults to the curated affirmative-requirements query)
# ---------------------------------------------------------------------------
st.subheader("2. Job description (affirmative requirements only)")
st.caption(
    "Note: only affirmative ('we want X') requirements should go here -- a "
    "bag-of-words similarity model can't distinguish a wanted requirement "
    "from an explicitly *unwanted* one, so negative criteria are handled "
    "separately as rule-based penalties, not via this text box."
)
jd_text = st.text_area(
    "JD text used for semantic similarity",
    value=config.JD_POSITIVE_QUERY_TEXT.strip(),
    height=160,
)

top_n_default = min(20, len(candidates)) if candidates else 10
top_n = st.slider(
    "How many ranked candidates to show",
    min_value=1,
    max_value=len(candidates) if candidates else 50,
    value=top_n_default if candidates else 10,
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
st.subheader("3. Run the ranker")

if st.button("🚀 Rank candidates", type="primary", disabled=candidates is None):
    if not candidates:
        st.error("No candidates loaded.")
    else:
        with st.spinner(f"Ranking {len(candidates)} candidates..."):
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir) / "demo_candidates.jsonl"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    for c in candidates:
                        f.write(json.dumps(c) + "\n")

                df = run_pipeline(
                    candidates_path=tmp_path,
                    artifacts_dir=Path(tmp_dir) / "artifacts",
                    jd_text=jd_text,
                    top_n=min(top_n, len(candidates)),
                    use_cache=False,  # small sample -- cheap enough to rebuild every run
                    verbose=False,
                )

        st.success(f"Ranked {len(candidates)} candidates in the sandbox.")
        st.dataframe(df, use_container_width=True, hide_index=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download ranked CSV",
            data=csv_bytes,
            file_name="sandbox_ranked_candidates.csv",
            mime="text/csv",
        )

        st.caption(
            "On the full 100,000-candidate dataset, `python rank.py --candidates "
            "data/candidates.jsonl --out submission.csv` produces the official "
            "top-100 submission file using this same pipeline."
        )
else:
    st.info("Choose a candidate source above, then click **Rank candidates**.")
