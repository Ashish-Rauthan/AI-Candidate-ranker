"""
semantic.py
===========
The "understand the role, not just match keywords" layer.

We use TF-IDF + Truncated SVD (a classic Latent Semantic Analysis pipeline)
rather than a neural sentence-transformer embedding model. This is a
deliberate engineering tradeoff, not a default:

  - The submission spec's compute constraints are hard: <=5 min wall-clock,
    CPU-only, no network access during ranking. A transformer encoder
    (sentence-transformers/BGE/E5) needs model weights pulled from the
    network or vendored into the repo (100s of MB), plus torch as a
    dependency -- fragile to reproduce inside an unfamiliar sandboxed
    Docker container at Stage 3.
  - TF-IDF+SVD has zero network dependency, fits in scikit-learn (already
    a near-universal dependency), and empirically runs in well under two
    minutes for the full 100,000-candidate corpus on a single CPU core
    (measured: ~90s fit+transform end-to-end during development).
  - It still produces genuine dense semantic vectors and genuine cosine
    similarity -- candidates are matched by *meaning* across a shared
    latent space, not by literal keyword overlap (LSA's whole point).
    The n-gram features additionally capture short multi-word phrases
    ("vector search", "learning to rank") that unigram keyword matching
    would miss.

This is the "semantic search / vector embeddings" component the hackathon
brief explicitly allows ("semantic search, LLM ranking, vector embeddings,
hybrid scoring -- bring whatever you think works best"), combined with the
rule-based features.py layer into a hybrid score in scoring.py.

CRITICAL: the JD query text used here (config.JD_POSITIVE_QUERY_TEXT) is
built ONLY from the JD's affirmative requirements. A bag-of-words model
cannot distinguish "we want X" from "we do NOT want X" -- both increase
similarity to documents mentioning X. Negative JD criteria (title-chasers,
framework-tutorial GitHubs, consulting-only, CV/speech-only, closed-source-
only) are handled exclusively as explicit rule-based penalties in
features.py, never folded into this similarity query. See config.py's
comment on JD_POSITIVE_QUERY_TEXT for the full rationale.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config

ARTIFACT_VERSION = 2  # bump if build_document() or vectorizer params change


def build_document(candidate: dict) -> str:
    """Turn one candidate record into a single weighted text document for
    TF-IDF. Repetition is used as a crude but effective term-weighting
    trick (headline/title/skills repeated => higher TF-IDF mass) without
    needing a custom weighted-vectorizer implementation.
    """
    profile = candidate.get("profile", {})
    parts: list[str] = [
        profile.get("headline", ""), profile.get("headline", ""),
        profile.get("current_title", ""), profile.get("current_title", ""),
        profile.get("current_title", ""),
        profile.get("summary", ""),
    ]
    for ch in candidate.get("career_history", []) or []:
        parts.append(ch.get("title", ""))
        parts.append(ch.get("description", ""))

    skill_names = [s.get("name", "") for s in candidate.get("skills", []) or []]
    skill_text = " ".join(skill_names)
    parts.append(skill_text)
    parts.append(skill_text)  # weight skills x2

    for e in candidate.get("education", []) or []:
        parts.append(e.get("field_of_study", ""))

    return " ".join(p for p in parts if p)


@dataclass
class SemanticIndex:
    candidate_ids: list[str]
    embeddings: np.ndarray          # shape (n_candidates, n_components)
    vectorizer: TfidfVectorizer
    svd: TruncatedSVD

    def similarity_to_jd(self, jd_text: str = config.JD_POSITIVE_QUERY_TEXT) -> np.ndarray:
        jd_vec = self.vectorizer.transform([jd_text])
        jd_reduced = self.svd.transform(jd_vec)
        sims = cosine_similarity(jd_reduced, self.embeddings)[0]
        return sims


def _artifact_paths(artifacts_dir: Path) -> dict[str, Path]:
    return {
        "vectorizer": artifacts_dir / "tfidf_vectorizer.pkl",
        "svd": artifacts_dir / "svd_model.pkl",
        "embeddings": artifacts_dir / "candidate_embeddings.npy",
        "ids": artifacts_dir / "candidate_ids.pkl",
        "meta": artifacts_dir / "meta.pkl",
    }


def load_cached_index(artifacts_dir: str | Path) -> SemanticIndex | None:
    artifacts_dir = Path(artifacts_dir)
    paths = _artifact_paths(artifacts_dir)
    if not all(p.exists() for p in paths.values()):
        return None
    try:
        with open(paths["meta"], "rb") as f:
            meta = pickle.load(f)
        if meta.get("version") != ARTIFACT_VERSION:
            return None
        with open(paths["vectorizer"], "rb") as f:
            vectorizer = pickle.load(f)
        with open(paths["svd"], "rb") as f:
            svd = pickle.load(f)
        with open(paths["ids"], "rb") as f:
            candidate_ids = pickle.load(f)
        embeddings = np.load(paths["embeddings"])
        return SemanticIndex(
            candidate_ids=candidate_ids,
            embeddings=embeddings,
            vectorizer=vectorizer,
            svd=svd,
        )
    except Exception as exc:  # noqa: BLE001 -- cache is best-effort
        print(f"[semantic] WARNING: failed to load cached artifacts ({exc}); rebuilding")
        return None


def save_index(index: SemanticIndex, artifacts_dir: str | Path) -> None:
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(artifacts_dir)
    with open(paths["vectorizer"], "wb") as f:
        pickle.dump(index.vectorizer, f)
    with open(paths["svd"], "wb") as f:
        pickle.dump(index.svd, f)
    with open(paths["ids"], "wb") as f:
        pickle.dump(index.candidate_ids, f)
    np.save(paths["embeddings"], index.embeddings)
    with open(paths["meta"], "wb") as f:
        pickle.dump({"version": ARTIFACT_VERSION, "n_candidates": len(index.candidate_ids)}, f)


def build_index(
    candidate_ids: list[str],
    documents: list[str],
    n_components: int = 150,
    verbose: bool = True,
) -> SemanticIndex:
    """Fit TF-IDF + TruncatedSVD over the full corpus of candidate
    documents. O(n) in candidates; measured at ~90s wall-clock for 100,000
    candidates on a single CPU core.
    """
    t0 = time.time()
    vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.6,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(documents)
    if verbose:
        print(f"[semantic] TF-IDF fit_transform: shape={X.shape}, {time.time()-t0:.1f}s")

    t1 = time.time()
    # n_components can't exceed min(n_samples, n_features) - 1 for SVD.
    n_components = min(n_components, X.shape[1] - 1, X.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    embeddings = svd.fit_transform(X)
    if verbose:
        print(
            f"[semantic] SVD: shape={embeddings.shape}, {time.time()-t1:.1f}s, "
            f"explained_variance={svd.explained_variance_ratio_.sum():.3f}"
        )

    return SemanticIndex(
        candidate_ids=candidate_ids,
        embeddings=embeddings.astype(np.float32),
        vectorizer=vectorizer,
        svd=svd,
    )
