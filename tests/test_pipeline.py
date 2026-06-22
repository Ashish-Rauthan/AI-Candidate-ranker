"""End-to-end pipeline tests on a small synthetic candidate pool (not the
full 100K file -- fast enough to run on every commit / in CI).
"""

from __future__ import annotations

import json

import pytest

from redrob_ranker.pipeline import run_pipeline

from .conftest import make_minimal_candidate


@pytest.fixture
def small_pool_path(tmp_path):
    candidates = []

    # A strong, genuine fit: AI Engineer with real retrieval/ranking depth.
    strong = make_minimal_candidate(candidate_id="CAND_0000001")
    strong["profile"].update(
        {
            "current_title": "Senior AI Engineer",
            "years_of_experience": 7.0,
            "location": "Pune, Maharashtra",
            "country": "India",
        }
    )
    strong["skills"] = [
        {"name": "Embeddings", "proficiency": "expert", "endorsements": 30, "duration_months": 48},
        {"name": "FAISS", "proficiency": "expert", "endorsements": 25, "duration_months": 40},
        {"name": "Python", "proficiency": "expert", "endorsements": 40, "duration_months": 84},
        {"name": "Learning to Rank", "proficiency": "advanced", "endorsements": 15, "duration_months": 30},
    ]
    strong["career_history"][0]["description"] = (
        "Shipped a production hybrid retrieval system combining BM25 and dense "
        "vectors, with NDCG and MRR offline evaluation and online A/B testing."
    )
    candidates.append(strong)

    # A keyword-stuffer trap: irrelevant title, AI buzzwords with near-zero
    # evidence behind them.
    trap = make_minimal_candidate(candidate_id="CAND_0000002")
    trap["profile"].update({"current_title": "HR Manager", "years_of_experience": 6.0})
    trap["skills"] = [
        {"name": "RAG", "proficiency": "advanced", "endorsements": 1, "duration_months": 1},
        {"name": "Pinecone", "proficiency": "advanced", "endorsements": 1, "duration_months": 1},
        {"name": "LLMs", "proficiency": "advanced", "endorsements": 1, "duration_months": 1},
    ]
    candidates.append(trap)

    # A honeypot: impossible years_of_experience vs career_history.
    honeypot = make_minimal_candidate(candidate_id="CAND_0000003")
    honeypot["profile"].update({"current_title": "Senior AI Engineer", "years_of_experience": 20.0})
    # career_history duration sums to only 36 months (3 years) by default.
    candidates.append(honeypot)

    # A weak/irrelevant candidate as filler.
    weak = make_minimal_candidate(candidate_id="CAND_0000004")
    weak["profile"].update({"current_title": "Graphic Designer", "years_of_experience": 4.0})
    weak["career_history"][0]["duration_months"] = 48
    weak["career_history"][0]["title"] = "Graphic Designer"
    weak["skills"] = [{"name": "Figma", "proficiency": "expert", "endorsements": 5, "duration_months": 24}]
    candidates.append(weak)

    path = tmp_path / "small_candidates.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")
    return path


def test_pipeline_runs_end_to_end_and_excludes_honeypot(small_pool_path, tmp_path):
    df = run_pipeline(
        candidates_path=small_pool_path,
        artifacts_dir=tmp_path / "artifacts",
        top_n=4,
        use_cache=False,
        verbose=False,
    )
    assert list(df.columns) == ["candidate_id", "rank", "score", "reasoning"]
    assert len(df) == 4
    # ranks 1..4 exactly once
    assert sorted(df["rank"].tolist()) == [1, 2, 3, 4]
    # scores non-increasing with rank
    scores = df.sort_values("rank")["score"].tolist()
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


def test_strong_genuine_candidate_outranks_keyword_stuffer(small_pool_path, tmp_path):
    df = run_pipeline(
        candidates_path=small_pool_path,
        artifacts_dir=tmp_path / "artifacts",
        top_n=4,
        use_cache=False,
        verbose=False,
    )
    ranks = dict(zip(df["candidate_id"], df["rank"]))
    assert ranks["CAND_0000001"] < ranks["CAND_0000002"]


def test_honeypot_ranks_last_among_otherwise_plausible_candidates(small_pool_path, tmp_path):
    df = run_pipeline(
        candidates_path=small_pool_path,
        artifacts_dir=tmp_path / "artifacts",
        top_n=4,
        use_cache=False,
        verbose=False,
    )
    ranks = dict(zip(df["candidate_id"], df["rank"]))
    # The honeypot claims a strong title but is internally impossible --
    # it should rank below the genuine strong candidate.
    assert ranks["CAND_0000003"] > ranks["CAND_0000001"]


def test_reasoning_is_non_empty_and_varies(small_pool_path, tmp_path):
    df = run_pipeline(
        candidates_path=small_pool_path,
        artifacts_dir=tmp_path / "artifacts",
        top_n=4,
        use_cache=False,
        verbose=False,
    )
    reasonings = df["reasoning"].tolist()
    assert all(isinstance(r, str) and len(r) > 0 for r in reasonings)
    assert len(set(reasonings)) == len(reasonings)  # all distinct
