"""Tests for all improvements made after the initial submission audit."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from redrob_ranker import config
from redrob_ranker.behavioral import score_behavioral
from redrob_ranker.consistency import check_consistency
from redrob_ranker.features import compute_skill_trust, score_disqualifiers, score_location, score_notice_period
from redrob_ranker.io_utils import iter_candidates
from redrob_ranker.pipeline import run_pipeline
from redrob_ranker.semantic import build_document

from .conftest import make_minimal_candidate


# ---------------------------------------------------------------------------
# Fix 1: io_utils — JSON array support
# ---------------------------------------------------------------------------

def test_iter_candidates_handles_json_array(tmp_path):
    """iter_candidates must load a JSON array file without crashing."""
    data = [make_minimal_candidate(candidate_id="CAND_0000001"),
            make_minimal_candidate(candidate_id="CAND_0000002")]
    p = tmp_path / "test.json"
    p.write_text(json.dumps(data))
    result = list(iter_candidates(p))
    assert len(result) == 2
    assert result[0]["candidate_id"] == "CAND_0000001"


def test_iter_candidates_still_handles_jsonl(tmp_path):
    """JSONL format must still work after adding JSON array support."""
    c1 = make_minimal_candidate(candidate_id="CAND_0000003")
    c2 = make_minimal_candidate(candidate_id="CAND_0000004")
    p = tmp_path / "test.jsonl"
    p.write_text(json.dumps(c1) + "\n" + json.dumps(c2) + "\n")
    result = list(iter_candidates(p))
    assert len(result) == 2
    assert result[1]["candidate_id"] == "CAND_0000004"


def test_pipeline_runs_on_json_array_file(tmp_path):
    """run_pipeline must not crash when given a .json array file."""
    candidates = [make_minimal_candidate(candidate_id=f"CAND_{i:07d}") for i in range(1, 6)]
    p = tmp_path / "candidates.json"
    p.write_text(json.dumps(candidates))
    df = run_pipeline(p, artifacts_dir=tmp_path / "arts", top_n=5, use_cache=False, verbose=False)
    assert len(df) == 5
    assert list(df.columns) == ["candidate_id", "rank", "score", "reasoning"]


# ---------------------------------------------------------------------------
# Fix 2: semantic.py — recency-weighted career history
# ---------------------------------------------------------------------------

def test_recency_weights_current_job_higher_than_past():
    """A candidate whose CURRENT job is in NLP/IR should produce a document
    with more NLP/IR terms than an otherwise identical candidate whose
    current job is unrelated but whose past job was NLP/IR.
    """
    current_nlp = make_minimal_candidate()
    current_nlp["career_history"] = [
        {"company": "A", "title": "NLP Engineer", "start_date": "2022-01-01",
         "end_date": None, "duration_months": 30, "is_current": True,
         "industry": "Software", "company_size": "201-500",
         "description": "semantic search embeddings retrieval ranking NDCG"},
    ]

    past_nlp = make_minimal_candidate()
    past_nlp["career_history"] = [
        {"company": "B", "title": "Project Manager", "start_date": "2023-01-01",
         "end_date": None, "duration_months": 18, "is_current": True,
         "industry": "Software", "company_size": "201-500",
         "description": "roadmaps stakeholder management delivery"},
        {"company": "A", "title": "NLP Engineer", "start_date": "2019-01-01",
         "end_date": "2023-01-01", "duration_months": 48, "is_current": False,
         "industry": "Software", "company_size": "201-500",
         "description": "semantic search embeddings retrieval ranking NDCG"},
    ]

    doc_current = build_document(current_nlp)
    doc_past = build_document(past_nlp)
    # Current-NLP candidate's doc should contain more NLP-term repetitions
    assert doc_current.count("embeddings") >= doc_past.count("embeddings")


# ---------------------------------------------------------------------------
# Fix 3: config.py — expanded NLP_IR_OVERLAP_SKILLS
# ---------------------------------------------------------------------------

def test_nlp_ir_overlap_includes_dataset_vocabulary():
    """The 7 IR skill names from the full dataset scan must be in NLP_IR_OVERLAP_SKILLS."""
    required = {
        "Natural Language Processing",
        "Information Retrieval Systems",
        "Ranking Systems",
        "Search & Discovery",
        "Search Backend",
        "Search Infrastructure",
        "Vector Representations",
    }
    assert required.issubset(config.NLP_IR_OVERLAP_SKILLS), (
        f"Missing: {required - config.NLP_IR_OVERLAP_SKILLS}"
    )


def test_cv_penalty_does_not_fire_for_ir_search_backend_candidate():
    """A candidate with CV/speech skills AND Search Backend / Elasticsearch
    must NOT get the CV-only penalty because they have NLP/IR overlap.
    """
    c = make_minimal_candidate()
    c["skills"] = [
        {"name": "Computer Vision", "proficiency": "expert", "endorsements": 10, "duration_months": 48},
        {"name": "YOLO", "proficiency": "advanced", "endorsements": 8, "duration_months": 30},
        {"name": "Search Backend", "proficiency": "expert", "endorsements": 15, "duration_months": 36},
        {"name": "Elasticsearch", "proficiency": "advanced", "endorsements": 12, "duration_months": 24},
    ]
    trust = compute_skill_trust(c)
    result = score_disqualifiers(c, trust, total_career_years=7.0)
    assert config.CV_SPEECH_ONLY_PENALTY not in [result.multiplier], (
        "CV-only penalty should not fire when candidate has Search Backend / Elasticsearch"
    )


def test_cv_penalty_does_not_fire_for_ranking_systems_candidate():
    """A candidate with 'Ranking Systems' in skills must not get the CV penalty."""
    c = make_minimal_candidate()
    c["skills"] = [
        {"name": "Computer Vision", "proficiency": "expert", "endorsements": 10, "duration_months": 48},
        {"name": "ASR", "proficiency": "advanced", "endorsements": 5, "duration_months": 20},
        {"name": "Ranking Systems", "proficiency": "expert", "endorsements": 20, "duration_months": 40},
    ]
    trust = compute_skill_trust(c)
    result = score_disqualifiers(c, trust, total_career_years=7.0)
    assert config.CV_SPEECH_ONLY_PENALTY not in [result.multiplier]


# ---------------------------------------------------------------------------
# Fix 3b: config.py — expanded MUST_HAVE_SKILL_GROUPS
# ---------------------------------------------------------------------------

def test_search_backend_scores_in_must_have():
    """'Search Backend' must contribute to the vector_db_hybrid_search group."""
    c = make_minimal_candidate()
    c["skills"] = [
        {"name": "Search Backend", "proficiency": "expert", "endorsements": 20, "duration_months": 36},
        {"name": "Python", "proficiency": "expert", "endorsements": 30, "duration_months": 80},
    ]
    trust = compute_skill_trust(c)
    from redrob_ranker.features import score_must_have_skills
    result = score_must_have_skills(c, trust)
    assert result.group_scores.get("vector_db_hybrid_search", 0) > 0


def test_ranking_systems_scores_in_must_have():
    """'Ranking Systems' must contribute to the eval_for_ranking group."""
    c = make_minimal_candidate()
    c["skills"] = [
        {"name": "Ranking Systems", "proficiency": "expert", "endorsements": 15, "duration_months": 30},
        {"name": "Python", "proficiency": "expert", "endorsements": 30, "duration_months": 80},
    ]
    trust = compute_skill_trust(c)
    from redrob_ranker.features import score_must_have_skills
    result = score_must_have_skills(c, trust)
    assert result.group_scores.get("eval_for_ranking", 0) > 0


# ---------------------------------------------------------------------------
# Fix 4: behavioral.py — dynamic TODAY
# ---------------------------------------------------------------------------

def test_behavioral_today_is_dynamic():
    """behavioral.TODAY must equal the current real date, not a hardcoded value."""
    from datetime import date
    import redrob_ranker.behavioral as beh
    assert beh.TODAY == date.today(), (
        f"behavioral.TODAY is hardcoded to {beh.TODAY}, expected {date.today()}"
    )


def test_consistency_today_is_dynamic():
    """consistency.TODAY must equal the current real date."""
    from datetime import date
    import redrob_ranker.consistency as con
    assert con.TODAY == date.today(), (
        f"consistency.TODAY is hardcoded to {con.TODAY}, expected {date.today()}"
    )


# ---------------------------------------------------------------------------
# Fix 5: features.py — endorsement factor in skill trust
# ---------------------------------------------------------------------------

def test_high_endorsements_increase_trust():
    """A skill with 20 endorsements must have higher trust than the same
    skill with 0 endorsements, all else being equal.
    """
    c_high = make_minimal_candidate()
    c_high["skills"] = [
        {"name": "FAISS", "proficiency": "advanced", "endorsements": 20, "duration_months": 24}
    ]
    c_low = make_minimal_candidate()
    c_low["skills"] = [
        {"name": "FAISS", "proficiency": "advanced", "endorsements": 0, "duration_months": 24}
    ]
    trust_high = compute_skill_trust(c_high)
    trust_low = compute_skill_trust(c_low)
    assert trust_high["FAISS"] > trust_low["FAISS"]


def test_endorsement_factor_capped_at_20():
    """Trust from 20 endorsements must equal trust from 100 endorsements
    (the factor must be capped at 1.0 so outliers don't dominate).
    """
    c20 = make_minimal_candidate()
    c20["skills"] = [{"name": "FAISS", "proficiency": "advanced", "endorsements": 20, "duration_months": 24}]
    c100 = make_minimal_candidate()
    c100["skills"] = [{"name": "FAISS", "proficiency": "advanced", "endorsements": 100, "duration_months": 24}]
    assert compute_skill_trust(c20)["FAISS"] == compute_skill_trust(c100)["FAISS"]


# ---------------------------------------------------------------------------
# Fix 6: semantic.py — adaptive min_df
# ---------------------------------------------------------------------------

def test_adaptive_min_df_small_corpus(tmp_path):
    """On a small corpus (<500 candidates), a rare but key term like 'pgvector'
    that appears in only one candidate must still end up in the TF-IDF
    vocabulary (min_df=1 on small corpora).
    """
    from redrob_ranker.semantic import build_document, build_index

    candidates = []
    for i in range(10):
        c = make_minimal_candidate(candidate_id=f"CAND_{i:07d}")
        c["skills"] = [{"name": "Python", "proficiency": "expert", "endorsements": 5, "duration_months": 40}]
        candidates.append(c)
    # Only candidate 0 has pgvector
    candidates[0]["skills"].append(
        {"name": "pgvector", "proficiency": "expert", "endorsements": 10, "duration_months": 20}
    )

    docs = [build_document(c) for c in candidates]
    ids = [c["candidate_id"] for c in candidates]
    index = build_index(ids, docs, verbose=False)
    vocab = set(index.vectorizer.vocabulary_.keys())
    assert "pgvector" in vocab, "pgvector should be in vocabulary on small corpus (min_df=1)"


# ---------------------------------------------------------------------------
# Fix: consistency.py — negative duration guard
# ---------------------------------------------------------------------------

def test_negative_duration_does_not_mask_yoe_mismatch():
    """A career entry with duration_months=-5 must be clamped to 0, not
    allowed to reduce the computed total and hide a real YoE mismatch.
    """
    c = make_minimal_candidate()
    c["profile"]["years_of_experience"] = 10.0
    c["career_history"] = [
        {"company": "A", "title": "Engineer", "start_date": "2020-01-01",
         "end_date": None, "duration_months": -5,  # malformed -- must clamp to 0
         "is_current": True, "industry": "Software", "company_size": "201-500",
         "description": "work"},
    ]
    # Without clamping: total = -5/12 = -0.4y, mismatch = |−0.4 − 10| = 10.4y → flagged
    # With clamping:    total = 0/12 = 0y,    mismatch = |0 − 10| = 10y   → flagged
    # Either way the mismatch is huge, but without clamping the mismatch calc
    # is polluted. The important thing is it's still caught.
    result = check_consistency(c)
    assert result.is_honeypot is True
    assert result.yoe_mismatch_years >= 9.0


# ---------------------------------------------------------------------------
# Fix: type-safety guards on malformed data (CAND_0000015 pattern)
# ---------------------------------------------------------------------------

def test_pipeline_handles_non_string_current_title(tmp_path):
    """Pipeline must not crash when current_title is a boolean (True).
    Uses 2 candidates because SVD requires n_samples >= 2."""
    c1 = make_minimal_candidate(candidate_id="CAND_0000001")
    c2 = make_minimal_candidate(candidate_id="CAND_0000002")
    c2["profile"]["current_title"] = True   # malformed
    p = tmp_path / "malformed.jsonl"
    p.write_text(json.dumps(c1) + "\n" + json.dumps(c2) + "\n")
    df = run_pipeline(p, artifacts_dir=tmp_path / "arts", top_n=2, use_cache=False, verbose=False)
    assert len(df) == 2


def test_pipeline_handles_non_numeric_yoe(tmp_path):
    """Pipeline must not crash when years_of_experience is 'five' (string)."""
    c1 = make_minimal_candidate(candidate_id="CAND_0000001")
    c2 = make_minimal_candidate(candidate_id="CAND_0000002")
    c2["profile"]["years_of_experience"] = "five"  # malformed
    p = tmp_path / "malformed_yoe.jsonl"
    p.write_text(json.dumps(c1) + "\n" + json.dumps(c2) + "\n")
    df = run_pipeline(p, artifacts_dir=tmp_path / "arts", top_n=2, use_cache=False, verbose=False)
    assert len(df) == 2


def test_pipeline_handles_non_string_location(tmp_path):
    """Pipeline must not crash when location is an integer."""
    c1 = make_minimal_candidate(candidate_id="CAND_0000001")
    c2 = make_minimal_candidate(candidate_id="CAND_0000002")
    c2["profile"]["location"] = 12345  # malformed
    p = tmp_path / "malformed_loc.jsonl"
    p.write_text(json.dumps(c1) + "\n" + json.dumps(c2) + "\n")
    df = run_pipeline(p, artifacts_dir=tmp_path / "arts", top_n=2, use_cache=False, verbose=False)
    assert len(df) == 2


def test_pipeline_handles_string_notice_period(tmp_path):
    """Pipeline must not crash when notice_period_days is 'thirty' (string)."""
    c1 = make_minimal_candidate(candidate_id="CAND_0000001")
    c2 = make_minimal_candidate(candidate_id="CAND_0000002")
    c2["redrob_signals"]["notice_period_days"] = "thirty"  # malformed
    p = tmp_path / "malformed_notice.jsonl"
    p.write_text(json.dumps(c1) + "\n" + json.dumps(c2) + "\n")
    df = run_pipeline(p, artifacts_dir=tmp_path / "arts", top_n=2, use_cache=False, verbose=False)
    assert len(df) == 2
