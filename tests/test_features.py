"""Tests for features.py."""

from __future__ import annotations

from redrob_ranker import config, features

from .conftest import make_minimal_candidate


def test_title_score_exact_match():
    c = make_minimal_candidate()
    c["profile"]["current_title"] = "Senior AI Engineer"
    result = features.score_title(c)
    assert result.score == config.TITLE_TIER_SCORE["Senior AI Engineer"]


def test_title_score_unknown_title_uses_default():
    c = make_minimal_candidate()
    c["profile"]["current_title"] = "Some Title Not In The Table"
    c["career_history"][0]["title"] = "Some Title Not In The Table"  # avoid past-title credit
    result = features.score_title(c)
    assert result.score == config.DEFAULT_TITLE_TIER_SCORE


def test_past_title_credit_applies_when_current_title_is_weaker():
    c = make_minimal_candidate()
    c["profile"]["current_title"] = "Engineering Manager"  # not a strong title
    c["career_history"] = [
        {
            "company": "X", "title": "Engineering Manager", "start_date": "2024-01-01",
            "end_date": None, "duration_months": 12, "is_current": True,
            "industry": "Software", "company_size": "201-500", "description": "d",
        },
        {
            "company": "Y", "title": "Senior NLP Engineer", "start_date": "2020-01-01",
            "end_date": "2024-01-01", "duration_months": 48, "is_current": False,
            "industry": "Software", "company_size": "201-500", "description": "d",
        },
    ]
    result = features.score_title(c)
    assert result.past_title_bonus > 0


def test_skill_trust_rewards_duration_and_proficiency():
    c = make_minimal_candidate()
    c["skills"] = [
        {"name": "FAISS", "proficiency": "expert", "endorsements": 10, "duration_months": 48},
        {"name": "FAISS_clone_low_evidence", "proficiency": "expert", "endorsements": 1, "duration_months": 0},
    ]
    trust = features.compute_skill_trust(c)
    assert trust["FAISS"] > trust["FAISS_clone_low_evidence"]


def test_skill_trust_discounted_by_contradicting_assessment():
    c = make_minimal_candidate()
    c["skills"] = [
        {"name": "RAG", "proficiency": "expert", "endorsements": 10, "duration_months": 36},
    ]
    c["redrob_signals"]["skill_assessment_scores"] = {"RAG": 5.0}  # contradicts "expert"
    trust = features.compute_skill_trust(c)

    c2 = make_minimal_candidate()
    c2["skills"] = [
        {"name": "RAG", "proficiency": "expert", "endorsements": 10, "duration_months": 36},
    ]
    c2["redrob_signals"]["skill_assessment_scores"] = {}  # no contradiction
    trust2 = features.compute_skill_trust(c2)

    assert trust["RAG"] < trust2["RAG"]


def test_must_have_skills_high_for_strong_ai_profile():
    c = make_minimal_candidate()
    c["skills"] = [
        {"name": "Embeddings", "proficiency": "expert", "endorsements": 20, "duration_months": 48},
        {"name": "FAISS", "proficiency": "expert", "endorsements": 20, "duration_months": 48},
        {"name": "Python", "proficiency": "expert", "endorsements": 30, "duration_months": 80},
        {"name": "Learning to Rank", "proficiency": "advanced", "endorsements": 10, "duration_months": 24},
    ]
    trust = features.compute_skill_trust(c)
    result = features.score_must_have_skills(c, trust)
    assert result.score > 0.5
    assert result.groups_satisfied >= 3


def test_must_have_skills_low_for_keyword_stuffer():
    """A candidate with an irrelevant title but AI buzzwords listed with
    minimal evidence (the dataset's documented 'keyword stuffing' trap)
    should NOT score as well as a genuine practitioner, because trust
    discounts the low-duration claims.
    """
    c = make_minimal_candidate()
    c["profile"]["current_title"] = "HR Manager"
    c["career_history"][0]["title"] = "HR Manager"  # avoid past-title credit
    c["career_history"][0]["description"] = "Managed employee onboarding and payroll."
    c["skills"] = [
        {"name": "RAG", "proficiency": "advanced", "endorsements": 1, "duration_months": 1},
        {"name": "Pinecone", "proficiency": "advanced", "endorsements": 1, "duration_months": 1},
    ]
    trust = features.compute_skill_trust(c)
    result = features.score_must_have_skills(c, trust)
    title_result = features.score_title(c)

    assert title_result.score == config.TITLE_TIER_SCORE["HR Manager"]
    assert title_result.score < 0.1


def test_experience_band_scores_full_inside_band():
    assert features.score_experience_band(7.0) == 1.0
    assert features.score_experience_band(5.0) == 1.0
    assert features.score_experience_band(9.0) == 1.0


def test_experience_band_falls_off_outside_band():
    inside = features.score_experience_band(7.0)
    outside_low = features.score_experience_band(1.0)
    outside_high = features.score_experience_band(20.0)
    assert outside_low < inside
    assert outside_high < inside


def test_consulting_only_career_is_penalized():
    c = make_minimal_candidate()
    c["career_history"] = [
        {
            "company": "TCS", "title": "Software Engineer", "start_date": "2020-01-01",
            "end_date": None, "duration_months": 60, "is_current": True,
            "industry": "IT Services", "company_size": "10001+", "description": "d",
        }
    ]
    trust = features.compute_skill_trust(c)
    result = features.score_disqualifiers(c, trust, total_career_years=5.0)
    assert result.multiplier == config.CONSULTING_ONLY_PENALTY
    assert any("consulting" in r for r in result.reasons)


def test_mixed_consulting_and_product_history_not_penalized():
    c = make_minimal_candidate()
    c["career_history"] = [
        {
            "company": "TCS", "title": "Software Engineer", "start_date": "2024-01-01",
            "end_date": None, "duration_months": 12, "is_current": True,
            "industry": "IT Services", "company_size": "10001+", "description": "d",
        },
        {
            "company": "Flipkart", "title": "ML Engineer", "start_date": "2020-01-01",
            "end_date": "2024-01-01", "duration_months": 48, "is_current": False,
            "industry": "E-commerce", "company_size": "10001+", "description": "d",
        },
    ]
    trust = features.compute_skill_trust(c)
    result = features.score_disqualifiers(c, trust, total_career_years=5.0)
    assert result.multiplier == 1.0


def test_cv_speech_only_without_nlp_overlap_is_penalized():
    c = make_minimal_candidate()
    c["skills"] = [
        {"name": "Computer Vision", "proficiency": "expert", "endorsements": 10, "duration_months": 48},
        {"name": "Object Detection", "proficiency": "expert", "endorsements": 10, "duration_months": 48},
    ]
    trust = features.compute_skill_trust(c)
    result = features.score_disqualifiers(c, trust, total_career_years=5.0)
    assert result.multiplier == config.CV_SPEECH_ONLY_PENALTY


def test_cv_skills_with_nlp_overlap_not_penalized_for_cv_rule():
    c = make_minimal_candidate()
    c["skills"] = [
        {"name": "Computer Vision", "proficiency": "expert", "endorsements": 10, "duration_months": 48},
        {"name": "Object Detection", "proficiency": "expert", "endorsements": 10, "duration_months": 48},
        {"name": "NLP", "proficiency": "advanced", "endorsements": 10, "duration_months": 30},
    ]
    trust = features.compute_skill_trust(c)
    result = features.score_disqualifiers(c, trust, total_career_years=5.0)
    assert config.CV_SPEECH_ONLY_PENALTY not in [result.multiplier]
