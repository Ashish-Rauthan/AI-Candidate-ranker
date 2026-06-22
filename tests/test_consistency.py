"""Tests for consistency.py -- the honeypot detector."""

from __future__ import annotations

from redrob_ranker.consistency import check_consistency

from .conftest import make_minimal_candidate


def test_consistent_candidate_is_not_flagged():
    c = make_minimal_candidate()
    result = check_consistency(c)
    assert result.is_honeypot is False
    assert result.reasons == []


def test_years_of_experience_mismatch_is_flagged():
    c = make_minimal_candidate()
    c["profile"]["years_of_experience"] = 16.0  # career_history only sums to 3 years
    result = check_consistency(c)
    assert result.is_honeypot is True
    assert any("years_of_experience" in r for r in result.reasons)


def test_expert_with_near_zero_duration_is_flagged():
    c = make_minimal_candidate()
    c["skills"] = [
        {"name": "Pinecone", "proficiency": "expert", "endorsements": 5, "duration_months": 0}
    ]
    result = check_consistency(c)
    assert result.is_honeypot is True
    assert any("expert" in r for r in result.reasons)


def test_multiple_current_jobs_is_flagged():
    c = make_minimal_candidate()
    c["career_history"] = [
        {
            "company": "A", "title": "Engineer", "start_date": "2022-01-01",
            "end_date": None, "duration_months": 24, "is_current": True,
            "industry": "Software", "company_size": "201-500", "description": "x",
        },
        {
            "company": "B", "title": "Engineer", "start_date": "2023-01-01",
            "end_date": None, "duration_months": 12, "is_current": True,
            "industry": "Software", "company_size": "201-500", "description": "y",
        },
    ]
    result = check_consistency(c)
    assert result.is_honeypot is True
    assert any("is_current" in r for r in result.reasons)


def test_real_honeypot_from_dataset_is_flagged(sample_by_id):
    """Regression check against an actual honeypot pattern empirically
    found in the real candidates.jsonl during development: a candidate
    whose stated years_of_experience is far larger than the sum of their
    career_history durations. We synthesize the same shape here using the
    public sample format rather than depending on the private full file,
    so this test is portable.
    """
    c = make_minimal_candidate(candidate_id="CAND_TEST_HONEYPOT")
    c["profile"]["years_of_experience"] = 16.2
    c["career_history"] = [
        {
            "company": "Meta", "title": "Senior Applied Scientist",
            "start_date": "2023-05-13", "end_date": None, "duration_months": 37,
            "is_current": True, "industry": "Internet", "company_size": "10001+",
            "description": "Owned the ranking pipeline.",
        },
        {
            "company": "Apple", "title": "Senior ML Engineer",
            "start_date": "2020-01-29", "end_date": "2023-05-13", "duration_months": 40,
            "is_current": False, "industry": "Consumer Electronics", "company_size": "10001+",
            "description": "Migrated retrieval to hybrid search.",
        },
    ]
    result = check_consistency(c)
    assert result.is_honeypot is True
    assert result.yoe_mismatch_years > 9.0
