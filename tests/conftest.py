from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

SAMPLE_PATH = ROOT / "data" / "sample_candidates.json"


@pytest.fixture(scope="session")
def sample_candidates() -> list[dict]:
    with open(SAMPLE_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def sample_by_id(sample_candidates) -> dict[str, dict]:
    return {c["candidate_id"]: c for c in sample_candidates}


def make_minimal_candidate(**overrides) -> dict:
    """Build a syntactically-valid minimal candidate dict for targeted unit
    tests, with sensible defaults that can be overridden per-test.
    """
    base = {
        "candidate_id": "CAND_TEST001",
        "profile": {
            "anonymized_name": "Test Candidate",
            "headline": "Senior AI Engineer",
            "summary": "Test summary",
            "location": "Pune, Maharashtra",
            "country": "India",
            "years_of_experience": 7.0,
            "current_title": "Senior AI Engineer",
            "current_company": "TestCo",
            "current_company_size": "201-500",
            "current_industry": "Software",
        },
        "career_history": [
            {
                "company": "TestCo",
                "title": "Senior AI Engineer",
                "start_date": "2019-01-01",
                "end_date": None,
                "duration_months": 84,
                "is_current": True,
                "industry": "Software",
                "company_size": "201-500",
                "description": "Built production retrieval and ranking systems.",
            }
        ],
        "education": [
            {
                "institution": "Test University",
                "degree": "B.Tech",
                "field_of_study": "Computer Science",
                "start_year": 2014,
                "end_year": 2018,
                "grade": "8.0 CGPA",
                "tier": "tier_2",
            }
        ],
        "skills": [
            {"name": "Python", "proficiency": "expert", "endorsements": 10, "duration_months": 60},
        ],
        "certifications": [],
        "languages": [{"language": "English", "proficiency": "native"}],
        "redrob_signals": {
            "profile_completeness_score": 80.0,
            "signup_date": "2025-01-01",
            "last_active_date": "2026-06-10",
            "open_to_work_flag": True,
            "profile_views_received_30d": 10,
            "applications_submitted_30d": 1,
            "recruiter_response_rate": 0.7,
            "avg_response_time_hours": 24.0,
            "skill_assessment_scores": {},
            "connection_count": 100,
            "endorsements_received": 5,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 20.0, "max": 30.0},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": 50.0,
            "search_appearance_30d": 20,
            "saved_by_recruiters_30d": 2,
            "interview_completion_rate": 0.9,
            "offer_acceptance_rate": 0.5,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        },
    }
    for key, value in overrides.items():
        base[key] = value
    return base
