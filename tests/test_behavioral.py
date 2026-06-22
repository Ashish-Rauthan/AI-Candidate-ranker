"""Tests for behavioral.py."""

from __future__ import annotations

from redrob_ranker import config
from redrob_ranker.behavioral import score_behavioral

from .conftest import make_minimal_candidate


def test_active_responsive_candidate_gets_high_multiplier():
    c = make_minimal_candidate()
    c["redrob_signals"]["last_active_date"] = "2026-06-17"  # very recent
    c["redrob_signals"]["recruiter_response_rate"] = 0.9
    c["redrob_signals"]["open_to_work_flag"] = True
    result = score_behavioral(c)
    assert result.multiplier > 0.9


def test_dormant_unresponsive_candidate_gets_low_multiplier():
    """The JD's own example: a perfect-on-paper candidate who hasn't
    logged in for 6 months and has a 5% recruiter response rate should be
    down-weighted."""
    c = make_minimal_candidate()
    c["redrob_signals"]["last_active_date"] = "2025-12-01"  # ~6 months stale
    c["redrob_signals"]["recruiter_response_rate"] = 0.05
    c["redrob_signals"]["open_to_work_flag"] = False
    c["redrob_signals"]["avg_response_time_hours"] = 200.0
    c["redrob_signals"]["interview_completion_rate"] = 0.2
    result = score_behavioral(c)
    assert result.multiplier < 0.76


def test_multiplier_always_within_configured_bounds():
    c = make_minimal_candidate()
    result = score_behavioral(c)
    assert config.BEHAVIORAL_MULT_FLOOR <= result.multiplier <= config.BEHAVIORAL_MULT_CEILING


def test_active_beats_dormant_holding_everything_else_equal():
    active = make_minimal_candidate()
    active["redrob_signals"]["last_active_date"] = "2026-06-18"

    dormant = make_minimal_candidate()
    dormant["redrob_signals"]["last_active_date"] = "2025-01-01"

    r_active = score_behavioral(active)
    r_dormant = score_behavioral(dormant)
    assert r_active.multiplier > r_dormant.multiplier
