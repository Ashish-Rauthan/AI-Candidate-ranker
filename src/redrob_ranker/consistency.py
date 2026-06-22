"""
consistency.py
===============
Detects internally-impossible candidate profiles -- the "honeypots" the
submission spec warns about ("~80 honeypot candidates with subtly
impossible profiles ... forced to relevance tier 0 in the ground truth").

Design philosophy: a honeypot is impossible *on its own terms* -- it
contradicts itself -- not merely "weak". We deliberately check for
internal contradictions (stated experience vs. actual career-history
duration; "expert" proficiency with near-zero time spent on the skill;
two simultaneously-current jobs; impossible date ranges) rather than
guessing at "suspicious-looking" profiles, because the latter risks
flagging genuinely strong-but-unusual candidates.

These thresholds were derived empirically against the real
candidates.jsonl (not the public sample) and validated two ways:
  1. They flag ~70 candidates out of 100,000 -- closely matching the "~80
     honeypots" the spec documents.
  2. The flagged count is stable across a wide band of nearby threshold
     choices (1.5-3.0 years for the mismatch check, 1-3 for the expert-zero
     count), meaning we're sitting in a genuine gap between "normal noise"
     and "deliberately impossible", not finely overfit to one number.

One concrete validated example: CAND_0039754, "Senior Applied Scientist"
at Meta, ranks in the top 15 by raw semantic similarity to the JD (career
history mentions BGE, FAISS, NDCG/MRR, A/B testing -- everything the JD
wants) yet claims years_of_experience=16.2 while career_history sums to
only ~8.2 years. This module is what catches that candidate before it
reaches the final ranking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from . import config

TODAY = date(2026, 6, 18)  # hackathon submission window reference date


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def _spans_overlap(a_start: date, a_end: date, b_start: date, b_end: date) -> bool:
    return a_start < b_end and b_start < a_end


@dataclass
class ConsistencyResult:
    is_honeypot: bool = False
    reasons: list[str] = field(default_factory=list)
    yoe_mismatch_years: float = 0.0


def check_consistency(candidate: dict) -> ConsistencyResult:
    """Run all internal-consistency checks on one raw candidate record."""
    reasons: list[str] = []

    profile = candidate.get("profile", {})
    career = candidate.get("career_history", []) or []
    education = candidate.get("education", []) or []
    skills = candidate.get("skills", []) or []

    # --- Check 1: stated years_of_experience vs. summed career_history ----
    yoe = float(profile.get("years_of_experience", 0) or 0)
    total_months = sum(int(c.get("duration_months", 0) or 0) for c in career)
    total_years = total_months / 12.0
    mismatch = abs(total_years - yoe)
    if mismatch > config.YOE_MISMATCH_THRESHOLD_YEARS:
        reasons.append(
            f"years_of_experience ({yoe:.1f}) vs. career_history total "
            f"({total_years:.1f}) differ by {mismatch:.1f} years"
        )

    # --- Check 2: "expert" proficiency with near-zero time spent ----------
    expert_zero = [
        s.get("name", "?")
        for s in skills
        if s.get("proficiency") == "expert"
        and int(s.get("duration_months", 999) or 0) <= config.EXPERT_ZERO_DURATION_MAX_MONTHS
    ]
    if len(expert_zero) >= config.EXPERT_ZERO_DURATION_MIN_COUNT:
        reasons.append(
            f"'expert' proficiency claimed with <= "
            f"{config.EXPERT_ZERO_DURATION_MAX_MONTHS} month(s) of use: {expert_zero}"
        )

    # --- Check 3: more than one simultaneously-current job ----------------
    n_current = sum(1 for c in career if c.get("is_current"))
    if n_current > 1:
        reasons.append(f"{n_current} career_history entries marked is_current")

    # --- Check 4: overlapping full-time employment date ranges ------------
    spans = []
    for c in career:
        s = _parse_date(c.get("start_date"))
        e = _parse_date(c.get("end_date")) or TODAY
        if s is not None:
            spans.append((s, e))
    spans.sort(key=lambda x: x[0])
    for i in range(len(spans) - 1):
        if _spans_overlap(spans[i][0], spans[i][1], spans[i + 1][0], spans[i + 1][1]):
            reasons.append("overlapping employment date ranges in career_history")
            break

    # --- Check 5: education end_year before start_year --------------------
    for e in education:
        sy, ey = e.get("start_year"), e.get("end_year")
        if sy is not None and ey is not None and ey < sy:
            reasons.append(f"education end_year ({ey}) precedes start_year ({sy})")
            break

    return ConsistencyResult(
        is_honeypot=len(reasons) > 0,
        reasons=reasons,
        yoe_mismatch_years=round(mismatch, 2),
    )
