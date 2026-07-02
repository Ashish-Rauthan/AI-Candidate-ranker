from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from . import config

TODAY = date.today()  # always use the real current date


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
    try:
        yoe = float(profile.get("years_of_experience", 0) or 0)
    except (TypeError, ValueError):
        yoe = 0.0
    # Guard: negative duration_months (malformed data) must not reduce the
    # computed total and mask a real mismatch -- clamp each entry at 0.
    total_months = sum(max(0, int(c.get("duration_months", 0) or 0)) for c in career)
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
