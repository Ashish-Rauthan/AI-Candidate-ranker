from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from . import config

TODAY = date.today()  # always use the real current date


def _parse_date(s: str | None) -> date | None:
    if not s or not isinstance(s, str):
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def _recency_score(last_active: str | None) -> float:
    """Exponential decay: 1.0 if active today, 0.5 at the configured
    half-life, asymptoting toward 0 for long-dormant profiles."""
    d = _parse_date(last_active)
    if d is None:
        return 0.5  # unknown -- neutral, don't punish missing data
    days_inactive = max(0, (TODAY - d).days)
    half_life = config.LAST_ACTIVE_HALF_LIFE_DAYS
    return 0.5 ** (days_inactive / half_life)


def _notice_period_score(days: int | None) -> float:
    if days is None:
        return 0.7
    try:
        days = int(days)
    except (TypeError, ValueError):
        return 0.7
    for threshold, score in config.NOTICE_PERIOD_SCORE_BREAKS:
        if days <= threshold:
            return score
    return 0.30  # longer than the worst documented break


@dataclass
class BehavioralBreakdown:
    multiplier: float
    recency_score: float
    responsiveness_score: float
    trust_score: float
    open_to_work: bool
    notice_period_days: int | None
    recruiter_response_rate: float | None
    last_active_date: str | None
    summary: str


def _safe_float(v, default=0.0):
    """Coerce v to float, returning default on None or non-numeric strings.
    Pass default=None to distinguish 'unparseable string' from a real zero."""
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def score_behavioral(candidate: dict) -> BehavioralBreakdown:
    sig = candidate.get("redrob_signals", {}) or {}

    # --- Recency / intent ---------------------------------------------
    recency = _recency_score(sig.get("last_active_date"))
    open_to_work = bool(sig.get("open_to_work_flag", False))
    intent = 1.0 if open_to_work else 0.65
    recency_intent = 0.7 * recency + 0.3 * intent

    # --- Responsiveness --------------------------------------------------
    response_rate = _safe_float(sig.get("recruiter_response_rate"), default=0.5)
    resp_time = sig.get("avg_response_time_hours")
    # Map response time to [0,1]: <=24h great, >=120h poor, linear between.
    # Use a sentinel: if resp_time can't be parsed as a number (e.g. "fast"),
    # fall back to neutral 0.5 rather than 0.0 which would give a perfect score.
    if resp_time is None:
        resp_time_score = 0.5
    else:
        resp_time_f = _safe_float(resp_time, default=None)
        if resp_time_f is None:
            resp_time_score = 0.5  # unparseable string → neutral
        else:
            resp_time_score = max(0.0, min(1.0, 1.0 - (resp_time_f - 24.0) / 96.0))
    interview_completion = _safe_float(sig.get("interview_completion_rate"), default=0.7)

    responsiveness = (
        0.45 * response_rate + 0.30 * resp_time_score + 0.25 * interview_completion
    )

    # --- Trust / verification --------------------------------------------
    trust_flags = [
        bool(sig.get("verified_email", False)),
        bool(sig.get("verified_phone", False)),
        bool(sig.get("linkedin_connected", False)),
    ]
    trust = sum(trust_flags) / 3.0

    offer_accept = sig.get("offer_acceptance_rate", -1)
    offer_accept_f = _safe_float(offer_accept, default=-1.0)
    if offer_accept_f >= 0:
        # blend in offer-acceptance history, but lightly -- a low rate can
        # simply mean the candidate is selective, not unavailable.
        trust = 0.85 * trust + 0.15 * offer_accept_f

    # --- Notice period (logistics, but behavioral in spirit: "can we
    #     actually onboard this person on a reasonable timeline") ---------
    notice_score = _notice_period_score(sig.get("notice_period_days"))

    # --- Combine into a single availability score in [0, 1] ---------------
    availability = (
        0.40 * recency_intent
        + 0.30 * responsiveness
        + 0.15 * trust
        + 0.15 * notice_score
    )
    availability = max(0.0, min(1.0, availability))

    # --- Map [0,1] availability onto the configured multiplier range ------
    lo, hi = config.BEHAVIORAL_MULT_FLOOR, config.BEHAVIORAL_MULT_CEILING
    multiplier = lo + availability * (hi - lo)

    summary_bits = []
    d = _parse_date(sig.get("last_active_date"))
    if d is not None:
        days = (TODAY - d).days
        if days <= 7:
            summary_bits.append("active in the last week")
        elif days <= 30:
            summary_bits.append("active in the last month")
        elif days <= 90:
            summary_bits.append(f"last active {days} days ago")
        else:
            summary_bits.append(f"inactive for {days} days")
    if response_rate is not None:
        summary_bits.append(f"recruiter response rate {response_rate:.0%}")
    summary = "; ".join(summary_bits)

    return BehavioralBreakdown(
        multiplier=round(multiplier, 4),
        recency_score=round(recency, 4),
        responsiveness_score=round(responsiveness, 4),
        trust_score=round(trust, 4),
        open_to_work=open_to_work,
        notice_period_days=sig.get("notice_period_days"),
        recruiter_response_rate=response_rate,
        last_active_date=sig.get("last_active_date"),
        summary=summary,
    )