"""
behavioral.py
=============
Converts `redrob_signals` (the 23 platform-activity fields) into a single
bounded multiplier in [BEHAVIORAL_MULT_FLOOR, BEHAVIORAL_MULT_CEILING],
applied on top of the skills/title/experience fit score.

Per redrob_signals_doc.docx, these signals answer a different question than
skill-fit: not "is this person qualified" but "can we actually hire this
person right now". The doc's own example is the one we anchor to: a
perfect-on-paper candidate who hasn't logged in for 6 months and has a 5%
recruiter response rate is, for hiring purposes, not actually available.

We deliberately keep this as a *multiplier*, not an additive term, and cap
its range fairly tight (0.55x-1.05x by default). Two reasons:
  1. The JD's own framing is "down-weight them appropriately" -- a
     well-qualified-but-quiet candidate should drop several ranks, not get
     buried below a 0.3-skill-match candidate who happens to be very active.
  2. A multiplier that could swing to 0 would let pure availability noise
     dominate the ranking, which is exactly the over-reliance on engagement
     metrics the JD's "vibe check" section implicitly warns against (it asks
     for systems thinking, not engagement-chasing).

Signals used, grouped by what they actually measure:
  - Recency / intent:     last_active_date, open_to_work_flag, signup recency
  - Responsiveness:       recruiter_response_rate, avg_response_time_hours,
                           interview_completion_rate
  - Market interest in them: profile_views_received_30d, search_appearance_30d,
                           saved_by_recruiters_30d (light signal only --
                           this reflects how others rate them, somewhat
                           circular for a ranking system to lean on heavily)
  - Trust / verification:  verified_email, verified_phone, linkedin_connected
  - Outcome history:       offer_acceptance_rate (handle -1 sentinel = no
                           offer history -> neutral, not negative)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from . import config

TODAY = date(2026, 6, 18)


def _parse_date(s: str | None) -> date | None:
    if not s:
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


def score_behavioral(candidate: dict) -> BehavioralBreakdown:
    sig = candidate.get("redrob_signals", {}) or {}

    # --- Recency / intent ---------------------------------------------
    recency = _recency_score(sig.get("last_active_date"))
    open_to_work = bool(sig.get("open_to_work_flag", False))
    intent = 1.0 if open_to_work else 0.65
    recency_intent = 0.7 * recency + 0.3 * intent

    # --- Responsiveness --------------------------------------------------
    response_rate = sig.get("recruiter_response_rate")
    response_rate = 0.5 if response_rate is None else float(response_rate)
    resp_time = sig.get("avg_response_time_hours")
    # Map response time to [0,1]: <=24h great, >=120h poor, linear between.
    if resp_time is None:
        resp_time_score = 0.5
    else:
        resp_time_score = max(0.0, min(1.0, 1.0 - (float(resp_time) - 24.0) / 96.0))
    interview_completion = sig.get("interview_completion_rate")
    interview_completion = 0.7 if interview_completion is None else float(interview_completion)

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
    if offer_accept is not None and offer_accept >= 0:
        # blend in offer-acceptance history, but lightly -- a low rate can
        # simply mean the candidate is selective, not unavailable.
        trust = 0.85 * trust + 0.15 * float(offer_accept)

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
