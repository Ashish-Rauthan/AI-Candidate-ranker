from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from . import config
from .behavioral import BehavioralBreakdown
from .consistency import ConsistencyResult
from .features import DisqualifierResult, MustHaveResult, TitleScoreResult


def percentile_rank(values: np.ndarray) -> np.ndarray:
    """Map an array of raw scores to their percentile rank in [0, 1].

    Used to normalize the semantic-similarity component: raw TF-IDF/SVD
    cosine similarity is heavily right-skewed (most candidates cluster near
    0, a small relevant set sits much higher), so percentile rank within
    the actual candidate pool is a more meaningful [0,1] score than a fixed
    linear rescale would be, and it composes naturally with the other
    already-[0,1] feature scores.
    """
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values))
    if len(values) > 1:
        ranks = ranks / (len(values) - 1)
    return ranks


@dataclass
class CandidateScoreBreakdown:
    candidate_id: str
    # Small set of display fields needed for reasoning / CSV output, copied
    # out at extraction time so downstream code never needs to hold the
    # full raw candidate dict (which matters across 100,000 records).
    title: str
    company: str
    years_of_experience: float

    base_fit: float
    final_score: float
    title_result: TitleScoreResult
    must_have_result: MustHaveResult
    nice_to_have_bonus: float
    experience_score: float
    location_score: float
    notice_score: float
    education_score: float
    semantic_similarity_raw: float
    semantic_score_normalized: float
    disqualifier_result: DisqualifierResult
    behavioral: BehavioralBreakdown
    consistency: ConsistencyResult
    component_scores: dict[str, float] = field(default_factory=dict)


def compute_base_fit(
    title_score: float,
    semantic_score: float,
    must_have_score: float,
    nice_to_have_bonus: float,
    experience_score: float,
    location_score: float,
    notice_score: float,
    education_score: float,
) -> tuple[float, dict[str, float]]:
    w = config.COMPOSITE_WEIGHTS
    components = {
        "title_tier": title_score * w["title_tier"],
        "semantic_similarity": semantic_score * w["semantic_similarity"],
        "must_have_skills": must_have_score * w["must_have_skills"],
        "experience_band": experience_score * w["experience_band"],
        "location": location_score * w["location"],
        "notice_period": notice_score * w["notice_period"],
        "education": education_score * w["education"],
    }
    base_fit = sum(components.values()) + nice_to_have_bonus
    base_fit = max(0.0, min(1.0, base_fit))
    return base_fit, components


def compute_final_score(
    base_fit: float,
    disqualifier_multiplier: float,
    behavioral_multiplier: float,
    is_honeypot: bool,
) -> float:
    score = base_fit * disqualifier_multiplier
    if is_honeypot:
        score *= config.HONEYPOT_SCORE_MULTIPLIER
    score *= behavioral_multiplier
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Reasoning text generation
# ---------------------------------------------------------------------------
_OPENERS = [
    "{title} with {yoe:.1f} years of experience at {company}",
    "{yoe:.1f}-year {title} currently at {company}",
    "{title} ({yoe:.1f}y experience), currently at {company}",
]
_STRONG_SKILL_FRAGMENTS = [
    "hands-on production background in {skills}",
    "direct experience with {skills}",
    "has shipped work involving {skills}",
]
_PARTIAL_SKILL_FRAGMENTS = [
    "some overlap with the JD's core stack ({skills}) but not across all areas",
    "partial coverage of the JD's must-haves, strongest in {skills}",
]
_WEAK_SKILL_FRAGMENTS = [
    "limited overlap with the JD's embeddings/retrieval/ranking requirements",
    "skill list does not show meaningful retrieval, vector-search, or ranking experience",
]


def _pick(seed_str: str, options: list[str], salt: str = "") -> str:
    """Deterministic-but-pseudo-random choice from a small set of phrasing
    options, so the 100 reasoning rows vary in sentence structure even
    when the underlying facts are similar. Determinism (vs. true
    randomness) keeps repeated runs of the pipeline producing identical
    output, which matters for Stage-3 reproducibility.
    """
    h = hashlib.sha256((seed_str + salt).encode()).hexdigest()
    idx = int(h, 16) % len(options)
    return options[idx]


def generate_reasoning(breakdown: CandidateScoreBreakdown) -> str:
    candidate_id = breakdown.candidate_id
    title = breakdown.title or "Unknown title"
    company = breakdown.company or "an unspecified company"
    yoe = float(breakdown.years_of_experience or 0)

    opener = _pick(candidate_id, _OPENERS, "opener").format(
        title=title, yoe=yoe, company=company
    )

    matched = breakdown.must_have_result.matched_skills
    groups_satisfied = breakdown.must_have_result.groups_satisfied
    if groups_satisfied >= 3 and matched:
        skill_str = ", ".join(matched[:4])
        skill_fragment = _pick(candidate_id, _STRONG_SKILL_FRAGMENTS, "skill").format(
            skills=skill_str
        )
    elif groups_satisfied >= 1 and matched:
        skill_str = ", ".join(matched[:3])
        skill_fragment = _pick(candidate_id, _PARTIAL_SKILL_FRAGMENTS, "skill").format(
            skills=skill_str
        )
    else:
        skill_fragment = _pick(candidate_id, _WEAK_SKILL_FRAGMENTS, "skill")

    sentence_1 = f"{opener}; {skill_fragment}."

    extra_clauses: list[str] = []

    # Behavioral / availability clause -- only surface when it's actually
    # informative (clearly strong or clearly weak), to avoid bloating every
    # single row with boilerplate.
    resp_rate = breakdown.behavioral.recruiter_response_rate
    if resp_rate is not None:
        if breakdown.behavioral.multiplier >= 0.95:
            extra_clauses.append(
                f"active on the platform with a {resp_rate:.0%} recruiter response rate"
            )
        elif breakdown.behavioral.multiplier <= 0.70:
            extra_clauses.append(
                f"availability is a concern ({breakdown.behavioral.summary or 'low recent activity'})"
            )

    # Notice period, when notably long.
    notice = breakdown.behavioral.notice_period_days
    try:
        notice_int = int(notice) if notice is not None else None
    except (TypeError, ValueError):
        notice_int = None
    if notice_int is not None and notice_int > 60:
        extra_clauses.append(f"{notice_int}-day notice period is on the longer side")

    # Disqualifier concerns, if any survived (these candidates can still be
    # in the top 100 if everything else is strong -- being honest about the
    # gap is exactly what the spec's "Honest concerns" check rewards).
    if breakdown.disqualifier_result.reasons:
        extra_clauses.append(breakdown.disqualifier_result.reasons[0])

    sentence_2 = ""
    if extra_clauses:
        # Keep it to at most two clauses to respect the 1-2 sentence limit.
        sentence_2 = " " + "; ".join(extra_clauses[:2]).capitalize() + "."

    reasoning = (sentence_1 + sentence_2).strip()
    # Hard safety cap so we never produce an unreasonably long cell.
    if len(reasoning) > 400:
        reasoning = reasoning[:397].rstrip() + "..."
    return reasoning
