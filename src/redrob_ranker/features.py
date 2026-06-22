"""
features.py
============
All rule-based, JD-grounded feature scoring that does NOT depend on the
TF-IDF/SVD semantic layer (that lives in semantic.py). Every function here
returns both a numeric score AND a short human-traceable explanation,
because the submission spec requires a `reasoning` column that references
*specific facts* from the candidate profile -- we want the scoring code
itself to already know which facts mattered, rather than reconstructing
that after the fact with a separate heuristic.

This module is the direct implementation of the JD's "skills inventory"
section and its disqualifier list, using config.py constants exclusively
so the weights/thresholds stay in one auditable place.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

from . import config

_PRODUCTION_EVIDENCE_RE = re.compile(config.PRODUCTION_EVIDENCE_PATTERN, re.IGNORECASE)
_EVAL_TEXT_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in config.MUST_HAVE_SKILL_GROUPS["eval_for_ranking"]["text_patterns"]
]


# ---------------------------------------------------------------------------
# Skill trust
# ---------------------------------------------------------------------------
def compute_skill_trust(candidate: dict) -> dict[str, float]:
    """Return {skill_name: trust_in_[0,1]} for every skill the candidate
    lists. Trust combines claimed proficiency, time-on-skill, and (when
    available) whether the candidate's own Redrob skill-assessment score
    contradicts the claim. This is the direct countermeasure to keyword
    stuffing: a "Pinecone (expert)" claim used for 1 month carries far less
    weight than the same claim backed by 4 years and a high assessment
    score.
    """
    assessment_scores = (candidate.get("redrob_signals", {}) or {}).get(
        "skill_assessment_scores", {}
    ) or {}

    trust: dict[str, float] = {}
    for s in candidate.get("skills", []) or []:
        name = s.get("name")
        if not name:
            continue
        prof = s.get("proficiency", "beginner")
        base = config.PROFICIENCY_BASE_TRUST.get(prof, 0.30)

        duration = float(s.get("duration_months", 0) or 0)
        duration_factor = min(1.0, duration / config.DURATION_TRUST_FULL_MONTHS)
        # Never let duration alone crush trust to 0 -- a brand-new hire on a
        # skill they're genuinely expert at (e.g. moved companies) is still
        # plausible; floor the duration factor at 0.25.
        duration_factor = max(0.25, duration_factor)

        value = base * duration_factor

        if name in assessment_scores and prof in ("advanced", "expert"):
            if assessment_scores[name] < config.ASSESSMENT_CONTRADICTION_THRESHOLD:
                value *= config.ASSESSMENT_CONTRADICTION_DISCOUNT

        trust[name] = round(min(1.0, value), 4)

    return trust


# ---------------------------------------------------------------------------
# Title tier
# ---------------------------------------------------------------------------
@dataclass
class TitleScoreResult:
    score: float
    past_title_bonus: float
    explanation: str


def score_title(candidate: dict) -> TitleScoreResult:
    profile = candidate.get("profile", {})
    current_title = profile.get("current_title", "")
    base = config.TITLE_TIER_SCORE.get(current_title, config.DEFAULT_TITLE_TIER_SCORE)

    # Credit strong past titles even if the current title has drifted --
    # e.g. someone who was "Senior NLP Engineer" then is now "Tech Lead".
    past_bonus = 0.0
    best_past_title = None
    for ch in candidate.get("career_history", []) or []:
        past_title = ch.get("title", "")
        if past_title == current_title:
            continue
        past_score = config.TITLE_TIER_SCORE.get(past_title)
        if past_score is not None:
            candidate_bonus = past_score * config.PAST_TITLE_CREDIT_WEIGHT
            if candidate_bonus > past_bonus:
                past_bonus = candidate_bonus
                best_past_title = past_title
    past_bonus = min(past_bonus, config.PAST_TITLE_CREDIT_CAP)

    explanation = f"current title '{current_title}' (base {base:.2f})"
    if past_bonus > 0:
        explanation += f", past title '{best_past_title}' adds +{past_bonus:.2f}"

    return TitleScoreResult(
        score=round(min(1.0, base + past_bonus), 4),
        past_title_bonus=round(past_bonus, 4),
        explanation=explanation,
    )


# ---------------------------------------------------------------------------
# Must-have skill groups (JD: "things you absolutely need")
# ---------------------------------------------------------------------------
@dataclass
class MustHaveResult:
    score: float
    group_scores: dict[str, float]
    matched_skills: list[str]
    groups_satisfied: int


def score_must_have_skills(
    candidate: dict, skill_trust: dict[str, float]
) -> MustHaveResult:
    career_text = " ".join(
        (ch.get("description") or "") for ch in candidate.get("career_history", []) or []
    )

    group_scores: dict[str, float] = {}
    matched_skills: list[str] = []
    groups_satisfied = 0

    for group_name, group_def in config.MUST_HAVE_SKILL_GROUPS.items():
        skill_list = group_def.get("skills", [])
        best_trust = 0.0
        for skill_name in skill_list:
            t = skill_trust.get(skill_name)
            if t is not None:
                if t > best_trust:
                    best_trust = t
                matched_skills.append(skill_name)

        # Secondary supporting evidence (e.g. PyTorch/TensorFlow support the
        # "Python used for real ML" claim without being a hard requirement).
        support_skills = group_def.get("support_skills", [])
        support_bonus = 0.0
        for skill_name in support_skills:
            t = skill_trust.get(skill_name)
            if t is not None:
                support_bonus = max(support_bonus, 0.15 * t)

        # Free-text evidence (NDCG/MRR/MAP/A-B-testing language) for the
        # evaluation-framework group, since this dataset models that group
        # mostly as career-history prose rather than a discrete skill tag.
        text_bonus = 0.0
        if group_def.get("text_patterns"):
            if any(p.search(career_text) for p in _EVAL_TEXT_PATTERNS):
                text_bonus = 0.6

        group_value = min(1.0, best_trust + support_bonus + text_bonus)
        group_scores[group_name] = round(group_value, 4)
        if group_value >= 0.35:
            groups_satisfied += 1

    weighted = sum(
        group_scores[g] * config.MUST_HAVE_SKILL_GROUPS[g]["weight"]
        for g in group_scores
    )

    return MustHaveResult(
        score=round(weighted, 4),
        group_scores=group_scores,
        matched_skills=sorted(set(matched_skills)),
        groups_satisfied=groups_satisfied,
    )


def score_nice_to_have(skill_trust: dict[str, float]) -> float:
    bonus = 0.0
    for skill_name in config.NICE_TO_HAVE_SKILLS:
        t = skill_trust.get(skill_name)
        if t is not None and t > 0.3:
            bonus += config.NICE_TO_HAVE_BONUS_PER_SKILL
    return round(min(config.NICE_TO_HAVE_BONUS_CAP, bonus), 4)


# ---------------------------------------------------------------------------
# Experience band fit
# ---------------------------------------------------------------------------
def score_experience_band(years_of_experience: float) -> float:
    if config.EXPERIENCE_BAND_MIN <= years_of_experience <= config.EXPERIENCE_BAND_MAX:
        return 1.0
    if years_of_experience < config.EXPERIENCE_BAND_MIN:
        gap = config.EXPERIENCE_BAND_MIN - years_of_experience
    else:
        gap = years_of_experience - config.EXPERIENCE_BAND_MAX
    score = 1.0 - gap * config.EXPERIENCE_FALLOFF_PER_YEAR
    return round(max(config.EXPERIENCE_FALLOFF_FLOOR, score), 4)


# ---------------------------------------------------------------------------
# Location fit
# ---------------------------------------------------------------------------
def score_location(profile: dict, redrob_signals: dict) -> float:
    location = (profile.get("location") or "").lower()
    country = (profile.get("country") or "").lower()
    willing_relocate = bool(redrob_signals.get("willing_to_relocate", False))

    if country == "india":
        for key, val in config.LOCATION_TIER_SCORE.items():
            if key in location:
                return val
        score = config.DEFAULT_INDIA_LOCATION_SCORE
        if willing_relocate:
            score = min(1.0, score + config.WILLING_RELOCATE_INDIA_BONUS)
        return round(score, 4)

    score = config.OUTSIDE_INDIA_BASE_SCORE
    if willing_relocate:
        score += config.OUTSIDE_INDIA_RELOCATE_BONUS
    return round(score, 4)


# ---------------------------------------------------------------------------
# Notice period fit (as a composite-score component, distinct from its
# smaller role inside the behavioral availability multiplier)
# ---------------------------------------------------------------------------
def score_notice_period(notice_period_days: int | None) -> float:
    if notice_period_days is None:
        return 0.7
    for threshold, score in config.NOTICE_PERIOD_SCORE_BREAKS:
        if notice_period_days <= threshold:
            return score
    return 0.30


# ---------------------------------------------------------------------------
# Education fit
# ---------------------------------------------------------------------------
def score_education(education: list[dict]) -> float:
    if not education:
        return 0.55
    best = 0.0
    for e in education:
        tier = e.get("tier", "unknown")
        val = config.EDUCATION_TIER_SCORE.get(tier, 0.65)
        field = (e.get("field_of_study") or "").lower()
        if any(rf in field for rf in config.EDUCATION_RELEVANT_FIELDS):
            val = min(1.0, val + config.EDUCATION_RELEVANT_FIELD_BONUS)
        best = max(best, val)
    return round(best, 4)


# ---------------------------------------------------------------------------
# JD disqualifier penalties -- combined into one multiplier, with reasons
# ---------------------------------------------------------------------------
@dataclass
class DisqualifierResult:
    multiplier: float = 1.0
    reasons: list[str] = field(default_factory=list)


def score_disqualifiers(
    candidate: dict, skill_trust: dict[str, float], total_career_years: float
) -> DisqualifierResult:
    multiplier = 1.0
    reasons: list[str] = []

    career = candidate.get("career_history", []) or []
    profile = candidate.get("profile", {})

    # --- Consulting-only career ------------------------------------------
    all_consulting = len(career) > 0 and all(
        (ch.get("industry") or "").strip().lower() in config.CONSULTING_INDUSTRY_LABELS
        or (ch.get("company") or "").strip().lower() in config.CONSULTING_COMPANY_NAMES
        for ch in career
    )
    if all_consulting:
        multiplier *= config.CONSULTING_ONLY_PENALTY
        reasons.append("entire career at consulting/IT-services firms only")

    # --- Recent LLM-wrapper pivot with no pre-LLM-era production depth ---
    ai_groups_present = any(
        skill_trust.get(s, 0) > 0.3
        for grp in ("embeddings_retrieval", "vector_db_hybrid_search")
        for s in config.MUST_HAVE_SKILL_GROUPS[grp]["skills"]
    )
    if career:
        most_recent = max(career, key=lambda c: c.get("start_date") or "")
        recent_ai_only = (
            ai_groups_present
            and (most_recent.get("duration_months") or 999) <= config.RECENT_PIVOT_MAX_MONTHS
            and (total_career_years - (most_recent.get("duration_months") or 0) / 12.0)
            < config.RECENT_PIVOT_MIN_PRIOR_YEARS
        )
        if recent_ai_only:
            multiplier *= config.RECENT_PIVOT_PENALTY
            reasons.append(
                "AI/retrieval experience appears concentrated in the most "
                "recent role only, with limited prior production history"
            )

    # --- CV/speech-only without NLP/IR overlap ----------------------------
    skill_names = {s.get("name") for s in candidate.get("skills", []) or []}
    cv_speech_hits = skill_names & config.CV_SPEECH_ONLY_SKILLS
    nlp_ir_hits = skill_names & config.NLP_IR_OVERLAP_SKILLS
    if len(cv_speech_hits) >= 2 and len(nlp_ir_hits) == 0:
        multiplier *= config.CV_SPEECH_ONLY_PENALTY
        reasons.append(
            f"skills concentrated in computer-vision/speech ({sorted(cv_speech_hits)}) "
            "with no NLP/IR overlap"
        )

    # --- Closed-source-only, long tenure, no external validation ----------
    redrob = candidate.get("redrob_signals", {}) or {}
    github_score = redrob.get("github_activity_score", -1)
    certifications = candidate.get("certifications", []) or []
    if (
        total_career_years >= config.CLOSED_SOURCE_MIN_YEARS
        and (github_score is None or github_score < 0)
        and not certifications
    ):
        multiplier *= config.CLOSED_SOURCE_PENALTY
        reasons.append(
            f"{total_career_years:.1f}y experience with no linked GitHub activity "
            "and no certifications on file (no external validation signal)"
        )

    # --- Title-chaser / job-hopping pattern --------------------------------
    if len(career) >= config.TITLE_CHASER_MIN_JOBS:
        avg_tenure = statistics.mean(c.get("duration_months", 0) or 0 for c in career)
        current_title_lower = (profile.get("current_title") or "").lower()
        is_senior_title = any(
            w in current_title_lower for w in config.TITLE_CHASER_SENIOR_WORDS
        )
        if avg_tenure < config.TITLE_CHASER_MAX_AVG_TENURE_MONTHS and is_senior_title:
            multiplier *= config.TITLE_CHASER_PENALTY
            reasons.append(
                f"{len(career)} employers at an average tenure of "
                f"{avg_tenure:.0f} months, reaching a senior-sounding title quickly"
            )

    # --- Pure-research career, no production deployment evidence ----------
    current_title_lower = (profile.get("current_title") or "").lower()
    if any(m in current_title_lower for m in config.PURE_RESEARCH_TITLE_MARKERS):
        career_text = " ".join((ch.get("description") or "") for ch in career)
        if not _PRODUCTION_EVIDENCE_RE.search(career_text):
            multiplier *= config.PURE_RESEARCH_PENALTY
            reasons.append(
                "research-flavored title with no production-deployment "
                "language found in career history"
            )

    return DisqualifierResult(multiplier=round(multiplier, 4), reasons=reasons)
