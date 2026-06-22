"""
pipeline.py
===========
End-to-end orchestration, written to comply with the submission spec's
compute constraints (<=5 min wall-clock, <=16GB RAM, CPU-only, no network)
on the full 100,000-candidate pool, on commodity hardware.

Memory design note: we never hold all 100,000 *raw* candidate dicts in
memory at once. We stream the file once with io_utils.iter_candidates,
and for each candidate immediately reduce it down to (a) a TF-IDF document
string and (b) a small `_InterimRecord` of already-computed feature
scores -- discarding the raw dict before moving to the next line. Only the
final top-100 candidates need their full reasoning text, and the few
display fields needed for that (title, company, years of experience) are
captured into `_InterimRecord` during the single streaming pass, so a
second file read is never required.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, features
from .behavioral import BehavioralBreakdown, score_behavioral
from .consistency import ConsistencyResult, check_consistency
from .features import DisqualifierResult, MustHaveResult, TitleScoreResult
from .io_utils import iter_candidates
from .scoring import (
    CandidateScoreBreakdown,
    compute_base_fit,
    compute_final_score,
    generate_reasoning,
    percentile_rank,
)
from .semantic import build_document, build_index, load_cached_index, save_index


@dataclass
class _InterimRecord:
    candidate_id: str
    title: str
    company: str
    years_of_experience: float
    title_result: TitleScoreResult
    must_have_result: MustHaveResult
    nice_to_have_bonus: float
    experience_score: float
    location_score: float
    notice_score: float
    education_score: float
    disqualifier_result: DisqualifierResult
    behavioral: BehavioralBreakdown
    consistency: ConsistencyResult


def _extract_interim(candidate: dict) -> tuple[_InterimRecord, str]:
    """Reduce one raw candidate dict to its feature record + TF-IDF
    document text. This is where the raw dict's lifetime ends -- nothing
    from `candidate` survives past this function except what's copied into
    the returned record/string.
    """
    profile = candidate.get("profile", {})
    redrob_signals = candidate.get("redrob_signals", {}) or {}
    career = candidate.get("career_history", []) or []

    skill_trust = features.compute_skill_trust(candidate)
    title_result = features.score_title(candidate)
    must_have_result = features.score_must_have_skills(candidate, skill_trust)
    nice_to_have_bonus = features.score_nice_to_have(skill_trust)

    total_career_years = sum(c.get("duration_months", 0) or 0 for c in career) / 12.0
    experience_score = features.score_experience_band(
        float(profile.get("years_of_experience", 0) or 0)
    )
    location_score = features.score_location(profile, redrob_signals)
    notice_score = features.score_notice_period(redrob_signals.get("notice_period_days"))
    education_score = features.score_education(candidate.get("education", []) or [])
    disqualifier_result = features.score_disqualifiers(candidate, skill_trust, total_career_years)
    behavioral = score_behavioral(candidate)
    consistency = check_consistency(candidate)

    record = _InterimRecord(
        candidate_id=candidate["candidate_id"],
        title=profile.get("current_title", ""),
        company=profile.get("current_company", ""),
        years_of_experience=float(profile.get("years_of_experience", 0) or 0),
        title_result=title_result,
        must_have_result=must_have_result,
        nice_to_have_bonus=nice_to_have_bonus,
        experience_score=experience_score,
        location_score=location_score,
        notice_score=notice_score,
        education_score=education_score,
        disqualifier_result=disqualifier_result,
        behavioral=behavioral,
        consistency=consistency,
    )
    doc_text = build_document(candidate)
    return record, doc_text


def run_pipeline(
    candidates_path: str | Path,
    artifacts_dir: str | Path = "artifacts",
    jd_text: str = config.JD_POSITIVE_QUERY_TEXT,
    top_n: int = 100,
    use_cache: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    t_start = time.time()

    # --- Pass 1: stream the full pool, extract features + doc text --------
    interim_records: list[_InterimRecord] = []
    documents: list[str] = []
    candidate_ids: list[str] = []

    n = 0
    for candidate in iter_candidates(candidates_path):
        record, doc_text = _extract_interim(candidate)
        interim_records.append(record)
        documents.append(doc_text)
        candidate_ids.append(record.candidate_id)
        n += 1
        if verbose and n % 25000 == 0:
            print(f"[pipeline] processed {n} candidates ({time.time()-t_start:.1f}s elapsed)")

    if verbose:
        print(f"[pipeline] feature extraction complete: {n} candidates, {time.time()-t_start:.1f}s")

    # --- Semantic layer: load cached index if available, else build it ----
    index = load_cached_index(artifacts_dir) if use_cache else None
    if index is None or index.candidate_ids != candidate_ids:
        if verbose:
            print("[pipeline] no valid cached semantic index found; building one now")
        index = build_index(candidate_ids, documents, verbose=verbose)
        if use_cache:
            save_index(index, artifacts_dir)
    else:
        if verbose:
            print("[pipeline] loaded cached semantic index")

    sims_raw = index.similarity_to_jd(jd_text)
    sims_norm = percentile_rank(sims_raw)

    if verbose:
        print(f"[pipeline] semantic similarity computed, {time.time()-t_start:.1f}s elapsed")

    # --- Pass 2 (in-memory, cheap): assemble final composite scores -------
    breakdowns: list[CandidateScoreBreakdown] = []
    for i, rec in enumerate(interim_records):
        base_fit, components = compute_base_fit(
            title_score=rec.title_result.score,
            semantic_score=float(sims_norm[i]),
            must_have_score=rec.must_have_result.score,
            nice_to_have_bonus=rec.nice_to_have_bonus,
            experience_score=rec.experience_score,
            location_score=rec.location_score,
            notice_score=rec.notice_score,
            education_score=rec.education_score,
        )
        final_score = compute_final_score(
            base_fit=base_fit,
            disqualifier_multiplier=rec.disqualifier_result.multiplier,
            behavioral_multiplier=rec.behavioral.multiplier,
            is_honeypot=rec.consistency.is_honeypot,
        )
        breakdowns.append(
            CandidateScoreBreakdown(
                candidate_id=rec.candidate_id,
                title=rec.title,
                company=rec.company,
                years_of_experience=rec.years_of_experience,
                base_fit=round(base_fit, 4),
                final_score=round(final_score, 6),
                title_result=rec.title_result,
                must_have_result=rec.must_have_result,
                nice_to_have_bonus=rec.nice_to_have_bonus,
                experience_score=rec.experience_score,
                location_score=rec.location_score,
                notice_score=rec.notice_score,
                education_score=rec.education_score,
                semantic_similarity_raw=round(float(sims_raw[i]), 6),
                semantic_score_normalized=round(float(sims_norm[i]), 6),
                disqualifier_result=rec.disqualifier_result,
                behavioral=rec.behavioral,
                consistency=rec.consistency,
                component_scores=components,
            )
        )

    # --- Rank and select top N ---------------------------------------------
    # Secondary deterministic tiebreak (per submission_spec.docx section 3):
    # ties broken by candidate_id ascending. Important: we tie-break on the
    # *rounded* score (4 decimals) -- the same precision written to the CSV
    # -- not the raw internal float, since two candidates can be distinct
    # internally but display identically once rounded; the spec's tie-break
    # rule is about what's visible in the submission file.
    breakdowns.sort(key=lambda b: (-round(b.final_score, 4), b.candidate_id))
    top = breakdowns[:top_n]

    if verbose:
        n_honeypots_in_top = sum(1 for b in top if b.consistency.is_honeypot)
        print(
            f"[pipeline] ranking complete, {time.time()-t_start:.1f}s elapsed. "
            f"Honeypots in top {top_n}: {n_honeypots_in_top}"
        )

    rows = []
    for rank, b in enumerate(top, start=1):
        rows.append(
            {
                "candidate_id": b.candidate_id,
                "rank": rank,
                "score": round(b.final_score, 4),
                "reasoning": generate_reasoning(b),
            }
        )

    df = pd.DataFrame(rows, columns=["candidate_id", "rank", "score", "reasoning"])

    if verbose:
        print(f"[pipeline] DONE in {time.time()-t_start:.1f}s total")

    return df
