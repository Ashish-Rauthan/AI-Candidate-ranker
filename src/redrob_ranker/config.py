from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. TITLE TIER SCORE
# ---------------------------------------------------------------------------
# The dataset has exactly 47 distinct `current_title` values (verified by a
# full scan of candidates.jsonl). Each is mapped to a base relevance score
# in [0, 1] reflecting how directly the title maps to the JD's mandate:
# "own the intelligence layer -- ranking, retrieval, and matching systems."
#
# This score is a STARTING POINT, not a verdict -- features.py layers
# skill-trust, production-evidence, and disqualifier checks on top, so a
# Tier-2 "Data Engineer" with genuine retrieval/ranking career history can
# still out-score a Tier-4 title with a thin, unverified skill list.
TITLE_TIER_SCORE: dict[str, float] = {
    # --- Tier 5: exact match to the JD's "ideal candidate" titles ---------
    "Senior AI Engineer": 1.00,
    "Lead AI Engineer": 1.00,
    "Senior Applied Scientist": 0.97,
    "Staff Machine Learning Engineer": 0.97,
    "Senior Machine Learning Engineer": 0.95,
    "Senior NLP Engineer": 0.95,

    # --- Tier 4: strong mid/senior roles squarely in retrieval/ranking/NLP -
    "Search Engineer": 0.88,                 # IR/search is the JD's core mandate
    "Recommendation Systems Engineer": 0.88,  # ranking systems = core JD mandate
    "AI Engineer": 0.85,
    "NLP Engineer": 0.85,
    "Applied ML Engineer": 0.82,
    "Senior Software Engineer (ML)": 0.78,
    "Machine Learning Engineer": 0.80,
    "Senior Data Scientist": 0.75,

    # --- Tier 3: relevant, but the JD raises specific cautions here --------
    "Data Scientist": 0.62,
    "ML Engineer": 0.60,
    # JD: "pure research environments ... without production deployment" --
    # this title needs the production_evidence check, so its base is modest.
    "AI Research Engineer": 0.55,
    "AI Specialist": 0.55,
    # "Junior" implies below the 5-9y band the JD targets.
    "Junior ML Engineer": 0.45,
    # JD: "primary expertise is computer vision ... without significant
    # NLP/IR exposure" -- explicit caution, base score is deliberately low;
    # must_have_skill_score gives credit back if NLP/IR skills are present.
    "Computer Vision Engineer": 0.40,

    # --- Tier 2: adjacent engineering -- possible "Tier-5 hidden gem" if
    # career history shows real retrieval/ranking/eval work (JD explicitly
    # warns against dismissing these purely on title) -------------------
    "Senior Data Engineer": 0.38,
    "Data Engineer": 0.35,
    "Analytics Engineer": 0.35,
    "Data Analyst": 0.30,
    "Senior Software Engineer": 0.33,
    "Backend Engineer": 0.32,
    "Software Engineer": 0.28,
    "Cloud Engineer": 0.22,
    "DevOps Engineer": 0.20,
    "Full Stack Developer": 0.18,

    # --- Tier 1: generic software engineering, unlikely but possible -------
    "Frontend Engineer": 0.12,
    "Java Developer": 0.12,
    "Mobile Developer": 0.10,
    "QA Engineer": 0.10,
    ".NET Developer": 0.10,

    # --- Tier 0: explicitly unrelated to the role ---------------------------
    "Project Manager": 0.06,
    "Business Analyst": 0.04,
    "HR Manager": 0.04,
    "Mechanical Engineer": 0.04,
    "Operations Manager": 0.04,
    "Content Writer": 0.04,
    "Marketing Manager": 0.04,
    "Accountant": 0.03,
    "Customer Support": 0.03,
    "Sales Executive": 0.03,
    "Civil Engineer": 0.03,
    "Graphic Designer": 0.03,
}
# Defensive fallback if a title appears that isn't in the table above
# (e.g. the hidden eval set has titles not present in this public sample).
DEFAULT_TITLE_TIER_SCORE = 0.15

# Credit for *past* titles in career_history that are themselves strong
# AI/IR titles, even if the candidate's current title has drifted (e.g.
# promoted into a generic "Tech Lead" label). Small, capped contribution.
PAST_TITLE_CREDIT_WEIGHT = 0.5          # past-title tier score is scaled by this
PAST_TITLE_CREDIT_CAP = 0.25            # and capped at this absolute bonus

# ---------------------------------------------------------------------------
# 2. MUST-HAVE SKILL GROUPS
# ---------------------------------------------------------------------------
# Directly from job_description.docx, "Things you absolutely need":
#   1. embeddings-based retrieval systems (sentence-transformers, OpenAI
#      embeddings, BGE, E5, or similar)
#   2. vector databases or hybrid search infra (Pinecone, Weaviate, Qdrant,
#      Milvus, OpenSearch, Elasticsearch, FAISS, or similar)
#   3. Strong Python
#   4. evaluation frameworks for ranking systems (NDCG, MRR, MAP,
#      offline-to-online correlation, A/B test interpretation)
#
# Skill names below are drawn from the *actual* 133-value skill vocabulary
# found in candidates.jsonl (verified by a full scan), not guessed.
MUST_HAVE_SKILL_GROUPS: dict[str, dict] = {
    "embeddings_retrieval": {
        "skills": [
            "Embeddings", "Sentence Transformers", "Semantic Search",
            "Vector Search", "Hugging Face Transformers", "Fine-tuning LLMs",
            "RAG", "LlamaIndex", "Haystack",
            "Vector Representations",       # dataset vocab — synonym for dense embeddings
        ],
        "weight": 0.30,
    },
    "vector_db_hybrid_search": {
        "skills": [
            "Pinecone", "FAISS", "Weaviate", "Milvus", "Qdrant",
            "OpenSearch", "Elasticsearch", "pgvector", "BM25",
            "Search Backend",               # dataset vocab — infra for search systems
            "Search Infrastructure",        # dataset vocab — same cluster
            "Search & Discovery",           # dataset vocab — product-layer search
        ],
        "weight": 0.30,
    },
    "python_production": {
        "skills": ["Python"],
        # secondary evidence that Python is used for real production ML,
        # not just listed -- contributes a smaller supporting boost.
        "support_skills": [
            "PyTorch", "TensorFlow", "scikit-learn", "FastAPI", "Flask", "Django",
        ],
        "weight": 0.20,
    },
    "eval_for_ranking": {
        "skills": [
            "Learning to Rank",
            "Ranking Systems",              # dataset vocab — directly maps to JD mandate
            "Information Retrieval Systems",# dataset vocab — IR umbrella term
        ],
        # NDCG / MRR / MAP / A-B-testing are not modeled as discrete `skills`
        # entries in this dataset -- they show up as free text inside
        # career_history descriptions instead, so we check both.
        "text_patterns": [
            r"NDCG", r"MRR", r"\bMAP\b", r"A/?B test",
            r"offline.{0,15}online", r"recall@",
        ],
        "weight": 0.20,
    },
}

# JD: "Things we'd like you to have but won't reject you for" -- small
# additive bonus, not a hard requirement. Covers all four JD nice-to-have
# categories: LLM fine-tuning, MLOps/experiment tracking, distributed systems
# infra, and recommendation systems. Skill names drawn from the verified 133-
# term dataset vocabulary. Per-skill bonus is lowered slightly (0.025 → 0.018)
# so stacking many non-core nice-to-haves cannot leapfrog genuine must-have
# skill coverage; cap raised (0.06 → 0.09) to allow up to five to contribute.
NICE_TO_HAVE_SKILLS = [
    # LLM fine-tuning (JD explicit)
    "LoRA", "QLoRA", "PEFT",
    # MLOps / experiment tracking (production ML maturity signal)
    "Weights & Biases", "MLflow", "Kubeflow",
    # Distributed systems / scale infra (JD explicit category)
    "Kafka", "Spark", "Redis", "Kubernetes", "Docker",
    # Recommendation systems (JD explicit nice-to-have)
    "Recommendation Systems",
]
NICE_TO_HAVE_BONUS_PER_SKILL = 0.018   # lowered from 0.025 to prevent leapfrogging
NICE_TO_HAVE_BONUS_CAP = 0.09          # raised from 0.06; allows up to 5 skills to fire

# ---------------------------------------------------------------------------
# 3. SKILL TRUST WEIGHTING
# ---------------------------------------------------------------------------
# A listed skill is only as credible as the evidence behind it. We trust a
# skill claim more when proficiency, duration, endorsements, and (if present)
# the candidate's own Redrob skill_assessment_score all agree. This is the
# direct countermeasure to the "keyword stuffer" trap described in the JD's
# hackathon note and confirmed empirically (irrelevant-titled candidates
# carry far more *unverified* AI buzzwords than genuinely-titled ones).
PROFICIENCY_BASE_TRUST = {
    "expert": 1.00,
    "advanced": 0.80,
    "intermediate": 0.55,
    "beginner": 0.30,
}
# Duration scaling: a skill "used" for 0 months is barely credible no matter
# the claimed proficiency; trust ramps up to 1.0x by ~24 months.
DURATION_TRUST_FULL_MONTHS = 24.0
# If the candidate has a Redrob skill_assessment_score for this skill and it
# strongly contradicts the claimed proficiency, discount trust sharply.
ASSESSMENT_CONTRADICTION_DISCOUNT = 0.35   # multiplier applied when contradicted
ASSESSMENT_CONTRADICTION_THRESHOLD = 30.0  # assessment score below this for an
                                            # "advanced"/"expert" claim = contradiction

# ---------------------------------------------------------------------------
# 4. EXPERIENCE BAND FIT
# ---------------------------------------------------------------------------
# JD: "5-9 years ... a range, not a requirement ... we'll seriously consider
# candidates outside the band if other signals are strong." Modeled as a
# smooth plateau (1.0 inside the band) with gentle linear falloff outside it,
# not a hard cutoff.
EXPERIENCE_BAND_MIN = 5.0
EXPERIENCE_BAND_MAX = 9.0
EXPERIENCE_FALLOFF_PER_YEAR = 0.12     # score lost per year outside the band
EXPERIENCE_FALLOFF_FLOOR = 0.35        # never drops below this from years alone

# ---------------------------------------------------------------------------
# 5. CONSULTING / IT-SERVICES-ONLY CAREER PENALTY
# ---------------------------------------------------------------------------
# JD: "People who have only worked at consulting firms (TCS, Infosys, Wipro,
# Accenture, Cognizant, Capgemini, etc.) in their entire career ... If you're
# currently at one of these companies but have prior product-company
# experience, that's fine."
#
# The dataset tags each job's `industry` field directly (e.g. "IT Services",
# "Consulting"), which is more robust than a hardcoded company name list --
# but we keep the explicit JD-named companies too, for entries where the
# industry tag might be missing or different.
CONSULTING_INDUSTRY_LABELS = {"it services", "consulting"}
CONSULTING_COMPANY_NAMES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mindtree", "mphasis",
}
CONSULTING_ONLY_PENALTY = 0.30   # multiplier applied if EVERY job was at a
                                  # consulting/IT-services employer

# ---------------------------------------------------------------------------
# 6. OTHER JD-EXPLICIT DISQUALIFIER PENALTIES (soft multipliers, not hard 0)
# ---------------------------------------------------------------------------
# JD: "If your 'AI experience' consists primarily of recent (under 12
# months) projects using LangChain to call OpenAI ... we will probably not
# move forward, unless you can demonstrate substantial pre-LLM-era ML
# production experience."
RECENT_PIVOT_MAX_MONTHS = 12
RECENT_PIVOT_MIN_PRIOR_YEARS = 3.0
RECENT_PIVOT_PENALTY = 0.45

# JD: "people whose primary expertise is computer vision, speech, or
# robotics without significant NLP/IR exposure." Already partly reflected
# in the CV-title base score; this applies the same idea to *anyone* whose
# skill list is CV/speech-heavy with zero NLP/IR overlap.
CV_SPEECH_ONLY_SKILLS = {
    "Computer Vision", "Image Classification", "Object Detection", "CNN",
    "OpenCV", "YOLO", "GANs", "Diffusion Models", "ASR", "Speech Recognition", "TTS",
}
NLP_IR_OVERLAP_SKILLS = {
    "NLP", "Information Retrieval", "Semantic Search", "Vector Search",
    "Embeddings", "RAG", "LLMs", "Hugging Face Transformers",
    "Sentence Transformers", "BM25", "Learning to Rank", "Recommendation Systems",
    # Additional dataset vocabulary confirmed by full skill scan:
    "Natural Language Processing",   # synonym for NLP used in dataset
    "Information Retrieval Systems", # dataset term covering IR broadly
    "Ranking Systems",               # direct JD mandate — ranking pipelines
    "Search & Discovery",            # product-layer search — clear IR overlap
    "Search Backend",                # infra for search — clear IR overlap
    "Search Infrastructure",         # same cluster
    "Vector Representations",        # dense embedding outputs — synonym for Embeddings
}
CV_SPEECH_ONLY_PENALTY = 0.55

# JD: "work has been entirely on closed-source proprietary systems for 5+
# years without external validation (papers, talks, open-source)." Proxy:
# long tenure (>=5y) with github_activity_score == -1 (no GitHub linked)
# and no certifications on file.
CLOSED_SOURCE_MIN_YEARS = 5.0
CLOSED_SOURCE_PENALTY = 0.85   # mild -- this is the softest of the JD's
                                # disqualifiers ("most important" in tone,
                                # but hardest to verify reliably from data)

# JD: "Title-chasers ... career trajectory shows you optimizing for
# 'Senior' -> 'Staff' -> 'Principal' titles by switching companies every
# 1.5 years." Proxy: 3+ employers with average tenure under 18 months AND
# a senior-sounding current title (the combination is what signals
# title-by-hopping rather than e.g. an early-career candidate who simply
# hasn't settled yet). Validated against the real dataset: flags 54 of
# 100,000 candidates, a plausible "small deliberate minority" rate.
TITLE_CHASER_MIN_JOBS = 3
TITLE_CHASER_MAX_AVG_TENURE_MONTHS = 18.0
TITLE_CHASER_SENIOR_WORDS = ("senior", "staff", "lead", "principal", "head")
TITLE_CHASER_PENALTY = 0.55

# JD: "spent your career in pure research environments ... without any
# production deployment." Proxy: title is research-flavored AND no
# career_history description contains production-deployment language.
# Validated against the dataset: this combination is rare here (most
# profiles, even "AI Research Engineer", describe shipped/production work),
# so this check is intentionally a low-frequency safety net rather than a
# primary scoring lever.
PURE_RESEARCH_TITLE_MARKERS = ("research",)
PRODUCTION_EVIDENCE_PATTERN = (
    r"\b(production|deployed|shipped|served \d|at scale|live system|"
    r"rollout|in prod)\b"
)
PURE_RESEARCH_PENALTY = 0.55

# ---------------------------------------------------------------------------
# 7. LOCATION FIT
# ---------------------------------------------------------------------------
# JD: "Pune/Noida ... flexible ... Candidates in Hyderabad, Pune, Mumbai,
# Delhi NCR welcome ... Outside India: case-by-case, but we don't sponsor
# work visas."
LOCATION_TIER_SCORE = {
    "noida": 1.00, "pune": 1.00,
    "hyderabad": 0.85, "mumbai": 0.85, "delhi": 0.85, "gurgaon": 0.85,
    "bangalore": 0.85, "bengaluru": 0.85,
}
DEFAULT_INDIA_LOCATION_SCORE = 0.55
WILLING_RELOCATE_INDIA_BONUS = 0.15
OUTSIDE_INDIA_BASE_SCORE = 0.30
OUTSIDE_INDIA_RELOCATE_BONUS = 0.10

# ---------------------------------------------------------------------------
# 8. NOTICE PERIOD FIT
# ---------------------------------------------------------------------------
# JD: "We'd love sub-30-day notice. We can buy out up to 30 days. 30+ day
# notice candidates are still in scope but the bar gets higher."
NOTICE_PERIOD_SCORE_BREAKS = [
    (30, 1.00),
    (60, 0.85),
    (90, 0.65),
    (180, 0.45),
]

# ---------------------------------------------------------------------------
# 9. EDUCATION
# ---------------------------------------------------------------------------
EDUCATION_TIER_SCORE = {
    "tier_1": 1.00, "tier_2": 0.85, "tier_3": 0.70, "tier_4": 0.60, "unknown": 0.65,
}
EDUCATION_RELEVANT_FIELDS = {
    "computer science", "data science", "statistics", "mathematics",
    "artificial intelligence", "machine learning", "information technology",
}
EDUCATION_RELEVANT_FIELD_BONUS = 0.10

# ---------------------------------------------------------------------------
# 10. BEHAVIORAL AVAILABILITY MULTIPLIER (the 23 redrob_signals)
# ---------------------------------------------------------------------------
# redrob_signals_doc.docx: "These behavioral signals are often *more
# predictive* of whether a candidate can actually be hired ... ranking
# systems can incorporate them as a multiplier or modifier on top of
# skill-match scoring." Modeled as a multiplier in [BEHAVIORAL_MULT_FLOOR,
# BEHAVIORAL_MULT_CEILING] so it meaningfully moves the final score without
# ever letting availability alone outrank genuine fit.
BEHAVIORAL_MULT_FLOOR = 0.55
BEHAVIORAL_MULT_CEILING = 1.05
LAST_ACTIVE_HALF_LIFE_DAYS = 45     # recency decay half-life

# ---------------------------------------------------------------------------
# 11. HONEYPOT / INTERNAL-CONSISTENCY THRESHOLDS
# ---------------------------------------------------------------------------
# Reverse-engineered and validated against the real candidates.jsonl: these
# thresholds flag ~70 candidates out of 100,000, closely matching the "~80
# honeypots" the submission spec documents, and they are stable across a
# wide range of nearby threshold choices (not finely overfit to one value).
YOE_MISMATCH_THRESHOLD_YEARS = 2.0          # |years_of_experience - sum(career_history)/12|
EXPERT_ZERO_DURATION_MAX_MONTHS = 1         # "expert" claimed with <=1 month of use
EXPERT_ZERO_DURATION_MIN_COUNT = 1          # even a single such claim is disqualifying
HONEYPOT_SCORE_MULTIPLIER = 0.02            # near-zero, not literally zero (keeps
                                             # sort/tie-break well-defined)

# ---------------------------------------------------------------------------
# 12. COMPOSITE SCORE WEIGHTS
# ---------------------------------------------------------------------------
# These four components are weighted-summed into a base fit score in [0, 1],
# then modified by disqualifier multipliers and the behavioral multiplier.
COMPOSITE_WEIGHTS = {
    "title_tier": 0.28,
    "semantic_similarity": 0.22,
    "must_have_skills": 0.25,
    "experience_band": 0.10,
    "location": 0.09,
    "notice_period": 0.03,
    "education": 0.03,
}
assert abs(sum(COMPOSITE_WEIGHTS.values()) - 1.0) < 1e-9, "COMPOSITE_WEIGHTS must sum to 1.0"

# ---------------------------------------------------------------------------
# 13. SEMANTIC LAYER -- CURATED POSITIVE-SIGNAL JD QUERY TEXT
# ---------------------------------------------------------------------------
# IMPORTANT DESIGN NOTE: this text is deliberately built ONLY from the JD's
# *affirmative* requirements ("what we want"), never from the "Things we
# explicitly do NOT want" section. A bag-of-words / TF-IDF model cannot
# distinguish affirmed mentions of a term from negated ones -- e.g. the JD
# says "If your GitHub is full of LangChain tutorials ... that's not what we
# need." Feeding that sentence into the query would *reward* LangChain
# tutorial experience, the opposite of the JD's intent. Negative criteria
# (title-chasers, framework enthusiasts, consulting-only, CV/speech-only,
# closed-source-only) are therefore handled exclusively as explicit
# rule-based penalties in features.py, never via the text-similarity query.
JD_POSITIVE_QUERY_TEXT = """
Senior AI Engineer founding team. Own the intelligence layer of a
recruiting product: ranking, retrieval, and matching systems that decide
what recruiters see when they search for candidates and what candidates
see when they search for roles. Deep technical depth in modern ML systems:
embeddings, retrieval, ranking, large language models, fine-tuning.

Production experience with embeddings-based retrieval systems:
sentence-transformers, OpenAI embeddings, BGE, E5 embedding models.
Handled embedding drift, index refresh, retrieval-quality regression in
production.

Production experience with vector databases or hybrid search
infrastructure: Pinecone, Weaviate, Qdrant, Milvus, OpenSearch,
Elasticsearch, FAISS, hybrid sparse and dense retrieval, BM25 plus dense
vectors, semantic search at scale.

Strong Python, production code quality, building and shipping services.

Hands-on experience designing evaluation frameworks for ranking systems:
NDCG, MRR, MAP, offline-to-online metric correlation, A/B test
interpretation, recommendation systems, learning to rank, re-ranking
models, search relevance.

Shipped an end-to-end ranking, search, or recommendation system to real
users at meaningful scale at a product company. Audited an existing BM25
plus rule-based scoring system and replaced it with a hybrid retrieval and
re-ranking pipeline that improved engagement metrics. Set up offline
benchmarks, online A/B testing, and feedback loops.

LLM fine-tuning with LoRA, QLoRA, PEFT. Learning-to-rank models, XGBoost
based or neural ranking. Prior exposure to HR-tech, recruiting technology,
or marketplace products. Distributed systems and large-scale inference
optimization. Open-source contributions in the AI and ML space.

Comfortable with scrappy product engineering: willing to ship a working
ranker quickly, learn from real users, then iterate on what to optimize.
"""
