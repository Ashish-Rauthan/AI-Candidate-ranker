# Redrob Candidate Ranker

A hybrid rule-based + semantic system that ranks candidate profiles
against a job description — built for Redrob's *Intelligent Candidate
Discovery & Ranking Challenge*.

> **One command produces the ranked output:**
> ```
> python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
> ```
> Runs in well under a minute on a warm cache, a few minutes on a cold
> one — CPU-only, no GPU, no network calls during ranking.

---

## Table of contents

1. [What this does](#what-this-does)
2. [How it works](#how-it-works)
3. [Windows setup — step by step](#windows-setup--step-by-step)
4. [Running the ranker](#running-the-ranker)
5. [Running the tests](#running-the-tests)
6. [The Streamlit sandbox demo](#the-streamlit-sandbox-demo)
7. [Deploying the sandbox to Streamlit Cloud](#deploying-the-sandbox-to-streamlit-cloud)
8. [Repository structure](#repository-structure)
9. [Design notes](#design-notes)

---

## What this does

Most naive candidate-ranking approaches are easy to fool: a profile that
lists every trendy keyword in its skills section can outscore someone
whose career history actually demonstrates the work, simply because
keyword search can't tell the difference between a skill that's been
used for years and one that's been pasted in for show. Profiles can also
contain subtle internal inconsistencies (claimed experience that doesn't
add up, credentials that don't hold together) that a purely semantic or
keyword-based system has no way to catch.

This system takes a more deliberate approach. Each candidate is scored
across several largely-independent signals — how relevant their actual
career history is to the role, how credible their listed skills are
(not just whether they're present), how their full profile compares
semantically to the role description, and practical fit factors like
experience level and logistics. On top of that, a set of guardrail
checks down-weight known failure patterns and flag profiles that are
internally inconsistent, and a final adjustment accounts for whether a
candidate looks reachable/available right now versus just well-qualified
on paper.

The output is a ranked shortlist with a short, fact-grounded explanation
for every candidate — built from the same signals that produced their
score, not reconstructed after the fact.

## How it works

```
 candidate pool (streamed; never fully loaded into memory at once)
            │
            ▼
 ┌─────────────────────────────────────────────────────────┐
 │  Feature extraction                                       │
 │  · role/title relevance        · experience-level fit      │
 │  · verified skill match         · location & logistics fit │
 │  · semantic similarity to the role description              │
 └─────────────────────────────────────────────────────────┘
            │
            ▼
 ┌─────────────────────────────────────────────────────────┐
 │  Guardrails                                                │
 │  · penalty rules for known anti-patterns                   │
 │  · internal-consistency / data-integrity checks             │
 │  · availability & responsiveness adjustment                 │
 └─────────────────────────────────────────────────────────┘
            │
            ▼
   composite score → sorted, ranked shortlist with reasoning
```

A few notable engineering choices:

- **No transformer embedding model.** Semantic similarity is computed
  with a classic TF-IDF + Truncated SVD pipeline rather than a neural
  sentence-embedding model. This was a deliberate tradeoff: judging
  environments for this kind of challenge are typically CPU-only,
  time-boxed, and offline, which makes a model that needs vendored
  weights and a deep-learning runtime fragile to reproduce. TF-IDF+SVD
  needs only `scikit-learn`, has zero network dependency, and still
  produces genuine dense vectors compared by cosine similarity — not
  literal keyword overlap.
- **No LLM calls during ranking.** The full pipeline is a deterministic,
  streaming Python program — no hosted API calls, no GPU, nothing that
  could fail due to a network hiccup or a rate limit mid-run.
- **Streaming, not load-everything.** The candidate pool is processed
  one record at a time and reduced to a small feature record immediately
  — peak memory stays low regardless of pool size or the machine running
  it.
- **Guardrails are soft penalties, not hard cutoffs.** A candidate that
  trips one penalty rule but is otherwise an excellent match can still
  surface near the top, with the concern noted honestly in their
  reasoning text rather than being silently dropped.

The exact scoring weights, detection thresholds, and the full reasoning
behind each one live in the source (`src/redrob_ranker/config.py`'s
inline comments) rather than here, since that level of detail belongs in
code you can actually trace and modify, not prose that can drift out of
sync with it.

---

## Windows setup — step by step

These are the exact commands for **Windows (PowerShell)**. Git Bash, WSL,
macOS, and Linux users can use the equivalent commands.

### Step 1 — Install Python (if you don't already have it)

Download Python 3.10+ from [python.org/downloads](https://www.python.org/downloads/).
**During install, check "Add python.exe to PATH."**

Verify in PowerShell:
```powershell
python --version
```

### Step 2 — Get this repository

```powershell
git clone https://github.com/YOUR_USERNAME/redrob-ranker.git
cd redrob-ranker
```

### Step 3 — Create and activate a virtual environment

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```
*(If PowerShell blocks the activation script, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` once, then
retry the line above.)*

### Step 4 — Install dependencies

```powershell
pip install -r requirements.txt
```

### Step 5 — Provide a candidate dataset

This repo ships only a small public sample (`data/sample_candidates.json`)
for testing. Place your full candidate pool at `data/candidates.jsonl`
(this path is gitignored and won't be committed):

```powershell
copy "C:\path\to\your\candidates.jsonl" "data\candidates.jsonl"
```

### Step 6 — Run the ranker

```powershell
python rank.py --candidates data\candidates.jsonl --out submission.csv
```

### Step 7 — Validate the output

```powershell
python validate_submission.py submission.csv
```

---

## Running the ranker

```bash
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
```

| Flag | Default | Meaning |
|---|---|---|
| `--candidates` | `data/candidates.jsonl` | Path to the candidate pool (`.jsonl` or `.jsonl.gz`) |
| `--out` | `submission.csv` | Output CSV path |
| `--artifacts-dir` | `artifacts` | Where the cached semantic index lives |
| `--no-cache` | off | Force a full rebuild of the semantic index |
| `--top-n` | `100` | Number of ranked rows to output |
| `--quiet` | off | Suppress progress logging |

---

## Running the tests

```bash
pytest tests/ -v
```

The test suite covers feature scoring, the internal-consistency checks,
the availability/behavioral adjustment, and a full end-to-end pipeline
run on a small synthetic candidate pool — independent of any private
dataset.

---

## The Streamlit sandbox demo

`app/streamlit_app.py` is a thin UI wrapper around the same ranking
pipeline `rank.py` uses — no separate "demo-only" logic.

```bash
streamlit run app/streamlit_app.py
```
Opens at `http://localhost:8501`. Defaults to the bundled public sample,
or upload your own small `.json`/`.jsonl` file.

## Deploying the sandbox to Streamlit Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, and click **"Create app" → "From an existing repo"**.
3. Select this repository, branch `main`, main file path
   `app/streamlit_app.py`.
4. Deploy. First build takes a few minutes.

---

## Repository structure

```
redrob-ranker/
├── rank.py                       # CLI entry point → submission.csv
├── requirements.txt
├── pyproject.toml
├── validate_submission.py        # Format validator
├── README.md
├── src/redrob_ranker/
│   ├── config.py                  # Scoring weights & thresholds
│   ├── io_utils.py                 # Streaming JSONL reader
│   ├── consistency.py              # Internal-consistency / anomaly checks
│   ├── behavioral.py               # Availability/responsiveness scoring
│   ├── features.py                 # Title, skills, experience, guardrails
│   ├── semantic.py                 # TF-IDF + SVD semantic similarity
│   ├── scoring.py                  # Composite score + reasoning text
│   └── pipeline.py                 # End-to-end orchestration
├── app/streamlit_app.py           # Hosted sandbox demo
├── tests/                         # pytest suite
├── data/
│   └── sample_candidates.json     # Small public sample (committed)
└── docs/                          # Methodology deck (PDF + source)
```

---

## Design notes

- **Reasoning text is generated from the same scores that drove the
  ranking**, not written separately — every claim in a candidate's
  explanation traces back to an actual computed feature.
- **Deterministic and reproducible.** Given the same input file, the
  pipeline produces byte-identical output every time — same ranking,
  same scores, same reasoning text.
- **Corpus-relative semantic layer.** The TF-IDF+SVD index is fit on the
  specific candidate pool plus the role description at run time, not a
  universal pretrained embedding space — a deliberate tradeoff for
  zero-network reproducibility, traded off against the broader semantic
  generalization a pretrained model would offer.
