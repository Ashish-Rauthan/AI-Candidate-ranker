```markdown
# Redrob Candidate Ranker

## Overview

This project is a hybrid candidate ranking system developed for the **Redrob Intelligent Candidate Discovery & Ranking Challenge**.

The objective is to rank candidate profiles based on how well they match a job description. Instead of relying only on keyword matching, the system combines rule-based scoring with semantic similarity to evaluate candidates using multiple aspects of their profile.

The output is a submission CSV containing the top 100 ranked candidates in the format specified by the challenge.

---

## Features

- Hybrid ranking using rule-based scoring and semantic similarity
- Processes large candidate datasets efficiently
- Detects inconsistent candidate profiles
- Generates reasoning for every ranked candidate
- Produces a submission-ready CSV
- Includes a validator to verify the output format

---

## Project Structure
```

redrob-ranker/
│
├── rank.py # Main entry point
├── validate_submission.py # Submission validator
├── requirements.txt
├── submission_metadata.yaml
├── README.md
│
├── src/
│ └── redrob_ranker/
│ ├── behavioral.py
│ ├── config.py
│ ├── consistency.py
│ ├── features.py
│ ├── io_utils.py
│ ├── pipeline.py
│ ├── scoring.py
│ └── semantic.py
│
├── app/
│ └── streamlit_app.py
│
├── tests/
│
├── data/
│ ├── sample_candidates.json
│ └── candidates.jsonl
│
└── artifacts/

````

---

## Installation

Clone the repository and create a virtual environment.

### Windows

```bash
python -m venv venv
venv\Scripts\activate
````

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the required packages.

```bash
pip install -r requirements.txt
```

---

## Dataset

Place the provided candidate dataset inside the `data` directory.

```
data/
└── candidates.jsonl
```

---

## Running the Project

Execute the ranking pipeline using:

```bash
python rank.py --candidates data/candidates.jsonl --out submission.csv
```

After execution, the generated file will be:

```
submission.csv
```

---

## Validating the Submission

Run the validator before submitting.

```bash
python validate_submission.py submission.csv
```

If the submission follows the required format, the output will be:

```
Submission is valid.
```

---

## Ranking Pipeline

The ranking process consists of the following steps:

1. Load candidate profiles from the dataset.
2. Extract candidate features such as:
   - Job title
   - Technical skills
   - Experience
   - Education
   - Location
   - Notice period

3. Compute semantic similarity between each profile and the job description.
4. Apply rule-based scoring using predefined weights.
5. Penalize inconsistent or low-quality profiles.
6. Combine all scores into a final ranking score.
7. Sort candidates and export the top 100.

---

## Scoring Components

The final score is calculated using several independent components.

| Component           | Purpose                                                               |
| ------------------- | --------------------------------------------------------------------- |
| Title Score         | Measures how closely the candidate's role matches the target position |
| Skills Score        | Evaluates required technical skills                                   |
| Semantic Similarity | Measures overall profile relevance to the job description             |
| Experience          | Rewards candidates within the preferred experience range              |
| Education           | Considers degree and field of study                                   |
| Location            | Gives preference to preferred locations                               |
| Notice Period       | Prefers candidates who can join sooner                                |
| Behavioral Signals  | Uses available engagement-related information                         |
| Consistency Check   | Penalizes contradictory or suspicious profiles                        |

---

## Design Decisions

### Hybrid Approach

Keyword matching alone cannot capture the complete relevance of a candidate profile. Combining semantic similarity with rule-based scoring produces more balanced rankings.

### TF-IDF + SVD

The semantic layer uses TF-IDF with Singular Value Decomposition (SVD). This approach works efficiently on CPU-only systems and avoids downloading large pretrained language models.

### Streaming Pipeline

Candidate profiles are processed efficiently without loading the entire dataset into memory at once.

---

## Running Tests

Run all unit tests using:

```bash
pytest tests/
```

The test suite covers:

- Feature extraction
- Candidate scoring
- Semantic similarity
- Consistency checks
- End-to-end pipeline execution

---

## Streamlit Demo

A simple Streamlit application is included for testing the ranking pipeline on smaller datasets.

Run it locally:

```bash
streamlit run app/streamlit_app.py
```

---

## Output Format

The generated CSV contains the following columns:

| Column       | Description                              |
| ------------ | ---------------------------------------- |
| candidate_id | Candidate identifier                     |
| rank         | Final ranking position                   |
| score        | Composite ranking score                  |
| reasoning    | Brief explanation for the assigned score |

---

## Limitations

- Semantic similarity is based on TF-IDF and SVD instead of transformer embeddings.
- Some consistency checks rely on heuristics.
- Ranking quality depends on the completeness and accuracy of the input data.

---

## Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- Pytest

---

## Summary

This project implements a hybrid candidate ranking system that combines semantic similarity with rule-based feature scoring to rank candidate profiles efficiently. The solution is designed to process large datasets, generate explainable rankings, and produce a submission-ready CSV while remaining within the computational constraints of the challenge.

```

```
