# FUTURE_ML_03 — Resume / Candidate Screening System

## Objective
Build an ML-assisted system that screens and ranks resumes against a given job role, while highlighting missing required skills.

## Data
The repository contains:
- one sample Junior Machine Learning Intern job description
- eight **synthetic sample resumes**

These resumes are fictional demonstration documents and do not contain real candidate personal data.

## Method
1. Read and normalize the job description and resumes.
2. Extract skills from a controlled technical-skill vocabulary.
3. Convert the job description and resumes into TF-IDF vectors.
4. Calculate cosine similarity between each resume and the job description.
5. Calculate required and preferred skill coverage.
6. Combine the signals into a transparent fit score:
   - 70% required skill coverage
   - 20% preferred skill coverage
   - 10% TF-IDF similarity
7. Rank candidates and show skill gaps.

## Output
`outputs/candidate_ranking.csv` contains:
- rank
- candidate
- fit score
- TF-IDF similarity
- required skill coverage
- preferred skill coverage
- matched skills
- skill gaps

## Responsible-use limitation
This is a portfolio demonstration, not an autonomous hiring decision system. Candidate screening tools can reproduce or amplify bias if trained or evaluated carelessly. A real deployment should include job-related validation, privacy controls, fairness testing, human review, and monitoring.

## Run
```bash
python src/resume_screening.py
```
