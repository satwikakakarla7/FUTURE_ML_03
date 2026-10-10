import re
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT/"data/job_description.txt"
RESUMES = ROOT/"data/resumes"
OUT = ROOT/"outputs"
OUT.mkdir(exist_ok=True)

SKILLS = [
    "python","machine learning","pandas","numpy","scikit-learn",
    "sql","git","statistics","nlp","tensorflow","matplotlib","jupyter"
]

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#.\-\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def extract_skills(text):
    clean = normalize(text)
    return sorted([s for s in SKILLS if s in clean])

def main():
    job_text = JOB.read_text(encoding="utf-8")
    required = ["python","machine learning","pandas","numpy","scikit-learn","sql","git","statistics"]
    preferred = ["nlp","tensorflow","matplotlib","jupyter"]

    files = sorted(RESUMES.glob("*.txt"))
    names, texts, skills_list = [], [], []
    for f in files:
        text = f.read_text(encoding="utf-8")
        names.append(text.splitlines()[0].strip())
        texts.append(text)
        skills_list.append(extract_skills(text))

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([job_text] + texts)
    similarity = cosine_similarity(matrix[0:1], matrix[1:]).ravel()

    rows = []
    for i, (name, skills) in enumerate(zip(names, skills_list)):
        required_match = len(set(skills) & set(required)) / len(required)
        preferred_match = len(set(skills) & set(preferred)) / len(preferred)
        fit_score = 0.70*required_match + 0.20*preferred_match + 0.10*similarity[i]
        missing = sorted(set(required) - set(skills))
        rows.append({
            "candidate": name,
            "fit_score": round(float(fit_score*100), 2),
            "tfidf_similarity": round(float(similarity[i]), 4),
            "required_skill_coverage_percent": round(float(required_match*100), 2),
            "preferred_skill_coverage_percent": round(float(preferred_match*100), 2),
            "matched_skills": ", ".join(sorted(set(skills))),
            "skill_gaps": ", ".join(missing) if missing else "None"
        })

    result = pd.DataFrame(rows).sort_values(
        ["fit_score","required_skill_coverage_percent"], ascending=False
    ).reset_index(drop=True)
    result.insert(0, "rank", range(1, len(result)+1))
    result.to_csv(OUT/"candidate_ranking.csv", index=False)
    result.to_html(OUT/"candidate_ranking.html", index=False)

    print(result.to_string(index=False))

if __name__ == "__main__":
    main()
