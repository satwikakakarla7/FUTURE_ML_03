from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "outputs" / "candidate_ranking.csv"

st.set_page_config(page_title="Resume Screening Dashboard", page_icon="📄", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {background: rgba(128,128,128,.08); border: 1px solid rgba(128,128,128,.18); padding: 14px 16px; border-radius: 12px;}
.hero {padding: 22px 24px; border-radius: 16px; background: linear-gradient(120deg, #14213d, #315c8c); color: white; margin-bottom: 18px;}
.hero p {color: #e8eef8; margin-bottom: 0;}
.small-note {font-size: .88rem; opacity: .8;}
</style>
<div class="hero">
  <h1>📄 Resume Screening & Candidate Fit Dashboard</h1>
  <p>Explore candidate rankings, required-skill coverage, preferred skills, and skill gaps for a sample Junior ML Intern role.</p>
</div>
""", unsafe_allow_html=True)

if not OUTPUT_FILE.exists():
    st.error("Cannot find outputs/candidate_ranking.csv. Run `python src/resume_screening.py` first.")
    st.stop()

df = pd.read_csv(OUTPUT_FILE)
for col in ["fit_score", "tfidf_similarity", "required_skill_coverage_percent", "preferred_skill_coverage_percent", "rank"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df["skill_gaps"] = df["skill_gaps"].fillna("None").replace("", "None")
df["matched_skills"] = df["matched_skills"].fillna("")

with st.sidebar:
    st.header("Dashboard filters")
    min_score, max_score = st.slider("Fit score range", 0.0, 100.0, (0.0, 100.0), 1.0)
    query = st.text_input("Search candidate or skill", placeholder="e.g. Python, Aarav")
    gap_filter = st.selectbox("Skill-gap status", ["All candidates", "No listed gaps", "Has skill gaps"])
    st.divider()
    st.caption("Demo data: all resumes are synthetic and fictional.")

filtered = df[df["fit_score"].between(min_score, max_score)].copy()
if query.strip():
    q = query.strip().lower()
    filtered = filtered[filtered.apply(lambda r: q in str(r["candidate"]).lower() or q in str(r["matched_skills"]).lower() or q in str(r["skill_gaps"]).lower(), axis=1)]
if gap_filter == "No listed gaps":
    filtered = filtered[filtered["skill_gaps"].str.lower().eq("none")]
elif gap_filter == "Has skill gaps":
    filtered = filtered[~filtered["skill_gaps"].str.lower().eq("none")]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Candidates in view", len(filtered))
c2.metric("Top fit score", f"{filtered['fit_score'].max():.1f}" if not filtered.empty else "—")
c3.metric("Avg. required-skill coverage", f"{filtered['required_skill_coverage_percent'].mean():.1f}%" if not filtered.empty else "—")
c4.metric("Candidates with gaps", int((~filtered["skill_gaps"].str.lower().eq("none")).sum()) if not filtered.empty else 0)

left, right = st.columns([1.25, 1])
with left:
    st.subheader("Candidate fit ranking")
    if filtered.empty:
        st.info("No candidates match these filters. Try widening the score range or clearing the search.")
    else:
        chart_df = filtered.sort_values("fit_score", ascending=True)
        fig = px.bar(chart_df, x="fit_score", y="candidate", orientation="h", text="fit_score",
                     color="fit_score", color_continuous_scale="Blues", range_x=[0, 100],
                     labels={"fit_score": "Fit score", "candidate": "Candidate"})
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside", cliponaxis=False)
        fig.update_layout(height=400, coloraxis_showscale=False, margin=dict(l=8, r=30, t=15, b=10))
        st.plotly_chart(fig, use_container_width=True)
with right:
    st.subheader("Skill coverage comparison")
    if not filtered.empty:
        cov = filtered[["candidate", "required_skill_coverage_percent", "preferred_skill_coverage_percent"]].melt(
            id_vars="candidate", var_name="coverage_type", value_name="coverage")
        cov["coverage_type"] = cov["coverage_type"].map({
            "required_skill_coverage_percent": "Required skills",
            "preferred_skill_coverage_percent": "Preferred skills"})
        fig2 = px.bar(cov, x="candidate", y="coverage", color="coverage_type", barmode="group",
                      labels={"coverage": "Coverage (%)", "candidate": "Candidate", "coverage_type": "Skill type"},
                      color_discrete_sequence=["#315c8c", "#63b3a5"])
        fig2.update_yaxes(range=[0, 105])
        fig2.update_layout(height=400, margin=dict(l=8, r=8, t=15, b=10), legend_title_text="")
        st.plotly_chart(fig2, use_container_width=True)

st.subheader("Candidate details")
show = filtered.sort_values("rank")[["rank", "candidate", "fit_score", "required_skill_coverage_percent", "preferred_skill_coverage_percent", "tfidf_similarity", "matched_skills", "skill_gaps"]].rename(columns={
    "rank": "Rank", "candidate": "Candidate", "fit_score": "Fit score",
    "required_skill_coverage_percent": "Required coverage (%)",
    "preferred_skill_coverage_percent": "Preferred coverage (%)",
    "tfidf_similarity": "TF-IDF similarity", "matched_skills": "Matched skills", "skill_gaps": "Skill gaps"})
st.dataframe(show, hide_index=True, use_container_width=True, column_config={
    "Fit score": st.column_config.ProgressColumn("Fit score", min_value=0, max_value=100, format="%.1f"),
    "Required coverage (%)": st.column_config.ProgressColumn("Required coverage (%)", min_value=0, max_value=100, format="%.0f%%"),
    "Preferred coverage (%)": st.column_config.ProgressColumn("Preferred coverage (%)", min_value=0, max_value=100, format="%.0f%%")})

if not filtered.empty:
    st.subheader("Inspect one candidate")
    selected = st.selectbox("Choose candidate", filtered.sort_values("rank")["candidate"].tolist())
    row = filtered[filtered["candidate"] == selected].iloc[0]
    a, b = st.columns(2)
    with a:
        st.markdown("**Matched skills**")
        skills = [s.strip() for s in str(row["matched_skills"]).split(",") if s.strip()]
        st.write(" · ".join(skills) if skills else "No skills listed")
    with b:
        st.markdown("**Skill gaps**")
        gaps = [s.strip() for s in str(row["skill_gaps"]).split(",") if s.strip() and s.strip().lower() != "none"]
        st.write(" · ".join(gaps) if gaps else "No gaps listed in this demo")

st.divider()
st.caption("Method note: the demo combines required-skill coverage (70%), preferred-skill coverage (20%), and TF-IDF similarity (10%). Scores are illustrative, based on synthetic resumes, and must not be used as the sole basis for hiring decisions. Real deployment requires validation, fairness testing, privacy controls, and human review.")

csv = filtered.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download filtered results (CSV)", data=csv, file_name="filtered_candidate_ranking.csv", mime="text/csv")
