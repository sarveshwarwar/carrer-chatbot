# career_ai_assistant.py
"""
AI Career Assistant (Advanced) - Streamlit app

Features:
- Career Chatbot (heuristic answers + optional OpenAI integration)
- Career Growth Roadmap suggestions for common developer/data roles
- Skill Gap Analyzer: compares user skills vs roadmap and gives prioritized learning tasks
- Resume Scorer: keyword overlap score + optional TF-IDF similarity (scikit-learn used if available)
- Exportable report download (TXT)

Recommended requirements (put in requirements.txt):
streamlit
pandas
numpy
scikit-learn
openai   # OPTIONAL - only if you want GPT integration
"""

from typing import List, Dict, Tuple
import streamlit as st
import re
import io
import json
import textwrap

# Optional ML imports
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# Optional OpenAI import
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

st.set_page_config(page_title="CareerPath AI — Advanced Career Assistant", layout="wide")

st.title("🤖 CareerPath AI — Advanced Career Assistant")
st.markdown(
    "Modules: Career Chatbot • Career Roadmap • Skill Gap Analyzer • Resume Scorer\n\n"
    "This app uses offline heuristics and light NLP. Optionally you can enable OpenAI in the sidebar for richer answers."
)

# ---------------------------
# Sidebar / Integration
# ---------------------------
st.sidebar.header("Settings & Integrations")
use_openai = st.sidebar.checkbox("Enable OpenAI responses (optional)", value=False)
openai_api_key = None
if use_openai:
    openai_api_key = st.sidebar.text_input("OpenAI API Key (sk-...)", type="password")
    if openai_api_key:
        if OPENAI_AVAILABLE:
            openai.api_key = openai_api_key
        else:
            st.sidebar.warning("`openai` package not installed in the environment; OpenAI disabled.")
            use_openai = False

# ---------------------------
# Knowledge: role -> skills + resources
# ---------------------------
ROLE_ROADMAPS: Dict[str, Dict] = {
    "web developer": {
        "title": "Web Developer (Frontend/Fullstack)",
        "core_skills": ["html", "css", "javascript", "react", "git"],
        "advanced_skills": ["typescript", "node", "express", "rest apis", "sql", "webpack"],
        "projects": [
            "Portfolio website with responsive UI (host on Netlify/Vercel)",
            "SPA using React and public API integration",
            "Fullstack app with Node/Express and a database"
        ],
        "resources": [
            ("freeCodeCamp - Responsive Web Design", "https://www.freecodecamp.org/learn/responsive-web-design/"),
            ("React Official Tutorial", "https://reactjs.org/tutorial/tutorial.html"),
            ("MDN Web Docs", "https://developer.mozilla.org/")
        ]
    },
    "backend developer": {
        "title": "Backend Developer",
        "core_skills": ["python", "flask", "django", "sql", "git"],
        "advanced_skills": ["rest apis", "docker", "kubernetes", "redis", "celery"],
        "projects": [
            "REST API with Flask/Django + JWT auth",
            "Background worker (Celery) for async tasks",
            "Deploy API using Docker on a cloud VM"
        ],
        "resources": [
            ("Flask Mega-Tutorial", "https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world"),
            ("Django Official Tutorial", "https://docs.djangoproject.com/en/stable/intro/tutorial01/")
        ]
    },
    "data scientist": {
        "title": "Data Scientist",
        "core_skills": ["python", "pandas", "numpy", "scikit-learn", "statistics"],
        "advanced_skills": ["feature engineering", "model evaluation", "deep learning", "deployment"],
        "projects": [
            "End-to-end ML project (data cleaning → model → evaluation → deployment)",
            "Classification/regression projects with clear metrics",
            "Kaggle competition participation"
        ],
        "resources": [
            ("Kaggle Learn", "https://www.kaggle.com/learn"),
            ("Coursera - Machine Learning by Andrew Ng", "https://www.coursera.org/learn/machine-learning")
        ]
    },
    "machine learning engineer": {
        "title": "Machine Learning Engineer",
        "core_skills": ["python", "numpy", "pytorch", "tensorflow", "ml fundamentals"],
        "advanced_skills": ["model serving", "onnx/tf-serving", "MLOps", "distributed training"],
        "projects": [
            "Train and deploy a model as an API",
            "Set up CI/CD for model updates",
            "Optimize model inference latency"
        ],
        "resources": [
            ("fast.ai", "https://www.fast.ai/"),
            ("MLOps guides", "https://mlops.community/")
        ]
    },
    "devops engineer": {
        "title": "DevOps / SRE",
        "core_skills": ["linux", "docker", "git", "bash"],
        "advanced_skills": ["kubernetes", "ci/cd", "cloud (aws/gcp/azure)", "monitoring"],
        "projects": [
            "Containerize an application and deploy to Kubernetes",
            "Implement CI/CD pipelines",
            "Create monitoring dashboards (Prometheus/Grafana)"
        ],
        "resources": [
            ("Docker docs", "https://docs.docker.com/"),
            ("Kubernetes docs", "https://kubernetes.io/docs/home/")
        ]
    },
    "data analyst": {
        "title": "Data Analyst",
        "core_skills": ["excel", "sql", "python", "pandas", "visualization"],
        "advanced_skills": ["power bi", "tableau", "statistical testing", "etl basics"],
        "projects": [
            "Sales dashboard with filters and KPIs",
            "Data cleaning and EDA notebook",
            "SQL-based reporting"
        ],
        "resources": [
            ("Mode SQL Tutorial", "https://mode.com/sql-tutorial/"),
            ("DataCamp - Data Analyst", "https://www.datacamp.com/")
        ]
    }
}

# Helper: flatten skill set and normalize
def normalize_skill(s: str) -> str:
    return re.sub(r'[^a-z0-9 ]', '', s.lower()).strip()

def role_exists(role: str) -> bool:
    return normalize_skill(role) in ROLE_ROADMAPS

def find_best_matching_role(role: str) -> Tuple[str, Dict]:
    key = normalize_skill(role)
    if key in ROLE_ROADMAPS:
        return key, ROLE_ROADMAPS[key]
    # try fuzzy containment
    for k in ROLE_ROADMAPS.keys():
        if key in k or k in key:
            return k, ROLE_ROADMAPS[k]
    # fallback: choose closest by token overlap
    best = None
    best_score = 0
    tokens = set(key.split())
    for k in ROLE_ROADMAPS.keys():
        score = len(tokens & set(k.split()))
        if score > best_score:
            best_score = score
            best = k
    if best:
        return best, ROLE_ROADMAPS[best]
    # default first
    return list(ROLE_ROADMAPS.keys())[0], ROLE_ROADMAPS[list(ROLE_ROADMAPS.keys())[0]]

# ---------------------------
# Chatbot: simple heuristics and optional OpenAI
# ---------------------------
def heuristic_chat_response(q: str) -> str:
    ql = q.lower()
    if "developer" in ql or "become dev" in ql or "be a developer" in ql:
        return (
            "To become a developer: pick a stack (frontend or backend), build 3 portfolio projects, "
            "learn Git & testing, practice coding problems, and contribute on GitHub."
        )
    if "data science" in ql or "data scientist" in ql:
        return (
            "To move into Data Science: learn Python, statistics, pandas, scikit-learn, "
            "complete end-to-end projects and share them on GitHub."
        )
    if "resume" in ql or "cv" in ql:
        return "Use action verbs, quantify impact (e.g., 'reduced latency by 30%'), keep it concise (1 page for freshers)."
    if "interview" in ql:
        return "Practice system design and behavioral questions, and solve 2–3 coding challenges daily in the run-up to interviews."
    return "Tell me your target role and current skills (comma-separated) so I can give tailored guidance."

def openai_chat_response(q: str) -> str:
    # Uses gpt-3.5-style if available (user supplies API key); keep conservative token usage
    if not OPENAI_AVAILABLE:
        return "OpenAI package not installed in environment; enable OpenAI in sidebar only on environments that have openai installed."
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini" if hasattr(openai, "ChatCompletion") else "gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a concise friendly career coach."},
                {"role": "user", "content": q}
            ],
            max_tokens=300,
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI call failed: {e}. Falling back to offline response."

# ---------------------------
# Skill gap analyzer
# ---------------------------
def analyze_skill_gap(user_skills: List[str], role_key: str) -> Dict:
    role = ROLE_ROADMAPS[role_key]
    core = [normalize_skill(s) for s in role["core_skills"]]
    adv = [normalize_skill(s) for s in role["advanced_skills"]]
    user = [normalize_skill(s) for s in user_skills]
    have_core = [s for s in core if s in user]
    missing_core = [s for s in core if s not in user]
    have_adv = [s for s in adv if s in user]
    missing_adv = [s for s in adv if s not in user]
    prioritized = missing_core + missing_adv
    return {
        "role_title": role["title"],
        "have_core": have_core,
        "missing_core": missing_core,
        "have_adv": have_adv,
        "missing_adv": missing_adv,
        "prioritized_learning": prioritized
    }

# ---------------------------
# Resume scoring (keywords + TF-IDF fallback)
# ---------------------------
def extract_keywords_from_role(role_key: str) -> List[str]:
    role = ROLE_ROADMAPS[role_key]
    kws = role["core_skills"] + role["advanced_skills"]
    # expand by tokens
    tokens = set()
    for k in kws:
        for tok in normalize_skill(k).split():
            tokens.add(tok)
    return sorted(tokens)

def simple_keyword_score(resume_text: str, role_tokens: List[str]) -> Tuple[float, Dict]:
    rt = normalize_skill(resume_text)
    found = []
    for tok in role_tokens:
        if re.search(r'\b' + re.escape(tok) + r'\b', rt):
            found.append(tok)
    score = (len(found) / len(role_tokens)) * 100 if role_tokens else 0.0
    return score, {"found": found, "missing": [t for t in role_tokens if t not in found]}

def tfidf_similarity_score(resume_text: str, role_text: str) -> float:
    if not SKLEARN_AVAILABLE:
        return 0.0
    try:
        vec = TfidfVectorizer(stop_words='english').fit([resume_text, role_text])
        m = vec.transform([resume_text, role_text])
        sim = cosine_similarity(m[0], m[1])[0][0]
        return float(sim) * 100.0  # percentage-like
    except Exception:
        return 0.0

def score_resume_for_role(resume_text: str, role_key: str) -> Dict:
    role_tokens = extract_keywords_from_role(role_key)
    kscore, kwres = simple_keyword_score(resume_text, role_tokens)
    # Build role descriptive text for TF-IDF
    role_desc = " ".join(ROLE_ROADMAPS[role_key]["core_skills"] + ROLE_ROADMAPS[role_key]["advanced_skills"] + ROLE_ROADMAPS[role_key]["projects"])
    tfidf_score = tfidf_similarity_score(resume_text, role_desc) if SKLEARN_AVAILABLE else 0.0
    # Combine scores: weighted (keywords 0.6, tfidf 0.4 if available)
    if SKLEARN_AVAILABLE:
        combined = 0.6 * kscore + 0.4 * tfidf_score
    else:
        combined = kscore
    # Clamp
    combined = max(0.0, min(100.0, combined))
    suggestions = []
    missing = kwres["missing"]
    if missing:
        suggestions.append("Add or highlight these keywords/skills: " + ", ".join(missing[:8]))
    if kscore < 50:
        suggestions.append("Add quantifiable achievements (numbers/impact) and relevant project details.")
    if SKLEARN_AVAILABLE and tfidf_score < 20:
        suggestions.append("Consider stronger role-specific language and project descriptions that mirror job descriptions.")
    return {
        "keyword_score": round(kscore, 2),
        "tfidf_score": round(tfidf_score, 2),
        "combined_score": round(combined, 2),
        "found_keywords": kwres["found"],
        "missing_keywords": missing,
        "suggestions": suggestions
    }

# ---------------------------
# UI Layout: Tabs
# ---------------------------
tabs = st.tabs(["💬 Career Chatbot", "🧭 Career Roadmap", "🧩 Skill Gap Analyzer", "📄 Resume Scorer & Report"])

# ---------------------------
# Tab 1: Career Chatbot
# ---------------------------
with tabs[0]:
    st.header("💬 Career Chatbot")
    st.write("Ask career questions, e.g. 'I want to be a developer' or 'How do I switch to data science?'")
    user_q = st.text_input("Ask a question", placeholder="e.g., I want to become a data scientist")
    if st.button("Get Advice", key="chat_ask"):
        if not user_q.strip():
            st.info("Type a question to get started.")
        else:
            if use_openai and openai_api_key:
                resp = openai_chat_response(user_q)
                st.markdown("**AI (OpenAI) Answer:**")
                st.write(resp)
            else:
                resp = heuristic_chat_response(user_q)
                st.markdown("**Career Assistant Answer:**")
                st.write(resp)

# ---------------------------
# Tab 2: Career Roadmap
# ---------------------------
with tabs[1]:
    st.header("🧭 Career Roadmap Suggestions")
    st.write("Choose a role to get a structured roadmap, projects, and useful resources.")
    role_input = st.selectbox("Pick a target role", options=list(ROLE_ROADMAPS.keys()))
    if st.button("Show Roadmap", key="roadmap_show"):
        rdata = ROLE_ROADMAPS[role_input]
        st.subheader(rdata["title"])
        st.markdown("**Core skills**")
        st.write(", ".join(rdata["core_skills"]))
        st.markdown("**Advanced skills**")
        st.write(", ".join(rdata["advanced_skills"]))
        st.markdown("**Suggested Projects**")
        for p in rdata["projects"]:
            st.write("- " + p)
        st.markdown("**Recommended Resources**")
        for name, link in rdata["resources"]:
            st.write(f"- [{name}]({link})")

# ---------------------------
# Tab 3: Skill Gap Analyzer
# ---------------------------
with tabs[2]:
    st.header("🧩 Skill Gap Analyzer")
    st.write("Enter your current skills (comma-separated), choose a target role, and get a prioritized learning list.")
    skills_text = st.text_area("Your skills (comma-separated)", placeholder="e.g., Python, SQL, React", height=100)
    target_role = st.selectbox("Target role", options=list(ROLE_ROADMAPS.keys()), index=0)
    if st.button("Analyze Skill Gap", key="gap"):
        user_skills = [s.strip() for s in re.split(r'[,\n]+', skills_text) if s.strip()]
        role_key, _ = find_best_matching_role(target_role)
        result = analyze_skill_gap(user_skills, role_key)
        st.subheader(f"Role: {result['role_title']}")
        st.markdown("**Core skills you have:**")
        st.write(result["have_core"] or "None detected")
        st.markdown("**Core skills missing (priority 1):**")
        st.write(result["missing_core"] or "None")
        st.markdown("**Advanced skills you have:**")
        st.write(result["have_adv"] or "None")
        st.markdown("**Advanced skills missing (priority 2):**")
        st.write(result["missing_adv"] or "None")
        st.markdown("### Prioritized learning plan (start at top):")
        if result["prioritized_learning"]:
            for i, s in enumerate(result["prioritized_learning"], 1):
                st.write(f"{i}. {s}")
        else:
            st.write("No missing skills detected — focus on projects & depth!")

# ---------------------------
# Tab 4: Resume Scorer & Report
# ---------------------------
with tabs[3]:
    st.header("📄 Resume Scorer & Report")
    st.write("Paste your resume text (or upload a TXT) and choose a target role to score relevance and get suggestions.")

    uploaded = st.file_uploader("Upload resume (TXT) or paste text below", type=["txt"])
    resume_text_area = st.text_area("Resume text (paste or leave blank if uploaded)", height=300)

    if uploaded:
        try:
            raw = uploaded.read()
            resume_text_area = raw.decode("utf-8", errors="ignore")
            st.success("Loaded uploaded resume text into the editor.")
        except Exception:
            st.error("Failed to read uploaded file. Paste text manually.")

    selected_role = st.selectbox("Select role to score against", options=list(ROLE_ROADMAPS.keys()))
    if st.button("Score Resume", key="score"):
        if not resume_text_area.strip():
            st.warning("Please paste resume text or upload a TXT resume.")
        else:
            role_key, _ = find_best_matching_role(selected_role)
            score_data = score_resume_for_role(resume_text_area, role_key)
            st.subheader(f"Scoring for role: {ROLE_ROADMAPS[role_key]['title']}")
            st.metric("Keyword Match Score", f"{score_data['keyword_score']}%")
            if SKLEARN_AVAILABLE:
                st.metric("TF-IDF Similarity", f"{score_data['tfidf_score']:.2f}%")
            st.metric("Combined Relevance Score", f"{score_data['combined_score']}%")
            st.markdown("**Found keywords:** " + (", ".join(score_data["found_keywords"]) or "None"))
            st.markdown("**Missing keywords (prioritize adding):** " + (", ".join(score_data["missing_keywords"]) or "None"))
            st.markdown("**Suggestions:**")
            for s in score_data["suggestions"]:
                st.write("- " + s)

            # Download a simple text report
            report_lines = [
                f"Resume Relevance Report for role: {ROLE_ROADMAPS[role_key]['title']}",
                f"Combined Score: {score_data['combined_score']}%",
                f"Keyword Score: {score_data['keyword_score']}%",
            ]
            if SKLEARN_AVAILABLE:
                report_lines.append(f"TF-IDF Similarity: {score_data['tfidf_score']}%")
            report_lines.append("")
            report_lines.append("Found keywords: " + (", ".join(score_data["found_keywords"]) or "None"))
            report_lines.append("Missing keywords: " + (", ".join(score_data["missing_keywords"]) or "None"))
            report_lines.append("")
            report_lines.append("Suggestions:")
            for s in score_data["suggestions"]:
                report_lines.append("- " + s)
            report_text = "\n".join(report_lines)
            st.download_button("Download Report (TXT)", report_text, file_name="resume_report.txt", mime="text/plain")

# Footer
st.markdown("---")
st.caption("CareerPath AI — advanced career assistant.")

