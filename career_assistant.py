
import streamlit as st
import re
import io
import os
from typing import List, Dict, Tuple
import pandas as pd

# Optional libraries
try:
    import PyPDF2
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False

try:
    import docx2txt
    DOCX_AVAILABLE = True
except Exception:
    DOCX_AVAILABLE = False

# Optional OpenAI support
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

st.set_page_config(page_title="🤖 AI Career Assistant", layout="wide")
st.title("🤖 AI Career Assistant")
st.markdown(
    "Your personal career guidance and resume advisor — powered by heuristics and optional open AI.\n\n"
    "Features: Career Q&A • Resume Advice • Job Role Recommendation • Career Guidance • Interview Prep"
)

# ---------------------------
# Sidebar: Options & API key
# ---------------------------
st.sidebar.header("Settings & Integrations")
use_openai = st.sidebar.checkbox("Enable OpenAI (optional)", value=False)
if use_openai:
    openai_api_key = st.sidebar.text_input("OpenAI API Key (sk-...)", type="password")
    if openai_api_key:
        if OPENAI_AVAILABLE:
            openai.api_key = openai_api_key
        else:
            st.sidebar.warning("`openai` package not installed. Install in requirements to use API.")

# Small curated skills taxonomy (expandable)
SKILL_TO_ROLES = {
    "python": ["Backend Developer", "Data Scientist", "ML Engineer", "Automation Engineer"],
    "javascript": ["Frontend Developer", "Fullstack Developer"],
    "react": ["Frontend Developer", "Fullstack Developer"],
    "node": ["Backend Developer", "Fullstack Developer"],
    "flask": ["Backend Developer", "Fullstack Developer"],
    "django": ["Backend Developer", "Fullstack Developer"],
    "sql": ["Data Engineer", "Backend Developer", "Data Analyst"],
    "pandas": ["Data Scientist", "Data Analyst"],
    "numpy": ["Data Scientist", "ML Engineer"],
    "tensorflow": ["ML Engineer", "Deep Learning Engineer"],
    "pytorch": ["ML Engineer", "Deep Learning Engineer"],
    "aws": ["Cloud Engineer", "DevOps Engineer", "Backend Developer"],
    "docker": ["DevOps Engineer", "Backend Developer"],
    "kubernetes": ["DevOps Engineer", "Site Reliability Engineer"],
    "html": ["Frontend Developer"],
    "css": ["Frontend Developer"],
    "typescript": ["Frontend Developer", "Fullstack Developer"],
    "git": ["All developer roles"],
    "c++": ["Systems Developer", "Embedded Engineer"],
    "java": ["Backend Developer", "Android Developer"],
    "android": ["Android Developer"],
    "ios": ["iOS Developer", "Mobile Developer"],
    "machine learning": ["ML Engineer", "Data Scientist"],
    "data analysis": ["Data Analyst", "Business Analyst"],
}

COMMON_SKILLS = list(SKILL_TO_ROLES.keys())

# ---------------------------
# Utility functions
# ---------------------------
def simple_text_clean(s: str) -> str:
    return re.sub(r'\s+', ' ', s.strip())

def extract_text_from_pdf(file_bytes: bytes) -> str:
    if not PDF_AVAILABLE:
        raise RuntimeError("PyPDF2 is not installed.")
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = []
    for p in reader.pages:
        try:
            text.append(p.extract_text() or "")
        except Exception:
            text.append("")
    return "\n".join(text)

def extract_text_from_docx(file_bytes: bytes) -> str:
    if not DOCX_AVAILABLE:
        raise RuntimeError("docx2txt is not installed.")
    # docx2txt only reads from path; write temp
    tmp_path = "tmp_resume.docx"
    with open(tmp_path, "wb") as f:
        f.write(file_bytes)
    text = docx2txt.process(tmp_path) or ""
    try:
        os.remove(tmp_path)
    except Exception:
        pass
    return text

def extract_skills(text: str, skill_list: List[str] = COMMON_SKILLS) -> List[str]:
    text_low = text.lower()
    found = []
    for skill in skill_list:
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, text_low):
            found.append(skill)
    return found

def recommend_roles(skills: List[str]) -> Dict[str, int]:
    role_scores = {}
    for skill in skills:
        mapped = SKILL_TO_ROLES.get(skill.lower(), [])
        for r in mapped:
            role_scores[r] = role_scores.get(r, 0) + 1
    # sort by score
    return dict(sorted(role_scores.items(), key=lambda x: x[1], reverse=True))

def make_learning_plan(role: str, skills: List[str]) -> List[str]:
    # Very simple heuristic plan (expandable)
    plan = []
    if "Data" in role or "ML" in role or "Data Scientist" in role:
        plan = [
            "Master Python and libraries: numpy, pandas, matplotlib",
            "Learn statistics and probability fundamentals",
            "Study machine learning basics (supervised/unsupervised)",
            "Practice projects: regression/classification projects",
            "Deploy models: Flask, FastAPI or cloud services",
        ]
    elif "Frontend" in role:
        plan = [
            "Master HTML, CSS, JavaScript",
            "Learn React and state management (Redux/Context)",
            "Build portfolio UI projects (responsive & accessible)",
            "Understand build tools (webpack, Vite) and testing",
        ]
    elif "Backend" in role or "Fullstack" in role:
        plan = [
            "Learn a backend language (Python/Node/Java)",
            "Understand REST APIs and databases (SQL/NoSQL)",
            "Practice authentication & security basics",
            "Containerize apps with Docker and deploy to cloud",
        ]
    elif "DevOps" in role or "SRE" in role:
        plan = [
            "Learn Linux internals and shell scripting",
            "Master Docker and Kubernetes basics",
            "Understand CI/CD pipelines (GitHub Actions, Jenkins)",
            "Get hands-on with cloud infra (AWS/GCP/Azure)",
        ]
    else:
        plan = [
            "Identify core primary technologies for the role",
            "Follow structured learning path: fundamentals → projects → deployment",
            "Contribute to open-source or build portfolio projects",
        ]
    # customize with user's existing skills
    custom = [p for p in plan if not any(s.lower() in p.lower() for s in skills)]
    return custom

def improve_resume_bullets(text: str, max_suggestions: int = 5) -> List[str]:
    """Find lines that look like bullets and return improved versions."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    bullet_lines = [l for l in lines if re.match(r'^[-•\*]\s+', l) or len(l.split()) > 5 and ('achiev' in l.lower() or 'respons' in l.lower() or '%' in l)]
    improved = []
    for i, b in enumerate(bullet_lines[:max_suggestions]):
        # naive improvement: add metrics and action words if missing
        base = re.sub(r'^[-•\*]\s*', '', b)
        if not re.search(r'\d', base):
            base = base + " (quantify achievements: e.g., % improvement, delivered on-time)"
        if len(base.split()) < 6:
            base = "Implemented " + base
        improved.append(f"- {base}")
    if not improved:
        improved = ["- Use action verbs (Implemented, Designed, Optimized) and quantify impact (e.g., reduced X by 30%)."]
    return improved

# ---------------------------
# UI: Tabs
# ---------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Career Q&A", "Resume Advice", "Job Role Recommendation", "Career Guidance & Plan"])

# ---------------------------
# Tab: Career Q&A
# ---------------------------
with tab1:
    st.header("Career Q&A")
    st.write("Ask a career-related question and get actionable guidance.")
    q = st.text_input("Ask a question (e.g., 'I want to be a developer', 'How do I switch to data science?')")

    if st.button("Get Answer"):
        user_q = simple_text_clean(q or "")
        if not user_q:
            st.info("Type a question to get started.")
        else:
            # If OpenAI enabled, use it, otherwise offline heuristics
            if use_openai and openai_api_key and OPENAI_AVAILABLE:
                try:
                    resp = openai.ChatCompletion.create(
                        model="gpt-4o-mini" if "gpt-4o-mini" in openai.Model.list().data[0].id else "gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a friendly career coach. Keep answers concise and actionable."},
                            {"role": "user", "content": user_q}
                        ],
                        max_tokens=400,
                        temperature=0.2,
                    )
                    ans = resp.choices[0].message.content
                    st.markdown("**AI (OpenAI) Answer:**")
                    st.write(ans)
                except Exception as e:
                    st.error(f"OpenAI call failed: {e}. Falling back to offline answer below.")
                    use_openai = False  # fallback to offline
            if not (use_openai and openai_api_key and OPENAI_AVAILABLE):
                # Offline heuristic responder (simple intent mapping)
                uq = user_q.lower()
                if "developer" in uq or "become dev" in uq or "be developer" in uq:
                    st.markdown("**Actionable plan to become a developer:**")
                    st.markdown("""
                    1. Pick a stack (Frontend: HTML/CSS/JS/React | Backend: Python/Flask/Django or Node.js).  
                    2. Learn fundamentals (data structures, algorithms basics, HTTP).  
                    3. Build 3 portfolio projects and put code on GitHub.  
                    4. Practice coding questions (LeetCode / HackerRank) for 1 hour daily.  
                    5. Prepare an ATS-friendly resume and apply to internships/entry-level roles.
                    """)
                elif "data science" in uq or "data scientist" in uq:
                    st.markdown("**Actionable plan to switch to Data Science:**")
                    st.markdown("""
                    1. Learn Python, pandas, numpy, matplotlib/seaborn.  
                    2. Study statistics & basic ML (scikit-learn).  
                    3. Do 2 end-to-end projects: cleaning, modelling, deployment.  
                    4. Share notebooks on GitHub and write a short blog.  
                    5. Apply to junior data roles or internships.
                    """)
                else:
                    # generic career guidance
                    st.markdown("**General career guidance:**")
                    st.markdown("""
                    - Identify your target role and required skills.  
                    - Build a 6-month learning plan with milestones.  
                    - Network on LinkedIn and seek mentorship.  
                    - Practice interviews and mock projects.
                    """)

# ---------------------------
# Tab: Resume Advice
# ---------------------------
with tab2:
    st.header("Resume Advice")
    st.write("Upload your resume (PDF, DOCX, or TXT). The assistant will extract skills, suggest improvements and generate better bullets.")

    uploaded_resume = st.file_uploader("Upload resume", type=["pdf", "docx", "txt"])
    resume_text = ""
    if uploaded_resume is not None:
        raw = uploaded_resume.read()
        name_hint = uploaded_resume.name.lower()
        try:
            if name_hint.endswith(".pdf"):
                if not PDF_AVAILABLE:
                    st.error("PyPDF2 not installed. Install it to parse PDFs.")
                else:
                    resume_text = extract_text_from_pdf(raw)
            elif name_hint.endswith(".docx"):
                if not DOCX_AVAILABLE:
                    st.error("docx2txt not installed. Install it to parse DOCX.")
                else:
                    resume_text = extract_text_from_docx(raw)
            else:
                resume_text = raw.decode("utf-8", errors="ignore")
        except Exception as e:
            st.error(f"Failed to parse resume: {e}")

        if resume_text:
            st.subheader("Extracted Resume Text (preview)")
            st.text_area("Resume text", value=resume_text[:5000], height=250)

            # Skills extraction
            detected_skills = extract_skills(resume_text)
            st.subheader("Detected Skills")
            if detected_skills:
                st.write(", ".join(detected_skills))
            else:
                st.info("No skills from taxonomy detected. Consider adding explicit skill keywords (python, react, aws, sql, etc.)")

            # Suggest role(s)
            role_scores = recommend_roles(detected_skills)
            if role_scores:
                st.subheader("Recommended Roles (based on detected skills)")
                for r, score in role_scores.items():
                    st.write(f"- {r} (score {score})")
            else:
                st.info("No direct role mapping found. Use Career Guidance tab to plan next steps.")

            # Improve bullets
            st.subheader("Resume Bullet Improvements (sample suggestions)")
            improvements = improve_resume_bullets(resume_text, max_suggestions=6)
            for s in improvements:
                st.write(s)

            # Generate one improved resume summary (concise)
            if st.button("Generate improved professional summary"):
                # build a short summary using detected skills and user text
                top_skills = detected_skills[:5]
                summary = "Experienced " + (role_scores and list(role_scores.keys())[0] or "professional")
                if top_skills:
                    summary += " with working knowledge of " + ", ".join(top_skills) + "."
                summary += " Demonstrated ability to deliver projects end-to-end and learn new technologies quickly."
                st.subheader("Suggested Professional Summary")
                st.write(summary)
                st.download_button("Download summary as text", data=summary, file_name="professional_summary.txt")
    else:
        st.info("Upload your resume to get suggestions.")

# ---------------------------
# Tab: Job Role Recommendation
# ---------------------------
with tab3:
    st.header("Job Role Recommendation")
    st.write("Type your skills (comma-separated) or paste resume text to get role suggestions.")

    skills_input = st.text_area("Enter skills (e.g., python, react, sql) or paste resume text", height=150)
    if st.button("Recommend roles"):
        text = skills_input.strip()
        if not text:
            st.info("Please enter skills or paste resume text.")
        else:
            # try detect as comma list
            if "," in text and len(text.split(",")) <= 30:
                input_skills = [s.strip().lower() for s in text.split(",") if s.strip()]
            else:
                input_skills = extract_skills(text)
                if not input_skills:
                    # fallback: take words that look like skills (simple heuristics)
                    words = re.findall(r'\b[a-zA-Z\+\#\-]{2,}\b', text.lower())
                    input_skills = [w for w in words if w in COMMON_SKILLS][:10]

            if not input_skills:
                st.warning("Couldn't find skills — try comma-separated list or paste more of your resume.")
            else:
                st.write("Detected / Provided skills:", ", ".join(input_skills))
                roles = recommend_roles(input_skills)
                if roles:
                    st.subheader("Top recommended roles")
                    for r, score in roles.items():
                        st.write(f"- {r} (match score: {score})")
                    # show learning plan for top role
                    top_role = next(iter(roles.keys()))
                    st.subheader(f"Learning & Roadmap for: {top_role}")
                    plan = make_learning_plan(top_role, input_skills)
                    for step in plan:
                        st.write(f"- {step}")
                else:
                    st.info("No role recommendations found for given skills.")

# ---------------------------
# Tab: Career Guidance & Plan
# ---------------------------
with tab4:
    st.header("Career Guidance & Personalized Plan")
    st.write("Get a 6-month study/apply plan based on a target role.")

    target_role = st.selectbox("Choose target role", sorted(list(set(sum([v for v in SKILL_TO_ROLES.values()], [])))))
    current_skills = st.text_input("Comma-separated current skills (e.g., python, sql, react)", value="")
    months = st.slider("Plan length (months)", 1, 12, 6)

    if st.button("Create personalized plan"):
        user_skills = [s.strip().lower() for s in current_skills.split(",") if s.strip()]
        plan = make_learning_plan(target_role, user_skills)
        # split into monthly milestones
        if not plan:
            st.info("No specific plan generated; try different role or add skills.")
        else:
            per_month = max(1, len(plan) // months)
            st.subheader(f"{months}-month plan for {target_role}")
            for m in range(months):
                st.markdown(f"**Month {m+1}**")
                start = m * per_month
                end = start + per_month
                chunk = plan[start:end]
                if not chunk:
                    chunk = ["Project & apply learnings; practice interviews."]
                for task in chunk:
                    st.write(f"- {task}")
            st.success("Plan generated — follow it consistently and update progress weekly.")

# ---------------------------
# Footer / help
# ---------------------------
st.markdown("---")
st.caption("Built with heuristics. For richer, human-like responses enable OpenAI in the sidebar and provide your API key. "
           "You can extend the skill-role mapping to match your local job market.")
