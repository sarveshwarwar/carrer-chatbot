import streamlit as st
from career_assistant import get_answer, resume_tips

st.set_page_config(page_title="AI Career Assistant", layout="wide")

st.title("🤖 AI Career Assistant")
st.write("Your personal career guidance and resume advisor, powered by open-source AI.")

# Tabs for different features
tab1, tab2, tab3 = st.tabs(["Career Q&A", "Resume Advice", "Job Role Recommendation"])

# --- Career Q&A ---
with tab1:
    st.header("Career Guidance")
    question = st.text_input("Ask a career-related question:")
    if st.button("Get Answer", key="qa"):
        if question.strip() != "":
            answer = get_answer(question)
            st.success(answer)
        else:
            st.warning("Please enter a question.")

# --- Resume Advice ---
with tab2:
    st.header("Resume Improvement Tips")
    resume_text = st.text_area("Paste your resume text here:")
    if st.button("Get Resume Tips", key="resume"):
        if resume_text.strip() != "":
            tips = resume_tips(resume_text)
            st.success(tips)
        else:
            st.warning("Please paste your resume text.")

# --- Job Role Recommendation ---
with tab3:
    st.header("Job Role Recommendation")
    skills = st.text_input("Enter your skills (comma-separated):")
    interest = st.text_input("Your area of interest:")
    if st.button("Recommend Roles", key="roles"):
        if skills.strip() != "" and interest.strip() != "":
            st.success(f"Based on your skills [{skills}] and interest [{interest}], you can consider roles like:\n- Data Analyst\n- Machine Learning Engineer\n- AI Specialist\n- Software Developer")
        else:
            st.warning("Please enter both skills and interests.")
