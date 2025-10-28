import streamlit as st
from career_assistant import get_answer, resume_tips

st.set_page_config(page_title="🤖 AI Career Assistant", layout="wide")

st.title("🤖 AI Career Assistant")
st.subheader("Your personal career guidance and resume advisor, powered by open-source AI.")

st.divider()

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to:", ["Career Q&A", "Resume Advice", "Job Role Recommendation"])

if page == "Career Q&A":
    st.header("💬 Career Guidance")
    question = st.text_input("Ask a career-related question:")
    if st.button("Ask"):
        if question.strip():
            answer = get_answer(question)
            st.success(answer)
        else:
            st.warning("Please type a question to get advice!")

elif page == "Resume Advice":
    st.header("📝 Resume Tips")
    if st.button("Show Resume Tips"):
        tips = resume_tips()
        st.info("\n".join([f"- {tip}" for tip in tips]))

elif page == "Job Role Recommendation":
    st.header("🎯 Job Role Recommendation")
    skills = st.text_input("Enter your top 3 skills (comma separated):")
    if st.button("Recommend Role"):
        if skills:
            skills = skills.lower()
            if "python" in skills and "ml" in skills:
                st.success("You could explore becoming a **Machine Learning Engineer**.")
            elif "javascript" in skills:
                st.success("You could be a **Frontend Developer** or **Fullstack Developer**.")
            elif "sql" in skills or "excel" in skills:
                st.success("You could be a **Data Analyst**.")
            else:
                st.success("Try exploring Software Engineering or Product Development roles!")
        else:
            st.warning("Please enter your skills.")

