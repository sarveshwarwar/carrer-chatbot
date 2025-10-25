from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os

# Load Q&A
def load_career_qa(file_path="career_qa.json"):
    if not os.path.exists(file_path):
        default_data = [
            {"question": "How to write a resume?", "answer": "Focus on achievements, use bullet points, quantify results, and keep it concise."},
            {"question": "How to prepare for coding interviews?", "answer": "Practice DSA questions, understand algorithms, and solve problems on platforms like LeetCode."},
            {"question": "How to choose a career path?", "answer": "Identify your interests, evaluate skills, research industries, and seek mentorship."}
        ]
        with open(file_path, "w") as f:
            json.dump(default_data, f, indent=2)
    with open(file_path) as f:
        return json.load(f)

data = load_career_qa()
questions = [item["question"] for item in data]
answers = [item["answer"] for item in data]

# Use smaller, fast model for cloud
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
question_embeddings = model.encode(questions, convert_to_numpy=True)

dimension = question_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(question_embeddings)

def get_answer(user_query):
    if not user_query.strip():
        return "Please enter a valid question."
    embedding = model.encode([user_query], convert_to_numpy=True)
    _, idx = index.search(embedding, k=1)
    return answers[idx[0][0]]

def resume_tips(resume_text):
    if not resume_text.strip():
        return "Please enter your resume text."
    tips = [
        "Highlight measurable achievements.",
        "Use action verbs.",
        "Keep it concise and clear.",
        "Tailor resume for each job application.",
        "Check formatting and grammar."
    ]
    return "Some tips to improve your resume:\n- " + "\n- ".join(tips)

