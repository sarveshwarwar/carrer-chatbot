import os
import json
import sys

# Auto-install missing packages
def install_package(package):
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    install_package("sentence-transformers")
    from sentence_transformers import SentenceTransformer

try:
    import faiss
except ImportError:
    install_package("faiss-cpu")
    import faiss

import numpy as np

# Load career QA safely
def load_career_qa(file_path="career_qa.json"):
    if not os.path.exists(file_path):
        # Create default file if missing
        default_data = [
            {"question": "How to write a resume?", "answer": "Focus on achievements, use bullet points, quantify results, and keep it concise."},
            {"question": "How to prepare for coding interviews?", "answer": "Practice DSA questions, understand algorithms, and solve problems on platforms like LeetCode."},
            {"question": "How to choose a career path?", "answer": "Identify your interests, evaluate skills, research industries, and seek mentorship."}
        ]
        with open(file_path, "w") as f:
            json.dump(default_data, f, indent=2)
    with open(file_path) as f:
        return json.load(f)

# Initialize model and FAISS index
def initialize_model():
    data = load_career_qa()
    questions = [item["question"] for item in data]
    answers = [item["answer"] for item in data]

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(questions, convert_to_numpy=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return model, index, answers

model, index, answers = initialize_model()

# Semantic search for career questions
def get_answer(user_query):
    if not user_query.strip():
        return "Please enter a valid question."
    try:
        embedding = model.encode([user_query], convert_to_numpy=True)
        _, idx = index.search(embedding, k=1)
        return answers[idx[0][0]]
    except Exception as e:
        return f"Error fetching answer: {e}"

# Resume tips
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

