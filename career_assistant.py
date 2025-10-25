import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load Q&A data
with open("career_qa.json") as f:
    data = json.load(f)

questions = [item["question"] for item in data]
answers = [item["answer"] for item in data]

# Create embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
question_embeddings = model.encode(questions, convert_to_numpy=True)

# FAISS index
dimension = question_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(question_embeddings)

# Semantic search
def get_answer(user_query):
    query_embedding = model.encode([user_query], convert_to_numpy=True)
    _, idx = index.search(query_embedding, k=1)
    return answers[idx[0][0]]


# Resume tips generator
def resume_tips(resume_text):
    tips = [
        "Highlight measurable achievements.",
        "Use action verbs.",
        "Keep it concise and clear.",
        "Tailor resume for each job application.",
        "Check formatting and grammar."
    ]
    return "Some tips to improve your resume:\n- " + "\n- ".join(tips)
