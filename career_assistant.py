# career_assistant.py
import random

def get_answer(user_input):
    """Return career guidance based on keywords in user input."""
    responses = {
        "developer": "To become a developer, focus on building projects, learn Git, contribute to open-source, and master problem-solving on LeetCode.",
        "data": "A Data Science career needs Python, Pandas, NumPy, statistics, ML, and visualization tools like Power BI or Tableau.",
        "ai": "AI roles require Python, machine learning, deep learning (TensorFlow/PyTorch), and strong math foundations.",
        "cybersecurity": "Learn networking, ethical hacking, and tools like Wireshark, Burp Suite, and Metasploit.",
        "web": "For web development, learn HTML, CSS, JavaScript, React, and backend frameworks like Flask or Django.",
        "cloud": "Cloud engineers master AWS, Azure, or GCP, along with DevOps practices and Docker/Kubernetes."
    }
    for key, val in responses.items():
        if key in user_input.lower():
            return val
    return "Explore your interests — software, data, AI, or design. I can guide you with tailored resources!"

def resume_tips():
    """Return 3 random resume improvement tips."""
    tips = [
        "Keep your resume concise — ideally one page.",
        "Add measurable achievements like 'Increased accuracy by 15%'.",
        "Include GitHub or portfolio links to showcase projects.",
        "Use keywords from the job description to pass ATS filters.",
        "Highlight recent and relevant experience first.",
        "Add certifications and skills relevant to your desired job."
    ]
    return random.sample(tips, 3)
