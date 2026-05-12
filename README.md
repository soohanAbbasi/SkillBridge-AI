# 🌟 SkillBridge AI
### AI-Powered Career & Learning Recommendation System

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0-red)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-green)](https://langchain.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> Built for **DevNetwork AI/ML Hackathon 2026** | Social Good Track

---

## 🎯 Problem Statement

Millions of people, especially in developing countries like Pakistan, don't know:
- What career path to take based on their skills
- What courses to learn
- Where to find free learning resources

**SkillBridge AI solves this with a complete AI-powered career roadmap.**

---

## ✨ Features

| Feature | Description |
|---|---|
| 💼 **Job Recommendations** | 124,000+ LinkedIn job postings matched via TF-IDF |
| 📚 **Course Suggestions** | 6,600+ Coursera courses with ratings |
| ▶️ **YouTube Tutorials** | Live YouTube API integration for free learning |
| 🤖 **5 AI Agents** | LangChain + LLaMA 3.3 powered agents |
| 📊 **Career Dashboard** | Visual charts for job matches & salary ranges |
| ⚖️ **Bias Detection** | Job market bias analysis (location, experience, salary) |
| 🧠 **Explainable AI** | SHAP values showing why each job was recommended |

---

## 🤖 AI Agents

1. **Skills Analyzer Agent** — Evaluates skill level & identifies gaps
2. **Career Coach Agent** — Recommends career paths with salary ranges
3. **Learning Path Agent** — Creates 3-month personalized roadmap
4. **Market Trends Agent** — Analyzes current job market demand
5. **Resume Analyzer Agent** — Extracts skills from uploaded resume

---

## 🏗️ Architecture

```
User Input (Skills / Resume)
           ↓
┌─────────────────────────────┐
│      SkillBridge AI         │
│                             │
│  TF-IDF Recommendation      │
│  Engine (Jobs + Courses)    │
│           +                 │
│  5 LangChain AI Agents      │
│  (Groq/LLaMA 3.3)          │
│           +                 │
│  YouTube Data API v3        │
│           +                 │
│  Bias Detection (Pandas)    │
│           +                 │
│  Explainable AI (SHAP)      │
└─────────────────────────────┘
           ↓
    Streamlit Dashboard
```

---

## 📦 Tech Stack

- **Frontend:** Streamlit
- **ML/NLP:** Scikit-learn (TF-IDF, Cosine Similarity), SHAP
- **AI Agents:** LangChain + Groq (LLaMA 3.3-70b)
- **APIs:** YouTube Data API v3
- **Data:** LinkedIn Job Postings (124K+), Coursera Courses (6.6K+)
- **Visualization:** Matplotlib

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/SkillBridge-AI.git
cd SkillBridge-AI
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set API Keys
Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key
YOUTUBE_API_KEY=your_youtube_api_key
```

### 4. Download Datasets
- [LinkedIn Job Postings](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)
- [Coursera Courses 2024](https://www.kaggle.com/datasets/azraimohamad/coursera-courses-2024)

Place CSV files in the project root directory.

### 5. Run the app
```bash
streamlit run app.py
```

---

## 📊 Dataset

| Dataset | Source | Size |
|---|---|---|
| LinkedIn Job Postings | Kaggle (arshkon) | 124,000+ jobs |
| Coursera Courses 2024 | Kaggle (azraimohamad) | 6,600+ courses |

---

## 🔍 Key Findings (Bias Detection)

- 📍 **Location Bias:** US-based jobs dominate 41.7% of postings
- 👔 **Experience Bias:** Mid-Senior level dominates 43.9% of jobs
- 💼 **Work Type Bias:** Full-time jobs = 79.8% of all postings
- 💰 **Salary Bias:** Contract workers earn more than permanent employees

---

## 🧠 Explainable AI Results

Most important factors in job recommendation (SHAP analysis):
1. **Skill Match Score** — #1 factor
2. **Experience Level** — #2 factor
3. **Work Type** — #3 factor

---

## 📁 Project Structure

```
SkillBridge-AI/
│
├── app.py                  # Streamlit web application
├── notebook.ipynb          # Kaggle notebook (full pipeline)
├── requirements.txt        # Python dependencies
├── .env.example           # API keys template
├── .gitignore             # Git ignore file
└── README.md              # Project documentation
```

---

## 👩‍💻 About the Developer

- **Role:** Computer Science Lecturer (5+ years)
- **Experience:** Recommendation Systems, Agentic AI (CrewAI), NLP
- **Built for:** Social Good — helping underserved communities find career paths

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🏆 Hackathon Submission

- **Event:** DevNetwork AI/ML Hackathon 2026
- **Track:** Social Good / Community
- **Submission Date:** May 2026
