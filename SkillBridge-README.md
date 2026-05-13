# SkillBridge AI

**AI-Powered Career and Learning Recommendation System**

Built for the **DevNetwork AI/ML Hackathon 2026** | Social Good Track

---

## Problem Statement

Millions of people, especially in developing countries, face three core challenges when navigating their careers:

- They do not know what career path suits their current skills
- They cannot find relevant, free learning resources
- They have no access to personalized career coaching

SkillBridge AI addresses all three with a single, intelligent system.

---

## What It Does

Enter your skills and SkillBridge AI will:

1. Match you with the most relevant job postings from 124,000+ LinkedIn jobs
2. Recommend the best Coursera courses for your skill set
3. Pull live YouTube tutorials for each of your skills
4. Run 5 AI agents to analyze your profile, coach your career, and build a learning roadmap
5. Detect bias in the job market (location, experience, salary)
6. Explain recommendations using SHAP values

---

## System Architecture

```
User Input (skills or resume)
          |
          v
TF-IDF Recommendation Engine
          |
    |-----|-----|
    |           |
Jobs Index   Courses Index
(124K jobs)  (6600 courses)
          |
          v
5 LangChain AI Agents (Groq / LLaMA 3.3-70b)
   |
   |-- Skills Analyzer Agent
   |-- Career Coach Agent
   |-- Learning Path Agent
   |-- Market Trends Agent
   |-- Resume Analyzer Agent
          |
          v
YouTube Data API v3 (live tutorials)
          |
          v
Bias Detection + SHAP Explainability
          |
          v
Streamlit Dashboard
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| Recommendation Engine | TF-IDF + Cosine Similarity (Scikit-learn) |
| AI Agents | LangChain + Groq (LLaMA 3.3-70b) |
| Explainability | SHAP + Random Forest |
| Video Tutorials | YouTube Data API v3 |
| Bias Detection | Pandas + Matplotlib |
| Jobs Dataset | LinkedIn Job Postings (124,000+ records) |
| Courses Dataset | Coursera Courses 2024 (6,600+ records) |

---

## Features

| Feature | Description |
|---|---|
| Job Recommendations | Top matching jobs via TF-IDF cosine similarity |
| Course Suggestions | Best Coursera courses for your skill set |
| YouTube Tutorials | Live free tutorials fetched per skill |
| Skills Analyzer Agent | Identifies skill level and gaps |
| Career Coach Agent | Recommends career paths with salary ranges |
| Learning Path Agent | Builds a 3-month personalized roadmap |
| Market Trends Agent | Analyzes job market demand and outlook |
| Resume Analyzer Agent | Extracts and evaluates skills from resume |
| Bias Detection | Identifies location, experience, and salary bias |
| Explainable AI | SHAP values showing top recommendation factors |

---

## Key Findings

**Bias Detection Results:**
- Location bias: US-based jobs account for 41.7% of all postings
- Experience bias: Mid-Senior level roles make up 43.9% of jobs
- Work type bias: Full-time positions represent 79.8% of all postings
- Salary bias: Contract workers earn more than permanent employees on average

**SHAP Explainability Results:**
- Skill match score is the number one recommendation factor
- Experience level is the second most important factor
- Work type is the third most important factor

---

## Datasets

- LinkedIn Job Postings: [kaggle.com/datasets/arshkon/linkedin-job-postings](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)
- Coursera Courses 2024: [kaggle.com/datasets/azraimohamad/coursera-courses-2024](https://www.kaggle.com/datasets/azraimohamad/coursera-courses-2024)

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/soohanAbbasi/SkillBridge-AI.git
cd SkillBridge-AI
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up API keys

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key
YOUTUBE_API_KEY=your_youtube_api_key
```

Get your keys here:
- Groq API: [console.groq.com](https://console.groq.com)
- YouTube Data API v3: [console.cloud.google.com](https://console.cloud.google.com)

### 4. Download datasets

Download both CSV files from Kaggle and place them in the project root:
- `postings.csv` (LinkedIn jobs)
- `coursera_course_2024.csv` (Coursera courses)
- `skills.csv` and `job_skills.csv` (from the LinkedIn dataset)

### 5. Run the app

```bash
streamlit run app.py
```

---

## Project Structure

```
SkillBridge-AI/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env.example            # API keys template
├── .gitignore              # Git ignore rules
├── LICENSE                 # MIT license
└── README.md               # Project documentation
```

---

## About the Author

**Soohan Abbasi**
- Computer Science Lecturer with 5 years of experience
- Research background in recommendation systems, agentic AI, and NLP
- GitHub: [github.com/soohanAbbasi](https://github.com/soohanAbbasi)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Hackathon

- **Event:** DevNetwork AI/ML Hackathon 2026
- **Track:** Social Good
- **Submission:** May 2026
