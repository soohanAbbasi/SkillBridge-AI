import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from googleapiclient.discovery import build
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="SkillBridge AI",
    page_icon="🌟",
    layout="wide"
)

# ===== CUSTOM CSS =====
st.markdown("""
<style>
    .main { background-color: #0F1117; }
    .stApp { background-color: #0F1117; }
    h1, h2, h3 { color: #00D4FF; }
    .metric-card {
        background-color: #1E2130;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #00D4FF;
        margin: 10px 0;
    }
    .job-card {
        background-color: #1E2130;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #51CF66;
        margin: 8px 0;
    }
    .course-card {
        background-color: #1E2130;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #FFD43B;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ===== LOAD DATA =====
@st.cache_data
def load_data():
    jobs_df = pd.read_csv("postings.csv")
    skills_df = pd.read_csv("skills.csv")
    job_skills_df = pd.read_csv("job_skills.csv")
    coursera_df = pd.read_csv("coursera_course_2024.csv")

    job_skills_merged = job_skills_df.merge(skills_df, on="skill_abr", how="left")
    job_skills_grouped = job_skills_merged.groupby("job_id")["skill_name"].apply(
        lambda x: " ".join(x.dropna())).reset_index()
    job_skills_grouped.columns = ["job_id", "skills_text"]

    jobs_clean = jobs_df[["job_id", "title", "description", "skills_desc",
                           "formatted_experience_level", "formatted_work_type",
                           "location", "job_posting_url"]].copy()
    jobs_clean = jobs_clean.merge(job_skills_grouped, on="job_id", how="left")
    jobs_clean["combined_text"] = (
        jobs_clean["title"].fillna("") + " " +
        jobs_clean["skills_desc"].fillna("") + " " +
        jobs_clean["skills_text"].fillna("")
    )

    coursera_clean = coursera_df[["title", "Skills", "Description",
                                   "Level", "rating", "URL"]].copy()
    coursera_clean["combined_text"] = (
        coursera_clean["title"].fillna("") + " " +
        coursera_clean["Skills"].fillna("") + " " +
        coursera_clean["Description"].fillna("")
    )

    return jobs_clean, coursera_clean

@st.cache_resource
def build_models(jobs_clean, coursera_clean):
    job_tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    job_tfidf_matrix = job_tfidf.fit_transform(jobs_clean["combined_text"].fillna(""))
    course_tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    course_tfidf_matrix = course_tfidf.fit_transform(coursera_clean["combined_text"].fillna(""))
    return job_tfidf, job_tfidf_matrix, course_tfidf, course_tfidf_matrix

# Load data and models
jobs_clean, coursera_clean = load_data()
job_tfidf, job_tfidf_matrix, course_tfidf, course_tfidf_matrix = build_models(
    jobs_clean, coursera_clean)

# Initialize LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=GROQ_API_KEY,
    temperature=0.7
)

# ===== FUNCTIONS =====
def recommend(user_skills, top_jobs=5, top_courses=3):
    """Main recommendation function"""
    user_job_vec = job_tfidf.transform([user_skills])
    job_scores = cosine_similarity(user_job_vec, job_tfidf_matrix).flatten()
    top_job_indices = job_scores.argsort()[::-1][:top_jobs]
    recommended_jobs = jobs_clean.iloc[top_job_indices][
        ["title", "location", "formatted_experience_level",
         "formatted_work_type", "job_posting_url"]].copy()
    recommended_jobs["match_score"] = job_scores[top_job_indices].round(2)

    user_course_vec = course_tfidf.transform([user_skills])
    course_scores = cosine_similarity(user_course_vec, course_tfidf_matrix).flatten()
    top_course_indices = course_scores.argsort()[::-1][:top_courses]
    recommended_courses = coursera_clean.iloc[top_course_indices][
        ["title", "Level", "rating", "URL"]].copy()
    recommended_courses["match_score"] = course_scores[top_course_indices].round(2)

    return recommended_jobs, recommended_courses

def get_youtube_videos(skill, max_results=2):
    """Fetch YouTube tutorials"""
    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        request = youtube.search().list(
            part="snippet",
            q=f"{skill} tutorial for beginners",
            type="video",
            maxResults=max_results,
            relevanceLanguage="en",
            order="relevance"
        )
        response = request.execute()
        return [{
            "title": item["snippet"]["title"],
            "channel": item["snippet"]["channelTitle"],
            "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}"
        } for item in response["items"]]
    except:
        return []

def run_agent(prompt_system, prompt_human, input_dict):
    """Run a LangChain agent"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_system),
        ("human", prompt_human)
    ])
    return (prompt | llm).invoke(input_dict).content

# ===== MAIN UI =====
st.markdown("<h1 style=text-align:center>🌟 SkillBridge AI</h1>",
            unsafe_allow_html=True)
st.markdown(
    "<p style=text-align:center;color:#888>AI-Powered Career & Learning Recommendation System</p>",
    unsafe_allow_html=True)
st.markdown("---")

# ===== SIDEBAR =====
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("### Enter Your Skills")
    user_skills = st.text_input(
        "Skills (comma separated)",
        placeholder="python, machine learning, data analysis")
    top_jobs = st.slider("Number of Jobs", 3, 10, 5)
    top_courses = st.slider("Number of Courses", 1, 6, 3)
    run_button = st.button("🚀 Get Recommendations", type="primary")

    st.markdown("---")
    st.markdown("### 📊 Features")
    st.markdown("✅ Job Recommendations")
    st.markdown("✅ Course Suggestions")
    st.markdown("✅ YouTube Tutorials")
    st.markdown("✅ AI Career Coach")
    st.markdown("✅ Market Trends Analysis")
    st.markdown("✅ Bias Detection")
    st.markdown("✅ Explainable AI (SHAP)")

# ===== RESULTS =====
if run_button and user_skills:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💼 Jobs", "📚 Courses & Videos",
        "🤖 AI Agents", "📊 Dashboard", "⚖️ Bias & XAI"
    ])

    jobs_result, courses_result = recommend(user_skills, top_jobs, top_courses)

    # TAB 1: JOBS
    with tab1:
        st.markdown("## 💼 Top Job Recommendations")
        for _, row in jobs_result.iterrows():
            st.markdown(f"""
            <div class="job-card">
                <h3>🏢 {row["title"]}</h3>
                <p>📍 {row["location"]} | 💼 {row["formatted_work_type"]} |
                   📊 Match: {row["match_score"]:.0%}</p>
                <a href="{row["job_posting_url"]}" target="_blank">🔗 Apply Now</a>
            </div>
            """, unsafe_allow_html=True)

    # TAB 2: COURSES & VIDEOS
    with tab2:
        st.markdown("## 📚 Recommended Courses")
        for _, row in courses_result.iterrows():
            st.markdown(f"""
            <div class="course-card">
                <h3>📖 {row["title"]}</h3>
                <p>📈 Level: {row["Level"]} | ⭐ Rating: {row["rating"]}</p>
                <a href="{row["URL"]}" target="_blank">🔗 Enroll Now</a>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("## ▶️ YouTube Tutorials")
        skills_list = [s.strip() for s in user_skills.split(",")][:3]
        for skill in skills_list:
            with st.expander(f"🔍 {skill} tutorials"):
                videos = get_youtube_videos(skill)
                for v in videos:
                    st.markdown(f"📺 **{v['title']}**")
                    st.markdown(f"Channel: {v['channel']}")
                    st.markdown(f"[▶️ Watch]({v['url']})")
                    st.markdown("---")

    # TAB 3: AI AGENTS
    with tab3:
        st.markdown("## 🤖 AI Agent Analysis")
        col1, col2 = st.columns(2)

        with col1:
            with st.spinner("🔍 Analyzing skills..."):
                skills_analysis = run_agent(
                    "You are an expert career skills analyzer. Analyze skills and provide: 1) Skill level 2) Top 3 missing skills 3) Strength summary. Be concise.",
                    "Analyze: {skills}",
                    {"skills": user_skills}
                )
            st.markdown("### 🔍 Skills Analysis")
            st.markdown(skills_analysis)

            with st.spinner("📈 Analyzing market..."):
                market = run_agent(
                    "You are a job market expert. Provide: 1) Demand level 2) Top 3 industries 3) Future outlook 4) Emerging skills. Be concise.",
                    "Analyze market for: {skills}",
                    {"skills": user_skills}
                )
            st.markdown("### 📈 Market Trends")
            st.markdown(market)

        with col2:
            with st.spinner("🎯 Career coaching..."):
                career = run_agent(
                    "You are an expert career coach. Provide: 1) Top 3 career paths 2) Why each suits them 3) Salary range USD 4) Growth potential. Be concise.",
                    "Skills: {skills} Jobs: {jobs}",
                    {"skills": user_skills,
                     "jobs": ", ".join(jobs_result["title"].tolist()[:3])}
                )
            st.markdown("### 🎯 Career Coach")
            st.markdown(career)

            with st.spinner("📅 Creating learning path..."):
                learning = run_agent(
                    "You are a learning path designer. Create: 1) 3-month roadmap 2) Week by week plan 3) Key milestones. Be concise.",
                    "Skills: {skills} Courses: {courses}",
                    {"skills": user_skills,
                     "courses": ", ".join(courses_result["title"].tolist())}
                )
            st.markdown("### 📅 Learning Path")
            st.markdown(learning)

    # TAB 4: DASHBOARD
    with tab4:
        st.markdown("## 📊 Career Dashboard")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Top Match Score", f"{jobs_result['match_score'].max():.0%}")
        with col2:
            st.metric("Jobs Found", len(jobs_result))
        with col3:
            st.metric("Courses Found", len(courses_result))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor("#0F1117")

        ax1 = axes[0]
        ax1.set_facecolor("#1E2130")
        titles = [t[:20] + "..." if len(t) > 20 else t
                  for t in jobs_result["title"].tolist()]
        ax1.barh(titles, jobs_result["match_score"].tolist(),
                 color=plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(titles))))
        ax1.set_title("Job Match Scores", color="white")
        ax1.tick_params(colors="white", labelsize=8)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        ax2 = axes[1]
        ax2.set_facecolor("#1E2130")
        course_names = [t[:20] + "..." if len(t) > 20 else t
                        for t in courses_result["title"].tolist()]
        ax2.bar(course_names, courses_result["rating"].tolist(),
                color=["#51CF66", "#FFD43B", "#FF6B6B"])
        ax2.set_title("Course Ratings", color="white")
        ax2.set_ylim(0, 5)
        ax2.tick_params(colors="white", labelsize=7, axis="x", rotation=15)
        ax2.tick_params(colors="white", labelsize=8, axis="y")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)

    # TAB 5: BIAS & XAI
    with tab5:
        st.markdown("## ⚖️ Bias Detection & Explainable AI")
        st.info("Run the full analysis in the Kaggle notebook to see detailed charts.")

        st.markdown("### 📊 Key Bias Findings:")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h4>📍 Location Bias</h4>
                <p>US-based jobs dominate 41.7% of postings</p>
            </div>
            <div class="metric-card">
                <h4>👔 Experience Bias</h4>
                <p>Mid-Senior level dominates 43.9% of jobs</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4>💼 Work Type Bias</h4>
                <p>Full-time jobs = 79.8% of all postings</p>
            </div>
            <div class="metric-card">
                <h4>🧠 XAI Finding</h4>
                <p>Skill Match Score is #1 recommendation factor</p>
            </div>
            """, unsafe_allow_html=True)

elif not user_skills and run_button:
    st.warning("⚠️ Please enter your skills first!")
else:
    st.markdown("""
    <div style="text-align:center; padding:50px;">
        <h2 style="color:#00D4FF">👈 Enter your skills in the sidebar to get started!</h2>
        <p style="color:#888">SkillBridge AI will recommend jobs, courses,
        and create your personalized career roadmap</p>
    </div>
    """, unsafe_allow_html=True)
