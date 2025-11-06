import streamlit as st
import pandas as pd

# Basic page config
st.set_page_config(page_title="CareerConnect", layout="wide")

# Title + description
st.title("CareerConnect 🧠")
st.markdown("AI-powered job and upskilling assistant")

# Resume upload
uploaded_file = st.file_uploader("📄 Upload your resume (PDF)", type=["pdf"])

if uploaded_file:
    st.success("Resume uploaded successfully!")

    # Mock: Extracted skills
    extracted_skills = ["Python", "SQL", "Machine Learning"]
    st.subheader("🧠 Extracted Skills")
    st.write(", ".join(extracted_skills))

    # Mock: Recommended jobs
    st.subheader("💼 Recommended Jobs")
    job_df = pd.DataFrame({
        "Job Title": ["Data Analyst", "ML Engineer", "Business Analyst"],
        "Company": ["Google", "Amazon", "Deloitte"],
        "Location": ["Remote", "Seattle", "New York"]
    })
    st.table(job_df)

    # Mock: Missing skills + course recommendations
    st.subheader("📚 Upskilling Suggestions")
    st.write("Missing Skill: Tableau → Suggested Course: [Tableau for Beginners](https://udemy.com)")

    # Mock: Generated summary
    st.subheader("🧾 Career Summary")
    st.write("You are well-suited for Data Analyst roles and could strengthen visualization skills (Tableau).")
