import streamlit as st
import pandas as pd
import requests

# Backend API endpoint
API_URL = "http://127.0.0.1:8000/api/process_resume/"

# Page setup
st.set_page_config(page_title="CareerConnect", layout="wide")

# Title and description
st.title("CareerConnect")
st.markdown("AI-powered job and upskilling assistant")

# Resume upload
uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

if uploaded_file:
    st.success("Resume uploaded successfully!")

    # Send PDF to FastAPI backend
    with st.spinner("Analyzing your resume..."):
        files = {"file": uploaded_file.getvalue()}
        try:
            response = requests.post(API_URL, files=files)
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to backend. Make sure FastAPI is running on port 8000.")
            st.stop()

    if response.status_code == 200:
        data = response.json()

        # Display extracted skills
        st.subheader("Extracted Skills")
        st.write(", ".join(data.get("skills", [])))

        # Display recommended jobs
        st.subheader("Recommended Jobs")
        job_df = pd.DataFrame(data.get("jobs", []))
        if not job_df.empty:
            st.table(job_df)
        else:
            st.write("No job recommendations available.")

        # Display upskilling suggestions
        st.subheader("Upskilling Suggestions")
        course = data.get("course", {})
        if course:
            st.write(f"Missing Skill: {course.get('skill')} → Suggested Course: [{course.get('name')}]({course.get('url')})")
        else:
            st.write("No upskilling suggestions available.")

        # Display career summary
        st.subheader("Career Summary")
        st.write(data.get("summary", "No summary generated."))
    else:
        st.error(f"Backend returned an error: {response.status_code}")
        st.text(response.text)
