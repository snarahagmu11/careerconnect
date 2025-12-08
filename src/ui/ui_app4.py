# src/ui/ui_app.py

import streamlit as st
import pandas as pd
import requests
from pathlib import Path

API_URL = "http://127.0.0.1:8010/api/process_resume/"

st.set_page_config(page_title="CareerConnect", layout="wide")

# Logo
LOGO_PATH = Path(__file__).parent / "logo.png"
if LOGO_PATH.exists():
    st.image(str(LOGO_PATH), use_container_width=True)

# Title
st.markdown("""
<h1 style="text-align:center; font-size:40px;">CareerConnect</h1>
<p style="text-align:center; font-size:18px; color:#777;">
AI-powered job & upskilling assistant
</p>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

if uploaded:
    st.success("Resume uploaded!")

    with st.spinner("Analyzing your resume..."):
        resp = requests.post(
            API_URL,
            files={
                "file": (
                    uploaded.name,
                    uploaded.getvalue(),
                    uploaded.type or "application/pdf",
                )
            }
        )

    if resp.status_code != 200:
        st.error(f"Backend error {resp.status_code}")
        st.text(resp.text)
        st.stop()

    data = resp.json()

    tabs = st.tabs(["Skills", "Jobs", "Upskilling", "Summary"])

    # ---------------- SKILLS ------------------
    with tabs[0]:
        st.subheader("Extracted Skills")
        st.write(", ".join(data.get("skills", [])))

    # ---------------- JOBS --------------------
    with tabs[1]:
        st.subheader("Recommended Jobs")

        jobs = pd.DataFrame(data.get("jobs", []))

        if not jobs.empty:
            # Make URL clickable
            if "Url" in jobs.columns:
                jobs["Url"] = jobs["Url"].apply(
                    lambda x: f"[Link]({x})" if x != "N/A" else "N/A"
                )

            st.dataframe(jobs, use_container_width=True)
        else:
            st.info("No job matches found.")

    # ---------------- UPSKILLING --------------
    with tabs[2]:
        st.subheader("Upskilling Courses")

        courses = pd.DataFrame(data.get("upskilling", []))

        if not courses.empty:
            courses["Url"] = courses["Url"].apply(
                lambda x: f"[Link]({x})" if x else "N/A"
            )
            st.dataframe(courses, use_container_width=True)
        else:
            st.info("No upskilling suggestions found.")

    # ---------------- SUMMARY -----------------
    with tabs[3]:
        st.subheader("Career Summary")
        st.markdown(data.get("summary", "No summary generated."), unsafe_allow_html=True)

