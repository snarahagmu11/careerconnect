# -----------------------------
# FIX IMPORT PATH FOR STREAMLIT
# -----------------------------
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd
import requests

API_URL = "http://127.0.0.1:8010/api/process_resume/"

st.set_page_config(page_title="CareerConnect", layout="wide")

# Logo
LOGO = Path(__file__).parent / "logo.png"
if LOGO.exists():
    st.image(str(LOGO), use_container_width=True)

st.title("CareerConnect – AI Job & Upskilling Assistant")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if uploaded_file:
    st.success("Resume uploaded!")

    with st.spinner("Analyzing..."):
        r = requests.post(
            API_URL,
            files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        )

    if r.status_code != 200:
        st.error("Backend error")
        st.text(r.text)
        st.stop()

    data = r.json()

    tabs = st.tabs(["Skills", "Jobs", "Upskilling", "Summary"])

    # ---------------- SKILLS ----------------
    with tabs[0]:
        st.subheader("Extracted Skills")
        st.write(", ".join(data.get("skills", [])))

    # ---------------- JOBS ----------------
    with tabs[1]:
        st.subheader("Recommended Jobs")
        jobs = data.get("jobs", [])

        if jobs:
            df = pd.DataFrame(jobs)
            df["url"] = df["url"].apply(lambda u: f"[Link]({u})" if u else "")
            st.markdown(df.to_markdown(index=False), unsafe_allow_html=True)
        else:
            st.info("No jobs found.")

    # ---------------- UPSKILLING ----------------
    with tabs[2]:
        st.subheader("Upskilling Suggestions")

        courses = data.get("upskilling", [])
        if courses:
            df = pd.DataFrame(courses)
            df["url"] = df["url"].apply(lambda u: f"[Link]({u})")
            st.markdown(df.to_markdown(index=False), unsafe_allow_html=True)
        else:
            st.info("No course suggestions.")

    # ---------------- SUMMARY ----------------
    with tabs[3]:
        st.subheader("Career Summary")
        st.markdown(data.get("summary", ""), unsafe_allow_html=True)

