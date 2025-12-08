import streamlit as st
import pandas as pd
import requests
from pathlib import Path

API_URL = "http://127.0.0.1:8010/api/process_resume/"

st.set_page_config(page_title="CareerConnect", layout="wide")

LOGO = Path(__file__).parent / "logo.png"
if LOGO.exists():
    st.image(str(LOGO), use_container_width=True)

st.title("CareerConnect – AI Job & Upskilling Assistant")

file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if file:
    with st.spinner("Processing..."):
        r = requests.post(
            API_URL,
            files={"file": (file.name, file.getvalue(), "application/pdf")},
        )

    data = r.json()

    tabs = st.tabs(["Skills", "Jobs", "Upskilling", "Summary"])

    # --- Skills ---
    with tabs[0]:
        st.write(", ".join(data.get("skills", [])))

    # --- Jobs ---
    with tabs[1]:
        jobs = pd.DataFrame(data.get("jobs", []))
        if not jobs.empty:
            jobs["url"] = jobs["url"].apply(
                lambda u: f'<a href="{u}" target="_blank">Link</a>' if u else "N/A"
            )
            st.write(jobs.to_html(escape=False, index=False), unsafe_allow_html=True)
        else:
            st.info("No jobs found.")

     # --- Courses ---
    with tabs[2]:
        courses = pd.DataFrame(data.get("courses", []))

        if not courses.empty:

            # Normalize column names (handle Url/url/URL)
            courses.columns = courses.columns.str.lower()

            # Replace missing URLs
            courses["url"] = courses["url"].apply(lambda u: u if u not in [None, "", "N/A"] else "N/A")

            # Make URLs clickable
            courses["url"] = courses["url"].apply(
                lambda u: f'<a href="{u}" target="_blank">Course</a>' 
                if u != "N/A" else "N/A"
            )

            st.write(courses.to_html(escape=False, index=False), unsafe_allow_html=True)

        else:
            st.info("No courses found.")

    # --- Summary ---
    with tabs[3]:
        st.markdown(data.get("summary", ""))

