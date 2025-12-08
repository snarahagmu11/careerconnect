import streamlit as st
import pandas as pd
import requests
from pathlib import Path

API_URL = "http://127.0.0.1:8010/api/process_resume/"

st.set_page_config(page_title="CareerConnect", layout="wide")

LOGO_PATH = Path(__file__).parent / "logo.png"
if LOGO_PATH.exists():
    st.image(str(LOGO_PATH), use_container_width=True)

st.markdown("""
<h1 style="text-align:center;">CareerConnect</h1>
<p style="text-align:center; color:#888;">AI job & upskilling assistant</p>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload Resume PDF")

if uploaded:
    res = requests.post(API_URL, files={"file":(uploaded.name, uploaded.getvalue(),"application/pdf")})
    data = res.json()

    tab1,tab2,tab3,tab4 = st.tabs(["Skills","Jobs","Upskilling","Summary"])

    with tab1:
        st.write("**Detected Skills:**", ", ".join(data.get("skills",[])))

    with tab2:
        jobs = pd.DataFrame(data.get("jobs",[]))

        if not jobs.empty:
            jobs.drop(columns=["skills"], errors="ignore", inplace=True)

            jobs.columns = ["Title","Company","Location","Salary","Url","Description"]

            def make_click(u):
                return f'<a href="{u}" target="_blank">🔗 Link</a>' if u not in ["N/A",""] else "N/A"

            jobs["Url"] = jobs["Url"].apply(make_click)

            st.write(jobs.to_html(escape=False, index=False), unsafe_allow_html=True)
        else:
            st.warning("No Jobs Found!")

    with tab3:
        courses = pd.DataFrame(data.get("courses",[]))
        if not courses.empty:
            courses.columns = ["Title","Url"]
            courses["Url"] = courses["Url"].apply(lambda u: f'<a href="{u}" target="_blank">📘 {u}</a>' if u not in ["N/A",""] else "N/A")
            st.write(courses.to_html(escape=False, index=False), unsafe_allow_html=True)
        else:
            st.info("No Courses Found!")

    with tab4:
        st.markdown(data.get("summary","No summary"))

