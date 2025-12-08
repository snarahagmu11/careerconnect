# scripts/preprocess_data.py

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = RAW

def _exists(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")
    return p

def _series_or_empty(df: pd.DataFrame, col: str):
    if col and col in df.columns:
        return df[col].astype(str).fillna("")
    return pd.Series([""] * len(df), index=df.index)

# ----------------------------- LINKEDIN -----------------------------------
def process_linkedin(max_rows=None):
    p_post = _exists(RAW / "linkedin_jobs" / "postings.csv")
    p_sk = _exists(RAW / "linkedin_jobs" / "job_skills.csv")

    postings = pd.read_csv(p_post, nrows=max_rows, low_memory=False)
    skills = pd.read_csv(p_sk, low_memory=False)

    skill_col = "skill_name" if "skill_name" in skills.columns else "skill" if "skill" in skills.columns else ""
    if skill_col == "":
        skills["skill_name"] = ""
        skill_col = "skill_name"

    agg_sk = skills.groupby("job_id")[skill_col].apply(lambda s: ", ".join(sorted({str(x) for x in s.dropna()}))).reset_index(name="skills_joined")
    df = postings.merge(agg_sk, on="job_id", how="left")

    title = "title" if "title" in df.columns else "job_title" if "job_title" in df.columns else ""
    comp = "company_name" if "company_name" in df.columns else "company" if "company" in df.columns else ""
    loc = "job_location" if "job_location" in df.columns else "location" if "location" in df.columns else ""
    desc = "description" if "description" in df.columns else "job_description" if "job_description" in df.columns else ""
    url = "job_posting_url" if "job_posting_url" in df.columns else ""

    df["text"] = (
        "Job Title: " + _series_or_empty(df, title) +
        "\nCompany: " + _series_or_empty(df, comp) +
        "\nLocation: " + _series_or_empty(df, loc) +
        "\nSkills: " + df["skills_joined"].fillna("") +
        "\nURL: " + _series_or_empty(df, url) +
        "\nDescription: " + _series_or_empty(df, desc)
    )

    df_out = pd.DataFrame({
        "title": _series_or_empty(df, title),
        "company": _series_or_empty(df, comp),
        "location": _series_or_empty(df, loc),
        "skills": df["skills_joined"].fillna(""),
        "description": _series_or_empty(df, desc),
        "url": _series_or_empty(df, url),
        "text": df["text"],
    })
    df_out.to_csv(OUT / "linkedin_jobs_processed.csv", index=False)
    print("✅ LinkedIn → linkedin_jobs_processed.csv")

# ----------------------------- MONSTER -----------------------------------
def process_monster(max_rows=None):
    p = _exists(RAW / "monster_jobs" / "monster_com-job_sample.csv")
    df = pd.read_csv(p, nrows=max_rows, low_memory=False)

    jt = "job_title"
    comp = "organization"  # ✅ Fix applied based on dataset preview
    loc = "location"
    desc = "job_description"
    url = "page_url"
    sal = "salary"

    df["text"] = (
        "Job Title: " + _series_or_empty(df, jt) +
        "\nCompany: " + _series_or_empty(df, comp) +
        "\nLocation: " + _series_or_empty(df, loc) +
        "\nSalary: " + _series_or_empty(df, sal) +
        "\nURL: " + _series_or_empty(df, url) +
        "\nDescription: " + _series_or_empty(df, desc)
    )

    df_out = pd.DataFrame({
        "title": _series_or_empty(df, jt),
        "company": _series_or_empty(df, comp),
        "location": _series_or_empty(df, loc),
        "salary": _series_or_empty(df, sal),
        "description": _series_or_empty(df, desc),
        "url": _series_or_empty(df, url),
        "text": df["text"],
    })
    df_out.to_csv(OUT / "monster_jobs_processed.csv", index=False)
    print("✅ Monster → monster_jobs_processed.csv")

# ----------------------------- SE SALARIES -------------------------------
def process_se_salaries(max_rows=None):
    p = _exists(RAW / "se_jobs_salaries_2024" / "Software Engineer Salaries.csv")
    df = pd.read_csv(p, nrows=max_rows)

    df["text"] = (
        "Job Title: " + df["Job Title"] +
        "\nCompany: " + df["Company"] +
        "\nLocation: " + df["Location"] +
        "\nSalary: " + df["Salary"]
    )

    df_out = pd.DataFrame({
        "title": df["Job Title"],
        "company": df["Company"],
        "location": df["Location"],
        "salary": df["Salary"],
        "url": pd.Series(["N/A"] * len(df)),
        "text": df["text"],
    })
    df_out.to_csv(OUT / "se_jobs_salaries_processed.csv", index=False)
    print("✅ SE Salaries → se_jobs_salaries_processed.csv")

# ----------------------------- UDEMY -------------------------------------
def process_udemy(max_rows=None):
    p = _exists(RAW / "udemy_courses" / "udemy_online_education_courses_dataset.csv")
    df = pd.read_csv(p, nrows=max_rows).fillna("")

    keywords = df["course_title"].astype(str).str.lower().str.replace(r"[^a-z0-9 ]", " ", regex=True)

    df["text"] = (
        "Course Title: " + df["course_title"] +
        "\nURL: " + df["url"] +
        "\nKeywords: " + keywords
    )

    df_out = pd.DataFrame({
        "title": df["course_title"],
        "url": df["url"],
        "is_paid": df["is_paid"],
        "text": df["text"],
    })
    df_out.to_csv(OUT / "udemy_courses_processed.csv", index=False)
    print("✅ Udemy → udemy_courses_processed.csv")

# ----------------------------- MASTER MERGE ------------------------------
def write_master():
    files = [
        "linkedin_jobs_processed.csv",
        "monster_jobs_processed.csv",
        "se_jobs_salaries_processed.csv",
        "udemy_courses_processed.csv",
    ]
    frames = []
    for f in files:
        p = OUT / f
        if p.exists():
            df = pd.read_csv(p)
            df["source_file"] = f
            frames.append(df[["text", "source_file"]])

    master = pd.concat(frames, ignore_index=True)
    master.to_csv(OUT / "careerconnect_master.csv", index=False)
    print("✅ Master → careerconnect_master.csv")

# ----------------------------- MAIN --------------------------------------
def main():
    process_linkedin()
    process_monster()
    process_se_salaries()
    process_udemy()
    write_master()

if __name__ == "__main__":
    main()


