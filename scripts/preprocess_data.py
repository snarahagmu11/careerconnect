# scripts/preprocess_data.py  (patched)

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW  = ROOT / "data" / "raw"
OUT  = RAW

def _exists(p):
    if not p.exists():
        raise FileNotFoundError(p)
    return p

def _series_or_empty(df: pd.DataFrame, col: str) -> pd.Series:
    """Return df[col] if it exists, else an empty-string Series of the same length."""
    if col and col in df.columns:
        return df[col]
    return pd.Series([""] * len(df), index=df.index)

# ------------------- LINKEDIN (unchanged from your working version) -------------------
def process_linkedin(max_rows=None):
    p_post = _exists(RAW/"linkedin_jobs"/"postings.csv")
    p_sk   = _exists(RAW/"linkedin_jobs"/"job_skills.csv")

    postings = pd.read_csv(p_post, nrows=max_rows, low_memory=False)
    skills   = pd.read_csv(p_sk, low_memory=False)

    skcol = "skill_name" if "skill_name" in skills.columns else ("skill" if "skill" in skills.columns else None)
    if skcol is None:
        skills["skill_name"] = ""
        skcol = "skill_name"

    agg_sk = (
        skills.groupby("job_id")[skcol]
        .apply(lambda s: ", ".join(sorted({str(x) for x in s.dropna()})))
        .reset_index(name="skills_joined")
    )
    df = postings.merge(agg_sk, on="job_id", how="left")

    title = "title" if "title" in df.columns else "job_title" if "job_title" in df.columns else ""
    comp  = "company_name" if "company_name" in df.columns else "company" if "company" in df.columns else ""
    loc   = "job_location" if "job_location" in df.columns else "location" if "location" in df.columns else ""
    desc  = "description" if "description" in df.columns else "job_description" if "job_description" in df.columns else ""

    df["text"] = (
        "Job Title: " + _series_or_empty(df, title).astype(str).fillna("") +
        "\nCompany: " + _series_or_empty(df, comp).astype(str).fillna("") +
        "\nLocation: " + _series_or_empty(df, loc).astype(str).fillna("") +
        "\nSkills: " + df["skills_joined"].fillna("") +
        "\nDescription: " + _series_or_empty(df, desc).astype(str).fillna("")
    )

    out = OUT / "linkedin_jobs_processed.csv"
    df[[title or "text", comp or "text", loc or "text", "skills_joined", desc or "text", "text"]].rename(
        columns={title:"title", comp:"company", loc:"location", desc:"description", "skills_joined":"skills"}
    ).to_csv(out, index=False)
    print(f"✅ LinkedIn -> {out} ({len(df)} rows)")

# ------------------- MONSTER (patched) -------------------
def process_monster(max_rows=None):
    p = _exists(RAW/"monster_jobs"/"monster_com-job_sample.csv")
    df = pd.read_csv(p, nrows=max_rows, low_memory=False)

    # Try common column names; fall back to empty series when missing
    jt   = "job_title" if "job_title" in df.columns else (df.columns[0] if len(df.columns) else "")
    comp = "company_name" if "company_name" in df.columns else ""
    loc  = "location" if "location" in df.columns else ""
    desc = "description" if "description" in df.columns else ""

    df["text"] = (
        "Job Title: " + _series_or_empty(df, jt).astype(str).fillna("") +
        "\nCompany: " + _series_or_empty(df, comp).astype(str).fillna("") +
        "\nLocation: " + _series_or_empty(df, loc).astype(str).fillna("") +
        "\nDescription: " + _series_or_empty(df, desc).astype(str).fillna("")
    )

    out = OUT / "monster_jobs_processed.csv"
    df_out = pd.DataFrame({
        "title": _series_or_empty(df, jt),
        "company": _series_or_empty(df, comp),
        "location": _series_or_empty(df, loc),
        "description": _series_or_empty(df, desc),
        "text": df["text"],
    })
    df_out.to_csv(out, index=False)
    print(f"✅ Monster -> {out} ({len(df_out)} rows)")

# ------------------- SE SALARIES (unchanged but safe) -------------------
def process_se_salaries(max_rows=None):
    p = _exists(RAW/"se_jobs_salaries_2024"/"Software Engineer Salaries.csv")
    df = pd.read_csv(p, nrows=max_rows, low_memory=False)

    title="Job Title" if "Job Title" in df.columns else (df.columns[0] if len(df.columns) else "")
    comp ="Company" if "Company" in df.columns else ""
    loc  ="Location" if "Location" in df.columns else ""
    sal  ="Salary" if "Salary" in df.columns else ""
    score="Company Score" if "Company Score" in df.columns else ""

    df["text"] = (
        "Job Title: " + _series_or_empty(df, title).astype(str).fillna("") +
        "\nCompany: " + _series_or_empty(df, comp).astype(str).fillna("") +
        "\nLocation: " + _series_or_empty(df, loc).astype(str).fillna("") +
        "\nSalary: " + _series_or_empty(df, sal).astype(str).fillna("") +
        ( "\nCompany Score: " + _series_or_empty(df, score).astype(str).fillna("") if score else "" )
    )

    out = OUT / "se_jobs_salaries_processed.csv"
    df_out = pd.DataFrame({
        "title": _series_or_empty(df, title),
        "company": _series_or_empty(df, comp),
        "location": _series_or_empty(df, loc),
        "salary": _series_or_empty(df, sal),
        "text": df["text"],
    })
    df_out.to_csv(out, index=False)
    print(f"✅ SE Salaries -> {out} ({len(df_out)} rows)")

# ------------------- SKILLS EMBEDDINGS (unchanged) -------------------
def process_skills_embeddings(max_rows=None):
    p = _exists(RAW/"skills_embeddings"/"skill_embeddings.parquet")
    df = pd.read_parquet(p)
    if max_rows: df = df.head(max_rows)
    sk = "skill" if "skill" in df.columns else df.columns[0]
    df["text"] = "Skill: " + _series_or_empty(df, sk).astype(str).fillna("")
    out = OUT / "skills_embeddings_processed.csv"
    df[[sk, "text"]].rename(columns={sk:"skill"}).to_csv(out, index=False)
    print(f"✅ Skills embeddings -> {out} ({len(df)} rows)")

# ------------------- UDEMY (unchanged but safe) -------------------
def process_udemy(max_rows=None):
    p = _exists(RAW/"udemy_courses"/"udemy_online_education_courses_dataset.csv")
    df = pd.read_csv(p, nrows=max_rows, low_memory=False)

    title = "course_title" if "course_title" in df.columns else (df.columns[0] if len(df.columns) else "")
    subj  = "subject" if "subject" in df.columns else ""
    level = "level" if "level" in df.columns else ""
    price = "price" if "price" in df.columns else ""
    desc  = "content_duration" if "content_duration" in df.columns else ""

    df["text"] = (
        "Course Title: " + _series_or_empty(df, title).astype(str).fillna("") +
        ( "\nSubject: " + _series_or_empty(df, subj).astype(str).fillna("") if subj else "" ) +
        ( "\nLevel: " + _series_or_empty(df, level).astype(str).fillna("") if level else "" ) +
        ( "\nPrice: " + _series_or_empty(df, price).astype(str).fillna("") if price else "" ) +
        ( "\nDuration/Desc: " + _series_or_empty(df, desc).astype(str).fillna("") if desc else "" )
    )

    out = OUT / "udemy_courses_processed.csv"
    df_out = pd.DataFrame({
        "title": _series_or_empty(df, title),
        "subject": _series_or_empty(df, subj),
        "price": _series_or_empty(df, price),
        "text": df["text"],
    })
    df_out.to_csv(out, index=False)
    print(f"✅ Udemy -> {out} ({len(df_out)} rows)")

def write_master():
    frames=[]
    for f in [
        "linkedin_jobs_processed.csv",
        "monster_jobs_processed.csv",
        "se_jobs_salaries_processed.csv",
        "skills_embeddings_processed.csv",
        "udemy_courses_processed.csv",
    ]:
        p = OUT/f
        if p.exists():
            df = pd.read_csv(p)
            df["source_file"]=f
            frames.append(df[["text","source_file"]])
    if frames:
        master = pd.concat(frames, ignore_index=True)
        master.to_csv(OUT/"careerconnect_master.csv", index=False)
        print(f"✅ Master -> {OUT/'careerconnect_master.csv'} ({len(master)} rows)")
    else:
        print("⚠️ No processed files found; master not created.")

def main():
    max_rows = None  # set e.g. 100000 for trial runs on huge files
    process_linkedin(max_rows)
    process_monster(max_rows)        # <-- this was failing; now robust
    process_se_salaries(max_rows)
    process_skills_embeddings(max_rows)
    process_udemy(max_rows)
    write_master()

if __name__ == "__main__":
    main()

