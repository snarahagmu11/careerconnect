# src/eval/get_clean_titles.py

import pandas as pd
import re
from pathlib import Path

RAW = Path("data/raw")

# ---------------------------------------------------------------
# CLEANING LOGIC
# ---------------------------------------------------------------

def is_clean(title: str) -> bool:
    if not isinstance(title, str):
        return False

    t = title.strip()

    # Too short
    if len(t) < 4:
        return False

    # Reject titles whose FIRST token is 1 letter (A, B, C...)
    first = t.split()[0]
    if len(first) == 1:
        return False

    # Reject patterns like A/, A-, A&, A:
    if re.match(r"^[A-Za-z][\/\-\&\:\.]", t):
        return False

    # HTML garbage
    if "<" in t or ">" in t:
        return False

    # Very noisy job posting spam words
    noise = [
        "apply today", "immediate hire", "sign on bonus", "sign-on",
        "call now", "bonus", "urgently hiring"
    ]
    if any(n in t.lower() for n in noise):
        return False

    # Shift/pay patterns
    if re.search(r"(1st|2nd|3rd|shift|hr|hour|\$)", t.lower()):
        return False

    # Has numeric job ID tags
    if re.search(r"#\s*\d", t):
        return False

    # All uppercase = noise
    if t.isupper():
        return False

    return True

# ---------------------------------------------------------------
# LOAD ALL TITLES FROM LINKEDIN + MONSTER + SALARIES
# ---------------------------------------------------------------

job_titles = []

files = [
    RAW / "linkedin_jobs_processed.csv",
    RAW / "monster_jobs_processed.csv",
    RAW / "se_jobs_salaries_processed.csv",
]

for f in files:
    if f.exists():
        df = pd.read_csv(f)
        if "title" in df.columns:
            job_titles.extend(df["title"].astype(str).tolist())

print(f"Total titles found: {len(job_titles)}")

# ---------------------------------------------------------------
# APPLY CLEANING
# ---------------------------------------------------------------

clean_titles = [t for t in job_titles if is_clean(t)]
print(f"Clean titles after filtering: {len(clean_titles)}")

# Remove duplicates
clean_titles = sorted(set(clean_titles))

# ---------------------------------------------------------------
# SHOW SAMPLE TITLES
# ---------------------------------------------------------------

print("\n--- Sample 20 Clean Titles ---")
for t in clean_titles[:20]:
    print(t)

# ---------------------------------------------------------------
# Save if needed
# ---------------------------------------------------------------
pd.DataFrame({"title": clean_titles}).to_csv("clean_job_titles.csv", index=False)
print("\n✔ Saved → clean_job_titles.csv")

