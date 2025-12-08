# scripts/download_all_kaggle.py
import kagglehub, os, shutil, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST_ROOT = ROOT / "data" / "raw"
DEST_ROOT.mkdir(parents=True, exist_ok=True)

def copy_matches(src_dir: Path, dest_dir: Path, patterns: list[str], required=True):
    dest_dir.mkdir(parents=True, exist_ok=True)
    regs = [re.compile("^" + p.replace(".", r"\.").replace("*", ".*") + "$", re.IGNORECASE) for p in patterns]
    found = 0
    for root, _, files in os.walk(src_dir):
        for f in files:
            if any(rx.match(f) for rx in regs):
                shutil.copy(Path(root)/f, dest_dir/f)
                print(f"  ✓ {Path(root).relative_to(src_dir) / f} -> {dest_dir/f}")
                found += 1
    if not found and required:
        print(f"  ! No matches for {patterns}. Copying all files for inspection.")
        for root, _, files in os.walk(src_dir):
            for f in files:
                shutil.copy(Path(root)/f, dest_dir/f)
                found += 1
    return found

def dl(slug: str, dest: str, patterns: list[str], required=True):
    print(f"\n=== {slug} → {dest} ===")
    path = Path(kagglehub.dataset_download(slug))
    out = DEST_ROOT / dest
    n = copy_matches(path, out, patterns, required=required)
    print(f"Done ({n} files) → {out}")

def main():
    # 1) LinkedIn Job Postings (core)
    dl(
        "arshkon/linkedin-job-postings",
        "linkedin_jobs",
        patterns=[
            "postings.csv",       # main jobs table
            "job_skills.csv",     # skills per job (sometimes in jobs/ subfolder)
            "*.csv",              # fallback to capture variants
        ],
        required=True,
    )

    # 2) US jobs on Monster.com (PromptCloudHQ)
    dl(
        "PromptCloudHQ/us-jobs-on-monstercom",
        "monster_jobs",
        patterns=[
            "monster_com-job_sample.csv",
            "*.csv",
        ],
        required=True,
    )

    # 3) Software Engineer Jobs & Salaries 2024 (emreksz)
    dl(
        "emreksz/software-engineer-jobs-and-salaries-2024",
        "se_jobs_salaries_2024",
        patterns=[
            "*Software*Engineer*Salari*.csv",
            "*.csv",
        ],
        required=True,
    )

    # 4) Combined O*NET + ESCO Skills with Embeddings (paiky1995)
    dl(
        "paiky1995/skills-dataset-with-embeddings",
        "skills_embeddings",
        patterns=[
            "skill_embeddings.parquet",
            "*.parquet",
            "*.csv",
        ],
        required=True,
    )

    # 5) Udemy Online Education Courses (yusufdelikkaya)
    dl(
        "yusufdelikkaya/udemy-online-education-courses",
        "udemy_courses",
        patterns=[
            "udemy_online_education_*.csv",
            "*.csv",
        ],
        required=True,
    )

    print("\n🎉 All downloads complete. Files are under data/raw/*")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)

