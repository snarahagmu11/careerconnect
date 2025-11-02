from pathlib import Path
import pandas as pd
from pypdf import PdfReader

def load_paths(root: Path):
    return [p for p in root.rglob("*") if p.suffix.lower() in {".pdf", ".md", ".txt", ".csv"}]

def read_text(path: Path) -> list[dict]:
    suf = path.suffix.lower()
    if suf == ".pdf":
        reader = PdfReader(str(path))
        text = "\n".join((pg.extract_text() or "") for pg in reader.pages)
        return [{"source": str(path), "text": text}]
    if suf in {".md", ".txt"}:
        return [{"source": str(path), "text": path.read_text(encoding="utf-8", errors="ignore")}]
    if suf == ".csv":
        df = pd.read_csv(path)
        rows = []
        for i, row in df.iterrows():
            rows.append({"source": f"{path}#{i}", "text": " ".join(map(str, row.values))})
        return rows
    return []

