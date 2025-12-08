from pathlib import Path
from typing import List

def load_paths(root: Path) -> List[Path]:
    root = Path(root)
    picked: List[Path] = []
    for p in root.rglob("*"):
        if not p.is_file(): 
            continue
        s = p.suffix.lower()
        if s in {".txt", ".md", ".pdf"}:
            picked.append(p)
        elif s == ".csv":
            if p.name == "careerconnect_master.csv" or p.name.endswith("_processed.csv"):
                picked.append(p)
    return sorted(picked)

