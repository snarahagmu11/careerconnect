from pathlib import Path
import yaml

def load_config(path: str = "config/settings.yaml"):
    cfg_path = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found at: {cfg_path}")
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

