from typing import List, Dict
from huggingface_hub import InferenceClient
import os
from pathlib import Path

_SYSTEM_PATH = Path(__file__).resolve().parents[1] / "templates" / "system_prompt.txt"
_SYSTEM = _SYSTEM_PATH.read_text(encoding="utf-8") if _SYSTEM_PATH.exists() else "You are a helpful RAG assistant."

HF_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_MODEL = os.getenv("HF_GEN_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
HF_MAX_NEW_TOKENS = int(os.getenv("HF_MAX_NEW_TOKENS", "256"))
HF_TEMPERATURE = float(os.getenv("HF_TEMPERATURE", "0.2"))

def _ctx_block(chunks: List[Dict], limit: int = 5) -> str:
    blocks = []
    for i, c in enumerate(chunks[:limit], 1):
        src = c.get("source", "unknown")
        t = c.get("chunk", "")[:1600]
        blocks.append(f"[{i}] SOURCE: {src}\n{t}")
    return "\n\n".join(blocks)

def _format_prompt(user_query: str, contexts: List[Dict]) -> str:
    ctx = _ctx_block(contexts)
    return (
        f"<system>\n{_SYSTEM}\n</system>\n\n"
        f"<context>\n{ctx}\n</context>\n\n"
        f"<user>\n{user_query}\n</user>\n\n"
        "Instructions:\n"
        "- Use ONLY the context when answering. If the answer is not present, say you do not have that info.\n"
        "- Cite with [1], [2], ... when you use a specific chunk."
    )

def answer(user_query: str, contexts: List[Dict]) -> str:
    prompt = _format_prompt(user_query, contexts)

    if not HF_TOKEN:
        return f"(HF_API_TOKEN not set)\n\nPROMPT PREVIEW:\n{prompt[:1200]}"

    client = InferenceClient(model=HF_MODEL, token=HF_TOKEN)
    try:
        out = client.text_generation(
            prompt,
            max_new_tokens=HF_MAX_NEW_TOKENS,
            temperature=HF_TEMPERATURE,
            do_sample=HF_TEMPERATURE > 0,
            return_full_text=False,
        )
        return out.strip()
    except Exception as e:
        return f"(HF generation error: {e})"

