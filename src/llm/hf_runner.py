# src/llm/hf_runner.py

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

HF_MODEL = os.getenv("HF_GEN_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
HF_MAX_NEW_TOKENS = int(os.getenv("HF_MAX_NEW_TOKENS", "512"))

_SYSTEM = (
    "You are a helpful RAG assistant for jobs and courses. "
    "Use ONLY the provided context to answer. "
    "If the answer is not in the context, say you don't know. "
    "Cite context chunks using [1], [2], etc."
)

class HuggingFaceLLM:
    """
    Local Mistral runner: downloads the model to the node and runs it
    via transformers (no Hugging Face Inference API / router).
    """

    def __init__(self, model_id: str | None = None, max_new_tokens: int | None = None):
        self.model_id = model_id or HF_MODEL
        self.max_new_tokens = max_new_tokens or HF_MAX_NEW_TOKENS

        # Load tokenizer & model locally
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        # Some Mistral variants may have no pad_token; use eos as pad if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Use fp16 on GPU, fall back to fp32 on CPU
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Force GPU explicitly
        cuda_available = torch.cuda.is_available()

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if cuda_available else torch.float32,
            device_map={"": 0} if cuda_available else "cpu",   # MIG-safe explicit mapping
        )

        if cuda_available:
            self.model.to("cuda")


        # Simple text-generation pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def generate(self, prompt: str) -> str:
        """
        `prompt` already includes context + question (from build_prompt).
        We wrap it with a system header and run local generation.
        """
        full_prompt = f"[INST] <<SYS>>\n{_SYSTEM}\n<</SYS>>\n{prompt}\n[/INST]"

        try:
            out = self.pipe(
                full_prompt,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )[0]["generated_text"]

            # Strip the prompt from the beginning of the generated text
            if out.startswith(full_prompt):
                out = out[len(full_prompt):]
            return out.strip()

        except Exception as e:
            return f"(local HF generation error: {e})"

