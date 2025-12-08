# src/llm/prompt_builder.py

def build_prompt(context_chunks, resume_text):
    """
    Build a structured LLM prompt for CareerConnect that extracts:
    - Skills
    - Certifications
    - Experience Highlights
    - Job Recommendations
    - Career Summary
    """

    # Combine top context snippets
    labeled = [f"[{i+1}] {c}" for i, c in enumerate(context_chunks[:5])]
    ctx = "\n\n".join(labeled)

    system = """
You are CareerConnect — an expert AI assistant that analyzes resumes 
and retrieved job/course context to produce structured, factual career insights.

RULES:
- Extract ONLY facts that appear in the resume or context.
- NEVER invent job titles, companies, or certifications.
- ALWAYS extract certifications exactly as written in the resume.
- Use retrieved context chunks when recommending jobs.
- If information is missing, say "not provided".
- Be concise and structured.
- DO NOT hallucinate additional details.
"""

    return f"""
[INST] <<SYS>>
{system}
<</SYS>>

================ RESUME TEXT ================
{resume_text}

================ CONTEXT CHUNKS =============
{ctx}

================ TASK =================
From the RESUME (not the context), extract:

1. **Skills**  
   - List one per line.

2. **Certifications**  
   - EXACT names as written in the resume.

3. **Experience Highlights**  
   - Short bullet points summarizing actual experience.

Then using BOTH the resume and the retrieved context chunks:

4. **Top Job Recommendations (max 5)**  
   Format each as:
   - Job Title at Company
   - Reason: why this matches the resume

5. **Career Summary**  
   - 5–7 lines  
   - Must mention skills, certifications, and experience  
   - No hallucinations

================ OUTPUT FORMAT (MANDATORY) ================
Skills:
- ...

Certifications:
- ...

Experience:
- ...

Job Recommendations:
1. Title at Company  
   Reason: ...

Career Summary:
<paragraph>

[/INST]
"""

