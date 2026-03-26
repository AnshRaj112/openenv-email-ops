import os

def llm_call(prompt):
    if os.getenv("LLM_PROVIDER") == "gemini":
        from .gemini_client import generate as gemini_generate
        return gemini_generate(prompt)
    from .groq_client import generate as groq_generate
    return groq_generate(prompt)