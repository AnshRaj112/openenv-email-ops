import os


def llm_call(prompt):
    provider = os.getenv("LLM_PROVIDER", "local").lower()
    if provider == "groq":
        from .groq_client import generate as groq_generate

        return groq_generate(prompt)
    if provider == "local":
        from .local_client import generate as local_generate

        return local_generate(prompt)
    from .local_client import generate as local_generate

    return local_generate(prompt)
