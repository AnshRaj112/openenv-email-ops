import os


def llm_call(prompt):
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider in {"openai", "local", "groq"}:
        # Keep legacy provider aliases mapped to the same deterministic client.
        from .local_client import generate as local_generate

        return local_generate(prompt)
    from .local_client import generate as local_generate

    return local_generate(prompt)
