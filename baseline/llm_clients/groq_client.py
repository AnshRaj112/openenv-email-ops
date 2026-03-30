import os

from groq import Groq

MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def generate(prompt: str) -> str:
    api_key = (os.getenv("GROQ_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set")

    client = Groq(api_key=api_key)
    res = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are an enterprise email triage agent. Return only the final answer.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return (res.choices[0].message.content or "").strip()
