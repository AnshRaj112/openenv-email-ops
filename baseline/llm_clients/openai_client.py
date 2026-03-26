import os

from openai import OpenAI

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def generate(prompt):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)
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
