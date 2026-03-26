import os
from google import genai

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

def generate(prompt):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set")
    client = genai.Client(api_key=api_key)
    res = client.models.generate_content(model=MODEL, contents=prompt)
    return res.text or ""