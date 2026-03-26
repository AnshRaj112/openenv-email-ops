from fastapi import FastAPI
from dotenv import load_dotenv

from backend.analytics import analyze_run
from backend.leaderboard import get_leaderboard, save_score
from backend.runner import run_episode

load_dotenv()

app = FastAPI()


@app.get("/")
def home():
    return {"status": "AI Ops Lab Running"}


@app.post("/run")
def run(model: str = "groq"):
    result = run_episode(model)
    score = result["score"]
    save_score(model, score)
    analytics = analyze_run(result["steps"])
    return {"score": score, "steps": result["steps"], "analytics": analytics}


@app.get("/leaderboard")
def leaderboard():
    return get_leaderboard()
