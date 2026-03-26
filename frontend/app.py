import gradio as gr
import requests

API = "http://localhost:8000"


def run_model(provider):
    try:
        response = requests.post(f"{API}/run", params={"model": provider}, timeout=60)
        response.raise_for_status()
        res = response.json()
        return res.get("score", 0), res.get("analytics", {}), res.get("steps", [])
    except Exception as exc:
        return 0, {"error": str(exc)}, []


def get_lb():
    try:
        response = requests.get(f"{API}/leaderboard", timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        return [{"model": "unavailable", "score": 0, "error": str(exc)}]


with gr.Blocks() as app:
    gr.Markdown("# AI Ops Lab Dashboard")

    provider = gr.Dropdown(["local", "groq", "gemini"], value="local", label="Provider")
    run_btn = gr.Button("Run Agent")

    score = gr.Number(label="Score")
    analytics = gr.JSON(label="Analytics")
    steps = gr.JSON(label="Step Trace")

    run_btn.click(run_model, inputs=provider, outputs=[score, analytics, steps])

    gr.Markdown("## Leaderboard")
    lb = gr.JSON(label="Top Runs")
    refresh = gr.Button("Refresh Leaderboard")
    refresh.click(get_lb, outputs=lb)

app.launch(server_name="0.0.0.0", server_port=7860)
