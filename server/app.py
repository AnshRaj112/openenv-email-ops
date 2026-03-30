from __future__ import annotations

import os

import uvicorn
import gradio as gr

try:
    # Primary server entry: FastAPI app for the environment.
    from backend.main import app as fastapi_app
except Exception as exc:  # pragma: no cover
    # Keep import error visible if the container build is incomplete.
    fastapi_app = None
    _import_error = exc
else:  # pragma: no cover
    _import_error = None

try:
    from backend.runner import run_episode
except Exception:  # pragma: no cover
    run_episode = None


def _run_episode_ui(task_id: str, provider: str):
    # Always returns a graded score + trace; when provider keys are missing,
    # `run_episode()` will fall back to the same behavior as `LLM_PROVIDER=local`.
    if run_episode is None:  # pragma: no cover
        raise RuntimeError("Failed to import backend.runner.run_episode")
    result = run_episode(task_id=task_id, provider=provider)
    return float(result["score"]), result.get("steps", [])


_task_choices = [
    "email-triage-easy",
    "email-triage-medium",
    "email-triage-hard",
]

_provider_choices = ["local", "groq"]


_demo = gr.Interface(
    fn=_run_episode_ui,
    inputs=[
        gr.Dropdown(choices=_task_choices, value="email-triage-easy", label="Task"),
        gr.Dropdown(choices=_provider_choices, value="local", label="Provider"),
    ],
    outputs=[gr.Number(label="Score"), gr.JSON(label="Step Trace")],
    title="OpenEnv Email Triage",
    description="Runs one episode and shows the deterministic score and step trace.",
)

def main() -> None:
    """
    OpenEnv multi-mode entrypoint.

    This is intentionally thin: it just starts the FastAPI server defined in
    `backend/main.py` so that `openenv validate` sees a callable `main()`.
    """

    if fastapi_app is None:  # pragma: no cover
        raise RuntimeError(f"Failed to import backend.main:app: {_import_error!r}")

    # Serve UI at root while keeping API endpoints exposed from FastAPI.
    app = gr.mount_gradio_app(fastapi_app, _demo, path="/")

    # HF Spaces typically expects the app to listen on $PORT (default to 7860).
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()

