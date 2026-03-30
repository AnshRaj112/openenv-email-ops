from __future__ import annotations

import os

import uvicorn

try:
    # Primary server entry: FastAPI app for the environment.
    from backend.main import app as fastapi_app
except Exception as exc:  # pragma: no cover
    # Keep import error visible if the container build is incomplete.
    fastapi_app = None
    _import_error = exc
else:  # pragma: no cover
    _import_error = None


def main() -> None:
    """
    OpenEnv multi-mode entrypoint.

    This is intentionally thin: it just starts the FastAPI server defined in
    `backend/main.py` so that `openenv validate` sees a callable `main()`.
    """

    if fastapi_app is None:  # pragma: no cover
        raise RuntimeError(f"Failed to import backend.main:app: {_import_error!r}")

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(fastapi_app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()

