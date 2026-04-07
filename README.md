---
title: OpenEnv AI Ops Lab
emoji: "🤖"
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
tags:
  - openenv
  - enterprise-ai
  - email-triage
---

# OpenEnv AI Ops: Email Triage Environment

Real-world OpenEnv environment for enterprise email operations.  
An agent must triage incoming support/compliance emails using `respond`, `escalate`, or `archive` while minimizing risk and maximizing completion quality.

## Why This Environment

Email triage is a real operational workflow in support, trust/safety, and compliance teams:
- classify routine vs high-risk requests,
- respond safely when appropriate,
- escalate legal/payment/security issues,
- avoid destructive or low-value actions.

This repository implements a full OpenEnv-style environment with typed models, deterministic task graders, and reproducible baseline inference.

## OpenEnv Interface

Core API is implemented in `env/environment.py`:
- `reset() -> Observation`
- `step(action: Action) -> (Observation, Reward, done, info)`
- `state() -> dict`

Typed models are in `env/models.py`:
- `Observation`
- `Action`
- `Reward`

## Observation Space

`Observation` contains:
- `inbox`: pending email objects (`id`, `subject`, `body`, `priority`, `sender`, `category`)
- `current_email`: the next pending email (or `None`)
- `history`: processed actions (`<email_id>:<action_type>`)
- `step_count`
- `remaining_count`
- `instructions`

## Action Space

`Action` contains:
- `type`: one of `respond | escalate | archive`
- `email_id`: target email id (optional, defaults to current email)
- `content`: response text (used for `respond`)
- `rationale`: optional trace field

## Reward Function

Reward is dense and trajectory-aware.  
`Reward.value` is clipped to `[-1.0, 1.0]`, with components:
- `decision_quality`: correct routing reward
- `response_quality`: keyword-based quality for response actions
- `safety_penalty`: penalties for risky routing (e.g. not escalating legal/payment/security)
- `loop_penalty`: penalties for invalid/repeated actions
- `completion_bonus`: reward for finishing all emails

This gives partial progress signal while still penalizing unsafe behavior.

## Tasks and Graders

Task definitions are in `env/tasks.py`, graders in `env/grader.py`.

- **Easy (`email-triage-easy`)**
  - routine support + low-risk inbox handling
  - `max_steps=6`
- **Medium (`email-triage-medium`)**
  - mixed routine + security/payment escalation decisions
  - `max_steps=9`
- **Hard (`email-triage-hard`)**
  - legal/compliance/security heavy triage
  - `max_steps=12`

Episode score is normalized to `[0.0, 1.0]` via deterministic grader logic.

## Submission inference (`inference.py`)

Phase 2 expects a root-level `inference.py` that:

- Uses the **OpenAI Python client** with the injected proxy:
  - `base_url` from `API_BASE_URL`
  - `api_key` from `API_KEY` or `HF_TOKEN` (either is accepted)
- Runs the three tasks (`EasyTask`, `MediumTask`, `HardTask`) against `EmailEnv`.
- Prints structured lines to **stdout** (with `flush=True`), one episode per task:
  - `[START] task=... env=... model=...`
  - `[STEP] step=... action=... reward=... done=... error=...` (once per `env.step`)
  - `[END] success=... steps=... score=... rewards=...`

### Mandatory environment variables (hackathon / validator)

Define these in the platform’s environment configuration (or a local `.env` for testing):

| Variable | Purpose |
|----------|---------|
| `API_BASE_URL` | LLM API base URL (LiteLLM / OpenAI-compatible proxy). |
| `API_KEY` or `HF_TOKEN` | Auth for the proxy; `inference.py` uses one or the other. |
| `MODEL_NAME` | Model id passed to `chat.completions.create`. |
| `LOCAL_IMAGE_NAME` | Only required if you load the env via `from_docker_image(...)`; otherwise optional. |

Do not hardcode secrets in the repo; rely on injected env vars in CI.

## Baseline script (optional, local dev)

`baseline/run_baseline.py` runs all three tasks and prints plain score lines (not the submission stdout format).

The baseline router defaults to provider `openai` and maps legacy `local` / `groq` names to the same deterministic fallback client used for offline runs.

If a remote call fails, stderr logs once and the run falls back to the same heuristic as the offline client (no label oracle).

**Run baseline locally:**

```bash
py -m pip install -r requirements.txt
py baseline/run_baseline.py
```

Expected baseline output format:

```text
email-triage-easy: 0.xxxx
email-triage-medium: 0.xxxx
email-triage-hard: 0.xxxx
overall: 0.xxxx
```

Baseline scores (deterministic offline provider; intentionally imperfect; no external API calls):
```text
email-triage-easy: 0.8750
email-triage-medium: 0.7129
email-triage-hard: 0.8240
overall: 0.8040
```

## Local Setup

```bash
py -m pip install -r requirements.txt
```

Optional API/UI stack:

```bash
py -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
py frontend/app.py
```

## OpenEnv Metadata

Environment metadata is declared in `openenv.yaml`.
Validate with:

```bash
openenv validate
```

Pre-submission repo validation (does local spec + API smoke checks):

```bash
py scripts/pre_submission_check.py
```

## Docker / Hugging Face Spaces

The `Dockerfile` uses a public mirror base image (`public.ecr.aws/docker/library/python:3.10-slim`) to reduce Docker Hub pull issues in CI.

Build (requires Docker Desktop or another engine running):

```bash
docker build -t openenv-email-ops:latest .
```

Set `LOCAL_IMAGE_NAME=openenv-email-ops:latest` only if your workflow uses that image name with `from_docker_image(...)`.

Run:

```bash
docker run --env-file .env -p 8000:8000 -p 7860:7860 openenv-email-ops:latest
```

## Space HTTP Endpoints (Docker)

The container runs on port `7860`:
- UI: `GET /`
- API: `GET /health` returns healthy status
- API: `POST /reset?task_id=email-triage-easy` returns the initial observation
- API: `POST /step` accepts an `Action` and returns `{observation, reward, done, info}`
- API: `GET /state` returns the current internal state
- API: `GET /tasks` returns available tasks + the action schema
- API: `GET /baseline` runs the repo baseline over all 3 tasks
- API: `POST /grader?task_id=...&provider=openai` runs one episode and returns a deterministic score in `0.0–1.0`

---
HF Space notes:
- SDK: `Docker`
- Add repo tag: `openenv`
- Set secrets / environment variables for inference and the app as required by the platform, for example:
  - `API_BASE_URL`
  - `API_KEY` or `HF_TOKEN`
  - `MODEL_NAME`
  - Optional: `LOCAL_IMAGE_NAME` (only if using docker-image–based env loading)
