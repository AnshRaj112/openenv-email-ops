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

## Baseline Inference (Groq / Gemini / Local)

Baseline script: `baseline/run_baseline.py` runs all three tasks and prints scores in `0.0–1.0`.

Remote inference uses **only** `GROQ_API_KEY` (Groq) and `GEMINI_API_KEY` (Google Gemini).

- **Providers:** `groq`, `gemini`, or `local`.
- **Groq:** `GROQ_API_KEY`, optional `GROQ_MODEL` (default `llama-3.3-70b-versatile`).
- **Gemini:** `GEMINI_API_KEY`, optional `GEMINI_MODEL` (default `gemini-2.0-flash`).
- **Auto-detect (no `LLM_PROVIDER`):** Groq if `GROQ_API_KEY` is set, else Gemini if `GEMINI_API_KEY` is set, else `local`.
- **Local:** `LLM_PROVIDER=local` or no API keys.

If a remote call fails, stderr logs once and the run falls back to the same heuristic as `local` (no label oracle).

**Run with Groq:**

```bash
py -m pip install -r requirements.txt
set LLM_PROVIDER=groq
set GROQ_API_KEY=gsk_your_key_here
py baseline/run_baseline.py
```

**Run with Gemini:**

```bash
py -m pip install -r requirements.txt
set LLM_PROVIDER=gemini
set GEMINI_API_KEY=your_key_here
py baseline/run_baseline.py
```

Expected output format:

```text
email-triage-easy: 0.xxxx
email-triage-medium: 0.xxxx
email-triage-hard: 0.xxxx
overall: 0.xxxx
```

Baseline scores (deterministic `local` provider; intentionally imperfect; no external API calls):
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

Build:

```bash
docker build -t openenv-email-ops .
```

Run:

```bash
docker run --env-file .env -p 8000:8000 -p 7860:7860 openenv-email-ops
```

## Space HTTP Endpoints (Docker)

The container runs on port `7860`:
- UI: `GET /` (redirects to Gradio UI at `/ui`)
- UI: `GET /ui`
- API: `GET /health` returns healthy status
- API: `POST /reset?task_id=email-triage-easy` returns the initial observation
- API: `POST /step` accepts an `Action` and returns `{observation, reward, done, info}`
- API: `GET /state` returns the current internal state
- API: `GET /tasks` returns available tasks + the action schema
- API: `GET /baseline` runs the repo baseline over all 3 tasks
- API: `POST /grader?task_id=...&provider=local` runs one episode and returns a deterministic score in `0.0–1.0`

---
HF Space notes:
- SDK: `Docker`
- Add repo tag: `openenv`
- Set secrets:
  - `GROQ_API_KEY` (optional)
  - `GEMINI_API_KEY` (optional)
