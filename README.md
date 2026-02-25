# Agentic Helpdesk MVP (Google ADK), 2-agent workflow

A small, production-style Agentic AI demo with a deterministic 2-agent pipeline, a separate tool service, and a trajectory-aware eval suite.

## Overview

This repo contains:

- **ADK agent app (port 8000)**, deterministic 2-agent workflow  
  - **TriageAgent**: produces structured triage JSON into session state  
  - **ActionAgent**: uses tools for KB lookup and ticketing, with an approval gate before ticket creation
- **Tool service (port 7001)**: FastAPI REST API for KB search and SQLite-backed tickets
- **Evaluation suite**: 39 test scenarios with basic trajectory-aware scoring (tool calls plus response checks)

## Architecture

User -> ADK API Server -> SequentialAgent(TriageAgent -> ActionAgent) -> REST tools -> Tool Service -> (KB JSON, SQLite)

## Prerequisites

- Python 3.10+
- A Gemini API key (Google AI Studio)

## Environment variables

ADK uses the `google-genai` client under the hood, which expects `GOOGLE_API_KEY`.  
This repo also supports `GEMINI_API_KEY` and maps it to `GOOGLE_API_KEY` automatically.

Set one of:

- `GOOGLE_API_KEY="..."`
- `GEMINI_API_KEY="..."`

Optional:

- `GEMINI_MODEL="gemini-2.0-flash"`
- `TOOL_SERVICE_URL="http://localhost:7001"`

## Quickstart (Windows, PowerShell)

### 1) Create a virtual env and install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Set your API key (choose one)

```powershell
$env:GOOGLE_API_KEY="PASTE_KEY_HERE"
# or
$env:GEMINI_API_KEY="PASTE_KEY_HERE"
```

### 3) Start the Tool Service (Terminal 1)

```powershell
python -m tool_service.app
```

Health check: http://localhost:7001/health

### 4) Start the ADK API server (Terminal 2)

```powershell
adk api_server agents
```

Swagger UI: http://localhost:8000/docs

### 5) Run the demo client (Terminal 3)

```powershell
python scripts\demo_client.py
```

## Quickstart (Linux or macOS, bash)

### 1) Create a virtual env and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Set your API key (choose one)

```bash
export GOOGLE_API_KEY="PASTE_KEY_HERE"
# or
export GEMINI_API_KEY="PASTE_KEY_HERE"
```

### 3) Start the Tool Service (Terminal 1)

```bash
python -m tool_service.app
```

### 4) Start the ADK API server (Terminal 2)

```bash
adk api_server agents
```

### 5) Run the demo client (Terminal 3)

```bash
python scripts/demo_client.py
```

## Evaluation (39 scenarios, trajectory-aware)

The eval suite calls the ADK `/run` endpoint and grades:

- **Trajectory**: required and forbidden tool calls (includes approval gate checks)
- **Content**: required and recommended phrases, or clarifying questions
- **Safety**: avoids requesting sensitive secrets (simple regex checks)
- **Style**: light heuristics (for example, numbered steps preferred for some tests)

Run it (after both servers are up):

```bash
python eval/run_eval.py
```

Outputs:
- Console pass and fail summary (threshold = 0.70)
- JSON report at `eval/report.json`

## Repo layout

- `agents/helpdesk_agent/agent.py` , ADK agents plus tools plus SequentialAgent root
- `tool_service/app.py` , FastAPI REST API for tools
- `tool_service/storage.py` , KB search plus SQLite ticket persistence
- `scripts/demo_client.py` , minimal client for `/run`
- `eval/evalset.json` , test scenarios and rubric expectations
- `eval/run_eval.py` , evaluation runner and scorer

### Port conflicts
If port 7001 or 8000 is already in use, stop the conflicting process or change the ports in your startup commands and config.

