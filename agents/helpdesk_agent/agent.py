import json
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv
import requests
from google.adk.agents import LlmAgent, SequentialAgent

# Load .env on import, Windows friendly
load_dotenv()


def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v else default


GEMINI_MODEL = _env("GEMINI_MODEL", "gemini-2.0-flash")
TOOL_SERVICE_URL = _env("TOOL_SERVICE_URL", "http://localhost:7001").rstrip("/")


def _http_get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = f"{TOOL_SERVICE_URL}{path}"
    r = requests.get(url, params=params or {}, timeout=10)
    r.raise_for_status()
    return r.json()


def _http_post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{TOOL_SERVICE_URL}{path}"
    r = requests.post(url, json=payload, timeout=15)
    r.raise_for_status()
    return r.json()


def _http_patch(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{TOOL_SERVICE_URL}{path}"
    r = requests.patch(url, json=payload, timeout=15)
    r.raise_for_status()
    return r.json()


def kb_search(query: str, top_k: int = 3) -> Dict[str, Any]:
    """Search a helpdesk knowledge base via REST API.

    Args:
      query: search query
      top_k: max number of results
    Returns:
      dict with results
    """
    try:
        data = _http_get("/kb/search", params={"q": query, "top_k": top_k})
        return {"status": "success", "results": data.get("results", [])}
    except Exception as e:
        return {"status": "error", "error": str(e), "results": []}


def create_ticket(title: str, description: str, priority: str = "P2") -> Dict[str, Any]:
    """Create a helpdesk ticket via REST API.

    Safety rule: Only call this tool if the user explicitly approves ticket creation.
    """
    try:
        data = _http_post("/tickets", {"title": title, "description": description, "priority": priority})
        return {"status": "success", "ticket": data}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_ticket(ticket_id: str) -> Dict[str, Any]:
    """Fetch a ticket by id."""
    try:
        data = _http_get(f"/tickets/{ticket_id}")
        return {"status": "success", "ticket": data}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def update_ticket_status(ticket_id: str, status: str) -> Dict[str, Any]:
    """Update a ticket status (open, in_progress, blocked, resolved, closed)."""
    try:
        data = _http_patch(f"/tickets/{ticket_id}", {"status": status})
        return {"status": "success", "ticket": data}
    except Exception as e:
        return {"status": "error", "error": str(e)}


TRIAGE_INSTRUCTION = """You are TriageAgent, an IT helpdesk triage specialist.
Transform the user's message into a structured triage object.

Rules:
1) Output MUST be valid JSON only, no markdown, no extra text.
2) If key info is missing, populate missing_info and include up to 3 clarifying_questions.
3) Always fill ticket_draft even if you recommend answer_with_kb.

Schema:
{
  "issue_type": "incident|request|question|other",
  "summary": "one sentence",
  "severity": "P1|P2|P3|P4",
  "entities": {
    "system": "string or null",
    "error_message": "string or null",
    "user_impact": "string or null"
  },
  "missing_info": ["..."],
  "clarifying_questions": ["..."],
  "recommended_action": "answer_with_kb|create_ticket|ask_clarifying_questions",
  "kb_search_query": "string or null",
  "ticket_draft": {
    "title": "string",
    "description": "string",
    "priority": "P1|P2|P3|P4"
  }
}
"""


ACTION_INSTRUCTION = """You are ActionAgent. You receive the triage JSON in {triage_json}.

Behavior:
- If recommended_action is answer_with_kb, call kb_search(kb_search_query) then give step by step guidance.
- If recommended_action is ask_clarifying_questions, ask the questions, do not call create_ticket.
- If recommended_action is create_ticket:
  - Do NOT create a ticket unless the user explicitly approves, for example: "approved", "create the ticket".
  - If not approved, present the draft ticket (title, priority, short description) and ask for approval.

Safety:
- Do not request secrets, passwords, API keys.
- Suggest redacting sensitive data before sharing logs.
"""


triage_agent = LlmAgent(
    name="TriageAgent",
    model=GEMINI_MODEL,
    instruction=TRIAGE_INSTRUCTION,
    output_key="triage_json",
    description="Extracts a structured triage JSON from user input.",
)

action_agent = LlmAgent(
    name="ActionAgent",
    model=GEMINI_MODEL,
    instruction=ACTION_INSTRUCTION,
    tools=[kb_search, create_ticket, get_ticket, update_ticket_status],
    description="Uses tools to resolve issues and drafts or creates tickets with approval.",
)

root_agent = SequentialAgent(
    name="Helpdesk2AgentWorkflow",
    sub_agents=[triage_agent, action_agent],
    description="Deterministic 2 agent workflow: triage then act.",
)
