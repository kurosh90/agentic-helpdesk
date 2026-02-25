import json
import uuid
import requests


API_BASE = "http://localhost:8000"


def create_session(app_name: str, user_id: str, session_id: str) -> None:
    url = f"{API_BASE}/apps/{app_name}/users/{user_id}/sessions/{session_id}"
    r = requests.post(url, json={}, timeout=10)
    r.raise_for_status()


def run(app_name: str, user_id: str, session_id: str, text: str) -> str:
    url = f"{API_BASE}/run"
    payload = {
        "appName": app_name,
        "userId": user_id,
        "sessionId": session_id,
        "newMessage": {
            "role": "user",
            "parts": [{"text": text}],
        },
    }
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    events = r.json()

    for ev in reversed(events):
        parts = ev.get("content", {}).get("parts", [])
        for p in parts:
            if "text" in p and p["text"]:
                return p["text"]
    return json.dumps(events, indent=2)


def main() -> None:
    app_name = "helpdesk_agent"
    user_id = "u_demo"
    session_id = f"s_{uuid.uuid4().hex[:8]}"
    create_session(app_name, user_id, session_id)

    msg = "My VPN cannot connect, it says certificate expired. Can you help?"
    print(run(app_name, user_id, session_id, msg))

    msg2 = "Approved, please create the ticket, this blocks my work today."
    print(run(app_name, user_id, session_id, msg2))


if __name__ == "__main__":
    main()
