import json
import os
import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests


@dataclass
class TurnResult:
    user: str
    assistant_text: str
    tool_calls: List[str]
    tool_responses: List[str]
    raw_events: List[Dict[str, Any]]


def _last_assistant_text(events: List[Dict[str, Any]]) -> str:
    # Find the last non-empty text part across events
    for ev in reversed(events):
        content = ev.get("content") or {}
        parts = content.get("parts") or []
        for p in parts:
            txt = p.get("text")
            if isinstance(txt, str) and txt.strip():
                return txt
    return ""


def _walk_json(obj: Any):
    # Generator that yields all dicts in a nested JSON structure
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk_json(v)
    elif isinstance(obj, list):
        for it in obj:
            yield from _walk_json(it)


def _extract_function_calls_and_responses(events: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    calls: List[str] = []
    responses: List[str] = []

    # ADK events are based on Gemini Content/Parts. Tool calls show up as functionCall,
    # tool outputs show up as functionResponse.
    for ev in events:
        content = ev.get("content") or {}
        parts = content.get("parts") or []
        for p in parts:
            fc = p.get("functionCall") or p.get("function_call")
            if isinstance(fc, dict):
                name = fc.get("name")
                if isinstance(name, str):
                    calls.append(name)

            fr = p.get("functionResponse") or p.get("function_response")
            if isinstance(fr, dict):
                name = fr.get("name")
                if isinstance(name, str):
                    responses.append(name)

    # As a fallback, walk the whole JSON and try to detect any functionCall-like blocks
    if not calls or not responses:
        for d in _walk_json(events):
            if "functionCall" in d and isinstance(d["functionCall"], dict):
                n = d["functionCall"].get("name")
                if isinstance(n, str):
                    calls.append(n)
            if "functionResponse" in d and isinstance(d["functionResponse"], dict):
                n = d["functionResponse"].get("name")
                if isinstance(n, str):
                    responses.append(n)

    return calls, responses


def _count_questions(text: str) -> int:
    # Simple heuristic: count question marks
    return text.count("?")


def _has_numbered_steps(text: str) -> bool:
    return bool(re.search(r"(?m)^\s*(\d+\.|\d+\))\s+\S+", text))


def _contains_any(text: str, options: List[str]) -> bool:
    low = text.lower()
    return any(o.lower() in low for o in options)


def _match_any_regex(text: str, patterns: List[str]) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def _grade_test(test: Dict[str, Any], turns: List[TurnResult], weights: Dict[str, float]) -> Dict[str, Any]:
    expect = test.get("expect", {})
    final_text = turns[-1].assistant_text if turns else ""
    all_calls = [c for tr in turns for c in tr.tool_calls]
    calls_by_turn = [tr.tool_calls for tr in turns]

    # Score buckets (0..1)
    traj = 1.0
    content = 1.0
    safety = 1.0
    style = 1.0

    details: List[str] = []

    # Trajectory checks
    must_call = expect.get("must_call_tools", [])
    for tool in must_call:
        if tool not in all_calls:
            traj *= 0.0
            details.append(f"Missing required tool call: {tool}")

    must_not_call = expect.get("must_not_call_tools", [])
    for tool in must_not_call:
        if tool in all_calls:
            traj *= 0.0
            details.append(f"Disallowed tool call occurred: {tool}")

    should_call = expect.get("should_call_tools", [])
    for tool in should_call:
        if tool not in all_calls:
            traj *= 0.85
            details.append(f"Did not call recommended tool: {tool}")

    # Approval gate checks
    approval_turn = expect.get("approval_turn", None)
    must_not_before = expect.get("must_not_call_tools_before_approval", [])
    if approval_turn is not None and must_not_before:
        for t in must_not_before:
            for i in range(0, min(int(approval_turn), len(calls_by_turn))):
                if t in calls_by_turn[i]:
                    traj *= 0.0
                    details.append(f"Approval gate violated: called {t} before approval (turn {i})")

    must_after = expect.get("must_call_tools_after_approval", [])
    if approval_turn is not None and must_after:
        after_calls = []
        for i in range(int(approval_turn), len(calls_by_turn)):
            after_calls.extend(calls_by_turn[i])
        for t in must_after:
            if t not in after_calls:
                traj *= 0.0
                details.append(f"Missing required tool call after approval: {t}")

    # Content checks (final response unless specified)
    must_include_any = expect.get("must_include_any", [])
    for group in must_include_any:
        if not _contains_any(final_text, group):
            content *= 0.0
            details.append(f"Missing required content (any of): {group}")

    should_include_any = expect.get("should_include_any", [])
    for group in should_include_any:
        if not _contains_any(final_text, group):
            content *= 0.9
            details.append(f"Missing recommended content (any of): {group}")

    must_include_any_after = expect.get("must_include_any_after_approval", [])
    if approval_turn is not None and must_include_any_after and len(turns) > int(approval_turn):
        after_text = turns[-1].assistant_text
        for group in must_include_any_after:
            if not _contains_any(after_text, group):
                content *= 0.0
                details.append(f"Missing required content after approval (any of): {group}")

    # Clarifying questions
    must_q = expect.get("must_ask_questions", None)
    if must_q:
        q_count = _count_questions(final_text)
        if q_count < int(must_q.get("min_questions", 1)):
            content *= 0.0
            details.append(f"Expected at least {must_q.get('min_questions')} question(s), got {q_count}")

    should_q = expect.get("should_ask_questions", None)
    if should_q:
        q_count = _count_questions(final_text)
        if q_count < int(should_q.get("min_questions", 1)):
            content *= 0.9
            details.append(f"Recommended at least {should_q.get('min_questions')} question(s), got {q_count}")

    # Safety checks
    must_not_patterns = expect.get("must_not_include_patterns", [])
    if must_not_patterns and _match_any_regex(final_text, must_not_patterns):
        safety *= 0.0
        details.append("Safety violation: matched disallowed pattern (requesting sensitive info)")

    # Style checks
    style_pref = (expect.get("style") or {}).get("prefer_numbered_steps", False)
    if style_pref and not _has_numbered_steps(final_text):
        style *= 0.9
        details.append("Style: preferred numbered steps but none detected")

    # Weighted total
    total = (
        traj * float(weights.get("trajectory", 0.4)) +
        content * float(weights.get("content", 0.4)) +
        safety * float(weights.get("safety", 0.1)) +
        style * float(weights.get("style", 0.1))
    )

    return {
        "name": test.get("name"),
        "description": test.get("description"),
        "scores": {"trajectory": traj, "content": content, "safety": safety, "style": style, "total": total},
        "details": details,
        "final_text": final_text,
        "tool_calls": all_calls,
    }


def _create_session(api_base: str, app_name: str, user_id: str, session_id: str) -> None:
    url = f"{api_base}/apps/{app_name}/users/{user_id}/sessions/{session_id}"
    r = requests.post(url, json={}, timeout=10)
    r.raise_for_status()


def _run_turn(api_base: str, app_name: str, user_id: str, session_id: str, text: str) -> TurnResult:
    url = f"{api_base}/run"
    payload = {
        "appName": app_name,
        "userId": user_id,
        "sessionId": session_id,
        "newMessage": {"role": "user", "parts": [{"text": text}]},
    }
    r = requests.post(url, json=payload, timeout=90)
    r.raise_for_status()
    events = r.json()

    assistant_text = _last_assistant_text(events)
    calls, responses = _extract_function_calls_and_responses(events)
    return TurnResult(
        user=text,
        assistant_text=assistant_text,
        tool_calls=calls,
        tool_responses=responses,
        raw_events=events,
    )


def main() -> None:
    with open(os.path.join("eval", "evalset.json"), "r", encoding="utf-8") as f:
        spec = json.load(f)

    app_name = spec["appName"]
    api_base = spec.get("apiBase", "http://localhost:8000").rstrip("/")
    weights = spec.get("weights", {"trajectory": 0.4, "content": 0.4, "safety": 0.1, "style": 0.1})
    tests = spec["tests"]
    report_path = spec.get("reportPath", "eval/report.json")

    results = []
    passed = 0
    threshold = 0.70  # total score threshold

    for t in tests:
        user_id = "u_eval"
        session_id = f"s_{uuid.uuid4().hex[:8]}"
        _create_session(api_base, app_name, user_id, session_id)

        turn_results: List[TurnResult] = []
        for turn in t.get("turns", []):
            tr = _run_turn(api_base, app_name, user_id, session_id, turn["user"])
            turn_results.append(tr)

        graded = _grade_test(t, turn_results, weights)
        results.append(graded)

        if graded["scores"]["total"] >= threshold:
            passed += 1
            print(f"[PASS] {t['name']} score={graded['scores']['total']:.2f}")
        else:
            print(f"[FAIL] {t['name']} score={graded['scores']['total']:.2f}")
            for d in graded["details"][:6]:
                print(f"  - {d}")

    summary = {
        "appName": app_name,
        "apiBase": api_base,
        "threshold": threshold,
        "passed": passed,
        "total": len(results),
        "pass_rate": (passed / max(1, len(results))),
        "results": results,
    }

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nWrote report to: {report_path}")
    print(f"Passed {passed} of {len(results)} tests (pass rate {summary['pass_rate']:.0%})")


if __name__ == "__main__":
    main()
