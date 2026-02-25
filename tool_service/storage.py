import json
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Ticket:
    id: str
    title: str
    description: str
    priority: str
    status: str
    created_at: float


class Storage:
    def __init__(self, db_path: str, kb_path: str) -> None:
        self.db_path = db_path
        self.kb_path = kb_path
        self._init_db()
        self._kb = self._load_kb()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tickets (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.commit()

    def _load_kb(self) -> List[Dict[str, Any]]:
        with open(self.kb_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod # TODO: Use an embedding model's tokenizer instead of manual tokenization
    def _tokenize(text: str) -> List[str]:
        out: List[str] = []
        cur: List[str] = []
        for ch in text.lower():
            if ch.isalnum():
                cur.append(ch)
            else:
                if cur:
                    out.append("".join(cur))
                    cur = []
        if cur:
            out.append("".join(cur))
        return out

    # TODO: Migrate to a vector DB once available
    def kb_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        q_tokens = set(self._tokenize(query))
        scored: List[Tuple[int, Dict[str, Any]]] = []
        for art in self._kb:
            hay = " ".join([art.get("title", ""), " ".join(art.get("tags", [])), art.get("body", "")])
            a_tokens = set(self._tokenize(hay))
            score = len(q_tokens.intersection(a_tokens))
            if score > 0:
                scored.append((score, art))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [a for _, a in scored[: max(1, top_k)]]

    def create_ticket(self, title: str, description: str, priority: str = "P2") -> Ticket:
        ticket_id = f"t_{int(time.time() * 1000)}"
        created_at = time.time()
        status = "open"
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO tickets (id, title, description, priority, status, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (ticket_id, title, description, priority, status, created_at),
            )
            conn.commit()
        return Ticket(id=ticket_id, title=title, description=description, priority=priority, status=status, created_at=created_at)

    def get_ticket(self, ticket_id: str) -> Optional[Ticket]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        if not row:
            return None
        return Ticket(
            id=row["id"],
            title=row["title"],
            description=row["description"],
            priority=row["priority"],
            status=row["status"],
            created_at=row["created_at"],
        )

    def update_ticket_status(self, ticket_id: str, status: str) -> Optional[Ticket]:
        with self._connect() as conn:
            conn.execute("UPDATE tickets SET status = ? WHERE id = ?", (status, ticket_id))
            conn.commit()
        return self.get_ticket(ticket_id)

    def list_tickets(self, limit: int = 20) -> List[Ticket]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM tickets ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        out: List[Ticket] = []
        for r in rows:
            out.append(
                Ticket(
                    id=r["id"],
                    title=r["title"],
                    description=r["description"],
                    priority=r["priority"],
                    status=r["status"],
                    created_at=r["created_at"],
                )
            )
        return out
