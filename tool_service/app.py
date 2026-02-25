import os
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .storage import Storage


KB_PATH = os.path.join(os.path.dirname(__file__), "data", "kb_articles.json")
DB_PATH = os.path.join(os.path.dirname(__file__), "data", "tickets.sqlite3")

storage = Storage(db_path=DB_PATH, kb_path=KB_PATH)
app = FastAPI(title="Agent Tool Service", version="0.1.0")


class KBSearchResponse(BaseModel):
    results: List[Dict[str, Any]]


class TicketCreateRequest(BaseModel):
    title: str = Field(..., min_length=3)
    description: str = Field(..., min_length=10)
    priority: str = Field("P2", pattern=r"^P[1-4]$")


class TicketUpdateRequest(BaseModel):
    status: str = Field(..., pattern=r"^(open|in_progress|blocked|resolved|closed)$")


class TicketResponse(BaseModel):
    id: str
    title: str
    description: str
    priority: str
    status: str
    created_at: float


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/kb/search", response_model=KBSearchResponse)
def kb_search(q: str, top_k: int = 3) -> KBSearchResponse:
    if not q.strip():
        return KBSearchResponse(results=[])
    results = storage.kb_search(query=q, top_k=top_k)
    return KBSearchResponse(results=results)


@app.post("/tickets", response_model=TicketResponse)
def create_ticket(req: TicketCreateRequest) -> TicketResponse:
    t = storage.create_ticket(title=req.title, description=req.description, priority=req.priority)
    return TicketResponse(**t.__dict__)


@app.get("/tickets", response_model=List[TicketResponse])
def list_tickets(limit: int = 20) -> List[TicketResponse]:
    return [TicketResponse(**t.__dict__) for t in storage.list_tickets(limit=limit)]


@app.get("/tickets/{ticket_id}", response_model=TicketResponse)
def get_ticket(ticket_id: str) -> TicketResponse:
    t = storage.get_ticket(ticket_id=ticket_id)
    if not t:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return TicketResponse(**t.__dict__)


@app.patch("/tickets/{ticket_id}", response_model=TicketResponse)
def update_ticket(ticket_id: str, req: TicketUpdateRequest) -> TicketResponse:
    t = storage.update_ticket_status(ticket_id=ticket_id, status=req.status)
    if not t:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return TicketResponse(**t.__dict__)


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7001, log_level="info")


if __name__ == "__main__":
    main()
