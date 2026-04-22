"""
AutoStream FastAPI Server
==========================
Exposes a /webhook POST endpoint that the frontend (or WhatsApp relay) can call.
Maintains per-session conversation state in memory.

For production: replace `state_store` with Redis or a database.
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import workflow

app = FastAPI(
    title="AutoStream AI Agent",
    description="Social-to-Lead agentic workflow for AutoStream",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


state_store: dict = {}



class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"  


class ChatResponse(BaseModel):
    response: str
    session_id: str
    collected: dict  



def _fresh_state() -> dict:
    return {
        "messages":  [],
        "intent":    "",
        "name":      "",
        "email":     "",
        "platform":  "",
        "response":  "",
    }


@app.get("/health")
async def health():
    """Simple liveness probe."""
    return {"status": "ok", "service": "AutoStream AI Agent"}


@app.post("/webhook", response_model=ChatResponse)
async def webhook(payload: ChatRequest):
    """
    Main chat endpoint.

    Accepts a user message + session_id, runs the LangGraph workflow,
    and returns the agent's response along with collected lead fields.
    """
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    session_id = payload.session_id

    if session_id not in state_store:
        state_store[session_id] = _fresh_state()

    state = state_store[session_id]
    state["messages"].append(message)

    try:
        result = workflow.invoke(state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(exc)}")

    state_store[session_id] = result

    return ChatResponse(
        response=result.get("response", "Sorry, I couldn't generate a response."),
        session_id=session_id,
        collected={
            "name":     result.get("name", ""),
            "email":    result.get("email", ""),
            "platform": result.get("platform", ""),
        },
    )


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    if session_id in state_store:
        del state_store[session_id]
    return {"status": "cleared", "session_id": session_id}


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    if session_id not in state_store:
        raise HTTPException(status_code=404, detail="Session not found.")
    s = state_store[session_id]
    return {
        "session_id": session_id,
        "messages":   s.get("messages", []),
        "intent":     s.get("intent", ""),
        "collected":  {
            "name":     s.get("name", ""),
            "email":    s.get("email", ""),
            "platform": s.get("platform", ""),
        },
    }