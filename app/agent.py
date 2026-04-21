"""
AutoStream Conversational AI Agent
====================================
LangGraph-powered agent that:
  1. Classifies user intent (greeting / pricing / high-intent / provide_info / farewell)
  2. Answers product/pricing questions via RAG
  3. Collects lead info (name, email, platform) across multiple turns
  4. Fires mock_lead_capture() only once ALL three fields are collected
"""

import re
import os
from typing import TypedDict, List, Optional

from langgraph.graph import StateGraph, END
from langchain_community.llms import Ollama
from dotenv import load_dotenv

from rag import query_knowledge_base
from tools import mock_lead_capture
from prompts import (
    CLASSIFICATION_PROMPT,
    RETRIEVAL_PROMPT,
    EXTRACTION_PROMPT,
    GREETING_PROMPT,
)

load_dotenv()

# ---------------------------------------------------------------------------
# LLM initialisation (Ollama — runs fully locally, no API key needed)
# ---------------------------------------------------------------------------

llm = Ollama(
    model="mistral:latest",
    base_url="http://localhost:11434",
    temperature=0.0,   # deterministic for intent classification
    timeout=300.0,
)

# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class State(TypedDict):
    messages: List[str]        # full conversation history
    intent: str                # current turn's classified intent
    name: Optional[str]        # lead: full name
    email: Optional[str]       # lead: email address
    platform: Optional[str]    # lead: creator platform
    response: str              # agent's reply for this turn


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

_GREETING_KEYWORDS = {
    "hi", "hello", "hey", "greetings", "howdy", "sup",
    "good morning", "good afternoon", "good evening",
}

def _is_greeting(message: str) -> bool:
    msg = message.lower().strip().rstrip("!.,")
    return msg in _GREETING_KEYWORDS or any(kw in msg for kw in _GREETING_KEYWORDS)


def _invoke_llm(prompt: str) -> str:
    """
    Invoke the Ollama LLM and return stripped plain-text output.
    Ollama returns a raw string (unlike ChatGoogleGenerativeAI which returns
    an AIMessage with a .content attribute), so we call .strip() directly.
    The hasattr guard handles any unexpected wrapper object gracefully.
    """
    result = llm.invoke(prompt)
    if hasattr(result, "content"):
        return result.content.strip()
    return str(result).strip()


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def classification_node(state: State) -> State:
    """Classify the intent of the latest user message."""
    last_message = state["messages"][-1]

    # Fast-path for unambiguous greetings (avoids an LLM call)
    if _is_greeting(last_message):
        return {**state, "intent": "greeting"}

    prompt = f"{CLASSIFICATION_PROMPT}\n\nCurrent message: {last_message}"
    try:
        raw = _invoke_llm(prompt).lower()
        # Match in priority order — farewell first to prevent history
        # bleed (e.g. "ok thanks" after mentioning Pro plan).
        for intent in ("farewell", "provide_info", "high-intent", "pricing", "greeting"):
            if intent in raw:
                return {**state, "intent": intent}
    except Exception as exc:
        print(f"[classification_node] LLM error: {exc}")

    # Fallback: if we were mid-collection and the LLM returned garbage,
    # keep the collection loop alive instead of dropping to END.
    missing = not (state.get("name") and state.get("email") and state.get("platform"))
    if state.get("intent") in ("high-intent", "provide_info") and missing:
        return {**state, "intent": "provide_info"}

    return {**state, "intent": "unknown"}


def greeting_node(state: State) -> State:
    """Return a warm, LLM-generated greeting."""
    prompt = GREETING_PROMPT.format(message=state["messages"][-1])
    try:
        response = _invoke_llm(prompt)
    except Exception:
        response = (
            "Hey there! Welcome to AutoStream 🎬 "
            "I can tell you about our plans, features, or help you get started. "
            "What can I do for you today?"
        )
    return {**state, "response": response}


def retrieval_node(state: State) -> State:
    """Answer product/pricing questions using RAG over the knowledge base."""
    query = state["messages"][-1]
    try:
        context = query_knowledge_base(query)
    except Exception:
        context = ""

    if not context:
        context = "No relevant information found in the knowledge base."

    prompt = RETRIEVAL_PROMPT.format(context=context, query=query)
    try:
        response = _invoke_llm(prompt)
    except Exception as exc:
        response = f"Sorry, I ran into an issue fetching that info. Please try again. ({exc})"

    return {**state, "response": response}


def extraction_node(state: State) -> State:
    """
    Extract name / email / platform from the full conversation.
    Persist already-collected values; only ask for what's still missing.
    """
    all_text = " ".join(state["messages"])
    prompt = EXTRACTION_PROMPT.format(text=all_text)

    new_state = {**state}

    try:
        raw = _invoke_llm(prompt)

        name_match     = re.search(r"^name:\s*(.+)$",     raw, re.IGNORECASE | re.MULTILINE)
        email_match    = re.search(r"^email:\s*(.+)$",    raw, re.IGNORECASE | re.MULTILINE)
        platform_match = re.search(r"^platform:\s*(.+)$", raw, re.IGNORECASE | re.MULTILINE)

        # Expanded invalid list
        _INVALID = {"none", "null", "empty", "value", "blank", "pro", "basic", "plan", "autostream", "n/a", "unknown", "not provided"}

        def _clean(match) -> Optional[str]:
            if not match:
                return None
            val = match.group(1).strip()
            # Strip brackets and quotes that the LLM might hallucinate
            clean_val = val.lower().strip(" .<>[]\"'")
            
            if not clean_val or clean_val in _INVALID:
                return None
            return val

        extracted_name     = _clean(name_match)
        extracted_email    = _clean(email_match)
        extracted_platform = _clean(platform_match)

        # Only update if we found something new and don't already have it
        if extracted_name and not new_state.get("name"):
            new_state["name"] = extracted_name
        if extracted_email and not new_state.get("email"):
            new_state["email"] = extracted_email
        if extracted_platform and not new_state.get("platform"):
            new_state["platform"] = extracted_platform

    except Exception as exc:
        print(f"[extraction_node] LLM error: {exc}")
        new_state["response"] = "Sorry, I had trouble processing that. Could you try again?"
        return new_state

    # Determine what's still missing
    missing = []
    if not new_state.get("name"):     missing.append("name")
    if not new_state.get("email"):    missing.append("email address")
    if not new_state.get("platform"): missing.append("creator platform (e.g. YouTube, Instagram)")

    if missing:
        if len(missing) == 3:
            new_state["response"] = (
                "The Pro plan is a great choice! 🎉 "
                "To get your workspace set up, I just need a few details. "
                f"Could you share your {missing[0]}?"
            )
        elif len(missing) == 1:
            new_state["response"] = f"Almost there! Just need your {missing[0]}."
        else:
            fields = " and ".join(missing)
            new_state["response"] = f"Got it! Could you also share your {fields}?"

    return new_state


def _extraction_router(state: State) -> str:
    """Proceed to execution only when ALL three fields are present and valid."""
    name = state.get("name")
    email = state.get("email")
    platform = state.get("platform")
    
    if (
        state.get("intent") in ("high-intent", "provide_info")
        and name and str(name).strip()
        and email and str(email).strip()
        and platform and str(platform).strip()
    ):
        return "execution"
    return END


def execution_node(state: State) -> State:
    """Fire the lead capture tool — only reached once ALL three fields are confirmed."""
    try:
        mock_lead_capture(
            name=state["name"],
            email=state["email"],
            platform=state["platform"],
        )
        response = (
            "You're all set! 🚀 I've captured your details and our team will reach out "
            "shortly to get your AutoStream workspace ready. "
            "Check your inbox — exciting things are coming!"
        )
    except Exception as exc:
        print(f"[execution_node] Error: {exc}")
        response = "Something went wrong saving your details. Could you try again?"

    return {**state, "response": response}


def farewell_node(state: State) -> State:
    """Friendly closing message."""
    return {
        **state,
        "response": "You're welcome! Feel free to reach out anytime. Have a great day! 👋",
    }


def unknown_node(state: State) -> State:
    """Graceful fallback for unrecognised intents."""
    return {
        **state,
        "response": (
            "I'm not sure I caught that. I can help you with AutoStream's pricing, "
            "features, or get you signed up. What would you like to know?"
        ),
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

graph = StateGraph(State)

graph.add_node("classification", classification_node)
graph.add_node("greeting",       greeting_node)
graph.add_node("retrieval",      retrieval_node)
graph.add_node("extraction",     extraction_node)
graph.add_node("execution",      execution_node)
graph.add_node("farewell",       farewell_node)
graph.add_node("unknown",        unknown_node)

graph.set_entry_point("classification")


def _classification_router(state: State) -> str:
    intent = state.get("intent", "unknown")
    return {
        "greeting":     "greeting",
        "pricing":      "retrieval",
        "high-intent":  "extraction",
        "provide_info": "extraction",  # user is supplying a missing field
        "farewell":     "farewell",
    }.get(intent, "unknown")




graph.add_conditional_edges("classification", _classification_router)
graph.add_conditional_edges("extraction",     _extraction_router)

graph.add_edge("greeting",  END)
graph.add_edge("retrieval", END)
graph.add_edge("execution", END)
graph.add_edge("farewell",  END)
graph.add_edge("unknown",   END)

workflow = graph.compile()