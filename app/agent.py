import re
import os
from typing import TypedDict, List, Optional

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
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

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.0,
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
    timeout=300.0
)

class State(TypedDict):
    messages: List[str]       
    intent: str               
    name: Optional[str]       
    email: Optional[str]       
    platform: Optional[str]   
    response: str             


_GREETING_KEYWORDS = {
    "hi", "hello", "hey", "greetings", "howdy", "sup",
    "good morning", "good afternoon", "good evening",
}

def _is_greeting(message: str) -> bool:
    msg = message.lower().strip().rstrip("!.,")
    return msg in _GREETING_KEYWORDS or any(kw in msg for kw in _GREETING_KEYWORDS)


def _invoke_llm(prompt: str) -> str:
    result = llm.invoke(prompt)
    if hasattr(result, "content"):
        return result.content.strip()
    return str(result).strip()


def classification_node(state: State) -> State:
    last_message = state["messages"][-1]

    if _is_greeting(last_message):
        return {**state, "intent": "greeting"}

    prompt = f"{CLASSIFICATION_PROMPT}\n\nCurrent message: {last_message}"
    try:
        raw = _invoke_llm(prompt).lower()
        for intent in ("farewell", "provide_info", "high-intent", "pricing", "greeting"):
            if intent in raw:
                return {**state, "intent": intent}
    except Exception as exc:
        print(f"LLM error: {exc}")
    missing = not (state.get("name") and state.get("email") and state.get("platform"))
    if state.get("intent") in ("high-intent", "provide_info") and missing:
        return {**state, "intent": "provide_info"}

    return {**state, "intent": "unknown"}

def greeting_node(state: State) -> State:
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
    all_text = " ".join(state["messages"])
    prompt = EXTRACTION_PROMPT.format(text=all_text)

    new_state = {**state}

    try:
        raw = _invoke_llm(prompt)

        name_match     = re.search(r"^name:\s*(.+)$",     raw, re.IGNORECASE | re.MULTILINE)
        email_match    = re.search(r"^email:\s*(.+)$",    raw, re.IGNORECASE | re.MULTILINE)
        platform_match = re.search(r"^platform:\s*(.+)$", raw, re.IGNORECASE | re.MULTILINE)

        _INVALID = {"none", "null", "empty", "value", "blank", "pro", "basic", "plan", "autostream", "n/a", "unknown", "not provided"}

        def _clean(match) -> Optional[str]:
            if not match:
                return None
            val = match.group(1).strip()
            clean_val = val.lower().strip(" .<>[]\"'")
            
            if not clean_val or clean_val in _INVALID:
                return None
            return val

        extracted_name     = _clean(name_match)
        extracted_email    = _clean(email_match)
        extracted_platform = _clean(platform_match)

        if extracted_name and not new_state.get("name"):
            new_state["name"] = extracted_name
        if extracted_email and not new_state.get("email"):
            new_state["email"] = extracted_email
        if extracted_platform and not new_state.get("platform"):
            new_state["platform"] = extracted_platform

    except Exception as exc:
        print(f"LLM error: {exc}")
        new_state["response"] = "Sorry, I had trouble processing that. Could you try again?"
        return new_state

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
        print(f"Error: {exc}")
        response = "Something went wrong saving your details. Could you try again?"

    return {**state, "response": response}

def farewell_node(state: State) -> State:
    return {
        **state,
        "response": "You're welcome! Feel free to reach out anytime. Have a great day! 👋",
    }

def unknown_node(state: State) -> State:
    return {
        **state,
        "response": (
            "I'm not sure I caught that. I can help you with AutoStream's pricing, "
            "features, or get you signed up. What would you like to know?"
        ),
    }

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
        "provide_info": "extraction", 
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