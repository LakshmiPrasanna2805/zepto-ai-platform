"""LangGraph flow:  classify_intent --(conditional edge)--> retrieve_and_answer | direct_answer

    START -> classify_intent --policy_question--> retrieve_and_answer -> END
                             \--general_question--> direct_answer -------> END

Every node's *generation* step branches on MOCK_LLM:
  * mock (default, graded): deterministic rules, no LLM call, no network call.
  * MOCK_LLM=0 (optional): the node prompts a real LLM instead.
Retrieval (embedding + ChromaDB) always runs for real in both modes.
"""
from typing import Literal, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

import vector_store
from config import SNIPPET_CHARS, mock_llm_enabled
from prompts import (ANSWER_PROMPT_TEMPLATE, CLASSIFY_PROMPT_TEMPLATE, DIRECT_PROMPT_TEMPLATE,
                     format_context)
from schemas import AskResponse

POLICY_KEYWORDS = ["delivery", "return", "refund", "membership", "tracking", "cancel",
                   "gift card", "support hours"]
GENERAL_ANSWER = "I can only answer questions about Zepto policies right now."


class AssistantState(TypedDict, total=False):
    query: str
    intent: Literal["policy_question", "general_question"]
    retrieved: list[dict]                 # chunks returned by ChromaDB
    response: Optional[AskResponse]       # the final, validated answer


# ------------------------------------------------------------------ helpers
def keyword_intent(query: str) -> str:
    q = query.lower()
    return "policy_question" if any(k in q for k in POLICY_KEYWORDS) else "general_question"


def make_snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    """First ~200 characters, cut at a word boundary."""
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "..."


# ------------------------------------------------------------------ node 1
def classify_intent(state: AssistantState) -> AssistantState:
    if mock_llm_enabled():
        intent = keyword_intent(state["query"])            # keyword heuristic, no LLM call
    else:
        import llm
        reply = llm.chat([{"role": "user",
                           "content": CLASSIFY_PROMPT_TEMPLATE.format(question=state["query"])}])
        intent = "general_question" if "general" in reply.lower() else "policy_question"
    return {"intent": intent}


# ------------------------------------------------------------------ node 2
def retrieve_and_answer(state: AssistantState) -> AssistantState:
    hits = vector_store.retrieve(state["query"])            # ALWAYS real: embed + ChromaDB top-3
    if mock_llm_enabled():
        top = hits[0]
        response = AskResponse(
            answer=f"Based on the retrieved context: {make_snippet(top['text'])}",
            sources=[h["chunk_id"] for h in hits],
            # deterministic: cosine similarity of the best chunk, clipped to 0-1
            confidence=round(min(max(top["similarity"], 0.0), 1.0), 3),
        )
    else:
        import llm
        prompt = ANSWER_PROMPT_TEMPLATE.format(context=format_context(hits), question=state["query"])
        response = llm.generate_structured(prompt)         # includes the retry-on-invalid-JSON logic
    return {"retrieved": hits, "response": response}


# ------------------------------------------------------------------ node 3
def direct_answer(state: AssistantState) -> AssistantState:
    if mock_llm_enabled():
        response = AskResponse(answer=GENERAL_ANSWER, sources=[], confidence=1.0)
    else:
        import llm
        response = llm.generate_structured(DIRECT_PROMPT_TEMPLATE.format(question=state["query"]))
        response.sources = []                               # no retrieval -> no sources
    return {"retrieved": [], "response": response}


# ------------------------------------------------------------------ routing (does NOT depend on MOCK_LLM)
def route_by_intent(state: AssistantState) -> str:
    return state["intent"]


def build_graph():
    graph = StateGraph(AssistantState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)

    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges("classify_intent", route_by_intent,
                                {"policy_question": "retrieve_and_answer",
                                 "general_question": "direct_answer"})
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)
    return graph.compile()


_app = None


def answer_query(query: str) -> dict:
    """Run the graph once. Returns the final state (intent, retrieved chunks, response)."""
    global _app
    if _app is None:
        _app = build_graph()
    return _app.invoke({"query": query})


if __name__ == "__main__":
    vector_store.ensure_index()
    for q in ["What is the delivery fee for small orders?", "Tell me a joke"]:
        state = answer_query(q)
        print(f"\nQ: {q}\n   intent  : {state['intent']}\n   response: {state['response'].model_dump_json()}")
