"""LangGraph: 5 agent + fan-out song song theo sub-query."""
from __future__ import annotations

from langgraph.graph import END, StateGraph
from langgraph.types import Send

from rag.agents.grader import after_grade, grader
from rag.agents.history import load_history
from rag.agents.persist import persist
from rag.agents.retriever import merge, retrieve
from rag.agents.supervisor import supervisor
from rag.agents.synthesizer import synthesizer
from rag.agents.verifier import after_verify, verifier
from rag.state import RAGState

_GRAPH = None


def fan_out(state: RAGState):
    queries = state.get("sub_queries") or [state.get("question", "")]
    return [Send("retrieve", {**state, "sub_query": q}) for q in queries]


def build_graph():
    g = StateGraph(RAGState)
    g.add_node("history", load_history)
    g.add_node("supervisor", supervisor)
    g.add_node("retrieve", retrieve)
    g.add_node("merge", merge)
    g.add_node("grader", grader)
    g.add_node("synthesizer", synthesizer)
    g.add_node("verifier", verifier)
    g.add_node("persist", persist)

    g.set_entry_point("history")
    g.add_edge("history", "supervisor")
    g.add_conditional_edges("supervisor", fan_out, ["retrieve"])
    g.add_edge("retrieve", "merge")
    g.add_edge("merge", "grader")
    g.add_conditional_edges(
        "grader", after_grade,
        {"supervisor": "supervisor", "synthesizer": "synthesizer"},
    )
    g.add_edge("synthesizer", "verifier")
    g.add_conditional_edges(
        "verifier", after_verify,
        {"synthesizer": "synthesizer", "persist": "persist"},
    )
    g.add_edge("persist", END)
    return g.compile()


def graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH
