from langgraph.graph import END, START, StateGraph

from .agents import judge, researcher, skeptic, writer
from .config import KEEP_THRESHOLD, MAX_ROUNDS, MIN_SUPPORTED
from .schemas import CrewState


def next_step(state: CrewState) -> str:
    """After the Judge: do we have enough good claims, or research again?"""
    kept = sum(v.final_confidence >= KEEP_THRESHOLD for v in state["verdicts"])
    if kept >= MIN_SUPPORTED or state["round"] >= MAX_ROUNDS:
        return "writer"
    return "researcher"


def initial_state(question: str) -> CrewState:
    return {"question": question, "round": 0, "sources": [], "claims": [],
            "objections": [], "verdicts": [], "answer": ""}


def build_graph():
    graph = StateGraph(CrewState)

    graph.add_node("researcher", researcher)
    graph.add_node("skeptic", skeptic)
    graph.add_node("judge", judge)
    graph.add_node("writer", writer)

    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "skeptic")
    graph.add_edge("skeptic", "judge")
    graph.add_conditional_edges("judge", next_step, ["researcher", "writer"])
    graph.add_edge("writer", END)

    return graph.compile()
