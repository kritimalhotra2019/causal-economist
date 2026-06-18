"""
Builds the LangGraph state machine.

    economist_design -> analyst_execute -> critic_evaluate -> (route)
        route: clean/agreed        -> finalize
               problem or conflict -> human_gate -> (route)
                   approve remedy  -> economist_revise -> analyst_execute (loop)
                   else            -> finalize
        bounded by max_iterations, then finalize (escalate).
"""
from __future__ import annotations

from functools import partial

from langgraph.graph import StateGraph, START, END

from . import agents
from .llm import make_llm
from .state import AnalysisState


def _route_after_critic(state) -> str:
    c = state.get("critic", {})
    if c.get("verdict") == "fully_satisfactory":
        return "finalize"
    if state.get("iteration", 0) >= state.get("max_iterations", 2):
        return "finalize"  # escalate: stop looping
    if c.get("remedy") is not None or c.get("conflict") is not None:
        return "human_gate"
    return "finalize"


def _route_after_human(state) -> str:
    c = state.get("critic", {})
    d = state.get("human_decision") or {}
    if c.get("remedy") is not None and d.get("approve"):
        return "economist_revise"
    return "finalize"


def build_graph(task, llm, checkpointer=None):
    # The economist uses the provided llm; the critic gets its own model
    # (DEFAULT_CRITIC_MODEL) in nebius mode. Stub shares one deterministic llm.
    econ_llm = llm
    critic_llm = llm if llm.provider == "stub" else make_llm(llm.provider, role="critic")

    g = StateGraph(AnalysisState)

    g.add_node("economist_design", partial(agents.economist_design, task=task, llm=econ_llm))
    g.add_node("analyst_execute", partial(agents.analyst_execute, task=task, llm=econ_llm))
    g.add_node("critic_evaluate", partial(agents.critic_evaluate, task=task, llm=critic_llm))
    g.add_node("economist_revise", partial(agents.economist_revise, task=task, llm=econ_llm))
    g.add_node("human_gate", partial(agents.human_gate, task=task, llm=econ_llm))
    g.add_node("finalize", partial(agents.finalize, task=task))

    g.add_edge(START, "economist_design")
    g.add_edge("economist_design", "analyst_execute")
    g.add_edge("analyst_execute", "critic_evaluate")
    g.add_conditional_edges("critic_evaluate", _route_after_critic,
                            {"human_gate": "human_gate", "finalize": "finalize"})
    g.add_conditional_edges("human_gate", _route_after_human,
                            {"economist_revise": "economist_revise", "finalize": "finalize"})
    g.add_edge("economist_revise", "analyst_execute")
    g.add_edge("finalize", END)

    return g.compile(checkpointer=checkpointer)
