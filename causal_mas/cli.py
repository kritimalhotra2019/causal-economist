"""
Run one analysis end-to-end.

  python -m causal_mas.cli --task lalonde                    # interactive, offline stub
  python -m causal_mas.cli --task thornton --auto            # headless
  python -m causal_mas.cli --task lalonde --provider nebius  # real LLM economist + critic

Interactive runs pause at each human gate and ask you to decide; state is
checkpointed to SQLite, so an interrupted run is resumable.
"""
from __future__ import annotations

import argparse

from dotenv import load_dotenv
from langgraph.types import Command

from .datasets import get_task
from .graph import build_graph
from .llm import make_llm

# Load .env so NEBIUS_API_KEY / NEBIUS_MODEL / ANTHROPIC_* are available without
# the user having to export them. Real (no-key) stub runs are unaffected.
load_dotenv()


def _ask_gate(payload: dict) -> dict:
    print("\n" + "=" * 60)
    print("  HUMAN GATE — the graph is paused, waiting on you")
    print("=" * 60)
    print(f"  critic verdict : {payload.get('verdict')}")
    if payload.get("proposed_remedy"):
        print(f"  proposed remedy: {payload['proposed_remedy']}")
    if payload.get("consequence"):
        print(f"  consequence    : {payload['consequence']}")
    if payload.get("current_estimate") is not None:
        print(f"  current estimate: {payload['current_estimate']:.3f}")
    if payload.get("conflict"):
        print(f"  DISAGREEMENT   : {payload['conflict']}")

    decision = {"source": "human"}
    if payload.get("proposed_remedy"):
        ans = input("  approve the remedy? [Y/n] ").strip().lower()
        decision["approve"] = ans in ("", "y", "yes")
    else:
        decision["approve"] = True
    if payload.get("conflict"):
        ans = input("  side with [e]conomist or [c]ritic? [e/c] ").strip().lower()
        decision["tie_break"] = "critic" if ans.startswith("c") else "economist"
    return decision


def run(task_id: str, provider: str, auto: bool, db: str):
    task = get_task(task_id)
    llm = make_llm(provider)
    cfg = {"configurable": {"thread_id": f"{task_id}-cli"}}
    init = {"task_id": task.id, "provider": provider, "plan": task.description,
            "available_covariates": task.covariates, "truth": task.truth,
            "units": task.units, "max_iterations": 2, "auto_resolve": auto, "ledger": []}

    from langgraph.checkpoint.sqlite import SqliteSaver
    with SqliteSaver.from_conn_string(db) as cp:
        app = build_graph(task, llm, checkpointer=cp)
        state = app.invoke(init, cfg)
        while "__interrupt__" in state:
            decision = ({"approve": True, "tie_break": "economist", "source": "auto"}
                        if auto else _ask_gate(state["__interrupt__"][0].value))
            state = app.invoke(Command(resume=decision), cfg)

    print("\n" + "=" * 60)
    print(state.get("report", "(no report)"))
    print("=" * 60)
    print("credibility ledger:")
    for e in state.get("ledger", []):
        extra = {k: v for k, v in e.items() if k not in ("t", "event")}
        print(f"  · {e['event']:18s} {extra}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="lalonde", choices=["lalonde", "thornton", "cai"])
    ap.add_argument("--provider", default="stub", choices=["stub", "nebius"])
    ap.add_argument("--auto", action="store_true", help="resolve gates automatically (no prompts)")
    ap.add_argument("--db", default="causal_mas_runs.sqlite")
    args = ap.parse_args()
    run(args.task, args.provider, args.auto, args.db)


if __name__ == "__main__":
    main()
