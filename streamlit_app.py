"""
streamlit_app.py — live UI for the causal multi-agent system.

Economist (DeepSeek-V4-Pro) proposes the adjustment-set design ->
Analyst runs the doubly-robust estimator + diagnostics (no LLM) ->
Critic (Nemotron-550B) writes the rationale over a deterministic rubric ->
human gate -> report + credibility ledger.

Run locally:   streamlit run streamlit_app.py
Deploy:        push to GitHub, deploy on share.streamlit.io, and set
               NEBIUS_API_KEY in the app's Secrets (leave NEBIUS_MODEL unset
               so the economist/critic per-role defaults apply).
"""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Local dev reads .env; on Streamlit Cloud the same names live in st.secrets —
# bridge them into os.environ because the package reads os.environ directly.
load_dotenv()
try:
    for _k, _v in dict(st.secrets).items():
        if _v and not os.environ.get(_k):
            os.environ[_k] = str(_v)
except Exception:
    pass

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from causal_mas.datasets import get_task
from causal_mas.graph import build_graph
from causal_mas.llm import make_llm, DEFAULT_ECONOMIST_MODEL, DEFAULT_CRITIC_MODEL

TASKS = {
    "lalonde": "LaLonde / NSW — job training → 1978 earnings (confounded observational)",
    "thornton": "Thornton — HIV-test incentive → result collection (Malawi)",
    "cai": "Cai — insurance information session → take-up (China)",
}

st.set_page_config(page_title="Causal MAS", page_icon="🧪", layout="wide")


def fmt(v, units: str) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    return f"${v:,.0f}" if units == "usd" else f"{v:+.4f}"


def run_pipeline(task_id: str, provider: str):
    """Run the full graph to completion (gates auto-resolved by policy)."""
    task = get_task(task_id)
    llm = make_llm(provider)
    app = build_graph(task, llm, checkpointer=MemorySaver())
    init = {"task_id": task.id, "provider": provider, "plan": task.description,
            "available_covariates": task.covariates, "truth": task.truth,
            "units": task.units, "max_iterations": 2, "auto_resolve": True,
            "ledger": []}
    cfg = {"configurable": {"thread_id": f"{task_id}-streamlit"}}
    state = app.invoke(init, cfg)
    # auto_resolve=True means no interrupt fires, but stay safe if that changes:
    while "__interrupt__" in state:
        state = app.invoke(
            Command(resume={"approve": True, "tie_break": "economist",
                            "source": "auto-policy"}), cfg)
    return task, state


# ----------------------------------------------------------------- UI
st.title("🧪 Causal Multi-Agent System")
st.markdown(
    "A multi-agent, human-in-the-loop pipeline for **observational causal "
    "inference**. The **economist** proposes the adjustment set, the **analyst** "
    "runs a doubly-robust estimator with overlap/balance/placebo diagnostics "
    "(no LLM — *facts are settled by numbers*), and the **critic** judges it. "
    "Every estimate is graded against an experimental ground truth.")

with st.sidebar:
    st.header("Run a study")
    task_id = st.selectbox("Task", list(TASKS), format_func=lambda t: TASKS[t])
    provider = st.radio(
        "Provider", ["nebius", "stub"], index=0,
        help="nebius = real LLM agents (billed). stub = offline, deterministic.")
    st.caption(
        f"**economist** → `{DEFAULT_ECONOMIST_MODEL}`  \n"
        f"**critic** → `{DEFAULT_CRITIC_MODEL}`" if provider == "nebius"
        else "stub: deterministic oracle, no API calls.")
    if provider == "nebius" and not os.environ.get("NEBIUS_API_KEY"):
        st.warning("No NEBIUS_API_KEY — set it in Secrets or switch to stub.")
    go = st.button("Run analysis", type="primary", use_container_width=True)

if not go:
    st.info("Pick a task and provider in the sidebar, then **Run analysis**.")
    st.stop()

with st.spinner(f"Running the multi-agent pipeline on **{task_id}** via **{provider}**…"):
    try:
        task, state = run_pipeline(task_id, provider)
    except Exception as e:
        st.error(f"Run failed: {type(e).__name__}: {e}")
        st.stop()

units = state.get("units", "")
r = state.get("results", {})
est = r.get("estimate")
truth = state.get("truth")
ci = r.get("ci")
err = abs(est - truth) if (est is not None and truth is not None and est == est) else None
covers = (ci and ci[0] == ci[0] and ci[0] <= truth <= ci[1]) if (truth is not None and ci) else None

st.subheader(f"Result — {task.name}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Estimate", fmt(est, units), help=f"estimand: {r.get('estimand')}")
c2.metric("Experimental truth", fmt(truth, units))
c3.metric("Absolute error", fmt(err, units))
c4.metric("95% CI covers truth", "✅ yes" if covers else ("❌ no" if covers is not None else "n/a"))
if ci:
    st.caption(f"95% CI: [{fmt(ci[0], units)}, {fmt(ci[1], units)}]  ·  estimand: {r.get('estimand')}")

crit = state.get("critic", {})
verdict = crit.get("verdict", "—")
icon = {"fully_satisfactory": "✅", "satisfactory_with_caveats": "⚠️",
        "not_satisfactory": "❌"}.get(verdict, "•")
st.markdown(f"**Critic verdict:** {icon} `{verdict}`")
if crit.get("rationale"):
    st.markdown(f"> {crit['rationale']}")
if crit.get("conflict"):
    d = state.get("human_decision") or {}
    st.markdown(f"**Disagreement at the human gate** → resolved by siding with the "
                f"**{d.get('tie_break', 'economist')}**.")
    st.caption(crit["conflict"])

design = state.get("design", {})
left, right = st.columns(2)
with left:
    st.markdown("**Adjustment set (included confounders)**")
    st.write(design.get("confounders", []))
with right:
    st.markdown("**Excluded** (with reason role)")
    exc = design.get("excluded", [])
    st.write(pd.DataFrame(exc)[["name", "role"]] if exc else "none")

diag = state.get("diagnostics", {})
if isinstance(diag, dict) and "overlap" in diag:
    d1, d2, d3 = st.columns(3)
    for col, key in [(d1, "overlap"), (d2, "balance"), (d3, "placebo")]:
        passed = diag.get(key, {}).get("passed")
        col.metric(key.capitalize(), "✅ pass" if passed else "❌ fail")

with st.expander("Full report"):
    st.code(state.get("report", "(no report)"))

st.subheader("Credibility ledger")
st.caption("The append-only audit trail — every step, and which model produced each LLM step.")
ledger = state.get("ledger", [])
if ledger:
    df = pd.DataFrame(ledger)
    front = [c for c in ("event", "model") if c in df.columns]
    df = df[front + [c for c in df.columns if c not in front]]
    st.dataframe(df, use_container_width=True, hide_index=True)
