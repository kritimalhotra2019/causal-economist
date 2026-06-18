"""
streamlit_app.py — live UI for the causal multi-agent system.

Two modes:
  • Benchmark study — the 3 RCT benchmarks, graded against experimental truth.
  • Your data — upload a CSV, ask a causal question in plain English; the
    economist maps it to a treatment/outcome/adjustment-set design (you confirm),
    then the full pipeline runs. No ground truth exists for your data, so the
    diagnostics + critic are the credibility signal, not a known answer.

Economist (DeepSeek-V4-Pro) proposes the design -> Analyst runs the doubly-robust
estimator + overlap/balance/placebo diagnostics (no LLM) -> Critic (Nemotron-550B)
judges over a deterministic rubric -> human gate -> report + credibility ledger.

Run locally:   streamlit run streamlit_app.py
Deploy:        push to GitHub, deploy on share.streamlit.io, set NEBIUS_API_KEY
               and APP_PASSWORD in the app's Secrets (leave NEBIUS_MODEL unset).
"""
from __future__ import annotations

import hmac
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

from causal_mas.datasets import get_task, build_task_from_df
from causal_mas.graph import build_graph
from causal_mas.llm import make_llm, DEFAULT_ECONOMIST_MODEL, DEFAULT_CRITIC_MODEL
from causal_mas.planner import map_question

TASKS = {
    "lalonde": "LaLonde / NSW — job training → 1978 earnings (confounded observational)",
    "thornton": "Thornton — HIV-test incentive → result collection (Malawi)",
    "cai": "Cai — insurance information session → take-up (China)",
}

st.set_page_config(page_title="Causal MAS", page_icon="🧪", layout="wide")


# ----------------------------------------------------------------- helpers
def check_password() -> None:
    """Gate behind APP_PASSWORD. Fail-closed: locked if APP_PASSWORD is unset,
    so a public deploy can never run billed calls unconfigured."""
    expected = os.environ.get("APP_PASSWORD")
    if not expected:
        st.title("🔒 Causal MAS")
        st.error("This app isn't configured. Set **APP_PASSWORD** in the app's "
                 "Secrets (or your local `.env`) to enable access.")
        st.stop()
    if st.session_state.get("auth_ok"):
        return
    st.title("🔒 Causal MAS")
    with st.form("login"):
        pw = st.text_input("Password", type="password")
        if st.form_submit_button("Enter"):
            if hmac.compare_digest(pw, expected):
                st.session_state["auth_ok"] = True
                st.rerun()
            else:
                st.error("Incorrect password.")
    st.stop()


def fmt(v, units: str) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    if units == "usd":
        return f"${v:,.0f}"
    if units == "prop":
        return f"{v:+.4f}"
    return f"{v:,.4g}"


def run_graph(task, provider: str):
    """Run the full graph to completion (gates auto-resolved by policy)."""
    llm = make_llm(provider)
    app = build_graph(task, llm, checkpointer=MemorySaver())
    init = {"task_id": task.id, "provider": provider, "plan": task.description,
            "available_covariates": task.covariates, "truth": task.truth,
            "units": task.units, "max_iterations": 2, "auto_resolve": True,
            "ledger": []}
    cfg = {"configurable": {"thread_id": f"{task.id}-streamlit"}}
    state = app.invoke(init, cfg)
    while "__interrupt__" in state:
        state = app.invoke(
            Command(resume={"approve": True, "tie_break": "economist",
                            "source": "auto-policy"}), cfg)
    return state


def render_results(task, state) -> None:
    units = state.get("units", "")
    r = state.get("results", {})
    est, truth, ci = r.get("estimate"), state.get("truth"), r.get("ci")
    err = abs(est - truth) if (est is not None and truth is not None and est == est) else None
    covers = (ci and ci[0] == ci[0] and ci[0] <= truth <= ci[1]) if (truth is not None and ci) else None

    st.subheader(f"Result — {task.name}")
    if truth is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Estimate", fmt(est, units), help=f"estimand: {r.get('estimand')}")
        c2.metric("Experimental truth", fmt(truth, units))
        c3.metric("Absolute error", fmt(err, units))
        c4.metric("95% CI covers truth", "✅ yes" if covers else ("❌ no" if covers is not None else "n/a"))
    else:
        c1, c2 = st.columns(2)
        c1.metric("Estimate", fmt(est, units), help=f"estimand: {r.get('estimand')}")
        c2.metric("95% CI", f"[{fmt(ci[0], units)}, {fmt(ci[1], units)}]" if ci else "n/a")
        st.caption("⚠️ No experimental ground truth for your data — this estimate "
                   "is only as valid as the unconfoundedness assumption. Read the "
                   "diagnostics and critic below, not a 'correct' number.")
    if ci and truth is not None:
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


# ----------------------------------------------------------------- UI
check_password()

st.title("🧪 Causal Multi-Agent System")
st.markdown(
    "A multi-agent, human-in-the-loop pipeline for **observational causal "
    "inference**. The **economist** proposes the adjustment set, the **analyst** "
    "runs a doubly-robust estimator with overlap/balance/placebo diagnostics "
    "(no LLM — *facts are settled by numbers*), and the **critic** judges it.")

mode = st.sidebar.radio("Mode", ["Benchmark study", "Your data — ask a question"])

# ============================================================ benchmark mode
if mode == "Benchmark study":
    with st.sidebar:
        task_id = st.selectbox("Task", list(TASKS), format_func=lambda t: TASKS[t])
        provider = st.radio("Provider", ["nebius", "stub"], index=0,
                            help="nebius = real LLM agents (billed). stub = offline, deterministic.")
        st.caption(
            f"**economist** → `{DEFAULT_ECONOMIST_MODEL}`  \n**critic** → `{DEFAULT_CRITIC_MODEL}`"
            if provider == "nebius" else "stub: deterministic oracle, no API calls.")
        if provider == "nebius" and not os.environ.get("NEBIUS_API_KEY"):
            st.warning("No NEBIUS_API_KEY — set it in Secrets or switch to stub.")
        go = st.button("Run analysis", type="primary", use_container_width=True)

    if not go:
        st.info("Pick a task and provider in the sidebar, then **Run analysis**.")
        st.stop()
    with st.spinner(f"Running the pipeline on **{task_id}** via **{provider}**…"):
        try:
            task = get_task(task_id)
            state = run_graph(task, provider)
        except Exception as e:
            st.error(f"Run failed: {type(e).__name__}: {e}")
            st.stop()
    render_results(task, state)

# ============================================================ your-data mode
else:
    st.subheader("Ask a causal question of your own data")
    if not os.environ.get("NEBIUS_API_KEY"):
        st.warning("Your-data mode needs **NEBIUS_API_KEY** — it uses the LLM to "
                   "map your question to the columns.")
        st.stop()

    up = st.file_uploader("Upload a CSV", type=["csv"])
    if up is None:
        st.info("Upload a CSV to begin. Then ask a question like "
                "*“Does attending the program increase income?”*")
        st.stop()
    try:
        df = pd.read_csv(up)
    except Exception as e:
        st.error(f"Couldn't read CSV: {e}")
        st.stop()
    st.caption(f"{len(df):,} rows × {len(df.columns)} columns")
    with st.expander("Preview"):
        st.dataframe(df.head(), use_container_width=True)

    question = st.text_input("Your causal question",
                             placeholder="e.g. Does enrolling in the training program raise earnings?")
    if st.button("Interpret with the economist", disabled=not question.strip()):
        with st.spinner("Mapping your question to the data…"):
            try:
                st.session_state["byo_mapping"] = map_question(make_llm("nebius"), question, df)
            except Exception as e:
                st.error(f"Mapping failed: {e}")

    mapping = st.session_state.get("byo_mapping")
    if not mapping:
        st.stop()

    st.markdown("**Confirm the design** — edit anything the model got wrong:")
    cols = list(df.columns)
    idx = lambda c: cols.index(c) if c in cols else 0
    treatment = st.selectbox("Treatment (must be binary)", cols, index=idx(mapping["treatment"]))
    outcome = st.selectbox("Outcome", cols, index=idx(mapping["outcome"]))
    numeric_cands = [c for c in cols if c not in (treatment, outcome)
                     and pd.api.types.is_numeric_dtype(df[c])]
    covars = st.multiselect(
        "Candidate covariates (the economist will refuse identifiers / mediators / colliders)",
        numeric_cands, default=numeric_cands)
    unit_opts = ["raw", "prop", "usd"]
    units = st.selectbox("Outcome units", unit_opts,
                         index=unit_opts.index(mapping["units"]) if mapping["units"] in unit_opts else 0)
    desc = st.text_input("Study description", value=mapping["study_description"])
    with st.expander("Economist's column notes (from your question)"):
        st.dataframe(pd.DataFrame([(c.name, c.description) for c in mapping["schema"]],
                                  columns=["column", "description"]),
                     use_container_width=True, hide_index=True)

    if st.button("Run analysis", type="primary"):
        keep = set(covars) | {treatment, outcome}
        schema = [cs for cs in mapping["schema"] if cs.name in keep]
        try:
            task = build_task_from_df(df, treatment, outcome, covars, schema=schema,
                                      units=units, name="Your study", description=desc)
        except Exception as e:
            st.error(f"Couldn't build the study: {e}")
            st.stop()
        with st.spinner("Running the multi-agent pipeline (nebius)…"):
            try:
                state = run_graph(task, "nebius")
            except Exception as e:
                st.error(f"Run failed: {type(e).__name__}: {e}")
                st.stop()
        render_results(task, state)
