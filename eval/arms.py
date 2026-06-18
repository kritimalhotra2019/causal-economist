"""
The four arms. The only honest test of "is MAS beneficial" is A vs C and A vs the
ablation — so all four share the same tasks and parsing, and the baseline prompts
are kept strictly free of diagnostic hints (telling them to check overlap would
smuggle the scaffold into the baseline).

  A  arm_mas            the LangGraph pipeline (deterministic via stub, or nebius)
  B  arm_oneshot_nocode one call, group summary only, no code        (Netflix baseline B)
  C  arm_oneshot_code   one call WITH a run_python tool, sees raw data (capability match)
     arm_ablation       like C, but a single agent TOLD to run the checklist + trim

Run A with stub for a deterministic reference. B/C/ablation need an API key.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

from langgraph.checkpoint.memory import MemorySaver

from causal_mas.graph import build_graph
from causal_mas.llm import make_llm, _parse_json

DEFAULT_ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")
CODE_TIMEOUT = 60


# ---------------------------------------------------------------- arm A: MAS
def arm_mas(task, provider="stub", seed=0) -> dict:
    llm = make_llm(provider)
    app = build_graph(task, llm, checkpointer=MemorySaver())
    init = {"task_id": task.id, "provider": provider, "plan": task.description,
            "available_covariates": task.covariates, "truth": task.truth,
            "units": task.units, "max_iterations": 2, "auto_resolve": True, "ledger": []}
    final = app.invoke(init, {"configurable": {"thread_id": f"{task.id}-{seed}"}})
    r = final.get("results", {})
    ci = r.get("ci") or (None, None)
    estimand_shifted = r.get("estimand", "") != "ATT (full sample)"
    flagged = estimand_shifted or final.get("critic", {}).get("verdict") != "fully_satisfactory"
    return {"arm": "A_mas", "estimate": r.get("estimate"),
            "ci_low": ci[0], "ci_high": ci[1],
            "flagged": bool(flagged), "estimand_disclosed": bool(estimand_shifted)}


# --------------------------------------------------- baseline LLM callers
def make_baseline_caller(provider="anthropic", model=None):
    """Returns call(system, user, tools=None, messages=None) -> anthropic-style response.
    For the no-code arm only the text path is used."""
    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        mdl = model or DEFAULT_ANTHROPIC_MODEL

        def call(system, messages, tools=None):
            kw = dict(model=mdl, max_tokens=1500, system=system, messages=messages)
            if tools:
                kw["tools"] = tools
            return client.messages.create(**kw)
        return call, "anthropic"

    if provider == "nebius":
        from openai import OpenAI
        from causal_mas.llm import NEBIUS_BASE_URL, DEFAULT_NEBIUS_MODEL
        client = OpenAI(base_url=NEBIUS_BASE_URL, api_key=os.environ["NEBIUS_API_KEY"])
        # `or`-chain (not get's default) so a set-but-empty NEBIUS_MODEL falls
        # through to the default instead of becoming "" — matches make_llm.
        mdl = model or os.environ.get("NEBIUS_MODEL") or DEFAULT_NEBIUS_MODEL

        def call(system, messages, tools=None):  # text-only path (no tools) for arm B
            oai_msgs = [{"role": "system", "content": system}] + messages
            return client.chat.completions.create(model=mdl, max_tokens=1500, messages=oai_msgs)
        return call, "nebius"

    raise ValueError(provider)


def _summary_prompt(task) -> str:
    cov = "\n".join(f"  {c}: treated {v['treated']}, control {v['control']}"
                    for c, v in task.summary["covariate_means"].items())
    s = task.summary
    return (f"Estimate the causal effect of \"{task.treatment_desc}\" on "
            f"\"{task.outcome_desc}\".\n\nStudy: {task.description}\n\n"
            f"Sample: {s['n']['treated']} treated, {s['n']['control']} control.\n"
            f"Outcome mean: treated {s['outcome_mean']['treated']}, control {s['outcome_mean']['control']}.\n"
            f"Covariate means by group:\n{cov}\n\n"
            "Reply with ONLY JSON: {\"point_estimate\": <number>, \"ci_low\": <number or null>, "
            "\"ci_high\": <number or null>, \"comparability_concern\": <true if the groups look too "
            "different to compare directly, else false>, \"note\": \"<=15 words\"}")


# ----------------------------------------------- arm B: one-shot, no code
def arm_oneshot_nocode(task, caller, seed=0) -> dict:
    call, provider = caller
    system = "You are a careful causal-inference analyst."
    user = _summary_prompt(task)
    try:
        if provider == "anthropic":
            resp = call(system, [{"role": "user", "content": user}])
            text = "".join(b.text for b in resp.content if b.type == "text")
        else:
            resp = call(system, [{"role": "user", "content": user}])
            text = resp.choices[0].message.content or ""
        out = _parse_json(text)
    except Exception as e:
        return {"arm": "B_oneshot_nocode", "estimate": None, "error": str(e), "flagged": False}
    return {"arm": "B_oneshot_nocode",
            "estimate": _num(out.get("point_estimate")),
            "ci_low": _num(out.get("ci_low")), "ci_high": _num(out.get("ci_high")),
            "flagged": out.get("comparability_concern") is True,
            "estimand_disclosed": False, "note": out.get("note")}


# ---------------------------------- arm C + ablation: one-shot WITH code
RUN_PYTHON_TOOL = {
    "name": "run_python",
    "description": "Execute Python. A pandas DataFrame `df` is preloaded with the study data. "
                   "Print whatever you need to see.",
    "input_schema": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
}


def _exec_python(code: str, csv_path: str, treat, outcome, covs) -> str:
    preamble = (
        "import pandas as pd, numpy as np\n"
        f"df = pd.read_csv({csv_path!r})\n"
        f"TREAT, OUTCOME, COVARIATES = {treat!r}, {outcome!r}, {covs!r}\n")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(preamble + code)
        path = f.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=CODE_TIMEOUT)
        return (p.stdout + (("\n[stderr]\n" + p.stderr) if p.stderr else ""))[:6000]
    except subprocess.TimeoutExpired:
        return "[timed out]"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _arm_code(task, caller, seed=0, with_checklist=False, max_turns=8) -> dict:
    call, provider = caller
    if provider != "anthropic":
        raise NotImplementedError("the code arms use the Anthropic tool loop")
    arm = "ablation_checklist" if with_checklist else "C_oneshot_code"

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        task.df.to_csv(f.name, index=False)
        csv_path = f.name

    base = (f"Estimate the causal effect of \"{task.treatment_desc}\" on "
            f"\"{task.outcome_desc}\" from observational data.\n"
            f"Study: {task.description}\n"
            f"The data is in `df`. Treatment column: {task.treat!r}. Outcome: {task.outcome!r}. "
            f"Covariates: {task.covariates}.\n"
            "Use the run_python tool to do the analysis. When finished, reply with ONLY JSON: "
            "{\"point_estimate\": <number>, \"ci_low\": <number or null>, \"ci_high\": <number or null>, "
            "\"flagged\": <true if you judged the naive comparison unreliable and corrected for it, else false>, "
            "\"estimand\": \"<what you estimated>\", \"note\": \"<=20 words\"}")
    if with_checklist:
        base += ("\n\nFollow this checklist: (1) fit a propensity model; (2) check overlap — the share of "
                 "units with propensity outside [0.1, 0.9]; (3) check covariate balance (standardized mean "
                 "differences, flag any > 0.2); (4) if overlap fails, trim to [0.1, 0.9] and note the estimand "
                 "changes; (5) use a doubly-robust estimator.")
    system = "You are a causal-inference analyst with a Python tool."
    messages = [{"role": "user", "content": base}]

    try:
        for _ in range(max_turns):
            resp = call(system, messages, tools=[RUN_PYTHON_TOOL])
            if resp.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": resp.content})
                results = []
                for blk in resp.content:
                    if blk.type == "tool_use" and blk.name == "run_python":
                        out = _exec_python(blk.input.get("code", ""), csv_path,
                                           task.treat, task.outcome, task.covariates)
                        results.append({"type": "tool_result", "tool_use_id": blk.id, "content": out})
                messages.append({"role": "user", "content": results})
                continue
            text = "".join(b.text for b in resp.content if b.type == "text")
            out = _parse_json(text)
            return {"arm": arm, "estimate": _num(out.get("point_estimate")),
                    "ci_low": _num(out.get("ci_low")), "ci_high": _num(out.get("ci_high")),
                    "flagged": bool(out.get("flagged")),
                    "estimand_disclosed": "overlap" in str(out.get("estimand", "")).lower()
                    or "subpop" in str(out.get("estimand", "")).lower(),
                    "note": out.get("note")}
        return {"arm": arm, "estimate": None, "error": "max turns reached", "flagged": False}
    finally:
        try:
            os.unlink(csv_path)
        except OSError:
            pass


def arm_oneshot_code(task, caller, seed=0):
    return _arm_code(task, caller, seed, with_checklist=False)


def arm_ablation(task, caller, seed=0):
    return _arm_code(task, caller, seed, with_checklist=True)


def _num(x):
    if isinstance(x, (int, float)):
        return float(x)
    if x is None:
        return None
    import re
    m = re.findall(r"-?\d+\.?\d*", str(x))
    return float(m[0]) if m else None
