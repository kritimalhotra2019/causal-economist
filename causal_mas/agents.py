"""
Agent node logic. Each function takes (state, task, llm) and returns a partial
state update; graph.py binds `task` and `llm` and wires them as nodes.

Division of labour:
  economist  -> design (treatment/outcome/confounders/estimand). LLM reasons here.
  analyst    -> runs the deterministic estimator + diagnostics. No LLM. Retries on error.
  critic     -> verdict + remedy + conflict. The verdict is a DETERMINISTIC rubric
                over the diagnostics (facts settled by numbers); the LLM only adds
                written rationale in nebius mode.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from . import backend as B
from .schema import candidate_schema, ANSWER_KEY, TREATMENT_OUTCOME
from .state import event

# ---- prompts (used only by the nebius provider) ----------------------------

ECON_SYSTEM = (
    "You are an econometrician choosing the adjustment set for an observational causal "
    "estimate under selection-on-observables. From the listed columns, select ONLY "
    "genuine PRE-treatment confounders. You MUST refuse:\n"
    "  - the treatment and the outcome themselves;\n"
    "  - identifiers (ids, codes, addresses);\n"
    "  - any second or parallel treatment / experimental arm;\n"
    "  - and most importantly, any variable measured AFTER the treatment or CAUSED BY it "
    "(a mediator), or any collider — adjusting for these induces bias even though the "
    "data will look perfectly fine.\n"
    "If a description says a variable is measured after / at follow-up, or that it can "
    "only occur for treated units, it is post-treatment: exclude it. "
    "Reply with ONLY a JSON object."
)


def _econ_user(task, schema) -> str:
    cols = "\n".join(f"  - {c.name}: {c.description}" for c in schema)
    return (
        f"Treatment: {task.treatment_desc}\n"
        f"Outcome: {task.outcome_desc}\n"
        f"Study: {task.description}\n\n"
        f"Columns available:\n{cols}\n\n"
        "Classify EVERY column. Return JSON:\n"
        "{\"columns\": [{\"name\": \"...\", "
        "\"role\": \"confounder|mediator|collider|post_treatment|outcome|treatment|identifier|other\", "
        "\"include\": true|false, \"reason\": \"<=12 words\"}], "
        "\"estimand\": \"ATT\", \"identification\": \"<one line>\", "
        "\"note\": \"<one line on the main threat to validity>\"}"
    )

CRITIC_SYSTEM = (
    "You are an independent referee for an observational causal estimate. You are given "
    "the diagnostics and the rubric verdict. Write a one-paragraph rationale a reviewer "
    "could audit. Reply with ONLY a JSON object: {\"rationale\": \"...\"}."
)

_ROLE_REASON = {
    "confounder": "pre-treatment baseline covariate",
    "outcome": "this is the outcome — never adjust for it",
    "treatment": "this is the treatment itself",
    "identifier": "an identifier, not a covariate",
    "co_treatment": "a separate randomized treatment, not a baseline confounder",
    "treatment_dose": "a form of the treatment, not a confounder",
    "mediator": "post-treatment, on the causal path — adjusting induces bias",
    "post_treatment": "measured after treatment — not a baseline covariate",
    "collider": "a collider — conditioning induces bias",
}


def _oracle_decisions(task_id: str) -> list[dict[str, Any]]:
    """Offline reference: the correct classification, from the (hidden) answer key."""
    return [{"name": c, "role": r, "include": r == "confounder",
             "reason": _ROLE_REASON.get(r, "")}
            for c, r in ANSWER_KEY[task_id].items()]


def _apply_decisions(decisions, task, treat_col, out_col):
    """Turn the economist's per-column calls into an adjustment set. Enforces only
    hard rules (never the treatment/outcome, must exist, must be numeric); otherwise
    TRUSTS the economist's judgement, so a bad design flows downstream and surfaces in
    the diagnostics rather than being silently corrected."""
    confounders, excluded = [], []
    present = set(task.df.columns)
    for d in decisions:
        name = d.get("name")
        role = d.get("role", "")
        reason = d.get("reason") or _ROLE_REASON.get(role, "")
        forced = None
        if name in (treat_col, out_col):
            forced = "hard rule: the treatment/outcome is never an adjustment variable"
        elif name not in present:
            forced = "not present in the data"
        elif not pd.api.types.is_numeric_dtype(task.df[name]):
            forced = "non-numeric — cannot enter the estimator"
        if forced:
            excluded.append({"name": name, "role": role, "reason": reason, "forced": forced})
        elif d.get("include"):
            confounders.append(name)
        else:
            excluded.append({"name": name, "role": role, "reason": reason, "forced": None})
    return confounders, excluded


# ---- nodes -----------------------------------------------------------------

def economist_design(state, task, llm) -> dict[str, Any]:
    # benchmark tasks use the hardcoded schema; user tasks carry their own.
    schema = task.schema or candidate_schema(task.id)
    treat_col, out_col = task.treat, task.outcome
    estimand, identification, note = ("ATT",
                                      "selection-on-observables (unconfoundedness)",
                                      "credible only if treatment is unconfounded given these covariates")

    if llm.provider == "stub":
        # the oracle only exists for benchmark tasks; for user data the stub
        # falls back to adjusting for every supplied covariate (safety net below).
        decisions = _oracle_decisions(task.id) if task.id in ANSWER_KEY else []
    else:
        decisions = []
        try:
            out = llm.complete_json(ECON_SYSTEM, _econ_user(task, schema))
            decisions = out.get("columns", []) or []
            estimand = out.get("estimand") or estimand
            identification = out.get("identification") or identification
            note = out.get("note") or note
        except Exception:
            decisions = []

    confounders, excluded = _apply_decisions(decisions, task, treat_col, out_col)
    if not confounders:  # safety net: the model returned nothing usable
        confounders = list(task.covariates)
        note = note + "  [fell back to the default covariate set]"

    design = {"treatment": task.treatment_desc, "outcome": task.outcome_desc,
              "confounders": confounders, "estimand": estimand,
              "identification": identification, "note": note, "excluded": excluded}
    return {"design": design, "spec": {"trim_bounds": None}, "iteration": 0,
            "ledger": [event("economist_design", confounders=confounders,
                             excluded=[{"name": e["name"], "role": e["role"]} for e in excluded],
                             model=getattr(llm, "model", None))]}


def analyst_execute(state, task, llm) -> dict[str, Any]:
    design = state["design"]
    spec = state.get("spec", {"trim_bounds": None})
    covs = design["confounders"]
    last_err = None
    for attempt in range(2):  # one retry on transient failure
        try:
            full = B.estimate_full(
                task.df, task.treat, task.outcome, covs,
                placebo_var=task.placebo_var, trim_bounds=spec.get("trim_bounds"),
                n_boot=200, seed=0)
            results = {k: full[k] for k in
                       ("estimate", "ci", "estimand", "n_treated", "n_control", "naive", "trim_bounds")}
            return {"results": results, "diagnostics": full["diagnostics"],
                    "ledger": [event("analyst_execute", attempt=attempt,
                                     estimate=results["estimate"], estimand=results["estimand"],
                                     trim_bounds=spec.get("trim_bounds"),
                                     overlap_passed=full["diagnostics"]["overlap"]["passed"],
                                     balance_passed=full["diagnostics"]["balance"]["passed"])]}
        except Exception as e:  # pragma: no cover
            last_err = str(e)
    # unrecoverable -> escalate via a not_satisfactory critic verdict downstream
    return {"results": {"estimate": float("nan"), "error": last_err},
            "diagnostics": {"all_gating_passed": False, "error": last_err},
            "ledger": [event("analyst_error", error=last_err)]}


def _rubric(diag, trim_applied, n_control) -> dict[str, Any]:
    """Deterministic verdict from the diagnostics — the 'facts settled by numbers'."""
    if "error" in diag:
        return {"verdict": "not_satisfactory", "remedy": None, "conflict": None,
                "rationale": f"estimation failed: {diag.get('error')}"}
    ov = diag["overlap"]["passed"]
    bal = diag["balance"]["passed"]
    plac = diag["placebo"]["passed"]
    remedy = None
    conflict = None

    if not ov and not trim_applied:
        verdict = "not_satisfactory"
        remedy = {"type": "trim", "bounds": list(B.OVERLAP_BOUNDS)}
        rationale = ("overlap is violated: a large share of comparison units have no "
                     "plausible counterpart in the treated group; trim to common support.")
    else:
        if ov and bal and plac:
            verdict = "fully_satisfactory"
            rationale = "overlap, balance, and placebo checks all pass."
        else:
            verdict = "satisfactory_with_caveats"
            caveats = []
            if not bal:
                caveats.append("residual covariate imbalance, absorbed by the doubly-robust adjustment")
            if not plac:
                caveats.append("placebo balance imperfect")
            if not ov:
                caveats.append("overlap still imperfect after trimming")
            rationale = "; ".join(caveats) or "minor caveats noted"
        # the economist-vs-critic disagreement: a thin overlap sample
        if trim_applied and n_control is not None and n_control < 500:
            conflict = (f"overlap sample is thin (n_control={n_control}): the critic recommends "
                        f"reporting this as low-confidence; the economist holds that the trimmed "
                        f"doubly-robust estimate is credible and should stand with a caveat.")
    return {"verdict": verdict, "remedy": remedy, "conflict": conflict, "rationale": rationale}


def critic_evaluate(state, task, llm) -> dict[str, Any]:
    diag = state["diagnostics"]
    results = state.get("results", {})
    trim_applied = bool(state.get("spec", {}).get("trim_bounds"))
    verdict = _rubric(diag, trim_applied, results.get("n_control"))

    if llm.provider != "stub" and "error" not in diag:
        try:
            payload = (f"diagnostics={diag}\nrubric_verdict={verdict['verdict']}\n"
                       f"estimate={results.get('estimate')} estimand={results.get('estimand')}")
            out = llm.complete_json(CRITIC_SYSTEM, payload)
            if out.get("rationale"):
                verdict["rationale"] = out["rationale"]
        except Exception:
            pass

    return {"critic": verdict,
            "ledger": [event("critic_evaluate", verdict=verdict["verdict"],
                             remedy=verdict["remedy"], conflict=bool(verdict["conflict"]),
                             model=getattr(llm, "model", None))]}


def economist_revise(state, task, llm) -> dict[str, Any]:
    """Apply an approved remedy to the spec (mechanical) and loop back to the analyst."""
    spec = dict(state.get("spec", {}))
    remedy = state["critic"].get("remedy") or {}
    note = "no-op"
    if remedy.get("type") == "trim":
        spec["trim_bounds"] = tuple(remedy["bounds"])
        note = f"trim to {spec['trim_bounds']} (estimand will narrow to the overlap subpopulation)"
    return {"spec": spec, "iteration": state.get("iteration", 0) + 1,
            "ledger": [event("economist_revise", applied=note)]}


def human_gate(state, task, llm) -> dict[str, Any]:
    """The single human-in-the-loop point. In eval/auto mode it resolves by a fixed
    policy (approve the remedy, side with the economist) with NO interrupt, so the
    only A-vs-C difference is enforced structure, not a human. Interactively it
    pauses via interrupt() and resumes with the human's decision."""
    critic = state["critic"]
    payload = {
        "verdict": critic["verdict"],
        "proposed_remedy": critic.get("remedy"),
        "consequence": ("trimming narrows the estimand to the comparable subpopulation"
                        if (critic.get("remedy") or {}).get("type") == "trim" else None),
        "conflict": critic.get("conflict"),
        "current_estimate": state.get("results", {}).get("estimate"),
    }
    if state.get("auto_resolve"):
        decision = {"approve": True, "tie_break": "economist", "source": "auto-policy"}
    else:
        from langgraph.types import interrupt
        decision = interrupt(payload)  # pauses the graph; resume with Command(resume=decision)

    return {"human_decision": decision,
            "ledger": [event("human_decision", approve=decision.get("approve"),
                             tie_break=decision.get("tie_break"), source=decision.get("source", "human"))]}


def finalize(state, task) -> dict[str, Any]:
    r = state.get("results", {})
    est = r.get("estimate")
    truth = state.get("truth")
    units = state.get("units", "")
    err = abs(est - truth) if (est is not None and truth is not None and est == est) else None
    ci = r.get("ci")
    covers = (ci and ci[0] == ci[0] and ci[0] <= truth <= ci[1]) if (truth is not None and ci) else None

    def f(v):
        if v is None or (isinstance(v, float) and v != v):
            return "n/a"
        return (f"${v:,.0f}" if units == "usd" else f"{v:+.3f}")

    lines = [
        f"Task: {task.name}",
        f"Design: {state['design']['estimand']} of [{task.treatment_desc}] on [{task.outcome_desc}]",
        f"  adjustment set: {', '.join(state['design']['confounders'])}",
        f"Estimate: {f(est)}   (estimand: {r.get('estimand')})",
        f"  95% CI: [{f(ci[0]) if ci else 'n/a'}, {f(ci[1]) if ci else 'n/a'}]" if ci else "",
        (f"Experimental truth: {f(truth)}   |error|: {f(err)}   CI covers truth: {covers}"
         if truth is not None else
         "No experimental ground truth (observational data) — judge credibility from "
         "the diagnostics and the critic verdict, not against a known answer."),
        f"Critic verdict: {state['critic']['verdict']}",
        f"  {state['critic']['rationale']}",
    ]
    if state["critic"].get("conflict"):
        d = state.get("human_decision") or {}
        lines.append(f"Resolved disagreement: you sided with the {d.get('tie_break','economist')}.")
    report = "\n".join(x for x in lines if x)
    return {"report": report, "ledger": [event("finalize", estimate=est, error=err, covers=covers)]}
