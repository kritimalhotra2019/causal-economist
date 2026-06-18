"""
Verify the economist's REASONING with a real model.

The mock tests in the repo prove the plumbing (parsing, hard-rule guards, that a
choice flows downstream). They do NOT prove the model makes good calls — that needs a
real LLM, your key, and a network this sandbox doesn't have. So you run this.

  export NEBIUS_API_KEY=...        # and optionally NEBIUS_MODEL=...
  python verify_economist_llm.py

It asks the model to classify every column for the three datasets and for a
hypothetical study with a planted mediator and collider, then grades the model's own
include/exclude flags against a hidden answer key and prints its stated reasons.
This is a reasoning unit-test: the probe is a data dictionary, not fabricated data.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

from causal_mas import agents
from causal_mas.datasets import get_task
from causal_mas.llm import make_llm
from causal_mas.schema import (ANSWER_KEY, PROBE_KEY, PROBE_SCHEMA,
                               PROBE_TREATMENT_OUTCOME, candidate_schema)

GREEN, RED, DIM, OFF = "\033[92m", "\033[91m", "\033[2m", "\033[0m"


def classify(llm, treat_desc, out_desc, study, schema):
    raw = llm.complete_json(
        agents.ECON_SYSTEM,
        agents._econ_user(SimpleNamespace(treatment_desc=treat_desc, outcome_desc=out_desc,
                                          description=study), schema))
    return {d.get("name"): d for d in raw.get("columns", [])}


def grade(decisions: dict, key: dict, title: str) -> bool:
    print(f"\n{'='*78}\n{title}\n{'='*78}")
    print(f"  {'column':22s} {'true role':14s} {'model':9s} {'verdict':8s} reason")
    all_ok = True
    for col, true_role in key.items():
        should = (true_role == "confounder")
        d = decisions.get(col, {})
        inc = bool(d.get("include"))
        ok = (inc == should)
        all_ok &= ok
        mark = f"{GREEN}OK{OFF}" if ok else f"{RED}WRONG{OFF}"
        verb = "include" if inc else "drop"
        print(f"  {col:22s} {true_role:14s} {verb:9s} {mark:17s} {DIM}{d.get('reason','—')}{OFF}")
    print(f"  → {'PASS' if all_ok else 'FAIL'}: the model "
          f"{'classified every column correctly' if all_ok else 'mis-classified at least one column'}.")
    return all_ok


def main():
    try:
        llm = make_llm("nebius")
    except Exception as e:
        print(f"could not initialise Nebius: {e}")
        sys.exit(1)
    print(f"model: {llm.model}")

    results = []
    for tid in ["lalonde", "thornton", "cai"]:
        t = get_task(tid)
        dec = classify(llm, t.treatment_desc, t.outcome_desc, t.description, candidate_schema(tid))
        results.append(grade(dec, ANSWER_KEY[tid], f"{t.name}  (real dataset schema)"))

    # the hard one: a mediator and a collider that look fine in the data
    dec = classify(llm, "enrolling in a job-training programme",
                   "annual earnings measured after the programme",
                   "A job-training evaluation with follow-up survey variables.", PROBE_SCHEMA)
    results.append(grade(dec, PROBE_KEY,
                         "Reasoning probe  (hypothetical study with a planted mediator + collider)"))

    print(f"\n{'='*78}\nOVERALL: {sum(results)}/{len(results)} schemas fully correct")
    print("The probe is the decisive one — it checks the model refuses post-treatment")
    print("variables (employed_at_followup, completed_program, received_certificate) that")
    print("the diagnostics could never catch.")


if __name__ == "__main__":
    main()
