"""
What the economist reasons over.

The whole point of making this an agent (not a script) is that picking the
adjustment set from an arbitrary column list is a reasoning problem with no closed
form: which columns are genuine pre-treatment confounders, and which must be refused
because they are the outcome, an identifier, a *second* randomized treatment, or —
the dangerous one — a mediator/collider that sits causally downstream of treatment.

Each `ColumnSpec` carries only a name and a neutral, data-dictionary-style
description. It deliberately does NOT carry the column's causal role: inferring that
from the description is exactly the economist's job. The correct roles live separately
in ANSWER_KEY (used only by the offline oracle and by the grader in
verify_economist_llm.py), so the model is never handed the answer.

Honesty note: the three public teaching extracts have had their post-treatment
variables stripped, so none of them contains a planted mediator. The columns the
economist must refuse on real data are therefore the outcome, identifiers, the
treatment itself, and (in Cai) the second randomized treatment arm. The harder
mediator/collider reasoning is exercised by PROBE_SCHEMA below — a hypothetical data
dictionary (no fabricated observations) used purely as a reasoning unit-test.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ColumnSpec:
    name: str
    description: str


# --------------------------------------------------------------------------- #
# Real candidate schemas (every column is actually present in the loaded data) #
# --------------------------------------------------------------------------- #

SCHEMAS: dict[str, list[ColumnSpec]] = {
    "lalonde": [
        ColumnSpec("data_id", "source/record identifier for the row"),
        ColumnSpec("treat", "1 if the person enrolled in the NSW training programme"),
        ColumnSpec("age", "age in years, recorded at baseline"),
        ColumnSpec("educ", "years of schooling at baseline"),
        ColumnSpec("black", "1 if the person is Black (baseline demographic)"),
        ColumnSpec("hisp", "1 if the person is Hispanic (baseline demographic)"),
        ColumnSpec("marr", "1 if married at baseline"),
        ColumnSpec("nodegree", "1 if no high-school degree at baseline"),
        ColumnSpec("re74", "real annual earnings in 1974, before the programme"),
        ColumnSpec("re75", "real annual earnings in 1975, before the programme"),
        ColumnSpec("re78", "real annual earnings in 1978, measured after the programme"),
    ],
    "thornton": [
        ColumnSpec("villnum", "village number the respondent belongs to"),
        ColumnSpec("got", "1 if the respondent collected their HIV test result"),
        ColumnSpec("distvct", "distance (km) from home to the results-collection centre, baseline"),
        ColumnSpec("tinc", "size of the cash incentive offered, in local currency"),
        ColumnSpec("any", "1 if the respondent was offered any cash incentive"),
        ColumnSpec("age", "age in years, recorded at baseline"),
        ColumnSpec("hiv2004", "HIV status from the 2004 testing round, before the incentive"),
    ],
    "cai": [
        ColumnSpec("address", "household address/identifier code"),
        ColumnSpec("village", "village code; sessions were organised village by village"),
        ColumnSpec("takeup_survey", "1 if the household purchased the weather insurance"),
        ColumnSpec("age", "age of the household head, baseline"),
        ColumnSpec("agpop", "agricultural household size, baseline"),
        ColumnSpec("ricearea_2010", "rice-cultivation area in 2010, baseline"),
        ColumnSpec("disaster_prob", "household's stated perceived probability of a disaster, baseline"),
        ColumnSpec("male", "1 if the household head is male, baseline"),
        ColumnSpec("default", "1 if the household was assigned the separate default/timing experimental condition"),
        ColumnSpec("intensive", "1 if the household attended the intensive information session"),
        ColumnSpec("risk_averse", "baseline risk-aversion score"),
        ColumnSpec("literacy", "baseline literacy indicator"),
        ColumnSpec("pre_takeup_rate", "insurance take-up rate in the household's village in the prior period"),
    ],
}

# treatment + outcome for each task (the two columns that must never be adjusted for)
TREATMENT_OUTCOME = {
    "lalonde": ("treat", "re78"),
    "thornton": ("any", "got"),
    "cai": ("intensive", "takeup_survey"),
}

# the answer key — NOT shown to the model. roles: confounder | outcome | treatment |
# identifier | co_treatment | treatment_dose
ANSWER_KEY: dict[str, dict[str, str]] = {
    "lalonde": {"data_id": "identifier", "treat": "treatment", "re78": "outcome",
                "age": "confounder", "educ": "confounder", "black": "confounder",
                "hisp": "confounder", "marr": "confounder", "nodegree": "confounder",
                "re74": "confounder", "re75": "confounder"},
    "thornton": {"villnum": "identifier", "got": "outcome", "any": "treatment",
                 "tinc": "treatment_dose", "distvct": "confounder", "age": "confounder",
                 "hiv2004": "confounder"},
    "cai": {"address": "identifier", "village": "identifier", "takeup_survey": "outcome",
            "intensive": "treatment", "default": "co_treatment", "age": "confounder",
            "agpop": "confounder", "ricearea_2010": "confounder", "disaster_prob": "confounder",
            "male": "confounder", "risk_averse": "confounder", "literacy": "confounder",
            "pre_takeup_rate": "confounder"},
}


def candidate_schema(task_id: str) -> list[ColumnSpec]:
    return SCHEMAS[task_id]


def correct_confounders(task_id: str) -> list[str]:
    return [c for c, r in ANSWER_KEY[task_id].items() if r == "confounder"]


# --------------------------------------------------------------------------- #
# Reasoning probe: a hypothetical data dictionary (NO fabricated observations)  #
# used only as a unit-test of the economist's mediator/collider reasoning.      #
# --------------------------------------------------------------------------- #

PROBE_TREATMENT_OUTCOME = ("program", "earnings")

PROBE_SCHEMA: list[ColumnSpec] = [
    ColumnSpec("participant_id", "unique identifier for the participant"),
    ColumnSpec("program", "1 if the participant enrolled in the job-training programme"),
    ColumnSpec("earnings", "annual earnings measured 18 months after enrolment"),
    ColumnSpec("age", "age in years at enrolment"),
    ColumnSpec("years_education", "years of schooling completed before enrolment"),
    ColumnSpec("prior_year_earnings", "annual earnings in the year before enrolment"),
    ColumnSpec("region", "region of residence at enrolment"),
    ColumnSpec("employed_at_followup", "1 if employed at the 12-month follow-up, after the programme"),
    ColumnSpec("completed_program", "1 if the participant finished the programme (only the enrolled can complete)"),
    ColumnSpec("received_certificate", "1 if awarded a completion certificate at the end of the programme"),
]

# roles for grading the probe (hidden from the model)
PROBE_KEY = {
    "participant_id": "identifier",
    "program": "treatment",
    "earnings": "outcome",
    "age": "confounder",
    "years_education": "confounder",
    "prior_year_earnings": "confounder",
    "region": "confounder",
    "employed_at_followup": "mediator",        # program -> employment -> earnings
    "completed_program": "post_treatment",      # defined only under treatment; conditioning selects
    "received_certificate": "mediator",         # downstream of treatment
}
