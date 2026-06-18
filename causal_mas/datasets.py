"""
The three benchmark tasks. Each is an RCT, so the experimental/randomized estimate
gives us ground truth — the thing you almost never have in observational inference,
and what makes this a scoreable eval.

- lalonde : a CONSTRUCTED observational study (NSW trainees vs a national CPS survey).
            Truth = the separate NSW experimental benchmark. Confounding is real.
- thornton: a clean RCT (Malawi HIV incentive). Truth = the randomized difference.
- cai     : a clean RCT (China weather insurance, intensive-session arm).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from causaldata import nsw_mixtape, cps_mixtape, thornton_hiv, social_insure


@dataclass
class Task:
    id: str
    name: str
    description: str
    treatment_desc: str
    outcome_desc: str
    df: pd.DataFrame                  # the frame to ESTIMATE on
    treat: str
    outcome: str
    covariates: list[str]
    truth: float                     # experimental / randomized ground truth
    units: str                       # "usd" or "prop"
    placebo_var: str | None = None
    is_rct_as_is: bool = True        # False when the estimation frame is constructed-observational
    summary: dict = field(default_factory=dict)


def _summary(df, treat, outcome, covariates) -> dict:
    g = {"n": {"treated": int((df[treat] == 1).sum()),
               "control": int((df[treat] == 0).sum())},
         "outcome_mean": {"treated": round(float(df.loc[df[treat] == 1, outcome].mean()), 3),
                          "control": round(float(df.loc[df[treat] == 0, outcome].mean()), 3)},
         "covariate_means": {}}
    for c in covariates:
        g["covariate_means"][c] = {
            "treated": round(float(df.loc[df[treat] == 1, c].mean()), 3),
            "control": round(float(df.loc[df[treat] == 0, c].mean()), 3)}
    return g


def load_lalonde() -> Task:
    nsw = nsw_mixtape.load_pandas().data
    cps = cps_mixtape.load_pandas().data
    covs = ["age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"]
    # Truth: NSW experimental benchmark (randomized treated vs randomized control)
    truth = float(nsw.loc[nsw.treat == 1, "re78"].mean() - nsw.loc[nsw.treat == 0, "re78"].mean())
    # Estimation frame: constructed observational (NSW treated + CPS comparison)
    obs = pd.concat([nsw[nsw.treat == 1], cps], ignore_index=True)
    return Task(
        id="lalonde", name="LaLonde / NSW",
        description=("US National Supported Work job-training programme. Effect of "
                     "enrolling on 1978 annual earnings. Treated = NSW participants; "
                     "comparison group = a national population survey (CPS)."),
        treatment_desc="enrolling in the NSW training programme",
        outcome_desc="1978 annual earnings (US dollars)",
        df=obs, treat="treat", outcome="re78", covariates=covs,
        truth=truth, units="usd", placebo_var="re74", is_rct_as_is=False,
        summary=_summary(obs, "treat", "re78", covs))


def load_thornton() -> Task:
    d = thornton_hiv.load_pandas().data.dropna(subset=["any", "got", "age", "distvct", "hiv2004"]).copy()
    d["any"] = d["any"].astype(int)
    covs = ["age", "distvct", "hiv2004"]
    truth = float(d.loc[d["any"] == 1, "got"].mean() - d.loc[d["any"] == 0, "got"].mean())
    return Task(
        id="thornton", name="Thornton (Malawi HIV)",
        description=("Randomized cash incentive to collect your HIV test result. "
                     "Outcome: collected results (1=yes)."),
        treatment_desc="receiving a randomized cash incentive",
        outcome_desc="collected HIV results (1=yes)",
        df=d, treat="any", outcome="got", covariates=covs,
        truth=truth, units="prop", is_rct_as_is=True,
        summary=_summary(d, "any", "got", covs))


def load_cai() -> Task:
    s = social_insure.load_pandas().data
    covs = ["age", "agpop", "ricearea_2010", "disaster_prob", "male",
            "risk_averse", "literacy", "pre_takeup_rate"]
    d = s.dropna(subset=["intensive", "takeup_survey"] + covs).copy()
    d["intensive"] = d["intensive"].astype(int)
    truth = float(d.loc[d["intensive"] == 1, "takeup_survey"].mean()
                  - d.loc[d["intensive"] == 0, "takeup_survey"].mean())
    return Task(
        id="cai", name="Cai (China insurance)",
        description=("Randomized intensive vs simple insurance information session. "
                     "Outcome: bought weather insurance (1=yes)."),
        treatment_desc="attending an intensive information session",
        outcome_desc="bought weather insurance (1=yes)",
        df=d, treat="intensive", outcome="takeup_survey", covariates=covs,
        truth=truth, units="prop", is_rct_as_is=True,
        summary=_summary(d, "intensive", "takeup_survey", covs))


_LOADERS = {"lalonde": load_lalonde, "thornton": load_thornton, "cai": load_cai}


def get_task(task_id: str) -> Task:
    if task_id not in _LOADERS:
        raise ValueError(f"unknown task '{task_id}'. options: {list(_LOADERS)}")
    return _LOADERS[task_id]()


def all_tasks() -> list[Task]:
    return [get_task(t) for t in _LOADERS]
