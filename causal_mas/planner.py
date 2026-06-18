"""
NL planner: map a free-text causal question + an uploaded dataframe onto a
study design (treatment / outcome / per-column descriptions) the existing graph
can run.

This is a SETUP step, not a determination. The user confirms and can edit the
mapping before anything runs, and the economist still independently classifies
the candidate columns (refusing identifiers / mediators / colliders).
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from .schema import ColumnSpec

MAP_SYSTEM = (
    "You translate a user's causal question and a dataset's column list into a "
    "study design for selection-on-observables estimation. Identify the single "
    "TREATMENT column (a binary intervention/exposure) and the single OUTCOME "
    "column the question asks about. For EVERY column, write a short, neutral "
    "data-dictionary description (<=12 words); if a column appears measured AFTER "
    "the treatment or caused by it, say so explicitly (the analyst must be able "
    "to refuse mediators/colliders). Never invent columns. Reply with ONLY a "
    "JSON object."
)


def _columns_brief(df: pd.DataFrame, max_samples: int = 5) -> str:
    lines = []
    for c in df.columns:
        s = df[c]
        try:
            samples = list(pd.unique(s.dropna()))[:max_samples]
        except Exception:
            samples = []
        lines.append(f"  - {c} (dtype={s.dtype}, n_unique={s.nunique(dropna=True)}, "
                     f"e.g. {samples})")
    return "\n".join(lines)


def map_question(llm, question: str, df: pd.DataFrame) -> dict[str, Any]:
    """Return {treatment, outcome, units, study_description, schema:[ColumnSpec]}.

    Raises ValueError if the model's treatment/outcome aren't real columns."""
    user = (
        f"Causal question: {question}\n\n"
        f"Dataset columns ({len(df.columns)} cols, {len(df)} rows):\n"
        f"{_columns_brief(df)}\n\n"
        "Return JSON:\n"
        '{"treatment": "<col>", "outcome": "<col>", "units": "usd|prop|raw", '
        '"study_description": "<one plain sentence>", '
        '"columns": [{"name": "<col>", "description": "<=12 words>"}]}'
    )
    out = llm.complete_json(MAP_SYSTEM, user)

    cols = set(df.columns)
    treatment, outcome = out.get("treatment"), out.get("outcome")
    if treatment not in cols or outcome not in cols:
        raise ValueError(
            f"Model proposed treatment={treatment!r}, outcome={outcome!r}, but "
            "both must be real columns. Edit the mapping manually below.")

    schema = [ColumnSpec(d["name"], d.get("description", ""))
              for d in out.get("columns", []) if d.get("name") in cols]
    have = {c.name for c in schema}
    for c in df.columns:                       # ensure every column has a spec
        if c not in have:
            schema.append(ColumnSpec(c, ""))

    return {"treatment": treatment, "outcome": outcome,
            "units": out.get("units", "raw"),
            "study_description": out.get("study_description", question),
            "schema": schema}
