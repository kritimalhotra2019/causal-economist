"""Shared LangGraph state and the credibility ledger."""
from __future__ import annotations

import operator
import time
from typing import Annotated, Any, Optional, TypedDict


class AnalysisState(TypedDict, total=False):
    # inputs / config
    task_id: str
    provider: str
    plan: str
    available_covariates: list[str]
    truth: float
    units: str
    max_iterations: int
    auto_resolve: bool            # eval mode: resolve gates by fixed policy, no interrupt

    # produced as the graph runs
    design: dict[str, Any]        # economist: treatment, outcome, confounders, estimand, identification
    spec: dict[str, Any]          # what the analyst executes: trim_bounds, etc.
    results: dict[str, Any]       # estimate, ci, estimand, n_treated, n_control, naive
    diagnostics: dict[str, Any]
    critic: dict[str, Any]        # verdict, remedy, conflict, rationale
    iteration: int
    human_decision: Optional[dict[str, Any]]
    report: Optional[str]

    # the audit trail — appended to, never overwritten
    ledger: Annotated[list[dict[str, Any]], operator.add]


def event(kind: str, **fields) -> dict:
    """One ledger entry."""
    return {"t": round(time.time(), 3), "event": kind, **fields}
