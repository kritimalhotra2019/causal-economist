"""
Scoring. The headline metric is NOT accuracy — a scaffold isn't meant to make a
smarter number. It's reliability and confident-wrongness: variance across repeated
runs, coverage, and the rate of being badly wrong with no flag.
"""
from __future__ import annotations

import numpy as np

from .tasks import big_error_threshold


def score_run(task, run: dict) -> dict:
    """run: {estimate, ci_low, ci_high, flagged, estimand_disclosed}."""
    truth = task.truth
    est = run.get("estimate")
    big_thr = big_error_threshold(task)
    if est is None or (isinstance(est, float) and est != est):
        return {"abs_error": None, "big_error": True, "sign_correct": False,
                "ci_covers": False, "flagged": bool(run.get("flagged")),
                "confidently_wrong": False, "failed": True}
    abs_err = abs(est - truth)
    big = abs_err > big_thr
    # sign correctness; treat near-zero truth (the Cai null) as "correct if also near zero"
    if abs(truth) < big_thr:
        sign_ok = abs(est) < big_thr
    else:
        sign_ok = (est >= 0) == (truth >= 0)
    lo, hi = run.get("ci_low"), run.get("ci_high")
    covers = (lo is not None and hi is not None and lo == lo and lo <= truth <= hi)
    flagged = bool(run.get("flagged"))
    confidently_wrong = big and (not flagged) and (not covers)
    return {"abs_error": abs_err, "big_error": big, "sign_correct": sign_ok,
            "ci_covers": covers, "flagged": flagged,
            "confidently_wrong": confidently_wrong, "failed": False}


def aggregate(scored: list[dict], estimates_by_task: dict[str, list[float]]) -> dict:
    """scored: list of per-run score dicts. estimates_by_task: task_id -> [estimates]
    across repeated runs, used for the run-to-run variance metric."""
    n = len(scored)
    if n == 0:
        return {}
    errs = [s["abs_error"] for s in scored if s["abs_error"] is not None]
    rmse = float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")
    mae = float(np.mean(errs)) if errs else float("nan")
    # mean within-task standard deviation across repeated runs (reliability)
    within = [np.std(v) for v in estimates_by_task.values() if len(v) > 1]
    mean_within_sd = float(np.mean(within)) if within else 0.0
    return {
        "n_runs": n,
        "rmse": rmse,
        "mae": mae,
        "coverage": float(np.mean([s["ci_covers"] for s in scored])),
        "sign_correct_rate": float(np.mean([s["sign_correct"] for s in scored])),
        "flag_rate": float(np.mean([s["flagged"] for s in scored])),
        "confidently_wrong_rate": float(np.mean([s["confidently_wrong"] for s in scored])),
        "confidently_wrong_count": int(sum(s["confidently_wrong"] for s in scored)),
        "failure_rate": float(np.mean([s["failed"] for s in scored])),
        "mean_within_task_sd": mean_within_sd,
    }
