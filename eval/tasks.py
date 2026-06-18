"""
The eval task library.

Three base tasks become many via a STRATIFIED bootstrap: resample treated and
control rows separately (keeps n_treated stable) so each replicate is a fresh
estimation task drawn from the same real rows, with the same experimental truth.
Nothing is fabricated — this is the LaLonde within-study-comparison logic scaled up.
"""
from __future__ import annotations

import copy
import numpy as np
import pandas as pd

from causal_mas.datasets import get_task, all_tasks, _summary, Task

BIG_ERROR = {"usd": 3000.0, "prop": 0.15}


def big_error_threshold(task: Task) -> float:
    return BIG_ERROR[task.units]


def _with_df(task: Task, df) -> Task:
    t = copy.copy(task)
    t.df = df
    t.summary = _summary(df, task.treat, task.outcome, task.covariates)
    return t


def bootstrap_task(task: Task, seed: int) -> Task:
    """One stratified-bootstrap replicate. Truth is unchanged (it's the experimental
    benchmark, not a property of the resampled frame)."""
    rng = np.random.default_rng(seed)
    df = task.df
    treated = df[df[task.treat] == 1]
    control = df[df[task.treat] == 0]
    ti = rng.integers(0, len(treated), len(treated))
    ci = rng.integers(0, len(control), len(control))
    boot = pd.concat([treated.iloc[ti], control.iloc[ci]], ignore_index=True)
    return _with_df(task, boot)


def make_task_library(task_ids=None, n_bootstrap=0):
    """Return a list of (task, seed) jobs. n_bootstrap=0 -> just the base tasks."""
    ids = task_ids or ["lalonde", "thornton", "cai"]
    jobs = []
    for tid in ids:
        base = get_task(tid)
        jobs.append((base, 0))
        for b in range(1, n_bootstrap + 1):
            jobs.append((bootstrap_task(base, seed=1000 + b), 1000 + b))
    return jobs
