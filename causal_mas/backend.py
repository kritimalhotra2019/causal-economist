"""
Deterministic causal-inference backend (the data analyst's tools).

No LLM anywhere in this file. Everything here is a plain function so results are
reproducible: the same spec on the same data always returns the same number.

Estimator: doubly-robust ATT (AIPW) with a logistic propensity model and an OLS
outcome model on the controls. EconML's DRLearner is a drop-in replacement if you
want cross-fitted ML nuisances; this hand-rolled version keeps dependencies light
and the numbers transparent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Default diagnostic thresholds (Netflix OCI conventions).
OVERLAP_BOUNDS = (0.1, 0.9)      # propensity must sit inside this band
OVERLAP_MAX_OUTSIDE = 0.10       # fail if >10% of units fall outside the band
SMD_THRESHOLD = 0.20             # standardized mean diff must be <= this
PLACEBO_SMD_THRESHOLD = 0.20     # balance on a pre-treatment placebo outcome
SENSITIVITY_THRESHOLD = 0.50     # leave-one-covariate-out: max relative move


def fit_propensity(df: pd.DataFrame, treat: str, covariates: list[str]) -> np.ndarray:
    """Logistic propensity on standardized covariates."""
    X = StandardScaler().fit_transform(df[covariates].values)
    model = LogisticRegression(max_iter=5000)
    model.fit(X, df[treat].values)
    return model.predict_proba(X)[:, 1]


def standardized_mean_diffs(df: pd.DataFrame, treat: str, covariates: list[str]) -> dict[str, float]:
    """|SMD| per covariate between treated and control."""
    out = {}
    for c in covariates:
        t = df.loc[df[treat] == 1, c]
        k = df.loc[df[treat] == 0, c]
        denom = np.sqrt((t.var() + k.var()) / 2)
        out[c] = float(abs(t.mean() - k.mean()) / denom) if denom > 0 else 0.0
    return out


def naive_difference(df: pd.DataFrame, treat: str, outcome: str) -> float:
    """Unadjusted difference in outcome means — the thing a one-shot tends to report."""
    return float(df.loc[df[treat] == 1, outcome].mean() - df.loc[df[treat] == 0, outcome].mean())


def att_aipw(df: pd.DataFrame, treat: str, outcome: str, covariates: list[str],
             ps: np.ndarray | None = None) -> float:
    """Doubly-robust ATT (AIPW): outcome model on controls + IPW correction."""
    if ps is None:
        ps = fit_propensity(df, treat, covariates)
    controls = df[df[treat] == 0]
    om = smf.ols(outcome + " ~ " + " + ".join(covariates), data=controls).fit()
    mu0 = om.predict(df).values
    t = df[treat].values
    y = df[outcome].values
    n1 = int((t == 1).sum())
    term_t = (y[t == 1] - mu0[t == 1]).sum()
    w = ps[t == 0] / (1 - ps[t == 0])
    term_c = (w * (y[t == 0] - mu0[t == 0])).sum()
    return float((term_t - term_c) / n1)


def bootstrap_ci(df: pd.DataFrame, treat: str, outcome: str, covariates: list[str],
                 n_boot: int = 200, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for the AIPW ATT."""
    rng = np.random.default_rng(seed)
    n = len(df)
    ests = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        b = df.iloc[idx]
        # need both arms present
        if b[treat].nunique() < 2:
            continue
        try:
            ests.append(att_aipw(b, treat, outcome, covariates))
        except Exception:
            continue
    if len(ests) < 20:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(ests, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def _sensitivity_robustness(df, treat, outcome, covariates, base_estimate) -> float:
    """Crude leave-one-covariate-out sensitivity: the largest relative move in the
    estimate when any single covariate is dropped. A proxy for how leaning the
    estimate is on the adjustment set."""
    if abs(base_estimate) < 1e-9 or len(covariates) < 2:
        return 0.0
    moves = []
    for c in covariates:
        rest = [x for x in covariates if x != c]
        try:
            e = att_aipw(df, treat, outcome, rest)
            moves.append(abs(e - base_estimate) / abs(base_estimate))
        except Exception:
            continue
    return float(max(moves)) if moves else 0.0


def run_diagnostics(df, treat, outcome, covariates, ps=None, placebo_var=None,
                    estimate=None) -> dict:
    """The four design diagnostics. Each returns a `passed` flag; overlap and
    balance are the gating checks, placebo gates only when a pre-treatment outcome
    is supplied, sensitivity is reported but not gating."""
    if ps is None:
        ps = fit_propensity(df, treat, covariates)
    t = df[treat].values

    # 1. Overlap
    outside = float(((ps < OVERLAP_BOUNDS[0]) | (ps > OVERLAP_BOUNDS[1])).mean())
    overlap = {"share_outside": outside, "bounds": OVERLAP_BOUNDS,
               "passed": outside <= OVERLAP_MAX_OUTSIDE}

    # 2. Balance
    smds = standardized_mean_diffs(df, treat, covariates)
    max_smd = max(smds.values()) if smds else 0.0
    worst = max(smds, key=smds.get) if smds else None
    balance = {"max_smd": max_smd, "worst_covariate": worst, "per_covariate": smds,
               "threshold": SMD_THRESHOLD, "passed": max_smd <= SMD_THRESHOLD}

    # 3. Placebo (only if a pre-treatment outcome is named)
    if placebo_var is not None and placebo_var in df.columns:
        pt = df.loc[df[treat] == 1, placebo_var]
        pk = df.loc[df[treat] == 0, placebo_var]
        denom = np.sqrt((pt.var() + pk.var()) / 2)
        psmd = float(abs(pt.mean() - pk.mean()) / denom) if denom > 0 else 0.0
        placebo = {"variable": placebo_var, "smd": psmd,
                   "threshold": PLACEBO_SMD_THRESHOLD, "passed": psmd <= PLACEBO_SMD_THRESHOLD}
    else:
        placebo = {"variable": None, "smd": None, "passed": True, "skipped": True}

    # 4. Sensitivity (reported, not gating)
    if estimate is None:
        estimate = att_aipw(df, treat, outcome, covariates, ps)
    rv = _sensitivity_robustness(df, treat, outcome, covariates, estimate)
    sensitivity = {"leave_one_out_max_move": rv, "threshold": SENSITIVITY_THRESHOLD,
                   "passed": rv <= SENSITIVITY_THRESHOLD}

    gating_passed = overlap["passed"] and balance["passed"] and placebo["passed"]
    return {"overlap": overlap, "balance": balance, "placebo": placebo,
            "sensitivity": sensitivity, "all_gating_passed": gating_passed}


def apply_trim(df, treat, covariates, bounds=OVERLAP_BOUNDS, ps=None):
    """Crump trimming to the common-support band. Returns the restricted frame and
    the relabeled estimand (effect on the overlap subpopulation)."""
    if ps is None:
        ps = fit_propensity(df, treat, covariates)
    keep = (ps >= bounds[0]) & (ps <= bounds[1])
    trimmed = df[keep].copy()
    return trimmed, "ATT (overlap subpopulation)"


def estimate_full(df, treat, outcome, covariates, placebo_var=None,
                  trim_bounds=None, n_boot=200, seed=0) -> dict:
    """One call that the analyst node uses: optionally trim, then estimate + CI +
    diagnostics. Returns everything the critic and ledger need."""
    estimand = "ATT (full sample)"
    work = df
    if trim_bounds is not None:
        ps_full = fit_propensity(df, treat, covariates)
        work, estimand = apply_trim(df, treat, covariates, trim_bounds, ps_full)

    ps = fit_propensity(work, treat, covariates)
    est = att_aipw(work, treat, outcome, covariates, ps)
    ci = bootstrap_ci(work, treat, outcome, covariates, n_boot=n_boot, seed=seed)
    diag = run_diagnostics(work, treat, outcome, covariates, ps=ps,
                           placebo_var=placebo_var, estimate=est)
    return {
        "estimate": est,
        "ci": ci,
        "estimand": estimand,
        "n_treated": int((work[treat] == 1).sum()),
        "n_control": int((work[treat] == 0).sum()),
        "naive": naive_difference(df, treat, outcome),
        "diagnostics": diag,
        "trim_bounds": trim_bounds,
    }
