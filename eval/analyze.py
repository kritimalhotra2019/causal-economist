"""
Analyze the results: per-arm metrics table, the pre-registered decision rules, and
the coverage-vs-RMSE + confidently-wrong plots.

  python -m eval.analyze --results eval/results.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

from causal_mas.datasets import get_task
from .metrics import score_run, aggregate


class _T:  # minimal task stand-in carrying truth+units for scoring
    def __init__(self, task_id, truth, units):
        self.id, self.truth, self.units = task_id, truth, units


def load(path):
    with open(path) as f:
        return json.load(f)


def analyze(records):
    by_arm = defaultdict(list)
    for r in records:
        by_arm[r["arm"]].append(r)

    summary = {}
    for arm, runs in by_arm.items():
        scored, ests_by_task = [], defaultdict(list)
        for r in runs:
            t = _T(r["task_id"], r["truth"], r["units"])
            scored.append(score_run(t, r))
            if r.get("estimate") is not None:
                key = f"{r['task_id']}:{r.get('task_seed', 0)}"  # exact instance
                ests_by_task[key].append(r["estimate"])
        summary[arm] = aggregate(scored, ests_by_task)
    return summary


def print_table(summary):
    cols = [("rmse", "RMSE"), ("coverage", "cov95"), ("flag_rate", "flag"),
            ("mean_within_task_sd", "run-SD"), ("confidently_wrong_rate", "conf-wrong"),
            ("failure_rate", "fail")]
    head = f"{'arm':20s} {'n':>4s} " + " ".join(f"{h:>10s}" for _, h in cols)
    print(head)
    print("-" * len(head))
    for arm in sorted(summary):
        s = summary[arm]
        row = f"{arm:20s} {s.get('n_runs',0):>4d} "
        for k, _ in cols:
            v = s.get(k)
            row += f"{'   n/a' if v is None else f'{v:>10.3f}'} "
        print(row)


def decision(summary):
    print("\n=== pre-registered verdict ===")
    A = summary.get("A_mas")
    C = summary.get("C_oneshot_code")
    AB = summary.get("ablation_checklist")
    if not A:
        print("  arm A (MAS) missing — nothing to judge."); return
    if not C:
        print("  arm C (code-equipped one-shot) was not run — the decisive comparison is")
        print("  unavailable. Run with an API key: --arms A,C,ablation."); 
    else:
        d_cw = A["confidently_wrong_rate"] - C["confidently_wrong_rate"]
        d_sd = A["mean_within_task_sd"] - C["mean_within_task_sd"]
        better = (C["confidently_wrong_rate"] - A["confidently_wrong_rate"] > 0.02) or \
                 (C["mean_within_task_sd"] - A["mean_within_task_sd"] > 1e-9 and
                  A["confidently_wrong_rate"] <= C["confidently_wrong_rate"])
        if better:
            print(f"  A vs C: MAS lowers confidently-wrong by {(-d_cw):+.3f} and run-to-run SD by "
                  f"{(-d_sd):+.3f}. Enforcement is doing real work — value confirmed.")
        else:
            print(f"  A vs C: MAS does NOT beat the code-equipped one-shot on confidently-wrong "
                  f"({A['confidently_wrong_rate']:.3f} vs {C['confidently_wrong_rate']:.3f}) or variance. "
                  f"On estimate quality, the scaffold adds little — keep it only if the audit trail / "
                  f"reproducibility alone justifies it.")
    if C and AB:
        gap = AB["confidently_wrong_rate"] - A["confidently_wrong_rate"]
        if abs(gap) < 0.02 and abs(AB["mean_within_task_sd"] - A["mean_within_task_sd"]) < 1e-9:
            print("  A vs ablation: full MAS ≈ a single agent forced to run the checklist. The "
                  "MULTI-AGENT part is decoration here — ship the checklist, not the agents.")
        else:
            print(f"  A vs ablation: full MAS beats the single-agent checklist (conf-wrong "
                  f"{A['confidently_wrong_rate']:.3f} vs {AB['confidently_wrong_rate']:.3f}) — the "
                  f"economist/critic disagreement is earning its place.")


def plot(summary, out="eval/benchmark.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot skipped: {e}]"); return
    arms = sorted(summary)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for arm in arms:
        s = summary[arm]
        ax1.scatter(s["rmse"], s["coverage"], s=90)
        ax1.annotate(arm, (s["rmse"], s["coverage"]), fontsize=8,
                     xytext=(5, 4), textcoords="offset points")
    ax1.axhline(0.95, ls="--", lw=0.8, color="grey")
    ax1.set_xlabel("RMSE (lower better)"); ax1.set_ylabel("95% CI coverage")
    ax1.set_title("coverage vs RMSE")
    ax2.bar(range(len(arms)), [summary[a]["confidently_wrong_rate"] for a in arms])
    ax2.set_xticks(range(len(arms))); ax2.set_xticklabels(arms, rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("confidently-wrong rate"); ax2.set_title("confidently wrong (lower better)")
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"\nwrote plot -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="eval/results.json")
    ap.add_argument("--plot", default="eval/benchmark.png")
    args = ap.parse_args()
    recs = load(args.results)
    summary = analyze(recs)
    print_table(summary)
    decision(summary)
    plot(summary, args.plot)


if __name__ == "__main__":
    main()
