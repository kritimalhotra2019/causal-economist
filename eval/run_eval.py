"""
Run the experiment.

  python -m eval.run_eval --arms A           --k 1                  # offline, deterministic MAS
  python -m eval.run_eval --arms A,B,C,ablation --bootstrap 50 --k 20  # the real eval (needs keys)

Arm A (MAS) runs with --mas-provider (stub|nebius). B/C/ablation use
--baseline-provider (anthropic|nebius) and are skipped if the key is missing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

from dotenv import load_dotenv

from causal_mas.datasets import get_task
from . import arms as A
from .tasks import make_task_library

# Load .env so NEBIUS_API_KEY / ANTHROPIC_API_KEY / *_MODEL are picked up without
# manual export — matches causal_mas.cli. The baseline arms gate on these keys.
load_dotenv()

ARM_FNS = {"A": "mas", "B": "oneshot_nocode", "C": "oneshot_code", "ablation": "ablation"}


def _baseline_available(provider):
    if provider == "anthropic":
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    if provider == "nebius":
        return bool(os.environ.get("NEBIUS_API_KEY"))
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A", help="comma list of A,B,C,ablation")
    ap.add_argument("--tasks", default="lalonde,thornton,cai")
    ap.add_argument("--bootstrap", type=int, default=0, help="bootstrap replicates per base task")
    ap.add_argument("--k", type=int, default=1, help="repeated runs per task per arm")
    ap.add_argument("--mas-provider", default="stub", choices=["stub", "nebius"])
    ap.add_argument("--baseline-provider", default="anthropic", choices=["anthropic", "nebius"])
    ap.add_argument("--out", default="eval/results.json")
    args = ap.parse_args()

    requested = [a.strip() for a in args.arms.split(",") if a.strip()]
    task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
    jobs = make_task_library(task_ids, n_bootstrap=args.bootstrap)

    # which arms can actually run?
    active, caller = [], None
    for a in requested:
        if a == "A":
            active.append(a)
        elif _baseline_available(args.baseline_provider):
            active.append(a)
        else:
            print(f"[skip] arm {a}: no {args.baseline_provider} key set", file=sys.stderr)
    if any(a in ("B", "C", "ablation") for a in active):
        caller = A.make_baseline_caller(args.baseline_provider)

    print(f"arms={active}  tasks={task_ids}  bootstrap={args.bootstrap}  k={args.k}  "
          f"jobs={len(jobs)}  mas={args.mas_provider}", file=sys.stderr)

    records = []
    for (task, tseed) in jobs:
        for a in active:
            for run_i in range(args.k):
                try:
                    if a == "A":
                        out = A.arm_mas(task, provider=args.mas_provider, seed=tseed * 100 + run_i)
                    elif a == "B":
                        out = A.arm_oneshot_nocode(task, caller, seed=run_i)
                    elif a == "C":
                        out = A.arm_oneshot_code(task, caller, seed=run_i)
                    elif a == "ablation":
                        out = A.arm_ablation(task, caller, seed=run_i)
                except Exception as e:  # keep going; record the failure
                    out = {"arm": a, "estimate": None, "error": f"{e}", "flagged": False}
                    traceback.print_exc()
                out.update({"task_id": task.id, "task_seed": tseed, "run": run_i,
                            "truth": task.truth, "units": task.units})
                records.append(out)
                print(f"  {task.id:9s} {a:18s} run{run_i}  est={out.get('estimate')}", file=sys.stderr)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nwrote {len(records)} records -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
