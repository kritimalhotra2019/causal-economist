# Pre-registration — does the multi-agent system actually help?

This is written **before** running the eval so the result can go against the system.
The honest null is real: it is entirely possible that a single frontier model with a
code interpreter does observational causal inference just as reliably, in which case
the multi-agent scaffold is orchestration with no estimate-quality value and should be
dropped (or kept only for its audit trail).

## The question, split in two

"Is the MAS beneficial?" hides two separate claims. We test both.

- **H1 (enforced rigor).** The MAS produces fewer *confidently wrong* answers and
  lower *run-to-run variance* than a single model that can run its own code but is
  **not** told to check anything. → tested by **A vs C**.
- **H2 (the multi-agent part specifically).** The full economist/critic/human
  structure beats a **single agent forced to run the same diagnostic checklist**. If
  a plain checklist captures the value, the multi-agent framing is decoration.
  → tested by **A vs the ablation**.

H1 can be true while H2 is false. That outcome (rigor helps, but a checklist is
enough) is a legitimate and useful finding, and the eval is built to surface it.

## Arms

| arm | what it is | sees raw data? | runs code? | told to run diagnostics? |
|---|---|---|---|---|
| **A — MAS** | the LangGraph pipeline (economist → analyst → critic → human gate), human gate replaced by a fixed auto-policy | yes | yes (deterministic) | yes (its design) |
| **B — one-shot, no code** | one model call, group summaries only | no | no | no |
| **C — one-shot + code** | one model call with a `run_python` tool, raw data | yes | yes | **no** |
| **ablation** | single agent with the code tool, **told** to run the checklist + trim rule | yes | yes | yes |

B is the weak Netflix-style baseline (a single call with no tools). **C is the
decisive comparison** — it is "I uploaded my data to a frontier model and asked,"
which is the thing a skeptical user would actually do instead.

## Ground truth

All three datasets are RCTs, so the randomized/experimental contrast is the true
effect (within-study-comparison logic, LaLonde 1986; Cook, Shadish & Wong 2008).
This is what makes the eval scoreable — observational inference normally has no
answer key.

- **lalonde** — constructed observational (NSW trainees vs CPS survey controls);
  truth = the NSW experimental benchmark. Confounding is real and severe.
- **thornton** — clean RCT; truth = the randomized difference.
- **cai** — clean RCT (intensive-session arm); truth ≈ 0 (a precise null).

Power: each base task is expanded by a **stratified bootstrap** (resample treated and
control rows separately; truth unchanged) into many task instances.

## Metrics

Accuracy is **not** the headline — a scaffold is not meant to produce a smarter
number. Reliability is.

- **Primary: confidently-wrong rate** — fraction of runs that are badly wrong
  (|error| over a per-task threshold: $3,000 for earnings, 0.15 for proportions),
  **not flagged**, and whose CI does **not** cover the truth. Unit-free, comparable
  across tasks. This is the failure mode that matters: wrong *and* unaware.
- **Secondary:** run-to-run standard deviation on a fixed task instance (stability);
  95% CI coverage; sign-correctness; failure rate.
- **Reported but not decisive:** RMSE/MAE. These mix dollars and proportions across
  tasks, so they are read **per task**, not pooled.

## Decision rules

| if the eval shows… | verdict |
|---|---|
| C ≈ A on confidently-wrong **and** variance | MAS adds no estimate-quality value — orchestration only. Keep it solely if the audit trail / reproducibility justifies the complexity. |
| A beats C on confidently-wrong and/or variance (paired, significant) | enforced rigor is doing real work — H1 supported. |
| ablation ≈ A | the multi-agent part is decoration — **ship the checklist**, not the agents. H2 rejected. |
| A beats ablation | the economist/critic disagreement catches what a flat checklist misses — H2 supported. |

"≈" threshold: confidently-wrong rates within 0.02 **and** run-to-run SD
indistinguishable.

## Fairness guards (so this isn't a stitch-up)

- Identical tasks and identical core prompt across arms.
- **Arm C's prompt contains no diagnostic hints** — no mention of overlap, balance,
  trimming, or propensity. Telling it to check would smuggle the scaffold into the
  baseline and is the single easiest way to rig this result.
- The human gate in arm A is replaced by a fixed auto-policy in the eval, so the only
  difference between A and C is *enforced structure*, not a human in the loop.
- Report a range over a few baseline prompt phrasings, not one lucky prompt.
- Same estimator math is available to every arm.

## Analysis plan

Paired by task instance. Because bootstrap replicates of one dataset are correlated,
treat the three datasets as the unit of generalization: a mixed-effects model with
random intercepts for dataset (and task instance nested within), outcome = confidently-
wrong indicator and |error|. Bootstrap CIs on the A−C and A−ablation differences.
With only three underlying datasets, conclusions about *populations* of studies are
weak — this is a demonstration on three well-understood cases, not a population claim.

## What would falsify the value of the MAS

A null on **both** H1 and H2: arm C matches MAS on confidently-wrong and variance,
**and** the single-agent checklist matches the full pipeline. If that is the result,
the honest report is "a code-equipped frontier model is sufficient; the value here is
the audit trail, not the agents." The code prints exactly that verdict when the
numbers support it.
