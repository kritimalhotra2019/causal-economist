# causal-mas

A multi-agent, human-in-the-loop system for **observational causal inference** —
and a pre-registered eval built to tell you whether the multi-agent part is actually
worth it.

It is a re-architecture of Netflix's open-source
[OCI agent](https://github.com/Netflix-Skunkworks/oci-agent)
([blog](https://netflixtechblog.medium.com/a-human-augmenting-agentic-workflow-for-causal-inference-4623f0a9c5af))
as a LangGraph state machine where the human is an **interactive arbiter at
estimand-changing moments**, not a post-hoc reviewer of artifacts.

---

## The problem

You have observational data and want a causal effect: does a job-training program
raise earnings? The naive thing — and what a one-shot "upload the CSV and ask"
tends to do — is compare the treated and untreated groups directly. When the groups
aren't comparable, that number can be wildly, **confidently** wrong.

The canonical demonstration is LaLonde's NSW program. The honest experimental answer
is **+\$1,794**. Comparing the trainees to a national survey of untreated people
gives **−\$8,498** — wrong *sign*, wrong by an order of magnitude — because the
comparison group is nothing like the trainees. The fix isn't a smarter model; it's a
**discipline**: check that the groups overlap, check covariate balance, trim to the
region where comparison is possible (which changes *what you're estimating*), and use
a doubly-robust estimator. After that discipline, this system lands at **+\$1,692**
with a 95% CI of roughly **[\$231, \$3,289]** that covers the truth.

The point of the scaffold is to make that discipline **enforced, inspectable, and
reproducible** — and to make the moments that require human judgement (approving a
change to the estimand, breaking a tie) explicit instead of buried.

## What's different from the Netflix agent

Netflix's agent is an actor–critic loop where the human inspects artifacts **after** a
draft→run→evaluate→revise cycle. Here:

- **The human is an interactive gate**, paused inside the graph (LangGraph
  `interrupt()`), consulted exactly when a decision changes the estimand or when the
  agents genuinely disagree — then the run resumes.
- **Three agents with non-overlapping mandates** that can disagree, with a principled
  rule for *who settles what*.
- **A credibility ledger**: every design choice, estimate, diagnostic, verdict, and
  human decision is appended to an audit trail you can read after the fact.
- **A pre-registered eval** ([`PREREGISTRATION.md`](PREREGISTRATION.md)) that can
  return "the multi-agent structure adds nothing."

## Architecture

```
                    ┌──────────────────┐
       START ─────▶ │ economist_design │  treatment / outcome / confounders / estimand   (LLM)
                    └────────┬─────────┘
                             ▼
              ┌────▶┌──────────────────┐
              │     │ analyst_execute  │  doubly-robust ATT + 4 diagnostics   (deterministic, retries)
              │     └────────┬─────────┘
              │              ▼
              │     ┌──────────────────┐
              │     │ critic_evaluate  │  3-tier verdict + remedy + conflict  (verdict = numbers)
              │     └────────┬─────────┘
              │       route  │
              │        ┌─────┴───────────────┐
              │   clean/agreed         problem OR disagreement
              │        │                     ▼
              │        │            ┌──────────────────┐
              │        │            │    human_gate    │  interrupt(): approve remedy / break tie
              │        │            └────────┬─────────┘  (fixed auto-policy in eval mode)
              │        │      route          │
              │        │        ┌────────────┴────────┐
              │        │   approve remedy           else
              │  ┌──────────────────┐                │
              └──│ economist_revise │                │   loop bounded by max_iterations
                 └──────────────────┘                ▼
                         │               ┌──────────────────┐
                         └──────────────▶│     finalize     │──▶ END   report + frozen ledger
                                         └──────────────────┘
```

### The three agents, and who settles what

- **Economist** — owns the *design*. Given the **full raw column list** (not a
  pre-cleaned covariate set), it classifies every column and selects only genuine
  pre-treatment confounders — refusing the outcome, identifiers, a second treatment
  arm, and any mediator/collider that sits downstream of treatment. This is the step
  where an LLM is load-bearing: the choice has no closed form and the diagnostics
  cannot catch a mediator. No code. (See "The design step" below.)
- **Data analyst** — owns *execution*: the only agent that touches data. Runs a
  deterministic doubly-robust (AIPW) estimator and the four diagnostics. Retries once
  on failure.
- **Critic** — owns *challenge*: a three-tier verdict (`fully_satisfactory` /
  `satisfactory_with_caveats` / `not_satisfactory`), a remedy when warranted, and a
  conflict flag.
- **Human** — arbiter **only** for irreducible judgement.

The conflict-resolution principle:

- **Facts are settled by the diagnostic numbers** — overlap and balance are
  thresholded deterministically; the LLM never overrules a pass/fail.
- **Design questions** (is this a confounder or a collider?) are settled by causal
  structure, in the economist's mandate.
- **Only genuine judgement with no ground truth** escalates to the human — e.g. "the
  overlap subsample is thin; report it or stand behind it?" There is deliberately **no
  fourth 'boss' agent**, because that would paper over disagreement instead of
  surfacing it. The loop is bounded: no convergence in `max_iterations` → escalate.

## The design step — why this is an agent, not a script

Once the design is fixed, everything downstream (estimate, diagnostics, trim, CI) is
deterministic — a function call, not an agent. The reasoning lives in **one place**:
choosing the adjustment set from an arbitrary column list. That is a genuine reasoning
problem with no closed form, and it is the most consequential one, because **adjusting
for a mediator biases the estimate and the diagnostics look fine either way** — you
cannot catch it from the numbers, only by reasoning that the variable sits causally
downstream of treatment.

So the economist is handed the **full real schema** — every column, with a neutral
description, including the ones it must refuse (the outcome, IDs, the second treatment
arm, post-treatment variables) — and must classify each and justify it
(`causal_mas/schema.py`, `economist_design` in `agents.py`). Hard rules are enforced
in code (it can never adjust for the literal treatment/outcome, a non-existent column,
or a non-numeric one); **every other call is the model's**, so a bad design flows
downstream and shows up in the diagnostics instead of being silently corrected. The
full include/exclude reasoning is written to the ledger.

**Honesty notes.**

- The offline `stub` economist is an **oracle** (it replays the correct classification)
  so the pipeline runs deterministically and the eval's MAS arm has zero variance. The
  *reasoning* is exercised only under `--provider nebius`.
- The three public teaching extracts have had their post-treatment variables **stripped**,
  so none contains a planted mediator. On real data here the economist's job is to refuse
  the outcome, identifiers, and (in Cai) the second randomized treatment arm. The harder
  mediator/collider reasoning is tested by a **reasoning probe** — a hypothetical data
  dictionary, not fabricated observations — in `verify_economist_llm.py`.
- The model's *judgement* (does it actually classify correctly?) can't be tested without
  a key, so it isn't tested in this repo — **you** verify it:
  `python verify_economist_llm.py` grades the model's own include/exclude flags against a
  hidden answer key, on all three datasets and the mediator/collider probe.



| dataset | what it is | what it demonstrates |
|---|---|---|
| **lalonde** | NSW job training vs a national survey (constructed observational) | **catches a broken comparison**: naive −\$8,498 → trimmed DR +\$1,692 (truth +\$1,794) |
| **thornton** | Malawi HIV-incentive RCT | **stays quiet**: balance passes, no false alarm, confirms +0.45 |
| **cai** | China weather-insurance RCT (intensive arm) | **confirms a precise null** (≈0) without inventing an effect |

A good referee both catches the bad case *and* doesn't cry wolf on the clean ones.

## Install

```bash
pip install -e .            # or: pip install -r requirements.txt
cp .env.example .env        # fill in keys only for the modes you'll use
```

The offline **stub** provider needs no keys or network.

## Quickstart — run the agent

```bash
# interactive: pauses at each human gate and asks you to decide (offline, deterministic)
python -m causal_mas.cli --task lalonde

# headless (gates auto-resolved)
python -m causal_mas.cli --task thornton --auto

# with real LLM agents via Nebius Token Factory (economist + critic)
python -m causal_mas.cli --task lalonde --provider nebius
```

On LaLonde you'll hit two gates: approve the trim (which narrows the estimand), then
break the economist-vs-critic tie about the thin overlap sample. State is checkpointed
to SQLite, so an interrupted run resumes.

### Providers: stub vs nebius

- `--provider stub` — fully deterministic, offline. The economist replays the correct
  classification (an **oracle**) and the critic uses the diagnostic rubric. Same input →
  same output every time. This is the eval's MAS reference arm (variance ~0, the point).
- `--provider nebius` — the economist reasons about the adjustment set and the critic
  writes an auditable rationale, via **Nebius Token Factory** (OpenAI-compatible).
  **Every pass/fail diagnostic is still computed in code** — the model reasons, it
  never decides a fact.

## Where the Week-3 requirements live

This is a real agent, not a one-shot or a RAG lookup:

- **Control flow** — conditional edges in `causal_mas/graph.py`
  (`_route_after_critic`, `_route_after_human`) with a bounded revise loop.
- **State across steps** — `causal_mas/state.py`: a typed state with an append-only
  ledger reducer; carried across nodes and persisted by the checkpointer.
- **Tool-failure recovery** — `analyst_execute` retries on error and, if estimation is
  unrecoverable, emits a `not_satisfactory` verdict that escalates rather than
  crashing.
- **Human-in-the-loop** — `human_gate` uses LangGraph `interrupt()`; the CLI resumes
  with `Command(resume=...)`. In the eval the same node switches to a fixed auto-policy.
- **Nebius Token Factory** — `causal_mas/llm.py` (`NebiusLLM`) powers the economist and
  critic.

## The eval — is the MAS actually worth it?

Read [`PREREGISTRATION.md`](PREREGISTRATION.md) first. Short version: four arms,
pre-registered, designed so it can disprove the system's value.

- **A — MAS** (this pipeline)
- **B — one-shot, no code** (the weak Netflix-style baseline)
- **C — one-shot WITH a code interpreter** — the decisive comparison: "I uploaded my
  data to a frontier model and asked." Its prompt contains **no diagnostic hints**.
- **ablation — single agent forced to run the checklist** — tests whether the
  *multi-agent* part adds anything beyond "always run the checks."

```bash
# offline, deterministic MAS only (no keys): proves the harness + MAS reliability
python -m eval.run_eval --arms A --bootstrap 20 --k 5 --mas-provider stub
python -m eval.analyze  --results eval/results.json

# the real comparison (needs an Anthropic key for the baselines)
python -m eval.run_eval --arms A,B,C,ablation --bootstrap 50 --k 20
python -m eval.analyze  --results eval/results.json
```

The headline metric is the **confidently-wrong rate** (badly wrong, unflagged, CI
misses truth) and **run-to-run variance** — not accuracy. `analyze.py` prints a
pre-registered verdict, including the honest "MAS adds no estimate-quality value;
keep it only for the audit trail" when the numbers say so.

## Repo layout

```
causal_mas/
  backend.py    deterministic engine: propensity, AIPW ATT, bootstrap CI, 4 diagnostics, trimming
  datasets.py   the three tasks; experimental truth computed from data
  schema.py     candidate column schemas + hidden answer key + the mediator/collider probe
  state.py      typed graph state + the credibility-ledger reducer
  llm.py        provider abstraction: NebiusLLM (real) + StubLLM (offline, deterministic)
  agents.py     economist (schema reasoning) / analyst / critic / revise / human_gate / finalize
  graph.py      LangGraph wiring: nodes, conditional edges, bounded loop, checkpointer
  cli.py        run one analysis (interactive or --auto)
eval/
  tasks.py      task library + stratified bootstrap
  arms.py       arms A / B / C / ablation + a local Python executor for the code arms
  metrics.py    scoring (confidently-wrong, coverage, variance, …)
  run_eval.py   run arms × tasks × K; skips arms whose keys are missing
  analyze.py    metrics table + pre-registered decision rules + plots
verify_economist_llm.py   run with your key: grades the model's column classification
PREREGISTRATION.md
```

## Limitations & extension points

- **The economist's reasoning is model-dependent and verified by you, not in CI.** The
  repo tests the plumbing (parsing, hard-rule guards, that a choice changes the estimate);
  whether the *model* classifies columns correctly needs a key and is checked by
  `verify_economist_llm.py`. Also, the curated RCT extracts contain no in-data mediator,
  so the mediator/collider case is exercised on a hypothetical schema probe rather than
  on these three datasets.
- **The economist-vs-critic conflict is generated deterministically in v1** (a thin
  overlap sample triggers the canonical LaLonde disagreement). Wiring *genuine*
  LLM-vs-LLM disagreement — the critic objects to a confounder, the economist defends
  it — is a clean extension; the prompts and routing already support an extra round.
- **Estimator.** A hand-rolled doubly-robust AIPW (transparent, light dependencies).
  EconML's `DRLearner` is a drop-in if you want cross-fitted ML nuisances.
- **Arm C / ablation need an API key** and run model-generated Python locally; run them
  in a sandbox. Arm A runs offline and deterministically.
- **Metrics across datasets mix units** (dollars vs proportions), so RMSE is read
  per-task; the unit-free confidently-wrong rate is the cross-task headline.
- **Three datasets are a demonstration, not a population claim** — see the analysis
  plan in the pre-registration.

## Provenance

LaLonde (1986); Dehejia & Wahba (1999, 2002) for the NSW/CPS construction; Thornton
(2008); Cai, de Janvry & Sadoulet (2015); Crump et al. (2009) for overlap trimming;
Cook, Shadish & Wong (2008) for within-study comparison. Data via the `causaldata`
package. Architecture adapted from Netflix's OCI agent.
