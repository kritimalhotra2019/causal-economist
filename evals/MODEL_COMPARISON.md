# Causal-MAS — model comparison

All model evals run for this project, consolidated. Models run on Nebius Token
Factory. This is causal-mas only (not the separate Amara tool-calling eval).

## 1. Quantitative — economist covariate-design benchmark

`verify_economist_llm.py` grades each model's adjustment-set choices against an
answer key over 3 real datasets (LaLonde, Thornton, Cai) + 1 planted
mediator/collider probe. This is the decisive, gradeable eval.

| Model | LaLonde | Thornton | Cai | Probe | Total | Runs | Failure mode |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **deepseek-ai/DeepSeek-V4-Pro** | ✅ | ✅ | ✅ | ✅ | **4/4** | 2× (stable) | — |
| zai-org/GLM-5.1 | ✅ | ✅ | ❌ | ✅ | 3/4 | 1× | Cai: dropped all 8 confounders (under-adjust) |
| Qwen/Qwen3-235B-A22B-Thinking-2507-fast | ✅ | ✅ | ❌ | ✅ | 3/4 | 1× | Cai: included `village` (identifier) |
| moonshotai/Kimi-K2.6 | ✅ | ❌ | ✅ | ✅ | 3/4 | 1× | Thornton: dropped `distvct`/`age`/`hiv2004` |
| **zai-org/GLM-5.2** | ✅ | ❌ | ❌ | ✅ | **2/4** | 1× | Thornton (drop confounders) + Cai (`village`) |
| nvidia/Nemotron-3-Ultra-550b-a55b | ✅ | ⚠️ | ⚠️ | ✅ | **2/4 ↔ 4/4** | 3× | UNSTABLE run-to-run (4/4 once, 2/4 twice) |

Notes:
- Every model passes the **probe** (none adjusted for a post-treatment
  mediator/collider) — the discriminators are the real datasets.
- **Cai is hardest** (4 of 6 fail it), in two opposite failure modes:
  under-adjustment (dropping real confounders → bias) vs. including an
  identifier like `village`.
- **DeepSeek-V4-Pro is the only clean, stable 4/4.**

## 2. Qualitative — concept-note review (head-to-head)

One flawed RCT note, reviewer role. n=1, subjective — illustrative, NOT
gradeable. Only these two were run as reviewer.

| Model | Caught fatal flaw | Threats found | Recommendations | Verdict |
|---|:---:|:---:|:---:|---|
| zai-org/GLM-5.2 | ✅ (2-district confound) | 7 (+ single-cluster inference, gen. equilibrium) | 10 | excellent, slightly more thorough |
| deepseek-ai/DeepSeek-V4-Pro | ✅ (2-district confound) | 6 (+ maturation/history) | 6 | excellent, more concise |

Result: roughly tied; GLM-5.2 marginally more granular. Not proven dominance.

## 3. Pipeline arm eval (context — not a model comparison)

`eval/run_eval --arms A,B` on nebius: arm A (MAS pipeline) vs arm B (one-shot
baseline). lalonde: MAS $1,692 (truth $1,794) vs baseline −$8,498;
thornton/cai baseline competitive. MAS rescues the confounded case; neutral
where confounding isn't the problem. Ran on the then-current models.

## Resulting config (split by job)

| Role | Model | Basis |
|---|---|---|
| Data economist | DeepSeek-V4-Pro | 4/4 quant, stable |
| Critic (all modes) | DeepSeek-V4-Pro | low-stakes (writes rationale only) |
| Concept-note reviewer | GLM-5.2 | more thorough qualitative style |

## Confidence caveat

The quant benchmark is 4 schemas, mostly single-run per model (DeepSeek 2×,
Nemotron 3×) — directional, not statistically powered. A `--k` / bootstrap
sweep would firm it up. The qualitative comparison is a single note.
