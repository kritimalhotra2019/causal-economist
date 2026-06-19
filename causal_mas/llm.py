"""
LLM providers.

`nebius`  -> Nebius Token Factory, OpenAI-compatible (the bootcamp requirement).
            Used for the economist's reasoning and the critic's written judgement.
`stub`    -> no network, fully deterministic. Used for offline tests and as the
            eval's MAS reference arm (its variance should be ~0, which is the point).

Only two things ever go to the model: the economist's design proposal and the
critic's rationale/conflict detection. Every pass/fail diagnostic is computed in
backend.py — the model never decides a fact.
"""
from __future__ import annotations

import json
import os
from typing import Any

NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
DEFAULT_ECONOMIST_MODEL = "deepseek-ai/DeepSeek-V4-Pro"   # quantitative data design (4/4)
DEFAULT_CRITIC_MODEL    = "deepseek-ai/DeepSeek-V4-Pro"   # red-team / rationale
DEFAULT_REVIEWER_MODEL  = "zai-org/GLM-5.2"               # qualitative concept-note review
DEFAULT_NEBIUS_MODEL    = DEFAULT_ECONOMIST_MODEL   # single-model / verifier fallback


class StubLLM:
    """Deterministic, offline. Nodes branch on `provider == 'stub'` and use their
    own canned logic, so this object is just a marker plus a no-op JSON method."""
    provider = "stub"
    model = "stub"

    def complete_json(self, system: str, user: str, **_) -> dict[str, Any]:
        return {}


class NebiusLLM:
    """Calls Nebius Token Factory via the OpenAI-compatible client."""
    provider = "nebius"

    def __init__(self, model: str | None = None, temperature: float = 0.1):
        from openai import OpenAI  # imported lazily so stub runs need no openai
        api_key = os.environ.get("NEBIUS_API_KEY")
        if not api_key:
            raise RuntimeError("NEBIUS_API_KEY is not set (see .env.example).")
        self.client = OpenAI(base_url=NEBIUS_BASE_URL, api_key=api_key)
        self.model = model or os.environ.get("NEBIUS_MODEL", DEFAULT_NEBIUS_MODEL)
        self.temperature = temperature

    def complete_json(self, system: str, user: str, **_) -> dict[str, Any]:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        text = resp.choices[0].message.content or ""
        return _parse_json(text)


ROLE_DEFAULTS = {
    "economist": DEFAULT_ECONOMIST_MODEL,
    "critic": DEFAULT_CRITIC_MODEL,
    "reviewer": DEFAULT_REVIEWER_MODEL,
}


def make_llm(provider: str, model: str | None = None, role: str | None = None) -> Any:
    if provider == "stub":
        return StubLLM()
    if provider == "nebius":
        # precedence: explicit model arg > NEBIUS_MODEL env (single-model override)
        #             > per-role default > generic fallback.
        chosen = (model or os.environ.get("NEBIUS_MODEL")
                  or ROLE_DEFAULTS.get(role) or DEFAULT_NEBIUS_MODEL)
        return NebiusLLM(model=chosen)
    raise ValueError(f"unknown provider '{provider}' (use 'stub' or 'nebius')")


def _parse_json(text: str) -> dict[str, Any]:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1] if "```" in t[3:] else t.strip("`")
        t = t[4:].strip() if t.lower().startswith("json") else t
    a, b = t.find("{"), t.rfind("}")
    if a >= 0 and b > a:
        t = t[a:b + 1]
    try:
        return json.loads(t)
    except Exception:
        return {}
