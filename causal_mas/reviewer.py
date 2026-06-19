"""
Concept-note review: an economist-reviewer reads an impact-evaluation /
quasi-experimental / causal-inference / cost-benefit note and critiques it; an
independent critic then red-teams that review.

Qualitative companion to the data pipeline — same spirit (an expert proposes,
a critic challenges), but over prose instead of a dataframe. Used by the
Streamlit "Review a concept note" mode.
"""
from __future__ import annotations

import json
from typing import Any

MAX_CHARS = 24000   # keep the note within a sane prompt budget

REVIEW_SYSTEM = (
    "You are a senior development economist and impact-evaluation methodologist "
    "reviewing a concept note. Be rigorous, specific, and honest — name concrete "
    "weaknesses, do not rubber-stamp. Cover, as applicable: the study type "
    "(RCT / quasi-experimental [DiD, RDD, IV, matching] / observational / "
    "cost-benefit / other); the identification strategy and its key assumptions; "
    "internal-validity threats; external validity; statistical power, sample "
    "size and minimum detectable effect; outcome measurement; cost-benefit "
    "analysis; how to estimate the intervention's contribution to GDP; data and "
    "feasibility; and ethics. Ground every claim in what the note actually says; "
    "if something is missing, say so rather than inventing it. Reply with ONLY a "
    "JSON object."
)

CRITIC_SYSTEM = (
    "You are an independent senior referee. You are given a concept note and "
    "another economist's review of it. Red-team the REVIEW: what did it miss, "
    "what did it overstate or get wrong, and what is the single most important "
    "issue with the note that the review under-weighted? Be concise and concrete. "
    "Reply with ONLY a JSON object: {\"missed\": [\"...\"], \"overstated\": "
    "[\"...\"], \"most_important_issue\": \"...\", \"verdict\": \"<one line>\"}."
)


def extract_text(name: str, data: bytes) -> str:
    """Extract plain text from an uploaded .txt/.md/.pdf/.docx file."""
    lower = name.lower()
    if lower.endswith((".txt", ".md")):
        return data.decode("utf-8", errors="ignore")
    if lower.endswith(".pdf"):
        import io
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    if lower.endswith(".docx"):
        import io
        import docx
        d = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in d.paragraphs)
    raise ValueError(f"Unsupported file type: {name} (use .txt/.md/.pdf/.docx)")


def _review_user(text: str) -> str:
    return (
        "Review this concept note. Return JSON with these keys:\n"
        '{"study_type": "...", "summary": "<=2 sentences", '
        '"identification": {"strategy": "...", "key_assumptions": ["..."], '
        '"assessment": "..."}, '
        '"internal_validity_threats": ["..."], "external_validity": "...", '
        '"power_and_sample": "...", "measurement": "...", '
        '"cost_benefit": "<assessment, or what is missing + how to do it>", '
        '"gdp_contribution": "<concrete method to estimate contribution to GDP '
        'for this intervention: e.g. value-added, fiscal/employment multipliers, '
        'input-output>", "ethics_feasibility": "...", "strengths": ["..."], '
        '"recommendations": ["prioritized, concrete"], "overall": "bottom line"}\n\n'
        f"CONCEPT NOTE:\n{text[:MAX_CHARS]}"
        + ("\n\n[note truncated for length]" if len(text) > MAX_CHARS else ""))


def review_concept_note(econ_llm, critic_llm, text: str) -> dict[str, Any]:
    """Return {review: {...}, critique: {...}}.  econ_llm reviews, critic_llm red-teams."""
    if not text or not text.strip():
        raise ValueError("The document has no extractable text "
                         "(is it a scanned PDF with no text layer?).")
    review = econ_llm.complete_json(REVIEW_SYSTEM, _review_user(text))
    critique = {}
    try:
        payload = (f"CONCEPT NOTE (excerpt):\n{text[:MAX_CHARS]}\n\n"
                   f"REVIEW TO CRITIQUE:\n{json.dumps(review, indent=1)}")
        critique = critic_llm.complete_json(CRITIC_SYSTEM, payload)
    except Exception:
        pass
    return {"review": review, "critique": critique}
