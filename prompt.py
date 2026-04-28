"""
inference/src/prompt.py
────────────────────────
Prompt templates for financial document summarisation inference.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a financial analyst expert. Your task is to produce concise, "
    "accurate summaries of financial documents. Focus on: key financial metrics "
    "(revenue, EPS, margins, guidance), risk factors, and material events. "
    "Be factual. Do not invent numbers not present in the document."
)

DOC_INSTRUCTIONS = {
    "10-K":   "Summarize this Annual Report (10-K): business overview, revenue and profitability trends, major risk factors, and management outlook.",
    "10-Q":   "Summarize this Quarterly Report (10-Q): quarterly performance vs prior period, notable changes, and guidance updates.",
    "earnings_transcript": "Summarize this earnings call: headline results vs expectations, management commentary, key guidance figures, and notable Q&A highlights.",
    "8-K":    "Summarize this 8-K: material event type, financial impact, and any guidance changes.",
    "default":"Summarize this financial document, capturing key metrics, risks, and outlook.",
}


def build_inference_prompt(
    document: str,
    doc_type: str = "10-K",
    use_chat_template: bool = True,
) -> str:
    """
    Build the Mistral-Instruct formatted prompt for inference.
    """
    instruction = DOC_INSTRUCTIONS.get(doc_type, DOC_INSTRUCTIONS["default"])
    user_content = f"{instruction}\n\nDocument:\n{document.strip()}"

    # Mistral [INST] format
    return (
        f"<s>[INST] {SYSTEM_PROMPT}\n\n{user_content} [/INST]"
    )


def build_consolidation_prompt_text(
    chunk_summaries: list[str],
    doc_type: str = "10-K",
) -> str:
    """Prompt for merging multiple chunk summaries into a final summary."""
    sections = "\n\n".join(
        f"[Section {i+1}]\n{s.strip()}"
        for i, s in enumerate(chunk_summaries)
        if s.strip()
    )
    instruction = (
        f"Below are summaries of consecutive sections of a {doc_type} document. "
        "Combine them into a single coherent, non-redundant summary. "
        "Include all key financial metrics, risk factors, and forward guidance. "
        "Be concise (250-400 words)."
    )
    user_content = f"{instruction}\n\nSection summaries:\n{sections}"
    return f"<s>[INST] {SYSTEM_PROMPT}\n\n{user_content} [/INST]"
