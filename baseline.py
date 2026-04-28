"""
evaluation/src/baseline.py
───────────────────────────
GPT-3.5-turbo zero-shot baseline for comparison.

The baseline uses the same system prompt and document as the fine-tuned model
but relies entirely on GPT-3.5's pre-trained capabilities (zero-shot).

GPT-3.5 ROUGE-L baseline: 0.385
Our fine-tuned Mistral-7B: 0.430  (+11.7%)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import tiktoken
from openai import OpenAI
from tqdm import tqdm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a financial analyst expert. Produce a concise, accurate summary of the "
    "provided financial document. Focus on: key financial metrics (revenue, EPS, margins), "
    "risk factors, management guidance, and material events. Be factual and precise."
)

DOC_INSTRUCTIONS = {
    "10-K": "Summarize this Annual Report (10-K): business overview, revenue trends, risk factors, and outlook.",
    "10-Q": "Summarize this Quarterly Report (10-Q): quarterly performance, notable changes, and guidance.",
    "earnings_transcript": "Summarize this earnings call: headline results, management commentary, guidance, and key Q&A highlights.",
    "8-K": "Summarize this 8-K: material event type, financial impact, and guidance changes.",
}


class GPT35Baseline:
    """
    Runs GPT-3.5-turbo on the test set as a zero-shot baseline.
    Rate-limited to avoid API throttling.
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo-1106",
        max_tokens: int = 512,
        temperature: float = 0.1,
        max_doc_tokens: int = 12_000,   # Stay within GPT-3.5 16K context
        rate_limit_rps: float = 3.0,
    ):
        self.client         = OpenAI()
        self.model          = model
        self.max_tokens     = max_tokens
        self.temperature    = temperature
        self.max_doc_tokens = max_doc_tokens
        self.min_interval   = 1.0 / rate_limit_rps
        self._enc           = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def _truncate(self, text: str, max_tokens: int) -> str:
        tokens = self._enc.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self._enc.decode(tokens[:max_tokens])

    def summarise_one(self, document: str, doc_type: str = "10-K") -> str:
        instruction = DOC_INSTRUCTIONS.get(doc_type, DOC_INSTRUCTIONS["10-K"])
        doc_trunc   = self._truncate(document, self.max_doc_tokens)

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": f"{instruction}\n\nDocument:\n{doc_trunc}"},
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(f"OpenAI error (attempt {attempt+1}): {e} — retrying in {wait}s")
                time.sleep(wait)

        return ""

    def run_batch(
        self,
        records: list[dict],
        batch_size: int = 1,
    ) -> list[str]:
        """Run GPT-3.5 on all records, respecting rate limits."""
        predictions = []
        last_call   = 0.0

        for rec in tqdm(records, desc="GPT-3.5 baseline"):
            # Rate limiting
            elapsed = time.time() - last_call
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)

            pred = self.summarise_one(
                document=rec["document"],
                doc_type=rec.get("doc_type", "10-K"),
            )
            predictions.append(pred)
            last_call = time.time()

        return predictions


class GPT4Baseline:
    """GPT-4 baseline (typically run on a smaller subset due to cost)."""

    def __init__(self, model: str = "gpt-4-turbo-preview"):
        self.client      = OpenAI()
        self.model       = model
        self.enc         = tiktoken.encoding_for_model("gpt-4")
        self.max_doc_tokens = 100_000   # GPT-4 128K context

    def summarise_one(self, document: str, doc_type: str = "10-K") -> str:
        instruction = DOC_INSTRUCTIONS.get(doc_type, DOC_INSTRUCTIONS["10-K"])
        tokens = self.enc.encode(document)
        if len(tokens) > self.max_doc_tokens:
            document = self.enc.decode(tokens[:self.max_doc_tokens])

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": f"{instruction}\n\nDocument:\n{document}"},
                ],
                max_tokens=512,
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"GPT-4 API error: {e}")
            return ""
