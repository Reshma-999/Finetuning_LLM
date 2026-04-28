"""
training/src/dataset.py
────────────────────────
HuggingFace Dataset builder for financial document summarisation.

Each training sample is a (document, summary) pair formatted as a
Mistral chat-style prompt:

  [INST] You are a financial analyst expert. Summarize the following
  {doc_type} document, focusing on key financial metrics, risks, and
  management outlook. Provide a concise, factual summary.

  Document:
  {document_chunk}
  [/INST]
  {summary}

Sources:
  • SEC 10-K filings (EDGAR)
  • Quarterly earnings call transcripts
  • 10-Q quarterly reports
  • 8-K current reports
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a financial analyst expert. Your task is to produce concise, "
    "accurate summaries of financial documents. Focus on: key financial metrics "
    "(revenue, EPS, margins), risk factors, management guidance, and material events. "
    "Be factual and precise. Do not hallucinate numbers not present in the document."
)

DOC_TYPE_INSTRUCTIONS = {
    "10-K": (
        "Summarize this Annual Report (10-K), covering: business overview, "
        "revenue and profitability trends, major risk factors, and management outlook."
    ),
    "10-Q": (
        "Summarize this Quarterly Report (10-Q), focusing on quarterly financial "
        "performance vs. prior periods, notable changes, and guidance updates."
    ),
    "earnings_transcript": (
        "Summarize this earnings call transcript, capturing: headline results "
        "vs. expectations, management commentary, key guidance figures, and "
        "analyst Q&A highlights."
    ),
    "8-K": (
        "Summarize this 8-K filing, identifying the material event type, "
        "financial impact, and any guidance changes."
    ),
}


def format_prompt(
    document: str,
    summary: str,
    doc_type: str = "10-K",
    tokenizer: Optional[PreTrainedTokenizer] = None,
    include_summary: bool = True,
) -> str:
    """
    Format a (document, summary) pair as a Mistral instruction prompt.

    Uses the [INST] ... [/INST] format expected by Mistral-Instruct models.
    """
    instruction = DOC_TYPE_INSTRUCTIONS.get(doc_type, DOC_TYPE_INSTRUCTIONS["10-K"])

    user_content = (
        f"{instruction}\n\n"
        f"Document:\n{document.strip()}"
    )

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]
        if include_summary:
            messages.append({"role": "assistant", "content": summary.strip()})
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=not include_summary,
        )

    # Fallback: manual [INST] format
    prompt = (
        f"[INST] {SYSTEM_PROMPT}\n\n{user_content} [/INST]"
    )
    if include_summary:
        prompt += f"\n{summary.strip()}"
    return prompt


# ── Dataset builder ───────────────────────────────────────────────────────────

@dataclass
class FinancialDatasetConfig:
    train_file: str = "training/data/splits/train.jsonl"
    eval_file: str  = "training/data/splits/val.jsonl"
    test_file: str  = "training/data/splits/test.jsonl"
    max_seq_length: int = 4096
    max_document_tokens: int = 3500    # Leave room for summary + prompt overhead
    text_column: str = "text"
    seed: int = 42
    val_size: float = 0.05
    test_size: float = 0.05


class FinancialDatasetBuilder:
    """Build train/val/test splits for financial summarisation fine-tuning."""

    def __init__(
        self,
        config: FinancialDatasetConfig,
        tokenizer: PreTrainedTokenizer,
    ):
        self.config    = config
        self.tokenizer = tokenizer

    def load_raw(self, path: str) -> list[dict]:
        """Load JSONL file with fields: document, summary, doc_type, ticker, year."""
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        logger.info(f"Loaded {len(records)} records from {path}")
        return records

    def format_records(self, records: list[dict]) -> list[dict]:
        """Format each record as a training example."""
        formatted = []
        skipped   = 0

        for rec in records:
            document = rec.get("document", "")
            summary  = rec.get("summary", "")
            doc_type = rec.get("doc_type", "10-K")

            if not document or not summary:
                skipped += 1
                continue

            # Truncate document to max tokens
            doc_tokens = self.tokenizer.encode(document)
            if len(doc_tokens) > self.config.max_document_tokens:
                document = self.tokenizer.decode(
                    doc_tokens[:self.config.max_document_tokens],
                    skip_special_tokens=True,
                )

            text = format_prompt(
                document=document,
                summary=summary,
                doc_type=doc_type,
                tokenizer=self.tokenizer,
                include_summary=True,
            )

            # Skip if total exceeds max_seq_length
            total_tokens = len(self.tokenizer.encode(text))
            if total_tokens > self.config.max_seq_length:
                skipped += 1
                continue

            formatted.append({
                "text":     text,
                "document": document,
                "summary":  summary,
                "doc_type": doc_type,
                "ticker":   rec.get("ticker", ""),
                "year":     rec.get("year", ""),
                "n_tokens": total_tokens,
            })

        logger.info(
            f"Formatted {len(formatted)} samples ({skipped} skipped as too long or empty)"
        )
        return formatted

    def build(self) -> DatasetDict:
        """Build train/val/test DatasetDict from JSONL files."""
        splits = {}

        for split, path in [
            ("train", self.config.train_file),
            ("validation", self.config.eval_file),
            ("test", self.config.test_file),
        ]:
            if Path(path).exists():
                records  = self.load_raw(path)
                formatted = self.format_records(records)
                splits[split] = Dataset.from_list(formatted)
                logger.info(
                    f"Split '{split}': {len(formatted)} examples, "
                    f"avg tokens: {sum(e['n_tokens'] for e in formatted) / max(1, len(formatted)):.0f}"
                )
            else:
                logger.warning(f"Split file not found: {path}")

        return DatasetDict(splits)


# ── Tokenization ──────────────────────────────────────────────────────────────

def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int = 4096,
    text_column: str = "text",
) -> DatasetDict:
    """Tokenize the dataset for SFTTrainer (returns input_ids + attention_mask)."""

    def tokenize_fn(batch):
        tokenized = tokenizer(
            batch[text_column],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_attention_mask=True,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    return dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=[
            c for c in dataset["train"].column_names
            if c not in ["input_ids", "attention_mask", "labels"]
        ],
        num_proc=4,
        desc="Tokenizing",
    )


# ── Data statistics ───────────────────────────────────────────────────────────

def dataset_stats(dataset: DatasetDict) -> dict:
    stats = {}
    for split, ds in dataset.items():
        if "n_tokens" not in ds.column_names:
            continue
        tokens = ds["n_tokens"]
        stats[split] = {
            "n_examples":  len(ds),
            "mean_tokens": sum(tokens) / len(tokens),
            "max_tokens":  max(tokens),
            "min_tokens":  min(tokens),
            "p95_tokens":  sorted(tokens)[int(0.95 * len(tokens))],
        }
        if "doc_type" in ds.column_names:
            from collections import Counter
            stats[split]["doc_type_dist"] = dict(Counter(ds["doc_type"]))
    return stats
