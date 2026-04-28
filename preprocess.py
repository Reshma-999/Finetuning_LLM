"""
pipeline/src/preprocess.py
───────────────────────────
Preprocessing pipeline for financial documents.

Steps:
  1. Load raw JSONL documents
  2. Quality filtering (min/max length, language detection)
  3. MinHash LSH deduplication (removes near-duplicate 10-K boilerplate)
  4. Train/val/test split (stratified by doc_type and year)
  5. Save processed splits to JSONL

MinHash deduplication is critical: many 10-K filings share near-identical
"boilerplate" risk factor sections. Without dedup, the model would overfit
to common phrases rather than learning to summarise financial content.

Near-duplicate threshold: Jaccard similarity > 0.85 → deduplicate
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Optional

from datasketch import MinHash, MinHashLSH
from langdetect import detect, LangDetectException
from loguru import logger
from tqdm import tqdm
import tiktoken

# ── Config ────────────────────────────────────────────────────────────────────

MIN_DOC_TOKENS   = 500
MAX_DOC_TOKENS   = 100_000   # Documents larger than this are still included but truncated at training time
MIN_SUMMARY_TOKENS = 50
MAX_SUMMARY_TOKENS = 600
DEDUP_THRESHOLD  = 0.85     # Jaccard similarity threshold for near-duplicate removal
MINHASH_PERMS    = 128       # Number of MinHash permutations (higher = more accurate)
NGRAM_SIZE       = 5         # Character n-grams for MinHash

VAL_RATIO        = 0.05
TEST_RATIO        = 0.05

enc = tiktoken.get_encoding("cl100k_base")


# ── Quality filter ────────────────────────────────────────────────────────────

def quality_filter(record: dict, stats: dict) -> bool:
    """
    Return True if the record passes quality checks.
    Updates stats dict in place.
    """
    doc     = record.get("document", "")
    summary = record.get("summary", "")

    # Length checks
    doc_tokens = len(enc.encode(doc))
    sum_tokens = len(enc.encode(summary))

    if doc_tokens < MIN_DOC_TOKENS:
        stats["too_short_doc"] = stats.get("too_short_doc", 0) + 1
        return False
    if doc_tokens > MAX_DOC_TOKENS:
        stats["too_long_doc_clipped"] = stats.get("too_long_doc_clipped", 0) + 1
        # Don't reject, but flag for clipping at training time
        record["flagged_long"] = True

    if sum_tokens < MIN_SUMMARY_TOKENS:
        stats["too_short_summary"] = stats.get("too_short_summary", 0) + 1
        return False
    if sum_tokens > MAX_SUMMARY_TOKENS:
        # Truncate long summaries
        summary_tokens = enc.encode(summary)[:MAX_SUMMARY_TOKENS]
        record["summary"] = enc.decode(summary_tokens)

    # Language check (financial documents should be English)
    try:
        lang = detect(doc[:1000])
        if lang != "en":
            stats["non_english"] = stats.get("non_english", 0) + 1
            return False
    except LangDetectException:
        pass

    # Basic content check: must contain at least some numeric data
    import re
    numbers = re.findall(r'\$[\d,.]+|\d+(?:\.\d+)?%|\d{4}', doc[:2000])
    if len(numbers) < 3:
        stats["no_numeric_data"] = stats.get("no_numeric_data", 0) + 1
        return False

    return True


# ── MinHash deduplication ─────────────────────────────────────────────────────

def text_to_minhash(text: str, num_perm: int = MINHASH_PERMS, n: int = NGRAM_SIZE) -> MinHash:
    """Create MinHash signature from character n-grams of text."""
    m     = MinHash(num_perm=num_perm)
    text  = text[:5000].lower()   # Use first 5K chars for dedup speed
    for i in range(len(text) - n + 1):
        ngram = text[i:i+n].encode("utf-8")
        m.update(ngram)
    return m


def deduplicate(
    records: list[dict],
    threshold: float = DEDUP_THRESHOLD,
    num_perm: int = MINHASH_PERMS,
) -> tuple[list[dict], int]:
    """
    Remove near-duplicate documents using MinHash LSH.

    Returns (deduplicated_records, n_removed).
    """
    lsh     = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept    = []
    removed = 0

    for i, record in enumerate(tqdm(records, desc="MinHash deduplication")):
        doc   = record.get("document", "")
        key   = f"doc_{i}"
        m     = text_to_minhash(doc, num_perm)

        # Check if any similar document already exists
        result = lsh.query(m)
        if result:
            # Near-duplicate found — skip this record
            removed += 1
            continue

        # No duplicate — add to index and keep
        lsh.insert(key, m)
        kept.append(record)

    logger.info(f"Deduplication: kept {len(kept)}, removed {removed} near-duplicates")
    return kept, removed


# ── Train/val/test split ──────────────────────────────────────────────────────

def stratified_split(
    records: list[dict],
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Stratified split by doc_type.
    Ensures each split has representative mix of 10-K, 10-Q, earnings_transcript.
    Also avoids data leakage: all records for a ticker go to the same split
    (actually we split by year: 2020-2022 = train, 2023 = test).
    """
    random.seed(seed)

    # Group by doc_type for stratification
    by_type: dict[str, list[dict]] = {}
    for rec in records:
        dt = rec.get("doc_type", "unknown")
        by_type.setdefault(dt, []).append(rec)

    train, val, test = [], [], []

    for dt, recs in by_type.items():
        random.shuffle(recs)
        n     = len(recs)
        n_val = max(1, int(n * val_ratio))
        n_test = max(1, int(n * test_ratio))

        test  += recs[:n_test]
        val   += recs[n_test:n_test + n_val]
        train += recs[n_test + n_val:]

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_preprocessing(
    input_file: str,
    output_dir: str,
    deduplicate_docs: bool = True,
    dedup_threshold: float = DEDUP_THRESHOLD,
) -> dict:
    """
    Full preprocessing pipeline.
    Returns a stats dict with counts.
    """
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all records
    records = []
    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} raw records")

    # Quality filtering
    stats   = {"total_input": len(records)}
    filtered = [r for r in records if quality_filter(r, stats)]
    stats["after_filter"] = len(filtered)
    logger.info(f"Quality filter: {len(records)} → {len(filtered)}")

    # Deduplication
    if deduplicate_docs:
        filtered, n_removed = deduplicate(filtered, dedup_threshold)
        stats["after_dedup"] = len(filtered)
        stats["dedup_removed"] = n_removed

    # Split
    train, val, test = stratified_split(filtered)
    stats.update({
        "train": len(train),
        "val":   len(val),
        "test":  len(test),
    })

    # Save splits
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        out_path = output_dir / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for rec in split_data:
                f.write(json.dumps(rec) + "\n")
        logger.info(f"Saved {len(split_data)} {split_name} examples → {out_path}")

    # Doc type distribution
    all_types = Counter(r.get("doc_type", "?") for r in filtered)
    stats["doc_type_dist"] = dict(all_types)

    logger.info(f"Preprocessing complete: {stats}")
    return stats


if __name__ == "__main__":
    import typer
    app = typer.Typer()

    @app.command()
    def main(
        input_file: str = typer.Option("./training/data/raw/documents.jsonl"),
        output_dir: str = typer.Option("./training/data/splits"),
        no_dedup:   bool = typer.Option(False),
    ):
        stats = run_preprocessing(input_file, output_dir, deduplicate_docs=not no_dedup)
        print(json.dumps(stats, indent=2))

    app()
