"""
inference/src/chunker.py
─────────────────────────
Sliding-window chunker for 50K+ token documents.

Financial documents (especially 10-K filings) can easily exceed 100K tokens.
Strategy:
  1. Split into semantic chunks (by section heading or paragraph boundary)
  2. Slide a 3500-token window with 256-token overlap
  3. Generate a summary for each chunk in parallel
  4. Merge chunk summaries with a final consolidation pass

This is what enables p95 latency < 2.5s for 50K-token documents:
  - Chunks are processed in parallel (vLLM continuous batching)
  - Section-aware chunking preserves semantic coherence
  - The consolidation summary is short, so the final pass is fast

Example:
  50K-token 10-K → 15 chunks of ~3.5K tokens → 15 parallel summaries
  → consolidation prompt (~2K tokens) → final 400-token summary
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import tiktoken


# ── Chunk ─────────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    text:       str
    start_char: int
    end_char:   int
    chunk_idx:  int
    section:    str = ""
    n_tokens:   int = 0

    @property
    def is_last(self) -> bool:
        return False  # Set externally by chunker


@dataclass
class ChunkingResult:
    chunks:     list[Chunk]
    n_tokens:   int
    n_chunks:   int
    doc_type:   str = ""
    sections:   list[str] = field(default_factory=list)


# ── Section detector ──────────────────────────────────────────────────────────

# Common 10-K / 10-Q section headers
SECTION_PATTERNS = [
    re.compile(r'^(ITEM\s+\d+[A-Z]?[\.\s]+[A-Z][A-Z\s]+)', re.MULTILINE | re.IGNORECASE),
    re.compile(r'^(PART\s+[IVX]+[\.\s]+[A-Z][A-Z\s]+)', re.MULTILINE | re.IGNORECASE),
    re.compile(r'^((?:MANAGEMENT.{0,20}DISCUSSION|RISK FACTORS|QUANTITATIVE|FINANCIAL STATEMENTS|CONTROLS AND PROCEDURES|LEGAL PROCEEDINGS|EXECUTIVE COMPENSATION).*)', re.MULTILINE | re.IGNORECASE),
    re.compile(r'^\n([A-Z][A-Z\s]{5,50})\n', re.MULTILINE),  # ALL-CAPS headers
]


def detect_sections(text: str) -> list[tuple[int, str]]:
    """Return list of (char_position, section_name) pairs."""
    sections = []
    for pattern in SECTION_PATTERNS:
        for m in pattern.finditer(text):
            sections.append((m.start(), m.group(1).strip()[:80]))
    return sorted(set(sections), key=lambda x: x[0])


# ── Chunker ───────────────────────────────────────────────────────────────────

class SlidingWindowChunker:
    """
    Chunk long financial documents into overlapping windows.

    Strategy:
      1. Detect section boundaries
      2. For each section: if fits in window, keep whole; else slide
      3. Overlap of `stride` tokens ensures context continuity across chunks

    Args:
        max_tokens:     Maximum tokens per chunk (default 3500 for 4096 context window)
        stride:         Overlap between adjacent chunks in tokens (default 256)
        model:          Tokenizer model name for tiktoken
        preserve_sections: Prefer not to break within a section
    """

    def __init__(
        self,
        max_tokens: int = 3500,
        stride:     int = 256,
        model:      str = "gpt2",          # tiktoken compatible
        preserve_sections: bool = True,
    ):
        self.max_tokens       = max_tokens
        self.stride           = stride
        self.preserve_sections = preserve_sections
        try:
            self._enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._enc = None

    def count_tokens(self, text: str) -> int:
        if self._enc:
            return len(self._enc.encode(text))
        return len(text) // 4  # Rough fallback

    def chunk(self, text: str, doc_type: str = "10-K") -> ChunkingResult:
        """
        Chunk the document. Returns ChunkingResult with all chunks.

        For short documents (< max_tokens): returns a single chunk.
        For long documents: sliding window, respecting section boundaries.
        """
        total_tokens = self.count_tokens(text)

        # Short document: single chunk
        if total_tokens <= self.max_tokens:
            chunks = [Chunk(text=text, start_char=0, end_char=len(text),
                            chunk_idx=0, n_tokens=total_tokens)]
            return ChunkingResult(
                chunks=chunks, n_tokens=total_tokens, n_chunks=1, doc_type=doc_type
            )

        # Detect sections for section-aware splitting
        sections = detect_sections(text) if self.preserve_sections else []

        chunks = self._sliding_window_chunk(text, sections)
        section_names = [s[1] for s in sections]

        return ChunkingResult(
            chunks=chunks,
            n_tokens=total_tokens,
            n_chunks=len(chunks),
            doc_type=doc_type,
            sections=section_names,
        )

    def _sliding_window_chunk(
        self,
        text: str,
        sections: list[tuple[int, str]],
    ) -> list[Chunk]:
        """
        Generate sliding window chunks from text.
        Uses tokenizer to measure window sizes accurately.
        """
        if self._enc is None:
            # Character-based fallback
            return self._char_based_chunk(text)

        tokens = self._enc.encode(text)
        n      = len(tokens)
        chunks = []
        start  = 0
        idx    = 0

        # Build a char→token index mapping for section boundaries
        # (simplified: use fractional positions)
        section_token_boundaries = set()
        for char_pos, _ in sections:
            # Approximate token position
            approx_token = int(char_pos / len(text) * n)
            # Snap to nearest paragraph boundary token
            section_token_boundaries.add(approx_token)

        while start < n:
            end = min(start + self.max_tokens, n)

            # If preserve_sections, try to snap end to a section boundary
            if self.preserve_sections and section_token_boundaries:
                for boundary in sorted(section_token_boundaries):
                    if start < boundary <= end:
                        end = boundary
                        break

            chunk_tokens = tokens[start:end]
            chunk_text   = self._enc.decode(chunk_tokens)

            # Find original char positions (approximate)
            start_char = int(start / n * len(text))
            end_char   = int(end   / n * len(text))

            # Find which section this chunk belongs to
            section = ""
            for char_pos, sec_name in reversed(sections):
                if char_pos <= start_char:
                    section = sec_name
                    break

            chunks.append(Chunk(
                text=chunk_text.strip(),
                start_char=start_char,
                end_char=end_char,
                chunk_idx=idx,
                section=section,
                n_tokens=len(chunk_tokens),
            ))

            # Advance with stride overlap
            start += self.max_tokens - self.stride
            idx   += 1

        return chunks

    def _char_based_chunk(self, text: str) -> list[Chunk]:
        """Fallback character-based chunking when tokenizer unavailable."""
        max_chars   = self.max_tokens * 4
        stride_chars = self.stride * 4
        chunks = []
        start  = 0
        idx    = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            chunks.append(Chunk(
                text=text[start:end],
                start_char=start,
                end_char=end,
                chunk_idx=idx,
                n_tokens=self.count_tokens(text[start:end]),
            ))
            start += max_chars - stride_chars
            idx   += 1
        return chunks


# ── Consolidation ─────────────────────────────────────────────────────────────

CONSOLIDATION_PROMPT = """You are a financial analyst expert. Below are summaries of
consecutive sections of a {doc_type} document. Combine them into a single coherent
summary that preserves all key financial metrics, risk factors, and guidance.
Eliminate redundancies. Be concise and factual.

Section summaries:
{section_summaries}"""


def build_consolidation_prompt(
    chunk_summaries: list[str],
    doc_type: str = "10-K",
) -> str:
    """Build the consolidation prompt from per-chunk summaries."""
    formatted = "\n\n".join(
        f"[Section {i+1}]\n{s.strip()}"
        for i, s in enumerate(chunk_summaries)
        if s.strip()
    )
    return CONSOLIDATION_PROMPT.format(
        doc_type=doc_type,
        section_summaries=formatted,
    )
