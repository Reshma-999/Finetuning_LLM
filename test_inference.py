"""
inference/tests/test_inference.py
───────────────────────────────────
Unit tests for the inference layer — chunker, prompt builder, app endpoints.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


# ── Chunker tests ─────────────────────────────────────────────────────────────

class TestSlidingWindowChunker:
    def setup_method(self):
        from inference.src.chunker import SlidingWindowChunker
        self.chunker = SlidingWindowChunker(max_tokens=500, stride=50)

    def test_short_document_single_chunk(self):
        doc = "Revenue increased 15% year-over-year to $1.2 billion. " * 10
        result = self.chunker.chunk(doc)
        assert result.n_chunks == 1
        assert result.chunks[0].text == doc.strip()

    def test_long_document_multiple_chunks(self):
        # ~5000 tokens worth of text
        doc = "This is a financial disclosure. Risk factors include market volatility. " * 300
        result = self.chunker.chunk(doc)
        assert result.n_chunks > 1
        assert all(c.n_tokens <= 500 + 50 for c in result.chunks)

    def test_chunks_cover_full_document(self):
        doc = "Section A content. " * 50 + "Section B content. " * 50
        result = self.chunker.chunk(doc)
        # All text should be covered
        covered = " ".join(c.text for c in result.chunks)
        # Every sentence from doc should appear somewhere in covered
        for sent in doc.split("."):
            sent = sent.strip()
            if sent:
                assert sent[:20] in covered, f"Missing: {sent[:20]}"

    def test_section_detection(self):
        from inference.src.chunker import detect_sections
        text = """
PART I

ITEM 1. BUSINESS OVERVIEW
Our company operates in the technology sector.

ITEM 1A. RISK FACTORS
Our business faces significant competition.

ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS
Revenue increased 15% this year.
"""
        sections = detect_sections(text)
        assert len(sections) >= 2
        # Should find at least one section
        section_names = [s[1] for s in sections]
        assert any("RISK" in name.upper() or "ITEM" in name.upper() for name in section_names)

    def test_chunk_indices_sequential(self):
        doc = "Financial data. " * 200
        result = self.chunker.chunk(doc)
        for i, chunk in enumerate(result.chunks):
            assert chunk.chunk_idx == i


# ── Prompt tests ──────────────────────────────────────────────────────────────

class TestPromptBuilder:
    def test_build_inference_prompt_contains_instruction(self):
        from inference.src.prompt import build_inference_prompt
        prompt = build_inference_prompt("Revenue was $1B.", "10-K")
        assert "[INST]" in prompt
        assert "[/INST]" in prompt
        assert "Revenue was $1B." in prompt
        assert "Annual Report" in prompt or "10-K" in prompt

    def test_build_inference_prompt_earnings(self):
        from inference.src.prompt import build_inference_prompt
        prompt = build_inference_prompt("Q3 earnings beat expectations.", "earnings_transcript")
        assert "earnings" in prompt.lower()

    def test_consolidation_prompt(self):
        from inference.src.prompt import build_consolidation_prompt_text
        summaries = [
            "Revenue grew 15% to $1.2B.",
            "Operating margins improved 200bps.",
            "Company guides for 10% growth next year.",
        ]
        prompt = build_consolidation_prompt_text(summaries, "10-K")
        assert "Section 1" in prompt
        assert "Section 2" in prompt
        assert "Revenue grew" in prompt
        assert "[INST]" in prompt


# ── FastAPI endpoint tests ─────────────────────────────────────────────────────

@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    # Mock the summarizer so we don't need a real model
    mock_result = MagicMock()
    mock_result.summary         = "Revenue increased 15% to $1.2B. EPS of $2.34 beat estimates."
    mock_result.n_tokens_input  = 500
    mock_result.n_tokens_output = 80
    mock_result.n_chunks        = 1
    mock_result.latency_ms      = 850.0

    with patch("inference.src.app.get_summarizer") as mock_get:
        mock_summarizer = MagicMock()
        mock_summarizer.summarise.return_value = mock_result
        mock_get.return_value = mock_summarizer

        from inference.src.app import app
        with TestClient(app) as c:
            yield c


class TestAPIEndpoints:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_summarize_success(self, client):
        resp = client.post("/summarize", json={
            "document": "Apple Inc. reported Q4 2023 revenue of $89.5 billion, a 1% decline year-over-year. "
                        "iPhone revenue was $43.8 billion. Services revenue reached a record $22.3 billion. "
                        "EPS of $1.46 beat analyst estimates of $1.39. The company guided for Q1 revenue of "
                        "$89-93 billion. CEO Tim Cook highlighted strong performance in emerging markets. " * 5,
            "doc_type": "earnings_transcript",
            "ticker": "AAPL",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "summary" in data
        assert data["doc_type"] == "earnings_transcript"
        assert data["ticker"]   == "AAPL"
        assert data["latency_ms"] > 0
        assert data["model_version"] == "mistral-7b-finance-v1"

    def test_summarize_short_document_rejected(self, client):
        resp = client.post("/summarize", json={
            "document": "Short.",
            "doc_type": "10-K",
        })
        assert resp.status_code == 422  # Validation error

    def test_summarize_invalid_doc_type(self, client):
        resp = client.post("/summarize", json={
            "document": "X" * 200,
            "doc_type": "INVALID_TYPE",
        })
        assert resp.status_code == 422

    def test_cache_hit(self, client):
        doc = "Revenue report. " * 50
        # First request
        resp1 = client.post("/summarize", json={"document": doc, "doc_type": "10-K", "use_cache": True})
        assert resp1.status_code == 200
        assert not resp1.json()["from_cache"]

        # Second request — should hit cache
        resp2 = client.post("/summarize", json={"document": doc, "doc_type": "10-K", "use_cache": True})
        assert resp2.status_code == 200
        assert resp2.json()["from_cache"]
        assert resp2.json()["latency_ms"] < resp1.json()["latency_ms"]

    def test_batch_summarize(self, client):
        docs = [
            {"document": "Annual report content. " * 20, "doc_type": "10-K", "ticker": f"TICK{i}"}
            for i in range(3)
        ]
        resp = client.post("/batch", json={"documents": docs})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 3
        assert data["total_latency_ms"] > 0


# ── Hallucination detector tests ──────────────────────────────────────────────

class TestHallucinationDetector:
    def test_no_hallucination_matching_numbers(self):
        from evaluation.src.metrics import detect_hallucinated_numbers
        doc = "Revenue was $1.2 billion. EPS was $2.34."
        summary = "The company reported $1.2B in revenue and EPS of $2.34."
        assert not detect_hallucinated_numbers(summary, doc)

    def test_hallucination_detected(self):
        from evaluation.src.metrics import detect_hallucinated_numbers
        doc = "Revenue was $1.2 billion."
        summary = "Revenue reached $5.6 billion."   # 5.6 not in source
        assert detect_hallucinated_numbers(summary, doc)

    def test_no_numbers_no_hallucination(self):
        from evaluation.src.metrics import detect_hallucinated_numbers
        doc = "The company faced significant headwinds this year."
        summary = "The company encountered challenges."
        assert not detect_hallucinated_numbers(summary, doc)
