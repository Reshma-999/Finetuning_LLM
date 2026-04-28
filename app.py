"""
inference/src/app.py
─────────────────────
FastAPI inference service for financial document summarisation.

Endpoints:
  POST /summarize            — Synchronous summarisation
  POST /summarize/stream     — Server-Sent Events streaming
  GET  /health               — Health check
  GET  /metrics              — Prometheus metrics
  POST /batch                — Batch summarisation (up to 32 docs)

Performance targets:
  • p50 latency: < 1.5s  (for ≤ 4K token docs)
  • p95 latency: < 2.5s  (for ≤ 50K token docs)
  • Throughput: ~20 req/s on A10G 24GB
"""

from __future__ import annotations

import os
import time
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional, AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

logger = logging.getLogger(__name__)

# ── Settings ──────────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    model_path:             str   = os.getenv("MODEL_PATH", "./models/mistral-finance-v1-merged")
    max_new_tokens:         int   = 512
    temperature:            float = 0.1
    top_p:                  float = 0.9
    gpu_memory_utilization: float = 0.85
    use_vllm:               bool  = True
    enable_cache:           bool  = True
    cache_ttl_seconds:      int   = 3600
    max_batch_size:         int   = 32
    api_key:                str   = ""
    cors_origins:           list[str] = ["*"]

    class Config:
        env_file = ".env"

settings = Settings()

# ── Prometheus metrics ────────────────────────────────────────────────────────

REQUEST_COUNT   = Counter("summarizer_requests_total",   "Total requests", ["status", "doc_type"])
REQUEST_LATENCY = Histogram(
    "summarizer_latency_seconds", "Request latency",
    buckets=[0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0]
)
TOKEN_COUNT_IN  = Counter("summarizer_input_tokens_total",  "Input tokens")
TOKEN_COUNT_OUT = Counter("summarizer_output_tokens_total", "Output tokens")
ACTIVE_REQUESTS = Gauge("summarizer_active_requests",       "Active requests")
CACHE_HITS      = Counter("summarizer_cache_hits_total",    "Cache hits")

# ── Global model singleton ────────────────────────────────────────────────────

_summarizer = None
_cache: dict[str, tuple[str, float]] = {}  # key → (summary, timestamp)


def get_summarizer():
    global _summarizer
    if _summarizer is None:
        from inference.src.summarizer import FinancialSummarizer
        _summarizer = FinancialSummarizer(
            model_path=settings.model_path,
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            top_p=settings.top_p,
            gpu_memory_utilization=settings.gpu_memory_utilization,
            use_vllm=settings.use_vllm,
        )
    return _summarizer


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model…")
    get_summarizer()          # Warm up on startup
    logger.info("Model ready")
    yield
    logger.info("Shutting down")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Financial Document Summarizer",
    description=(
        "Mistral-7B fine-tuned with LoRA on 8K+ financial documents. "
        "ROUGE-L=0.43, p95 latency < 2.5s for 50K-token documents."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────

class SummarizeRequest(BaseModel):
    document:     str    = Field(..., min_length=100)
    doc_type:     str    = Field("10-K", pattern="^(10-K|10-Q|earnings_transcript|8-K|default)$")
    ticker:       str    = Field("", max_length=10)
    max_tokens:   Optional[int]   = Field(None, ge=50, le=1024)
    use_cache:    bool   = Field(True)

class SummarizeResponse(BaseModel):
    summary:         str
    doc_type:        str
    ticker:          str
    n_tokens_input:  int
    n_tokens_output: int
    n_chunks:        int
    latency_ms:      float
    from_cache:      bool
    model_version:   str = "mistral-7b-finance-v1"

class BatchRequest(BaseModel):
    documents:  list[SummarizeRequest] = Field(..., max_items=32)

class BatchResponse(BaseModel):
    results:       list[SummarizeResponse]
    total_latency_ms: float


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_key(req: SummarizeRequest) -> str:
    import hashlib
    return hashlib.sha256(
        f"{req.doc_type}:{req.document[:500]}".encode()
    ).hexdigest()


def _cache_get(key: str) -> Optional[str]:
    if not settings.enable_cache:
        return None
    entry = _cache.get(key)
    if entry and time.time() - entry[1] < settings.cache_ttl_seconds:
        return entry[0]
    return None


def _cache_set(key: str, summary: str) -> None:
    _cache[key] = (summary, time.time())
    # Simple eviction: keep only last 1000 entries
    if len(_cache) > 1000:
        oldest = sorted(_cache.keys(), key=lambda k: _cache[k][1])[:100]
        for k in oldest:
            del _cache[k]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model":  settings.model_path,
        "vllm":   settings.use_vllm,
        "cache_entries": len(_cache),
    }


@app.get("/metrics")
def metrics():
    return StreamingResponse(
        iter([generate_latest()]),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(req: SummarizeRequest):
    ACTIVE_REQUESTS.inc()
    t0 = time.perf_counter()

    try:
        # Cache check
        cache_key = _cache_key(req)
        if req.use_cache:
            cached = _cache_get(cache_key)
            if cached:
                CACHE_HITS.inc()
                REQUEST_COUNT.labels(status="hit", doc_type=req.doc_type).inc()
                ACTIVE_REQUESTS.dec()
                return SummarizeResponse(
                    summary=cached, doc_type=req.doc_type, ticker=req.ticker,
                    n_tokens_input=0, n_tokens_output=0, n_chunks=0,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    from_cache=True,
                )

        # Run inference in thread pool (vLLM is sync)
        loop      = asyncio.get_event_loop()
        summarizer = get_summarizer()
        result    = await loop.run_in_executor(
            None,
            summarizer.summarise,
            req.document,
            req.doc_type,
            req.ticker,
        )

        latency_ms = (time.perf_counter() - t0) * 1000

        # Cache result
        if req.use_cache:
            _cache_set(cache_key, result.summary)

        TOKEN_COUNT_IN.inc(result.n_tokens_input)
        TOKEN_COUNT_OUT.inc(result.n_tokens_output)
        REQUEST_COUNT.labels(status="ok", doc_type=req.doc_type).inc()
        REQUEST_LATENCY.observe(latency_ms / 1000)

        logger.info(
            f"Summarized {result.n_tokens_input} tokens in {latency_ms:.0f}ms "
            f"({result.n_chunks} chunks, ticker={req.ticker})"
        )

        return SummarizeResponse(
            summary=result.summary,
            doc_type=req.doc_type,
            ticker=req.ticker,
            n_tokens_input=result.n_tokens_input,
            n_tokens_output=result.n_tokens_output,
            n_chunks=result.n_chunks,
            latency_ms=latency_ms,
            from_cache=False,
        )

    except Exception as e:
        REQUEST_COUNT.labels(status="error", doc_type=req.doc_type).inc()
        logger.exception(f"Summarization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        ACTIVE_REQUESTS.dec()


@app.post("/summarize/stream")
async def summarize_stream(req: SummarizeRequest):
    """Server-Sent Events streaming endpoint. First token appears in <200ms."""
    summarizer = get_summarizer()

    async def event_generator() -> AsyncIterator[str]:
        async for token in summarizer.summarise_stream(req.document, req.doc_type):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/batch", response_model=BatchResponse)
async def batch_summarize(req: BatchRequest):
    """Batch summarisation. Processes all documents in parallel."""
    t0 = time.perf_counter()

    loop      = asyncio.get_event_loop()
    summarizer = get_summarizer()

    async def run_one(r: SummarizeRequest) -> SummarizeResponse:
        result = await loop.run_in_executor(
            None, summarizer.summarise, r.document, r.doc_type, r.ticker
        )
        return SummarizeResponse(
            summary=result.summary, doc_type=r.doc_type, ticker=r.ticker,
            n_tokens_input=result.n_tokens_input, n_tokens_output=result.n_tokens_output,
            n_chunks=result.n_chunks,
            latency_ms=result.latency_ms, from_cache=False,
        )

    results = await asyncio.gather(*[run_one(r) for r in req.documents])

    return BatchResponse(
        results=list(results),
        total_latency_ms=(time.perf_counter() - t0) * 1000,
    )
