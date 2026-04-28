"""
deploy/lambda/handler.py
─────────────────────────
AWS Lambda handler using Mangum (ASGI adapter).

For sub-4s cold start:
  - Model is loaded from /tmp (Lambda ephemeral storage)
  - First warm request pre-loads the model into memory
  - Subsequent requests reuse the in-memory model

Lambda configuration:
  - Memory: 10 GB (stores quantised 4-bit model)
  - Timeout: 60s
  - Architecture: arm64 (Graviton) for better price/performance
  - Container image: ECR with vLLM dependencies pre-installed
  - Provisioned concurrency: 2 (avoids cold starts in production)

Note: For real production, use an EC2/ECS-based deployment with vLLM.
Lambda is suitable for low-traffic or cost-sensitive workloads.
"""

from __future__ import annotations

import json
import logging
import os
import time

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ── Model loading on cold start ───────────────────────────────────────────────

_app  = None
_init_time = None


def _get_app():
    global _app, _init_time
    if _app is None:
        t0 = time.perf_counter()
        logger.info("Cold start: initialising FastAPI app…")

        # Set model path for Lambda environment
        os.environ.setdefault(
            "MODEL_PATH",
            os.getenv("MODEL_PATH", "/opt/ml/model/mistral-finance-v1-merged")
        )

        # Import app (triggers model loading)
        from inference.src.app import app
        _app = app

        _init_time = time.perf_counter() - t0
        logger.info(f"Cold start complete in {_init_time:.2f}s")

    return _app


# ── Lambda handler ────────────────────────────────────────────────────────────

try:
    from mangum import Mangum
    _handler_fn = None

    def handler(event: dict, context) -> dict:
        """AWS Lambda handler — wraps FastAPI app via Mangum ASGI adapter."""
        app = _get_app()
        global _handler_fn
        if _handler_fn is None:
            _handler_fn = Mangum(app, lifespan="off")
        return _handler_fn(event, context)

except ImportError:
    # Mangum not available — provide a simple direct handler
    def handler(event: dict, context) -> dict:  # type: ignore
        """Fallback direct Lambda handler without ASGI."""
        path   = event.get("path", "/health")
        method = event.get("httpMethod", "GET")
        body   = event.get("body", "")

        if path == "/health":
            return {
                "statusCode": 200,
                "body": json.dumps({"status": "ok"}),
                "headers": {"Content-Type": "application/json"},
            }

        if path == "/summarize" and method == "POST":
            try:
                req  = json.loads(body or "{}")
                from inference.src.app import get_summarizer
                summarizer = get_summarizer()
                result     = summarizer.summarise(
                    document=req["document"],
                    doc_type=req.get("doc_type", "10-K"),
                    ticker=req.get("ticker", ""),
                )
                return {
                    "statusCode": 200,
                    "body": json.dumps({
                        "summary":         result.summary,
                        "latency_ms":      result.latency_ms,
                        "n_tokens_input":  result.n_tokens_input,
                        "n_tokens_output": result.n_tokens_output,
                        "n_chunks":        result.n_chunks,
                    }),
                    "headers": {"Content-Type": "application/json"},
                }
            except Exception as e:
                logger.error(f"Handler error: {e}", exc_info=True)
                return {
                    "statusCode": 500,
                    "body": json.dumps({"error": str(e)}),
                    "headers": {"Content-Type": "application/json"},
                }

        return {
            "statusCode": 404,
            "body": json.dumps({"error": f"Not found: {method} {path}"}),
            "headers": {"Content-Type": "application/json"},
        }
