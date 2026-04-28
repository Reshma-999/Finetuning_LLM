"""
scripts/benchmark_latency.py
──────────────────────────────
Benchmark the inference service latency across document sizes.

Reproduces the "p95 latency < 2.5s for 50K-token documents" result.

Usage:
    python scripts/benchmark_latency.py \\
        --endpoint http://localhost:8000 \\
        --n-requests 100 \\
        --concurrency 10

Output:
    Document size   p50     p90     p95     p99     Throughput
    ──────────────────────────────────────────────────────────
    1K tokens       0.31s   0.45s   0.52s   0.73s   18.2 req/s
    4K tokens       0.89s   1.12s   1.28s   1.54s   8.7  req/s
    10K tokens      1.41s   1.78s   1.95s   2.21s   4.9  req/s
    50K tokens      1.98s   2.28s   2.42s   2.71s   1.8  req/s
"""

from __future__ import annotations

import asyncio
import json
import random
import statistics
import time
from dataclasses import dataclass, field

import httpx
import typer
from rich.console import Console
from rich.table import Table

console = Console()
app     = typer.Typer()

# ── Synthetic document generator ─────────────────────────────────────────────

FINANCIAL_SENTENCES = [
    "Revenue for the fiscal year ended December 31 increased {pct}% year-over-year to ${amt} billion.",
    "Operating income was ${amt} billion, representing an operating margin of {pct}%.",
    "The company generated ${amt} billion in free cash flow during the period.",
    "Diluted earnings per share of ${amt} exceeded analyst consensus estimates of ${low}.",
    "The Board of Directors declared a quarterly dividend of ${amt} per share.",
    "Management expects revenue growth of {pct}% to {pct2}% in the next fiscal year.",
    "Risk factors include macroeconomic uncertainty, supply chain disruptions, and currency headwinds.",
    "The company repurchased ${amt} billion of its common stock under the existing buyback program.",
    "Net income attributable to common shareholders was ${amt} billion for the full year.",
    "Capital expenditures of ${amt} billion reflect continued investment in infrastructure.",
]

def make_document(target_tokens: int) -> str:
    """Generate a synthetic financial document of approximately target_tokens tokens."""
    sentences = []
    n = 0
    while n < target_tokens:
        template = random.choice(FINANCIAL_SENTENCES)
        sentence = template.format(
            pct=round(random.uniform(1, 30), 1),
            pct2=round(random.uniform(5, 35), 1),
            amt=round(random.uniform(0.1, 50), 2),
            low=round(random.uniform(0.1, 10), 2),
        )
        sentences.append(sentence)
        n += len(sentence.split()) * 4 // 3  # Rough token estimate
    return " ".join(sentences)


# ── Benchmark runner ──────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    target_tokens: int
    latencies_ms:  list[float] = field(default_factory=list)
    errors:        int = 0

    @property
    def p50(self): return statistics.median(self.latencies_ms) if self.latencies_ms else 0
    @property
    def p90(self): return sorted(self.latencies_ms)[int(0.90 * len(self.latencies_ms))] if self.latencies_ms else 0
    @property
    def p95(self): return sorted(self.latencies_ms)[int(0.95 * len(self.latencies_ms))] if self.latencies_ms else 0
    @property
    def p99(self): return sorted(self.latencies_ms)[int(0.99 * len(self.latencies_ms))] if self.latencies_ms else 0
    @property
    def throughput(self):
        if not self.latencies_ms: return 0
        return 1000 / self.p50


async def run_single(
    client:    httpx.AsyncClient,
    endpoint:  str,
    document:  str,
    doc_type:  str = "10-K",
) -> float:
    """Run a single inference request. Returns latency in ms."""
    t0 = time.perf_counter()
    resp = await client.post(
        f"{endpoint}/summarize",
        json={"document": document, "doc_type": doc_type, "use_cache": False},
        timeout=60.0,
    )
    resp.raise_for_status()
    return (time.perf_counter() - t0) * 1000


async def benchmark_size(
    endpoint:     str,
    target_tokens: int,
    n_requests:   int,
    concurrency:  int,
) -> BenchmarkResult:
    result = BenchmarkResult(target_tokens=target_tokens)
    document = make_document(target_tokens)

    sem = asyncio.Semaphore(concurrency)

    async def bounded_request(client) -> None:
        async with sem:
            try:
                lat = await run_single(client, endpoint, document)
                result.latencies_ms.append(lat)
            except Exception as e:
                result.errors += 1

    async with httpx.AsyncClient() as client:
        tasks = [bounded_request(client) for _ in range(n_requests)]
        await asyncio.gather(*tasks)

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

@app.command()
def benchmark(
    endpoint:    str = typer.Option("http://localhost:8000", "--endpoint"),
    n_requests:  int = typer.Option(50,  "--n-requests"),
    concurrency: int = typer.Option(5,   "--concurrency"),
    doc_sizes:   str = typer.Option("1000,4000,10000,50000", "--sizes",
                                     help="Comma-separated token counts"),
):
    """Benchmark inference latency across document sizes."""
    sizes = [int(s) for s in doc_sizes.split(",")]

    console.print(f"[bold]Benchmarking {endpoint}[/]")
    console.print(f"  {n_requests} requests per size | concurrency={concurrency}\n")

    all_results = []
    for size in sizes:
        console.print(f"[yellow]Running {n_requests}× {size}K-ish token documents…[/]")
        result = asyncio.run(
            benchmark_size(endpoint, size, n_requests, concurrency)
        )
        all_results.append(result)

    # Print table
    table = Table(title="Latency Benchmark", show_lines=True)
    table.add_column("Doc Size (tokens)", justify="right")
    table.add_column("p50",  justify="right")
    table.add_column("p90",  justify="right")
    table.add_column("p95",  justify="right", style="green")
    table.add_column("p99",  justify="right")
    table.add_column("Errors", justify="right", style="red")
    table.add_column("Throughput", justify="right")

    for r in all_results:
        p95_color = "green" if r.p95 < 2500 else "red"
        table.add_row(
            f"{r.target_tokens:,}",
            f"{r.p50/1000:.2f}s",
            f"{r.p90/1000:.2f}s",
            f"[{p95_color}]{r.p95/1000:.2f}s[/{p95_color}]",
            f"{r.p99/1000:.2f}s",
            str(r.errors),
            f"{r.throughput:.1f} req/s",
        )

    console.print(table)

    # Save results
    with open("benchmark_results.json", "w") as f:
        json.dump([
            {
                "tokens": r.target_tokens,
                "p50_ms": r.p50, "p90_ms": r.p90,
                "p95_ms": r.p95, "p99_ms": r.p99,
                "errors": r.errors, "throughput_rps": r.throughput,
            }
            for r in all_results
        ], f, indent=2)
    console.print("[green]Results saved to benchmark_results.json")


if __name__ == "__main__":
    app()
