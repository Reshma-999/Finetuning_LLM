"""
pipeline/src/ingest.py
───────────────────────
Data ingestion pipeline.

Sources:
  1. SEC EDGAR — 10-K, 10-Q, 8-K filings via EDGAR full-text search
  2. Earnings transcripts — via sec-api.io or open datasets
  3. Azure Blob Storage — cache raw files for reproducibility

Output: JSONL files with fields:
  {
    "id": "AAPL-10K-2023",
    "ticker": "AAPL",
    "company": "Apple Inc.",
    "doc_type": "10-K",
    "year": 2023,
    "document": "<full text>",
    "summary": "<analyst summary or extracted summary section>",
    "filing_date": "2023-11-03",
    "source_url": "https://..."
  }

Usage:
    python pipeline/src/ingest.py \\
        --source sec-edgar \\
        --years 2020 2021 2022 2023 \\
        --output ./training/data/raw
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Iterator, Optional

import requests
from bs4 import BeautifulSoup
from loguru import logger
from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────────────────

EDGAR_FULL_TEXT_SEARCH = "https://efts.sec.gov/LATEST/search-index?q=%22{query}%22&dateRange=custom&startdt={start}&enddt={end}&forms={form_type}"
EDGAR_SUBMISSIONS      = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
EDGAR_FILING           = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_path}/{primary_doc}"

SP500_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "BRK-B", "UNH", "JNJ",
    "JPM", "V", "PG", "MA", "HD", "CVX", "LLY", "MRK", "ABBV", "PEP", "KO",
    "AVGO", "COST", "TMO", "WMT", "ACN", "MCD", "CSCO", "NEE", "DHR", "TXN",
    "ADBE", "PM", "ABT", "MS", "VZ", "NFLX", "RTX", "BMY", "INTC", "ORCL",
    "QCOM", "HON", "T", "COP", "UPS", "IBM", "SPGI", "CAT", "GE", "BA",
]

HEADERS = {
    "User-Agent": "FinancialLLM Research research@financiallm.com",
    "Accept-Encoding": "gzip, deflate",
}


# ── EDGAR Ingestion ───────────────────────────────────────────────────────────

class EDGARIngestor:
    """Download and parse SEC filings from EDGAR."""

    def __init__(
        self,
        output_dir: Path,
        rate_limit_rps: float = 5.0,    # EDGAR allows 10 req/s; be conservative
    ):
        self.output_dir   = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._min_interval = 1.0 / rate_limit_rps
        self._last_request = 0.0

    def _get(self, url: str, **kwargs) -> requests.Response:
        elapsed = time.time() - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        resp = requests.get(url, headers=HEADERS, timeout=30, **kwargs)
        self._last_request = time.time()
        resp.raise_for_status()
        return resp

    def get_cik(self, ticker: str) -> Optional[int]:
        """Resolve ticker to CIK number."""
        url  = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=10-K&dateb=&owner=include&count=1&search_text="
        url2 = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&forms=10-K"
        try:
            resp = self._get(f"https://www.sec.gov/cgi-bin/browse-edgar?company=&CIK={ticker}&type=10-K&dateb=&owner=include&count=1&search_text=&action=getcompany")
            # Parse CIK from response
            m = re.search(r'/cgi-bin/browse-edgar\?action=getcompany&CIK=(\d+)', resp.text)
            if m:
                return int(m.group(1))
        except Exception as e:
            logger.warning(f"Could not resolve CIK for {ticker}: {e}")
        return None

    def iter_filings(
        self,
        ticker: str,
        form_types: list[str],
        years: list[int],
    ) -> Iterator[dict]:
        """Iterate over filings for a ticker and form types."""
        cik = self.get_cik(ticker)
        if not cik:
            logger.warning(f"Skipping {ticker} — could not resolve CIK")
            return

        try:
            resp = self._get(EDGAR_SUBMISSIONS.format(cik=cik))
            data = resp.json()
        except Exception as e:
            logger.error(f"Failed to get submissions for {ticker}: {e}")
            return

        filings = data.get("filings", {}).get("recent", {})
        forms       = filings.get("form", [])
        dates       = filings.get("filingDate", [])
        accessions  = filings.get("accessionNumber", [])
        primary_docs = filings.get("primaryDocument", [])

        for form, date, acc, doc in zip(forms, dates, accessions, primary_docs):
            if form not in form_types:
                continue
            year = int(date[:4])
            if year not in years:
                continue

            acc_path = acc.replace("-", "")
            url      = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_path[:10]}-{acc_path[10:12]}-{acc_path[12:]}/{doc}"
            yield {
                "ticker":       ticker,
                "cik":          cik,
                "form_type":    form,
                "filing_date":  date,
                "year":         year,
                "accession_no": acc,
                "url":          url,
            }

    def download_filing(self, filing_meta: dict) -> Optional[str]:
        """Download and extract plain text from an SEC filing."""
        try:
            resp = self._get(filing_meta["url"])
            return self._extract_text(resp.text, filing_meta["form_type"])
        except Exception as e:
            logger.warning(f"Failed to download {filing_meta['url']}: {e}")
            return None

    @staticmethod
    def _extract_text(html: str, form_type: str) -> str:
        """Extract clean text from SEC HTML filing."""
        soup = BeautifulSoup(html, "lxml")

        # Remove boilerplate elements
        for tag in soup.find_all(["script", "style", "ix:nonfraction", "ix:nonnumeric"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        # Clean up excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'\t+', ' ', text)

        return text.strip()


# ── Summary extraction ────────────────────────────────────────────────────────

def extract_summary_section(text: str, doc_type: str) -> Optional[str]:
    """
    Extract an existing summary or MD&A section to use as reference.
    Falls back to None if no suitable section found.
    """
    patterns = {
        "10-K": [
            r"(?:EXECUTIVE SUMMARY|SUMMARY OF OPERATIONS)(.*?)(?=\n\n[A-Z]{3,}|\Z)",
            r"(?:MANAGEMENT.{0,10}DISCUSSION.{0,10}ANALYSIS)(.*?)(?=ITEM \d+|\Z)",
        ],
        "earnings_transcript": [
            r"(PREPARED REMARKS.*?)(?=QUESTION AND ANSWER|\Z)",
        ],
    }

    for pattern in patterns.get(doc_type, patterns.get("10-K", [])):
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if m:
            section = m.group(1).strip()
            if 200 <= len(section) <= 3000:
                return section

    return None


# ── Full ingest pipeline ──────────────────────────────────────────────────────

def run_ingestion(
    source: str,
    years: list[int],
    output_dir: str,
    tickers: Optional[list[str]] = None,
    form_types: Optional[list[str]] = None,
    max_docs: Optional[int] = None,
) -> int:
    """
    Main ingestion function.
    Returns number of documents successfully ingested.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tickers    = tickers    or SP500_TICKERS
    form_types = form_types or ["10-K", "10-Q"]

    ingestor    = EDGARIngestor(output_dir / "raw_html")
    output_file = output_dir / "documents.jsonl"
    n_saved     = 0
    n_skipped   = 0

    with open(output_file, "a") as out_f:
        for ticker in tqdm(tickers, desc="Tickers"):
            if max_docs and n_saved >= max_docs:
                break

            for filing_meta in ingestor.iter_filings(ticker, form_types, years):
                if max_docs and n_saved >= max_docs:
                    break

                doc_id = f"{ticker}-{filing_meta['form_type']}-{filing_meta['year']}"

                text = ingestor.download_filing(filing_meta)
                if not text or len(text) < 1000:
                    n_skipped += 1
                    continue

                summary = extract_summary_section(text, filing_meta["form_type"])
                if not summary:
                    # Skip filings without extractable summaries
                    # (they can't be used for supervised fine-tuning without labels)
                    n_skipped += 1
                    continue

                record = {
                    "id":           doc_id,
                    "ticker":       ticker,
                    "doc_type":     filing_meta["form_type"],
                    "year":         filing_meta["year"],
                    "filing_date":  filing_meta["filing_date"],
                    "document":     text,
                    "summary":      summary,
                    "source_url":   filing_meta["url"],
                }

                out_f.write(json.dumps(record) + "\n")
                n_saved += 1

                logger.debug(f"Saved {doc_id} ({len(text):,} chars)")

    logger.info(f"Ingestion complete: {n_saved} saved, {n_skipped} skipped → {output_file}")
    return n_saved


if __name__ == "__main__":
    import typer
    app = typer.Typer()

    @app.command()
    def main(
        source:     str  = typer.Option("sec-edgar"),
        years:      list[int] = typer.Option([2021, 2022, 2023]),
        output_dir: str  = typer.Option("./training/data/raw"),
        max_docs:   int  = typer.Option(None),
    ):
        run_ingestion(source, years, output_dir, max_docs=max_docs)

    app()
