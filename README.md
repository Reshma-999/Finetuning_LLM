# Fine-Tuned LLM for Financial Document Summarization

**Mistral-7B fine-tuned with LoRA on 8K+ earnings transcripts and SEC 10-K filings.**

| Metric | Value |
|---|---|
| Base model | Mistral-7B-Instruct-v0.2 |
| Fine-tuning method | LoRA (rank=64, alpha=128) |
| Training corpus | 8,412 earnings transcripts + 10-K filings |
| ROUGE-L | **0.43** (vs GPT-3.5 zero-shot: 0.385, +12%) |
| BERTScore F1 | 0.891 |
| p95 inference latency | **< 2.5s** for 50K-token documents |
| Inference backend | FastAPI + vLLM |

---

## Architecture

```
SEC EDGAR / Earnings APIs
        │
        ▼
  Azure Blob Storage
  (raw documents)
        │
        ▼
  Data Pipeline
  ├── Chunker (sliding window, 2048 tokens)
  ├── Deduplicator (MinHash LSH)
  └── Quality Filter (length, language)
        │
        ▼
  Training Pipeline
  ├── Mistral-7B-Instruct-v0.2
  ├── LoRA (rank=64, alpha=128, target: q/k/v/o projections)
  ├── 4-bit QLoRA via BitsAndBytes
  └── Gradient checkpointing
        │
        ▼
  Evaluation Harness
  ├── ROUGE-L, ROUGE-1, ROUGE-2
  ├── BERTScore
  ├── FinBERT factual consistency
  └── GPT-3.5 baseline comparison
        │
        ▼
  Inference Service (FastAPI + vLLM)
  ├── AWS Lambda (cold start < 4s)
  ├── Streaming responses
  └── p95 latency < 2.5s
```

---

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Set: AZURE_STORAGE_CONNECTION_STRING, HF_TOKEN, OPENAI_API_KEY

# 3. Download and prepare data
python pipeline/src/ingest.py --source sec-edgar --years 2020 2021 2022 2023
python pipeline/src/preprocess.py

# 4. Train
python training/src/train.py \
  --config training/configs/lora_mistral7b.yaml \
  --output-dir ./checkpoints/mistral-finance-v1

# 5. Evaluate
python evaluation/src/evaluate.py \
  --model-path ./checkpoints/mistral-finance-v1 \
  --test-set ./training/data/splits/test.jsonl \
  --compare-gpt35

# 6. Run inference server
uvicorn inference.src.app:app --host 0.0.0.0 --port 8000 --workers 2
```

---

## Training Results

```
Epoch 1/3  loss=1.847  eval_rouge_l=0.312  eval_bert_f1=0.821
Epoch 2/3  loss=1.203  eval_rouge_l=0.391  eval_bert_f1=0.873
Epoch 3/3  loss=0.981  eval_rouge_l=0.430  eval_bert_f1=0.891

GPT-3.5 zero-shot baseline:
  ROUGE-L: 0.385   BERTScore F1: 0.851
  
Fine-tuned Mistral-7B (ours):
  ROUGE-L: 0.430   BERTScore F1: 0.891
  Improvement: +11.7% ROUGE-L, +4.7% BERTScore
```

---

## Project Structure

```
FinancialLLM/
├── training/
│   ├── src/
│   │   ├── train.py           # LoRA fine-tuning loop
│   │   ├── dataset.py         # HuggingFace dataset builder
│   │   ├── collator.py        # Dynamic padding collator
│   │   ├── callbacks.py       # W&B + checkpoint callbacks
│   │   └── lora_config.py     # LoRA/QLoRA configuration
│   ├── configs/
│   │   └── lora_mistral7b.yaml
│   └── scripts/
│       ├── merge_adapters.py  # Merge LoRA → base model
│       └── push_to_hub.py
├── inference/
│   ├── src/
│   │   ├── app.py             # FastAPI server
│   │   ├── summarizer.py      # vLLM inference engine
│   │   ├── chunker.py         # 50K-token sliding window
│   │   ├── prompt.py          # Prompt templates
│   │   └── cache.py           # Redis response cache
│   └── tests/
│       └── test_inference.py
├── evaluation/
│   └── src/
│       ├── evaluate.py        # Full eval harness
│       ├── metrics.py         # ROUGE, BERTScore, FinBERT
│       └── baseline.py        # GPT-3.5 baseline runner
├── pipeline/
│   └── src/
│       ├── ingest.py          # SEC EDGAR + earnings APIs
│       ├── preprocess.py      # Chunking + deduplication
│       └── upload.py          # Azure Blob Storage
├── deploy/
│   ├── lambda/handler.py      # AWS Lambda handler
│   ├── azure/                 # Azure Container config
│   └── docker/Dockerfile
└── scripts/
    └── benchmark_latency.py
```

---

## License

MIT
