"""
training/src/train.py
──────────────────────
Fine-tunes Mistral-7B-Instruct-v0.2 with LoRA adapters on financial documents.

Usage:
    python training/src/train.py \\
        --config training/configs/lora_mistral7b.yaml \\
        --output-dir ./checkpoints/mistral-finance-v1 \\
        [--resume-from-checkpoint ./checkpoints/mistral-finance-v1/checkpoint-300]

Requirements: A100 80GB (recommended) or 2× A10G 24GB with device_map="auto"
Training time: ~6 hours on A100 (3 epochs, 8K samples, max_seq_len=4096)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import typer
import yaml
from loguru import logger
from rich.console import Console
from rich.table import Table
from trl import SFTTrainer, SFTConfig

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from training.src.lora_config import LoRATrainingConfig, build_model_and_tokenizer
from training.src.dataset import (
    FinancialDatasetBuilder,
    FinancialDatasetConfig,
    dataset_stats,
)
from training.src.callbacks import RichProgressCallback, WandbSummaryCallback
from training.src.collator import DataCollatorForFinancialSummarization

console = Console()
app     = typer.Typer()


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_yaml_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def print_training_summary(cfg: dict) -> None:
    table = Table(title="Training Configuration", show_header=True)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    flat = {}
    for section, vals in cfg.items():
        if isinstance(vals, dict):
            for k, v in vals.items():
                flat[f"{section}.{k}"] = str(v)
        else:
            flat[section] = str(vals)

    for k, v in flat.items():
        table.add_row(k, v)
    console.print(table)


# ── Main training function ────────────────────────────────────────────────────

def train(config_path: str, output_dir: str, resume_from: str | None = None) -> None:
    cfg = load_yaml_config(config_path)
    print_training_summary(cfg)

    # ── W&B setup ─────────────────────────────────────────────────────────────
    import wandb
    wandb.init(
        project=cfg.get("wandb", {}).get("project", "financial-llm"),
        name=cfg.get("training", {}).get("run_name", "mistral-finance"),
        tags=cfg.get("wandb", {}).get("tags", []),
        config=cfg,
    )

    # ── Build model + tokenizer ───────────────────────────────────────────────
    lora_cfg = LoRATrainingConfig(
        base_model_id=cfg["model"]["base_model_id"],
        lora_r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        target_modules=cfg["lora"]["target_modules"],
        load_in_4bit=cfg["quantization"]["load_in_4bit"],
        use_flash_attention=cfg["model"].get("attn_implementation") == "flash_attention_2",
        max_seq_length=cfg["data"]["max_seq_length"],
    )

    with console.status("[bold green]Loading model and applying LoRA…"):
        model, tokenizer = build_model_and_tokenizer(lora_cfg)

    console.print(f"[green]✓ Model loaded: {cfg['model']['base_model_id']}")
    model.print_trainable_parameters()

    # ── Build dataset ─────────────────────────────────────────────────────────
    data_cfg = FinancialDatasetConfig(
        train_file=cfg["data"]["train_file"],
        eval_file=cfg["data"]["eval_file"],
        max_seq_length=cfg["data"]["max_seq_length"],
    )

    with console.status("[bold green]Building dataset…"):
        builder = FinancialDatasetBuilder(data_cfg, tokenizer)
        dataset = builder.build()

    stats = dataset_stats(dataset)
    for split, s in stats.items():
        console.print(
            f"[blue]{split}[/]: {s['n_examples']:,} examples | "
            f"mean {s['mean_tokens']:.0f} tokens | "
            f"p95 {s['p95_tokens']:.0f} tokens"
        )
    wandb.log({"dataset_stats": stats})

    # ── SFT training arguments ────────────────────────────────────────────────
    t = cfg["training"]
    sft_cfg = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        gradient_checkpointing=t["gradient_checkpointing"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        optim=t["optim"],
        bf16=t["bf16"],
        tf32=t.get("tf32", False),
        max_grad_norm=t["max_grad_norm"],
        eval_strategy=t["eval_strategy"],
        eval_steps=t["eval_steps"],
        save_strategy=t["save_strategy"],
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        load_best_model_at_end=t["load_best_model_at_end"],
        metric_for_best_model=t["metric_for_best_model"],
        greater_is_better=t["greater_is_better"],
        logging_steps=t["logging_steps"],
        report_to=t.get("report_to", ["wandb"]),
        run_name=t.get("run_name"),
        seed=t.get("seed", 42),
        group_by_length=t.get("group_by_length", True),
        # SFT-specific
        max_seq_length=cfg["data"]["max_seq_length"],
        dataset_text_field="text",
        packing=cfg["data"].get("packing", True),
        dataset_num_proc=cfg["data"].get("num_workers", 4),
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    from transformers import EarlyStoppingCallback
    callbacks = [
        RichProgressCallback(),
        WandbSummaryCallback(),
        EarlyStoppingCallback(early_stopping_patience=3),
    ]

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        args=sft_cfg,
        callbacks=callbacks,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    console.print("[bold green]Starting training…")
    train_result = trainer.train(resume_from_checkpoint=resume_from)

    # Log final metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    wandb.log({"train_metrics": metrics})

    # ── Save ──────────────────────────────────────────────────────────────────
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    console.print(f"[bold green]✓ Adapter saved to {output_dir}")
    console.print(f"[bold green]  train_loss={metrics.get('train_loss', 'N/A'):.4f}")

    # Save config for reproducibility
    import json
    with open(Path(output_dir) / "training_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    wandb.finish()
    return trainer


# ── CLI ───────────────────────────────────────────────────────────────────────

@app.command()
def main(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config"),
    output_dir: str = typer.Option("./checkpoints/model", "--output-dir", "-o"),
    resume_from: str = typer.Option(None, "--resume-from-checkpoint"),
    log_level: str = typer.Option("INFO", "--log-level"),
):
    """Fine-tune Mistral-7B with LoRA on financial documents."""
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))
    logger.info(f"Config: {config} → {output_dir}")
    train(config, output_dir, resume_from)


if __name__ == "__main__":
    app()
