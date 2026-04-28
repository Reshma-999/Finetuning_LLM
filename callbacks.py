"""
training/src/callbacks.py
──────────────────────────
Custom training callbacks for rich progress display and W&B summaries.
"""

from __future__ import annotations

import time
from typing import Any

import wandb
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

console = Console()


class RichProgressCallback(TrainerCallback):
    """
    Replaces the default HuggingFace tqdm bar with a Rich progress display.
    Shows: loss, eval ROUGE-L, learning rate, and estimated time remaining.
    """

    def __init__(self):
        self.progress = None
        self.train_task = None
        self.eval_task  = None
        self._start_time = None

    def on_train_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **_
    ):
        self._start_time = time.time()
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
        )
        self.progress.start()
        total_steps = state.max_steps or (
            args.num_train_epochs * (state.num_train_epochs or 1)
        )
        self.train_task = self.progress.add_task(
            f"[green]Epoch 1/{int(args.num_train_epochs)}",
            total=total_steps,
        )
        console.print(
            f"[bold]Training started[/] | "
            f"steps={state.max_steps} | "
            f"batch={args.per_device_train_batch_size}×"
            f"{args.gradient_accumulation_steps} | "
            f"lr={args.learning_rate}"
        )

    def on_step_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ):
        if self.train_task is None:
            return
        logs = kwargs.get("logs", {})
        loss = logs.get("loss", state.log_history[-1].get("loss", "?") if state.log_history else "?")
        lr   = logs.get("learning_rate", "?")

        epoch = int(state.epoch or 1)
        self.progress.update(
            self.train_task,
            advance=1,
            description=(
                f"[green]Epoch {epoch}/{int(args.num_train_epochs)} "
                f"loss={loss:.4f} lr={lr:.2e}"
                if isinstance(loss, float) else
                f"[green]Epoch {epoch}/{int(args.num_train_epochs)}"
            ),
        )

    def on_evaluate(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: dict, **_
    ):
        rouge_l = metrics.get("eval_rouge_l", metrics.get("eval_loss", "?"))
        bert_f1 = metrics.get("eval_bert_f1", "?")
        console.print(
            f"[cyan]Eval step {state.global_step}[/] | "
            f"rouge_l={rouge_l:.4f}" if isinstance(rouge_l, float) else
            f"[cyan]Eval step {state.global_step}[/]"
        )

    def on_train_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **_
    ):
        if self.progress:
            self.progress.stop()
        elapsed = time.time() - (self._start_time or time.time())
        console.print(
            f"[bold green]✓ Training complete[/] | "
            f"total_steps={state.global_step} | "
            f"elapsed={elapsed/3600:.1f}h"
        )


class WandbSummaryCallback(TrainerCallback):
    """Log best eval metrics to W&B run summary."""

    def __init__(self):
        self.best_rouge_l = 0.0
        self.best_step    = 0

    def on_evaluate(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: dict, **_
    ):
        rouge_l = metrics.get("eval_rouge_l", 0.0)
        if rouge_l > self.best_rouge_l:
            self.best_rouge_l = rouge_l
            self.best_step    = state.global_step
            if wandb.run:
                wandb.run.summary["best_rouge_l"]  = rouge_l
                wandb.run.summary["best_rouge_l_step"] = state.global_step
                wandb.run.summary["best_bert_f1"]  = metrics.get("eval_bert_f1", 0.0)
                wandb.run.summary["best_eval_loss"] = metrics.get("eval_loss", 0.0)

    def on_train_end(self, *args, **kwargs):
        if wandb.run:
            wandb.run.summary["final_best_rouge_l"] = self.best_rouge_l
            wandb.run.summary["final_best_step"]    = self.best_step


class GradientNormCallback(TrainerCallback):
    """Log gradient norm to W&B every N steps."""

    def __init__(self, log_every: int = 50):
        self.log_every = log_every

    def on_step_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ):
        if state.global_step % self.log_every != 0:
            return
        model = kwargs.get("model")
        if model is None:
            return
        import torch
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.detach().norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        if wandb.run:
            wandb.log({"grad_norm": total_norm}, step=state.global_step)
