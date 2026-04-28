"""
training/src/collator.py
─────────────────────────
Custom data collator that:
  1. Dynamically pads sequences to the longest in the batch (not global max_seq_len)
  2. Masks padding tokens in labels with -100 (ignored by cross-entropy)
  3. Masks the instruction/prompt tokens from the loss — we only train on completions
     (the summary). This is critical: without this, the model would also try to
     predict the prompt, which wastes capacity and hurts summarisation quality.

Prompt masking strategy:
  Find the [/INST] token boundary in each sequence.
  Set labels[: boundary_idx] = -100  (ignore prompt in loss)
  Set labels[boundary_idx :] = input_ids[boundary_idx :]  (learn summary)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import PreTrainedTokenizer


IGNORE_INDEX = -100

# Mistral [/INST] token representation in the tokenizer
RESPONSE_BOUNDARY_TOKENS = [
    "[/INST]",          # Mistral instruct format
    "<|im_start|>assistant",   # ChatML format fallback
]


@dataclass
class DataCollatorForFinancialSummarization:
    """
    Dynamic padding collator with prompt-masking.

    Args:
        tokenizer: The tokenizer (must have pad_token set)
        max_length: Hard truncation ceiling
        pad_to_multiple_of: Round up padding to this value (e.g. 8 for tensor cores)
        mask_prompt: If True, set prompt token labels to IGNORE_INDEX
    """

    tokenizer: PreTrainedTokenizer
    max_length: int = 4096
    pad_to_multiple_of: Optional[int] = 8
    mask_prompt: bool = True

    def __post_init__(self):
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract sequences
        input_ids_list = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        labels_list    = [torch.tensor(f.get("labels", f["input_ids"]), dtype=torch.long) for f in features]

        # Determine batch max length
        max_len = max(len(s) for s in input_ids_list)
        if self.pad_to_multiple_of:
            max_len = (max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of * self.pad_to_multiple_of
        max_len = min(max_len, self.max_length)

        pad_id = self.tokenizer.pad_token_id or 0

        input_ids_padded  = []
        attention_masks   = []
        labels_padded     = []

        for ids, labs in zip(input_ids_list, labels_list):
            # Truncate
            ids  = ids[:max_len]
            labs = labs[:max_len]
            seq_len = len(ids)
            pad_len = max_len - seq_len

            # Pad input_ids
            padded_ids = torch.cat([ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
            input_ids_padded.append(padded_ids)

            # Attention mask
            mask = torch.cat([
                torch.ones(seq_len, dtype=torch.long),
                torch.zeros(pad_len, dtype=torch.long),
            ])
            attention_masks.append(mask)

            # Labels: pad positions → -100
            padded_labs = torch.cat([labs, torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long)])

            # Prompt masking: set instruction tokens to -100
            if self.mask_prompt:
                padded_labs = self._mask_prompt_tokens(padded_ids, padded_labs)

            labels_padded.append(padded_labs)

        return {
            "input_ids":      torch.stack(input_ids_padded),
            "attention_mask": torch.stack(attention_masks),
            "labels":         torch.stack(labels_padded),
        }

    def _mask_prompt_tokens(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Find the end of the instruction prompt and mask everything before it.
        
        Strategy: search for the token ID sequence corresponding to [/INST].
        Set all labels at positions before [/INST] to IGNORE_INDEX.
        """
        # Find [/INST] token sequence
        boundary_ids = self.tokenizer.encode("[/INST]", add_special_tokens=False)
        
        if not boundary_ids:
            return labels

        # Search for the boundary in input_ids
        ids_list = input_ids.tolist()
        boundary_pos = self._find_subsequence(ids_list, boundary_ids)

        if boundary_pos == -1:
            # No boundary found — mask first 80% as a fallback heuristic
            fallback = int(len(ids_list) * 0.8)
            labels = labels.clone()
            labels[:fallback] = IGNORE_INDEX
            return labels

        # Mask everything up to (and including) [/INST]
        labels = labels.clone()
        labels[: boundary_pos + len(boundary_ids)] = IGNORE_INDEX
        return labels

    @staticmethod
    def _find_subsequence(seq: list[int], subseq: list[int]) -> int:
        """Return start index of last occurrence of subseq in seq, or -1."""
        n, m = len(seq), len(subseq)
        last = -1
        for i in range(n - m + 1):
            if seq[i:i+m] == subseq:
                last = i
        return last
