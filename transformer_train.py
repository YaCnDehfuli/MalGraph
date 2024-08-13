"""Self-supervised DistilBERT masked-LM pretraining over the assembly corpus.

Each line of corpus.txt is one basic block (instructions joined by [INS]) or one
import pseudo-block. We learn contextual instruction/opcode representations with
a small DistilBERT, then reuse the encoder to embed basic blocks downstream.

The loop is plain PyTorch on purpose. `transformers.Trainer` pulls in
`accelerate` and renames its own arguments between majors (`no_cuda` became
`use_cpu` in v5), which is exactly the kind of breakage that stops a research
repo from running a year later. Only the model and tokenizer classes come from
`transformers`, and those have been stable.
"""
import os
import json
import math
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    PreTrainedTokenizerFast,
    DistilBertConfig,
    DistilBertForMaskedLM,
)

# blocks are tiny (median 3 instr, p99 ~25); 4096 was copied from longformer and
# just wasted memory. 512 is already far more than any real block needs.
MAX_LEN = 512


def load_tokenizer(path="asm_tokenizer.json"):
    return PreTrainedTokenizerFast(
        tokenizer_file=path,
        unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
        sep_token="[SEP]", mask_token="[MASK]",
        additional_special_tokens=["[INS]", "[API]"],
    )


def read_corpus(path):
    with open(path, "r") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


class AsmMLMDataset(Dataset):
    """Tokenize each basic-block line; masking is applied by the collator."""

    def __init__(self, lines, tokenizer, max_len=MAX_LEN):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.lines = lines

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.lines[idx],
            truncation=True,
            max_length=self.max_len,
            return_attention_mask=True,
        )
        # a line that tokenizes to nothing would produce a zero-length row and
        # break padding; fall back to a single [UNK].
        ids = enc["input_ids"] or [self.tokenizer.unk_token_id]
        mask = enc["attention_mask"] or [1]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


def bucket_by_length(lines, boundaries=(16, 32, 64, 128)):
    """Group lines into length buckets so batches don't over-pad."""
    buckets = {b: [] for b in boundaries + (float("inf"),)}
    for line in lines:
        n = line.count("[INS]") + 1
        for b in buckets:
            if n <= b:
                buckets[b].append(line)
                break
    return buckets


def mlm_collator(features, tokenizer, mlm_prob=0.15):
    """Dynamic masking collator for masked-LM."""
    batch = tokenizer.pad(features, return_tensors="pt")
    input_ids = batch["input_ids"]
    labels = input_ids.clone()

    probability = torch.full(labels.shape, mlm_prob)
    masked = torch.bernoulli(probability).bool()
    input_ids[masked] = tokenizer.mask_token_id

    batch["input_ids"] = input_ids
    batch["labels"] = labels
    return batch
