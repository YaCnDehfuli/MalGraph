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

MAX_LEN = 4096


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
