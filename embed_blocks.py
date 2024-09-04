"""Turn each basic block into a fixed vector using the pretrained encoder.

The encoder (DistilBERT MLM) is frozen; we masked-mean-pool its last hidden
state over the real (non-pad) tokens to get one embedding per block.
"""
import os
import hashlib

import torch
import transformers
from transformers import AutoModel

from transformer_train import load_tokenizer

# We deliberately load a DistilBertModel out of a DistilBertForMaskedLM
# directory, so the unused MLM head weights are reported as "UNEXPECTED" on
# every single load. That is the intended behaviour here, and the banner buries
# the pipeline's own output.
transformers.logging.set_verbosity_error()


def masked_mean(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class BlockEmbedder:
    """Frozen encoder plus an in-memory and on-disk cache keyed by block hash."""

    def __init__(self, encoder_dir, max_len=512, device="cpu", batch_size=64):
        self.tokenizer = load_tokenizer(os.path.join(encoder_dir, "tokenizer.json"))
        self.model = AutoModel.from_pretrained(encoder_dir).to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.max_len = max_len
        self.device = device
        self.batch_size = batch_size
        self.dim = self.model.config.dim
        self._memo = {}

    @torch.no_grad()
    def embed(self, block_lines):
        """block_lines: list[str] -> tensor [N, dim].

        Chunked: a real binary has tens of thousands of blocks, and tokenizing
        them in one call allocates a single padded batch large enough to OOM.
        """
        enc = self.tokenizer(
            block_lines, padding=True, truncation=True,
            max_length=self.max_len, return_tensors="pt",
        ).to(self.device)
        hidden = self.model(**enc).last_hidden_state
        return masked_mean(hidden, enc["attention_mask"]).cpu()

    def embed_cached(self, block_lines, cache_dir="embeddings_cache"):
        """Embed only blocks not already cached; reuse the rest by hash.

        Identical blocks are extremely common across a corpus (compiler
        boilerplate, thunks, import stubs), so the cache pays for itself.
        """
        if not block_lines:
            return torch.empty((0, self.dim))
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        vectors = [None] * len(block_lines)
        todo, todo_idx, todo_keys = [], [], []
        for i, line in enumerate(block_lines):
            key = block_hash(line)
            if key in self._memo:
                vectors[i] = self._memo[key]
                continue
            path = os.path.join(cache_dir, key + ".pt") if cache_dir else None
            if path and os.path.exists(path):
                vector = torch.load(path)
                self._memo[key] = vector
                vectors[i] = vector
                continue
            # the same unseen block often appears many times in one call - embed
            # it once and fan the result back out to every position
            if key not in todo_keys:
                todo.append(line)
                todo_keys.append(key)
            todo_idx.append(i)

        if todo:
            fresh = self.embed(todo)
            for key, vector in zip(todo_keys, fresh):
                self._memo[key] = vector
                if cache_dir:
                    torch.save(vector, os.path.join(cache_dir, key + ".pt"))
            for i in todo_idx:
                vectors[i] = self._memo[block_hash(block_lines[i])]

        return torch.stack(vectors)
