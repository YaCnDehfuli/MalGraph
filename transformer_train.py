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


def build_model(tokenizer, n_layers=6, dim=384, n_heads=6, max_len=MAX_LEN):
    """A small DistilBERT masked-LM sized to the assembly vocabulary."""
    config = DistilBertConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=max_len,
        n_layers=n_layers,
        dim=dim,
        hidden_dim=dim * 4,
        n_heads=n_heads,
        pad_token_id=tokenizer.pad_token_id,
    )
    return DistilBertForMaskedLM(config)


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
    special = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in (tokenizer.pad_token_id, tokenizer.cls_token_id,
                     tokenizer.sep_token_id, tokenizer.mask_token_id):
        if token_id is not None:
            special |= input_ids == token_id
    masked = torch.bernoulli(probability).bool() & ~special
    if not bool(masked.any()):
        # a batch where nothing got masked has no contributing positions and the
        # cross-entropy comes back NaN; force one real position.
        candidates = (~special).nonzero()
        if len(candidates):
            row, col = candidates[0]
            masked[row, col] = True
    # only masked positions contribute to the loss; everything else is ignored
    labels[~masked] = -100
    input_ids[masked] = tokenizer.mask_token_id

    batch["input_ids"] = input_ids
    batch["labels"] = labels
    return batch


def _make_dummy_corpus(n=16, seed=0):
    import random
    rng = random.Random(seed)
    vocab = ["push rbx", "mov rax, IMM", "call ADDR", "test eax, eax",
             "je ADDR", "ret", "add rsp, IMM", "xor eax, eax", "cmp rbx, IMM"]
    lines = []
    for _ in range(n):
        k = rng.randint(2, 6)
        lines.append(" [INS] ".join(rng.choice(vocab) for _ in range(k)))
    return lines


def _lr_at(step, total_steps, base_lr, warmup):
    """Linear warmup then linear decay - the schedule Trainer used to supply."""
    if warmup and step < warmup:
        return base_lr * (step + 1) / warmup
    remaining = max(1, total_steps - warmup)
    return base_lr * max(0.0, (total_steps - step) / remaining)


def save_encoder(model, tokenizer, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def train_encoder(lines, tokenizer, output_dir, epochs=3, batch_size=32,
                  lr=5e-4, warmup=100, max_len=MAX_LEN, n_layers=6, dim=384,
                  n_heads=6, mlm_prob=0.15, device="cpu", resume=None,
                  log_every=50, seed=1337):
    """Pretrain the encoder and write it to `output_dir` in HF format."""
    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    dataset = AsmMLMDataset(lines, tokenizer, max_len=max_len)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=lambda feats: mlm_collator(feats, tokenizer, mlm_prob),
        generator=torch.Generator().manual_seed(seed),
    )

    model = build_model(tokenizer, n_layers=n_layers, dim=dim,
                        n_heads=n_heads, max_len=max_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    start_epoch, global_step = 0, 0
    total_steps = max(1, epochs * len(loader))
    history = []
    model.train()
    for epoch in range(start_epoch, epochs):
        running, n_batches = 0.0, 0
        for batch in loader:
            for group in optimizer.param_groups:
                group["lr"] = _lr_at(global_step, total_steps, lr, warmup)
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step_loss = float(loss.detach())
            running += step_loss
            n_batches += 1
            global_step += 1
            if log_every and global_step % log_every == 0:
                print(f"  step {global_step}/{total_steps} loss {step_loss:.4f}")

        mean_loss = running / max(1, n_batches)
        history.append({
            "epoch": epoch,
            "mlm_loss": mean_loss,
            "perplexity": math.exp(min(20.0, mean_loss)),
        })
        print(f"epoch {epoch}: mlm_loss {mean_loss:.4f} "
              f"ppl {history[-1]['perplexity']:.2f}")

    save_encoder(model, tokenizer, output_dir)
    with open(os.path.join(output_dir, "pretrain_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    return model, history


def smoke():
    """Tiny CPU run over dummy data to prove the pipeline executes end to end."""
    import tempfile
    from tokenizer_train import train_tokenizer

    tmp = tempfile.mkdtemp(prefix="asm_smoke_")
    corpus_path = os.path.join(tmp, "corpus.txt")
    lines = _make_dummy_corpus()
    with open(corpus_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    tok_path = os.path.join(tmp, "tok.json")
    train_tokenizer(corpus_path, tok_path, vocab_size=200, min_frequency=1)
    tokenizer = load_tokenizer(tok_path)

    train_encoder(lines, tokenizer, os.path.join(tmp, "enc"), epochs=1,
                  batch_size=4, max_len=64, n_layers=2, dim=64, n_heads=2,
                  warmup=2, log_every=0)
    print("smoke OK")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", default="corpus.txt")
    ap.add_argument("--tokenizer", default="asm_tokenizer.json")
    ap.add_argument("--output-dir", default="asm_encoder")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--smoke", action="store_true", help="tiny CPU dry run")
    ap.add_argument("--resume", default=None,
                    help="path to encoder_ckpt.pt, or None to start fresh")
    args = ap.parse_args(argv)

    if args.smoke:
        smoke()
        return

    tokenizer = load_tokenizer(args.tokenizer)
    lines = read_corpus(args.corpus)
    print(f"loaded {len(lines)} block sequences, vocab {len(tokenizer)}")
    train_encoder(
        lines, tokenizer, args.output_dir, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, warmup=args.warmup,
        max_len=args.max_len, n_layers=args.layers, dim=args.dim,
        n_heads=args.heads, device=args.device, resume=args.resume,
        seed=args.seed,
    )
    print(f"encoder written to {args.output_dir}")


if __name__ == "__main__":
    main()
