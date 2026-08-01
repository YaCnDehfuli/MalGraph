"""Train a WordPiece tokenizer over the assembly corpus (corpus.txt)."""
import argparse

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace


# [INS] separates instructions and [API] marks an import pseudo-block; [MASK] is
# needed for MLM; [PAD]/[CLS]/[SEP] for BERT. Whitespace pre-tokenization would
# otherwise shred "[INS]" into "[", "INS", "]".
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[INS]", "[API]"]


# NOTE ON REPRODUCIBILITY: this step is not bit-reproducible, and it is the only
# stage that isn't. `tokenizers` resolves ties at the vocabulary cutoff through a
# randomly-seeded Rust hash map, so two runs over an identical corpus learn
# vocabularies of the same size that differ in a few hundred rare subwords. The
# trainer exposes no seed, and RAYON_NUM_THREADS=1 does not change it. Every
# token id downstream shifts with it, which is why retraining end to end lands a
# few points away from a previous run. Save the tokenizer alongside the encoder
# and the checkpoint - that trio does reproduce predictions exactly.
def train_tokenizer(corpus_path, out_path="asm_tokenizer.json", vocab_size=8000,
                    min_frequency=2):
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordPieceTrainer(vocab_size=vocab_size,
                               min_frequency=min_frequency,
                               special_tokens=SPECIAL_TOKENS)
    tokenizer.train([corpus_path], trainer)
    tokenizer.save(out_path)
    return tokenizer


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", default="corpus.txt")
    ap.add_argument("--out", default="asm_tokenizer.json")
    ap.add_argument("--vocab-size", type=int, default=8000)
    ap.add_argument("--min-frequency", type=int, default=2)
    args = ap.parse_args(argv)
    tok = train_tokenizer(args.corpus, args.out, args.vocab_size, args.min_frequency)
    print(f"vocab size: {tok.get_vocab_size()} -> {args.out}")
    return tok


if __name__ == "__main__":
    main()
