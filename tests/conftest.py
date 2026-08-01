import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

FIXTURE = os.path.join(REPO_ROOT, "examples", "sample_pid.json")


@pytest.fixture(scope="session")
def repo_root():
    return REPO_ROOT


@pytest.fixture(scope="session")
def real_report():
    """The committed SMDA fixture - a real carved function, trimmed."""
    from data_tokenizer import load_report
    if not os.path.exists(FIXTURE):
        pytest.skip("examples/sample_pid.json is missing")
    return load_report(FIXTURE)


@pytest.fixture(scope="session")
def instruction_pool(real_report):
    from synth import harvest_instruction_pool, _FALLBACK_POOL
    return harvest_instruction_pool(real_report) + list(_FALLBACK_POOL)


@pytest.fixture(scope="session")
def synthetic_report(instruction_pool):
    from synth import generate_report, PROFILES_BY_NAME
    return generate_report(PROFILES_BY_NAME["Backdoor"], 11, instruction_pool)


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory, real_report):
    """A miniature end-to-end corpus: reports + trained encoder."""
    import synth
    from data_tokenizer import write_corpus_from_dir
    from tokenizer_train import train_tokenizer
    from transformer_train import (
        load_tokenizer, read_corpus, train_encoder,
    )

    root = tmp_path_factory.mktemp("dataset")
    reports_dir = root / "reports"
    spec = synth.DatasetSpec(n_per_profile=2, seed=5)
    labels = synth.generate_dataset(str(reports_dir), spec, real_report)

    import json
    labels_path = reports_dir / "labels.json"
    labels_path.write_text(json.dumps(labels))

    corpus = root / "corpus.txt"
    write_corpus_from_dir(str(reports_dir), str(corpus))

    tokenizer_path = root / "asm_tokenizer.json"
    train_tokenizer(str(corpus), str(tokenizer_path), vocab_size=400,
                    min_frequency=1)

    encoder_dir = root / "asm_encoder"
    train_encoder(read_corpus(str(corpus)), load_tokenizer(str(tokenizer_path)),
                  str(encoder_dir), epochs=1, batch_size=16, max_len=64,
                  n_layers=1, dim=32, n_heads=2, warmup=2, log_every=0)

    return {
        "root": str(root),
        "reports_dir": str(reports_dir),
        "labels": labels,
        "labels_path": str(labels_path),
        "corpus": str(corpus),
        "encoder_dir": str(encoder_dir),
    }
