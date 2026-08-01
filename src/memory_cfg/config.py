"""Experiment configuration + reproducibility helpers.

Every run writes its config next to its checkpoint and predictions, so a number
in a table can always be traced back to the settings that produced it.
"""
import os
import json
import random
from dataclasses import dataclass, asdict, field, fields


@dataclass
class EncoderConfig:
    max_len: int = 512
    n_layers: int = 6
    dim: int = 384
    n_heads: int = 6
    vocab_size: int = 8000
    mlm_probability: float = 0.15
    epochs: int = 3
    batch_size: int = 32
    lr: float = 5e-4


@dataclass
class ModelConfig:
    function_hidden: int = 128
    max_clusters: int = 16
    fcg_hidden: int = 128
    dropout: float = 0.2
    lr: float = 1e-3
    epochs: int = 10
    batch_size: int = 8      # binaries accumulated per optimizer step
    lam_family: float = 1.0
    lam_link: float = 0.1
    lam_entropy: float = 0.1


@dataclass
class ExperimentConfig:
    seed: int = 1337
    reports_dir: str = "reports/"
    encoder_dir: str = "asm_encoder/"
    output_dir: str = "runs/current"
    test_size: float = 0.3
    families: list = field(default_factory=list)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        return path

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_dict(cls, raw):
        """Rebuild from a saved dict, tolerating fields added since it was written."""
        raw = dict(raw)
        for key, klass in (("encoder", EncoderConfig), ("model", ModelConfig)):
            if isinstance(raw.get(key), dict):
                known = {f.name for f in fields(klass)}
                raw[key] = klass(**{k: v for k, v in raw[key].items() if k in known})
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        # CPU reductions are summed in thread-completion order, so the same
        # seed still drifts from run to run on a multicore box. A results table
        # that moves when nothing changed is worse than a slower one.
        torch.set_num_threads(1)
    except ImportError:
        pass
