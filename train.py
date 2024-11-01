"""Stage-2 training: freeze the encoder, train the CFG DiffPool encoder, the
function-graph GNN and the two classification heads end to end.

    L = L_binary
      + lam_family  * I[malware] * L_family
      + lam_link    * mean(L_diffpool_link)
      + lam_entropy * mean(L_diffpool_entropy)

A run is only reportable if it leaves behind the four artifacts this module
writes: `config.json`, `split.json`, `checkpoint.pt`, and `predictions.json`.
Everything downstream reads those files rather than whatever happened to be in
memory when the run ended.
"""
import os
import json
import time
import random
import argparse

import torch

from config import ExperimentConfig, set_seed
from Data_Loader import DataLoader
from embed_blocks import BlockEmbedder
from DiffPool import FunctionEncoder
from model import HierClassifier
from sample import build_sample


def load_labels(path):
    """Manifest: {report_filename: {"is_malware": 0|1, "family": "..."}}."""
    with open(path) as f:
        return json.load(f)


def families_from_labels(labels):
    return sorted({m["family"] for m in labels.values() if m.get("family")})


def train(report_dir, encoder_dir, labels, families=None, config=None,
          epochs=None, lr=None, lam_link=None, lam_ent=None, lam_family=None,
          train_names=None, cache_dir="embeddings_cache", device="cpu",
          batch_size=None, log=print):
    """Train the stage-2 model. Returns (function_encoder, classifier, history)."""
    config = config or ExperimentConfig()
    families = families if families is not None else families_from_labels(labels)
    epochs = config.model.epochs if epochs is None else epochs
    batch_size = config.model.batch_size if batch_size is None else batch_size
    lr = config.model.lr if lr is None else lr
    lam_link = config.model.lam_link if lam_link is None else lam_link
    lam_ent = config.model.lam_entropy if lam_ent is None else lam_ent
    lam_family = config.model.lam_family if lam_family is None else lam_family

    embedder = BlockEmbedder(encoder_dir, max_len=config.encoder.max_len,
                             device=device)
    in_dim = embedder.dim

    fenc = FunctionEncoder(in_dim, hidden_dim=config.model.function_hidden,
                           max_clusters=config.model.max_clusters).to(device)
    clf = HierClassifier(fenc.hidden_dim, config.model.fcg_hidden,
                         num_families=max(1, len(families)),
                         dropout=config.model.dropout).to(device)
    optimizer = torch.optim.Adam(
        list(fenc.parameters()) + list(clf.parameters()), lr=lr)
    fam_index = {f: i for i, f in enumerate(families)}

    names = train_names if train_names is not None else None
    n_pos = sum(1 for n, m in labels.items()
                if (names is None or n in names) and m["is_malware"])
    n_neg = max(1, (len(names) if names else len(labels)) - n_pos)
    # a corpus that is mostly malware would otherwise train a model that just
    # says "malware" - weight the positive class by the actual imbalance
    pos_weight = n_neg / max(1, n_pos)

    order = list(names) if names is not None else sorted(labels)
    parameters = list(fenc.parameters()) + list(clf.parameters())

    history = []
    for epoch in range(epochs):
        fenc.train()
        clf.train()
        total, n_seen, correct, started = 0.0, 0, 0, time.time()

        for batch in DataLoader(report_dir, mode="spatial", names=order):
            meta = labels[batch["name"]]
            s = build_sample(batch["report"], embedder, fenc,
                             meta["is_malware"], meta.get("family"),
                             cache_dir=cache_dir)

            bin_logit, fam_logits = clf(s.node_embeddings, s.fcg_edge_index)
            fam_idx = fam_index.get(s.family) if s.family else None

            cls_loss = clf.loss(bin_logit, fam_logits, s.is_malware, fam_idx,
                                lam_family=lam_family, pos_weight=pos_weight)
            # normalize per function so a 4k-function binary does not drown a
            # 20-function one, and weight link/entropy with their own lambdas
            # instead of collapsing both into one blended term
            n_functions = max(1, s.aux_loss["n_functions"])
            loss = (cls_loss
                    + lam_link * s.aux_loss["link"] / n_functions
                    + lam_ent * s.aux_loss["entropy"] / n_functions)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()

            total += float(loss.detach())
            n_seen += 1
            correct += int((float(torch.sigmoid(bin_logit.detach())) >= 0.5)
                           == bool(meta["is_malware"]))

        mean = total / max(1, n_seen)
        accuracy = correct / max(1, n_seen)
        history.append({"epoch": epoch, "loss": mean, "train_accuracy": accuracy,
                        "seconds": round(time.time() - started, 2)})
        log(f"epoch {epoch}: loss {mean:.4f} train_acc {accuracy:.3f} "
            f"({history[-1]['seconds']}s)")

    return fenc, clf, history
