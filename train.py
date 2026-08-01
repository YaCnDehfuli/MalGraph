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


def make_split(labels, test_size=0.3, seed=1337):
    """Stratified holdout over (is_malware, family) so both splits see every class.

    Grouping matters more than the exact ratio here: with a handful of binaries
    per family a random split routinely leaves a family entirely out of train,
    and the family head then has nothing to learn from.
    """
    rng = random.Random(seed)
    strata = {}
    for name, meta in sorted(labels.items()):
        key = (meta["is_malware"], meta.get("family") or "")
        strata.setdefault(key, []).append(name)

    train, test = [], []
    for key in sorted(strata):
        names = sorted(strata[key])
        rng.shuffle(names)
        n_test = max(1, round(len(names) * test_size)) if len(names) > 1 else 0
        test.extend(names[:n_test])
        train.extend(names[n_test:])
    return sorted(train), sorted(test)


def save_split(path, train, test, labels):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "train": train,
        "test": test,
        "labels": {name: labels[name] for name in train + test},
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def save_checkpoint(path, fenc, clf, families, config, encoder_dim):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "fenc": fenc.state_dict(),
        "clf": clf.state_dict(),
        "families": families,
        "encoder_dim": encoder_dim,
        "function_hidden": fenc.hidden_dim,
        "fcg_hidden": config.model.fcg_hidden,
        "max_clusters": config.model.max_clusters,
        "dropout": config.model.dropout,
    }, path)
    return path


@torch.no_grad()
def predict_split(names, report_dir, labels, embedder, fenc, clf, families,
                  cache_dir):
    """Score a list of reports; returns rows ready to be written to disk."""
    fenc.eval()
    clf.eval()
    rows = []
    for batch in DataLoader(report_dir, mode="spatial", names=names):
        meta = labels.get(batch["name"], {})
        s = build_sample(batch["report"], embedder, fenc,
                         meta.get("is_malware", 0), meta.get("family"),
                         cache_dir=cache_dir)
        bin_logit, fam_logits = clf(s.node_embeddings, s.fcg_edge_index)
        probability = float(torch.sigmoid(bin_logit))
        predicted = families[int(fam_logits.argmax())] if families else None
        rows.append({
            "report": batch["name"],
            "y_true": int(meta.get("is_malware", 0)),
            "y_score": probability,
            "family_true": meta.get("family"),
            "family_pred": predicted,
            "num_functions": s.num_functions,
            "num_nodes": len(s.node_ids),
        })
    return rows


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
    rng = random.Random(config.seed)
    parameters = list(fenc.parameters()) + list(clf.parameters())

    history = []
    for epoch in range(epochs):
        fenc.train()
        clf.train()
        # a fixed alphabetical order means every epoch sees the classes in the
        # same runs, which is exactly the correlation SGD should not get
        rng.shuffle(order)
        total, n_seen, correct, started = 0.0, 0, 0, time.time()
        optimizer.zero_grad()
        pending = 0

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

            # one binary is one graph, so a "batch" has to be accumulated by
            # hand; stepping per sample made the gradient pure noise
            (loss / batch_size).backward()
            pending += 1
            if pending == batch_size:
                torch.nn.utils.clip_grad_norm_(parameters, 5.0)
                optimizer.step()
                optimizer.zero_grad()
                pending = 0

            total += float(loss.detach())
            n_seen += 1
            correct += int((float(torch.sigmoid(bin_logit.detach())) >= 0.5)
                           == bool(meta["is_malware"]))

        if pending:
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            optimizer.zero_grad()

        mean = total / max(1, n_seen)
        accuracy = correct / max(1, n_seen)
        history.append({"epoch": epoch, "loss": mean, "train_accuracy": accuracy,
                        "seconds": round(time.time() - started, 2)})
        log(f"epoch {epoch}: loss {mean:.4f} train_acc {accuracy:.3f} "
            f"({history[-1]['seconds']}s)")

    return fenc, clf, history


def run_experiment(config, labels_path=None, device="cpu", log=print):
    """Full stage-2 experiment: split, train, score, and write every artifact."""
    set_seed(config.seed)
    labels_path = labels_path or os.path.join(config.reports_dir, "labels.json")
    labels = load_labels(labels_path)
    families = config.families or families_from_labels(labels)
    config.families = families

    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)
    config.save(os.path.join(output_dir, "config.json"))

    train_names, test_names = make_split(labels, config.test_size, config.seed)
    save_split(os.path.join(output_dir, "split.json"),
               train_names, test_names, labels)
    log(f"split: {len(train_names)} train / {len(test_names)} test, "
        f"families={families}")

    cache_dir = os.path.join(output_dir, "embeddings_cache")
    fenc, clf, history = train(
        config.reports_dir, config.encoder_dir, labels, families,
        config=config, train_names=train_names, cache_dir=cache_dir,
        device=device, log=log)

    embedder = BlockEmbedder(config.encoder_dir, max_len=config.encoder.max_len,
                             device=device)
    checkpoint = save_checkpoint(
        os.path.join(output_dir, "checkpoint.pt"), fenc, clf, families,
        config, embedder.dim)

    predictions = {
        "train": predict_split(train_names, config.reports_dir, labels,
                               embedder, fenc, clf, families, cache_dir),
        "test": predict_split(test_names, config.reports_dir, labels,
                              embedder, fenc, clf, families, cache_dir),
    }
    predictions_path = os.path.join(output_dir, "predictions.json")
    with open(predictions_path, "w") as f:
        json.dump(predictions, f, indent=2)

    with open(os.path.join(output_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    log(f"wrote {checkpoint} and {predictions_path}")
    return {"config": config, "history": history, "predictions": predictions,
            "checkpoint": checkpoint, "split": (train_names, test_names)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reports", default="reports/")
    ap.add_argument("--encoder", default="asm_encoder/")
    ap.add_argument("--labels", default=None,
                    help="label manifest (default <reports>/labels.json)")
    ap.add_argument("--out", default="runs/current")
    ap.add_argument("--config", default=None, help="a saved config.json")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--test-size", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    config = (ExperimentConfig.load(args.config) if args.config
              else ExperimentConfig())
    config.reports_dir = args.reports
    config.encoder_dir = args.encoder
    config.output_dir = args.out
    if args.epochs is not None:
        config.model.epochs = args.epochs
    if args.test_size is not None:
        config.test_size = args.test_size
    if args.seed is not None:
        config.seed = args.seed

    run_experiment(config, labels_path=args.labels, device=args.device)


if __name__ == "__main__":
    main()
