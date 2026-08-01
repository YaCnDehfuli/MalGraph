"""Simple, leakage-safe baselines to compare the hierarchical model against.

  - opcode n-gram: bag of mnemonic uni/bi-grams -> logistic regression
  - graph stats:   a handful of call-graph / CFG statistics -> logistic regression

Both read the SAME `split.json` the neural model trained on, and the feature
vocabulary is fitted on the training split only. A baseline that quietly saw the
test set is worse than no baseline, because it makes the real model look bad.
"""
import os
import json
import argparse
from collections import Counter

from .data_tokenizer import iter_blocks, load_report
from .CFG_Extractor import build_all_cfgs
from .fcg import build_call_graph


def opcode_ngram_features(report, n=2):
    counts = Counter()
    for instructions in iter_blocks(report):
        mnemonics = [ins[2] for ins in instructions]
        counts.update(mnemonics)                          # unigrams
        if n >= 2:
            for a, b in zip(mnemonics, mnemonics[1:]):    # bigrams
                counts[f"{a}_{b}"] += 1
    return counts


def api_features(report):
    """Imported symbol counts - the other obvious cheap signal."""
    counts = Counter()
    for func in report["xcfg"].values():
        for api_name in func.get("apirefs", {}).values():
            counts[f"api:{api_name}"] += 1
    return counts


def graph_stat_features(report):
    cfgs = build_all_cfgs(report, parallel=False)
    fcg = build_call_graph(report)
    n_funcs = len(cfgs)
    n_blocks = sum(g.number_of_nodes() for g in cfgs.values())
    n_edges = sum(g.number_of_edges() for g in cfgs.values())
    n_instructions = sum(len(instr or [])
                         for g in cfgs.values()
                         for _n, instr in g.nodes(data="instructions"))
    return {
        "num_functions": n_funcs,
        "num_blocks": n_blocks,
        "num_cfg_edges": n_edges,
        "num_instructions": n_instructions,
        "avg_blocks_per_func": n_blocks / max(n_funcs, 1),
        "avg_edges_per_func": n_edges / max(n_funcs, 1),
        "avg_instructions_per_block": n_instructions / max(n_blocks, 1),
        "num_fcg_nodes": fcg.number_of_nodes(),
        "num_fcg_edges": fcg.number_of_edges(),
        "fcg_density": fcg.number_of_edges() / max(1, fcg.number_of_nodes()),
    }


# fixed order, so a row means the same thing in train and test matrices
GRAPH_STAT_KEYS = (
    "num_functions", "num_blocks", "num_cfg_edges", "num_instructions",
    "avg_blocks_per_func", "avg_edges_per_func", "avg_instructions_per_block",
    "num_fcg_nodes", "num_fcg_edges", "fcg_density",
)


def build_vocab(reports, max_features=4000, use_apis=True):
    """Fit the n-gram vocabulary on the TRAINING reports only."""
    totals = Counter()
    for report in reports:
        totals.update(opcode_ngram_features(report))
        if use_apis:
            totals.update(api_features(report))
    return [token for token, _count in totals.most_common(max_features)]


def build_matrix(reports, vocab, use_apis=True):
    """reports: list[dict] -> dense list-of-lists over a fixed vocab order."""
    rows = []
    for report in reports:
        feats = opcode_ngram_features(report)
        if use_apis:
            feats.update(api_features(report))
        total = max(1, sum(feats.values()))
        # counts are normalized: otherwise the model mostly learns binary size
        rows.append([feats.get(token, 0) / total for token in vocab])
    return rows


def build_graph_matrix(reports):
    rows = []
    for report in reports:
        feats = graph_stat_features(report)
        rows.append([float(feats[key]) for key in GRAPH_STAT_KEYS])
    return rows


def fit_logreg(X, y):
    """Standardize, then fit.

    Without the scaler this is not a fair baseline: the n-gram features are
    normalized frequencies (order 1e-3) and the graph features mix counts in the
    thousands with densities near 1. Logistic regression then produced a perfect
    *ranking* (AUROC 1.0) whose probabilities never crossed 0.5, so F1 came out
    at 0.0 and the comparison was meaningless. The scaler is fitted on the
    training rows only and reused for test, so nothing leaks.
    """
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, class_weight="balanced"))
    clf.fit(X, y)
    return clf


def _scores(clf, X):
    """P(malware) for each row, whichever way sklearn ordered the classes."""
    classes = list(clf.classes_)
    if len(classes) == 1:
        return [float(classes[0])] * len(X)
    index = classes.index(1)
    return [row[index] for row in clf.predict_proba(X)]


def run_baselines(report_dir, split_path, out_path=None):
    """Train both baselines on the split's train half, score its test half."""
    from .eval import binary_metrics

    with open(split_path) as f:
        split = json.load(f)
    labels = split["labels"]

    def load(names):
        return [load_report(os.path.join(report_dir, name)) for name in names]

    train_reports = load(split["train"])
    test_reports = load(split["test"])
    y_train = [labels[n]["is_malware"] for n in split["train"]]
    y_test = [labels[n]["is_malware"] for n in split["test"]]

    results = {}

    vocab = build_vocab(train_reports)
    clf = fit_logreg(build_matrix(train_reports, vocab), y_train)
    results["opcode_ngram"] = binary_metrics(
        y_test, _scores(clf, build_matrix(test_reports, vocab)))
    results["opcode_ngram"]["num_features"] = len(vocab)

    clf = fit_logreg(build_graph_matrix(train_reports), y_train)
    results["graph_stats"] = binary_metrics(
        y_test, _scores(clf, build_graph_matrix(test_reports)))
    results["graph_stats"]["num_features"] = len(GRAPH_STAT_KEYS)

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reports", default="reports/")
    ap.add_argument("--split", default="runs/current/split.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    results = run_baselines(args.reports, args.split, args.out)
    for name, metrics in results.items():
        auroc = "n/a" if metrics["auroc"] is None else f"{metrics['auroc']:.3f}"
        print(f"{name:14s} accuracy={metrics['accuracy']:.3f} "
              f"f1={metrics['f1']:.3f} auroc={auroc} "
              f"({metrics['num_features']} features)")
    return results


if __name__ == "__main__":
    main()
