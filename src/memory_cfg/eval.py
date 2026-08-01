"""Evaluation metrics computed from saved predictions.

Everything here reads back arrays of predictions/labels so the test set is
scored exactly once per finalized experiment and every table is reproducible.
Nothing in this module touches a model: give it a `predictions.json` and it
gives you the same numbers on any machine.
"""
import json
import argparse


def binary_metrics(y_true, y_score, threshold=0.5):
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, f1_score, accuracy_score,
        matthews_corrcoef, confusion_matrix,
    )
    y_pred = [1 if s >= threshold else 0 for s in y_score]
    single_class = len(set(y_true)) < 2
    return {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        # AUROC/AUPRC are undefined when the split holds only one class; report
        # null instead of letting sklearn raise or, worse, silently mislead.
        "auroc": None if single_class else roc_auc_score(y_true, y_score),
        "auprc": None if single_class else average_precision_score(y_true, y_score),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "threshold": threshold,
    }


def family_metrics(y_true, y_pred, labels=None):
    from sklearn.metrics import (
        f1_score, balanced_accuracy_score, accuracy_score, confusion_matrix,
    )
    if not y_true:
        return {"n": 0}
    labels = labels or sorted(set(y_true) | set(y_pred))
    return {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro",
                             labels=labels, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "labels": labels,
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def score_rows(rows, threshold=0.5):
    """One split's prediction rows -> {"binary": ..., "family": ...}."""
    y_true = [r["y_true"] for r in rows]
    y_score = [r["y_score"] for r in rows]
    # the family head is only meaningful on samples that really are malware
    malicious = [r for r in rows if r["y_true"] == 1 and r.get("family_true")]
    return {
        "binary": binary_metrics(y_true, y_score, threshold),
        "family": family_metrics([r["family_true"] for r in malicious],
                                 [r["family_pred"] for r in malicious]),
    }


def score_file(predictions_path, threshold=0.5):
    with open(predictions_path) as f:
        predictions = json.load(f)
    return {split: score_rows(rows, threshold)
            for split, rows in predictions.items()}


def format_report(metrics):
    lines = []
    for split in sorted(metrics, reverse=True):
        binary, family = metrics[split]["binary"], metrics[split]["family"]
        lines.append(f"[{split}] n={binary['n']}")
        pieces = [f"accuracy={binary['accuracy']:.3f}",
                  f"f1={binary['f1']:.3f}", f"mcc={binary['mcc']:.3f}"]
        for key in ("auroc", "auprc"):
            pieces.append(f"{key}=" + ("n/a" if binary[key] is None
                                       else f"{binary[key]:.3f}"))
        lines.append("  binary : " + "  ".join(pieces))
        lines.append(f"  confusion (tn, fp / fn, tp): {binary['confusion']}")
        if family.get("n"):
            lines.append(
                f"  family : n={family['n']}  accuracy={family['accuracy']:.3f}"
                f"  macro_f1={family['macro_f1']:.3f}"
                f"  balanced_accuracy={family['balanced_accuracy']:.3f}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("predictions", help="predictions.json written by train.py")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default=None, help="write metrics.json here")
    args = ap.parse_args(argv)

    metrics = score_file(args.predictions, args.threshold)
    print(format_report(metrics))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nwrote {args.out}")
    return metrics


if __name__ == "__main__":
    main()
