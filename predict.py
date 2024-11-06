"""Run a trained model on a single normalized SMDA report -> prediction."""
import json
import argparse

import torch

from embed_blocks import BlockEmbedder
from DiffPool import FunctionEncoder
from model import HierClassifier
from sample import build_sample


def load_model(encoder_dir, checkpoint, device="cpu"):
    embedder = BlockEmbedder(encoder_dir, device=device)
    in_dim = embedder.dim
    state = torch.load(checkpoint, map_location=device, weights_only=False)

    fenc = FunctionEncoder(
        in_dim,
        hidden_dim=state.get("function_hidden", 128),
        max_clusters=state.get("max_clusters", 16),
    ).to(device)
    clf = HierClassifier(fenc.hidden_dim, state["fcg_hidden"],
                         num_families=max(1, len(state["families"])),
                         dropout=state.get("dropout", 0.2)).to(device)
    fenc.load_state_dict(state["fenc"])
    clf.load_state_dict(state["clf"])
    fenc.eval()
    clf.eval()
    return embedder, fenc, clf, state["families"]


@torch.no_grad()
def predict(report_path, encoder_dir, checkpoint, device="cpu",
            cache_dir="embeddings_cache", threshold=0.5):
    with open(report_path) as f:
        report = json.load(f)
    embedder, fenc, clf, families = load_model(encoder_dir, checkpoint, device)

    s = build_sample(report, embedder, fenc, is_malware=0,
                     cache_dir=cache_dir)  # label unknown at inference time
    bin_logit, fam_logits = clf(s.node_embeddings, s.fcg_edge_index)

    probability = float(torch.sigmoid(bin_logit))
    family_probabilities = torch.softmax(fam_logits, dim=-1).tolist()
    predicted = families[int(fam_logits.argmax())] if families else None
    return {
        "report": report_path,
        "malware_probability": probability,
        "prediction": "malware" if probability >= threshold else "benign",
        "predicted_family": predicted if probability >= threshold else None,
        "family_probabilities": (
            dict(zip(families, family_probabilities)) if families else {}),
        "num_functions": s.num_functions,
        "num_graph_nodes": len(s.node_ids),
        "num_call_edges": int(s.fcg_edge_index.shape[1]),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("report")
    ap.add_argument("--encoder", default="asm_encoder/")
    ap.add_argument("--checkpoint", default="checkpoints/model.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    result = predict(args.report, args.encoder, args.checkpoint,
                     device=args.device, threshold=args.threshold)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
    return result


if __name__ == "__main__":
    main()
