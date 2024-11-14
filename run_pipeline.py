#!/usr/bin/env python3
"""Run the full pipeline end to end: reports -> encoder -> classifier -> metrics.

This is the entrypoint for a complete experiment. It runs every stage - corpus,
tokenizer, masked-LM encoder, CFG pooling, call-graph classifier, evaluation,
baselines, inference - and leaves behind the artifact set this project requires
before any number may be quoted: config, split manifest, checkpoint,
predictions, metrics, baselines, and figures.

Point it at your own SMDA reports:

    python scripts/run_pipeline.py --reports reports/ --labels reports/labels.json

With no --reports it falls back to `memory_cfg.synth`, which emits SMDA-shaped
fixtures. That path is what the integration tests use, and what lets a fresh
checkout exercise every stage without a capture; it is not an experiment.
"""
import os
import sys
import json
import time
import shutil
import hashlib
import argparse
import subprocess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.isdir(os.path.join(REPO_ROOT, "src")):
    sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import synth                                    # noqa: E402
import viz                                      # noqa: E402
from config import ExperimentConfig, set_seed        # noqa: E402
from data_tokenizer import write_corpus_from_dir, load_report  # noqa: E402
from tokenizer_train import train_tokenizer          # noqa: E402
from transformer_train import (                      # noqa: E402
    load_tokenizer, read_corpus, train_encoder,
)
from train import run_experiment                     # noqa: E402
from eval import score_file, format_report           # noqa: E402
from baselines import run_baselines                  # noqa: E402
from predict import predict                          # noqa: E402
from CFG_Extractor import build_all_cfgs, graph_stats  # noqa: E402
from fcg import build_call_graph, call_graph_stats   # noqa: E402


def _banner(step, text):
    print(f"\n=== [{step}] {text} ===", flush=True)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
    except (OSError, subprocess.SubprocessError):
        return None


def build_config(args, output_dir, reports_dir, encoder_dir):
    config = ExperimentConfig()
    config.seed = args.seed
    config.reports_dir = reports_dir
    config.encoder_dir = encoder_dir
    config.output_dir = output_dir
    config.test_size = args.test_size
    # defaults are sized for a CPU box; raise --layers/--dim/--epochs for a
    # real corpus, where the encoder has enough text to justify the capacity
    config.encoder.n_layers = args.layers
    config.encoder.dim = args.dim
    config.encoder.n_heads = args.heads
    config.encoder.max_len = args.max_len
    config.encoder.vocab_size = args.vocab_size
    config.encoder.epochs = args.encoder_epochs
    config.encoder.batch_size = args.batch_size
    config.model.epochs = args.epochs
    config.model.function_hidden = args.function_hidden
    config.model.fcg_hidden = args.fcg_hidden
    config.model.max_clusters = args.max_clusters
    return config


def write_figures(fixture_path, assets_dir, reports_dir, labels):
    """Figures for whatever corpus this run actually used.

    Titles carry the report they came from rather than a hard-coded claim about
    provenance, so the same code produces honest captions for a capture, a
    system-binary corpus, or the test fixtures.
    """
    made = {}
    if os.path.exists(fixture_path):
        report = load_report(fixture_path)
        cfgs = build_all_cfgs(report, parallel=False)
        biggest = max(cfgs, key=lambda f: cfgs[f].number_of_nodes())
        made["sample_cfg"] = viz.draw_cfg(
            cfgs[biggest],
            title=(f"examples/sample_pid.json - function 0x{biggest:x} "
                   f"({cfgs[biggest].number_of_nodes()} blocks, "
                   f"{cfgs[biggest].number_of_edges()} edges)"),
            save_to=os.path.join(assets_dir, "sample-cfg-function.png"))
        made["sample_stats"] = graph_stats(cfgs)

    # the richest report in the corpus makes the most informative pictures
    names = sorted(labels)
    if not names:
        return made
    best, best_cfgs, best_size = None, None, -1
    for name in names:
        path = os.path.join(reports_dir, name)
        if not os.path.exists(path):
            continue
        cfgs = build_all_cfgs(load_report(path), parallel=False)
        size = sum(g.number_of_nodes() for g in cfgs.values())
        if size > best_size:
            best, best_cfgs, best_size = name, cfgs, size
    if best is None:
        return made

    # mid-sized functions: a 200-block CFG is a picture of nothing
    readable = {k: g for k, g in best_cfgs.items()
                if 8 <= g.number_of_nodes() <= 14} or best_cfgs
    pick = max(readable, key=lambda k: readable[k].number_of_edges())
    made["cfg_function"] = viz.draw_cfg(
        readable[pick],
        title=(f"{best} - function 0x{pick:x} "
               f"({readable[pick].number_of_nodes()} blocks, "
               f"{readable[pick].number_of_edges()} edges)"),
        save_to=os.path.join(assets_dir, "corpus-cfg-function.png"))
    grid = dict(sorted(readable.items(),
                       key=lambda kv: kv[1].number_of_edges(),
                       reverse=True)[:6])
    made["cfg_grid"] = viz.draw_cfg_grid(
        grid, columns=3, max_functions=6,
        title=f"Per-function control-flow graphs - {best}",
        save_to=os.path.join(assets_dir, "corpus-cfg-grid.png"))

    fcg = build_call_graph(load_report(os.path.join(reports_dir, best)))
    made["fcg"] = viz.draw_call_graph(
        fcg, max_nodes=90,
        title=f"Inter-function call graph - {best} (top-degree subgraph)",
        save_to=os.path.join(assets_dir, "corpus-call-graph.png"))
    made["fcg_stats"] = call_graph_stats(fcg)
    return made


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="runs/current", help="artifact directory")
    ap.add_argument("--reports", default=None,
                    help="use an existing report dir instead of generating one")
    ap.add_argument("--labels", default=None, help="label manifest for --reports")
    ap.add_argument("--per-profile", type=int, default=10,
                    help="generated fixtures per class when no --reports is given")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=12, help="stage-2 epochs")
    ap.add_argument("--encoder-epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--vocab-size", type=int, default=2000)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--function-hidden", type=int, default=64)
    ap.add_argument("--fcg-hidden", type=int, default=64)
    ap.add_argument("--max-clusters", type=int, default=8)
    ap.add_argument("--assets", default=None,
                    help="also write figures here (e.g. docs/assets)")
    ap.add_argument("--publish", default=None,
                    help="copy the small JSON artifacts here (e.g. docs/validation) "
                         "so the run is reviewable without rerunning it")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--keep-cache", action="store_true",
                    help="keep the block-embedding cache after the run")
    args = ap.parse_args(argv)

    started = time.time()
    output_dir = os.path.abspath(args.out)
    os.makedirs(output_dir, exist_ok=True)
    set_seed(args.seed)

    reports_dir = args.reports or os.path.join(output_dir, "reports")
    encoder_dir = os.path.join(output_dir, "asm_encoder")
    assets_dir = args.assets or os.path.join(output_dir, "figures")
    fixture = os.path.join(REPO_ROOT, "examples", "sample_pid.json")

    _banner(1, "SMDA reports")
    if args.reports:
        labels_path = args.labels or os.path.join(reports_dir, "labels.json")
        with open(labels_path) as handle:
            labels = json.load(handle)
        print(f"using {len(labels)} existing reports from {reports_dir}")
    else:
        seed_report = load_report(fixture) if os.path.exists(fixture) else None
        spec = synth.DatasetSpec(n_per_profile=args.per_profile, seed=args.seed)
        labels = synth.generate_dataset(reports_dir, spec, seed_report)
        labels_path = os.path.join(reports_dir, "labels.json")
        with open(labels_path, "w") as handle:
            json.dump(labels, handle, indent=2)
        n_mal = sum(m["is_malware"] for m in labels.values())
        print(f"generated {len(labels)} reports "
              f"({n_mal} malicious / {len(labels) - n_mal} benign) in {reports_dir}")

    _banner(2, "assembly corpus")
    corpus_path = os.path.join(output_dir, "corpus.txt")
    n_lines, n_reports = write_corpus_from_dir(reports_dir, corpus_path)
    print(f"{n_lines} deduplicated block/import lines from {n_reports} reports")

    _banner(3, "WordPiece tokenizer")
    tokenizer_path = os.path.join(output_dir, "asm_tokenizer.json")
    tokenizer = train_tokenizer(corpus_path, tokenizer_path,
                                vocab_size=args.vocab_size, min_frequency=1)
    print(f"vocab {tokenizer.get_vocab_size()} -> {tokenizer_path}")

    _banner(4, "DistilBERT masked-LM pretraining")
    config = build_config(args, output_dir, reports_dir, encoder_dir)
    _model, mlm_history = train_encoder(
        read_corpus(corpus_path), load_tokenizer(tokenizer_path), encoder_dir,
        epochs=config.encoder.epochs, batch_size=config.encoder.batch_size,
        lr=config.encoder.lr, warmup=20, max_len=config.encoder.max_len,
        n_layers=config.encoder.n_layers, dim=config.encoder.dim,
        n_heads=config.encoder.n_heads, device=args.device,
        log_every=100, seed=args.seed)

    _banner(5, "hierarchical classifier (CFG DiffPool -> FCG GNN)")
    result = run_experiment(config, labels_path=labels_path, device=args.device)

    _banner(6, "evaluation")
    predictions_path = os.path.join(output_dir, "predictions.json")
    metrics = score_file(predictions_path)
    print(format_report(metrics))
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as handle:
        json.dump(metrics, handle, indent=2)

    _banner(7, "classical baselines on the same split")
    baselines = run_baselines(reports_dir, os.path.join(output_dir, "split.json"),
                              os.path.join(output_dir, "baselines.json"))
    for name, scores in baselines.items():
        auroc = "n/a" if scores["auroc"] is None else f"{scores['auroc']:.3f}"
        print(f"{name:14s} accuracy={scores['accuracy']:.3f} "
              f"f1={scores['f1']:.3f} auroc={auroc}")

    _banner(8, "single-report inference from the saved checkpoint")
    with open(os.path.join(output_dir, "split.json")) as handle:
        split = json.load(handle)
    sample_report = os.path.join(reports_dir, split["test"][0])
    inference = predict(sample_report, encoder_dir,
                        os.path.join(output_dir, "checkpoint.pt"),
                        device=args.device,
                        cache_dir=os.path.join(output_dir, "embeddings_cache"))
    inference["ground_truth"] = split["labels"][split["test"][0]]
    print(json.dumps(inference, indent=2))
    with open(os.path.join(output_dir, "example_prediction.json"), "w") as handle:
        json.dump(inference, handle, indent=2)

    _banner(9, "figures")
    os.makedirs(assets_dir, exist_ok=True)
    figures = write_figures(fixture, assets_dir, reports_dir, labels)
    figures["training_curve"] = viz.draw_training_curves(
        result["history"], save_to=os.path.join(assets_dir, "training-curve.png"),
        title="Stage-2 training")
    figures["confusion"] = viz.draw_confusion(
        metrics["test"]["binary"]["confusion"], ["benign", "malware"],
        save_to=os.path.join(assets_dir, "confusion-binary.png"),
        title="Binary confusion matrix (held-out split)")
    family = metrics["test"]["family"]
    if family.get("n"):
        figures["confusion_family"] = viz.draw_confusion(
            family["confusion"], family["labels"],
            save_to=os.path.join(assets_dir, "confusion-family.png"),
            title="Family confusion matrix (held-out malicious samples)")
    for key, value in figures.items():
        if isinstance(value, str):
            print(f"  {key}: {os.path.relpath(value, REPO_ROOT)}")

    checkpoint_path = os.path.join(output_dir, "checkpoint.pt")
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "elapsed_seconds": round(time.time() - started, 1),
        "data": {
            "source": reports_dir if args.reports else "generated fixtures",
            "n_binaries": len(labels),
            "n_malicious": sum(m["is_malware"] for m in labels.values()),
            "families": config.families,
            "corpus_lines": n_lines,
            "vocab_size": tokenizer.get_vocab_size(),
        },
        "encoder": {
            "layers": config.encoder.n_layers, "dim": config.encoder.dim,
            "heads": config.encoder.n_heads, "epochs": config.encoder.epochs,
            "final_mlm_loss": mlm_history[-1]["mlm_loss"] if mlm_history else None,
            "final_perplexity": (mlm_history[-1]["perplexity"]
                                 if mlm_history else None),
        },
        "split": {"train": len(split["train"]), "test": len(split["test"])},
        "metrics": metrics,
        "baselines": baselines,
        "example_prediction": inference,
        "artifacts": {
            "config": "config.json",
            "split": "split.json",
            "checkpoint": "checkpoint.pt",
            "checkpoint_sha256": _sha256(checkpoint_path),
            "predictions": "predictions.json",
            "metrics": "metrics.json",
            "baselines": "baselines.json",
        },
    }
    summary_path = os.path.join(output_dir, "run_summary.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    if args.publish:
        # The checkpoint and the reports stay out of Git by policy; these JSON
        # files are what let a reader check the numbers without a rerun.
        published = os.path.abspath(args.publish)
        os.makedirs(published, exist_ok=True)
        for name in ("config.json", "split.json", "predictions.json",
                     "metrics.json", "baselines.json", "train_history.json",
                     "example_prediction.json", "run_summary.json"):
            source = os.path.join(output_dir, name)
            if os.path.exists(source):
                shutil.copy2(source, os.path.join(published, name))
        shutil.copy2(os.path.join(encoder_dir, "pretrain_history.json"),
                     os.path.join(published, "pretrain_history.json"))
        print(f"published run artifacts to {published}")

    if not args.keep_cache:
        shutil.rmtree(os.path.join(output_dir, "embeddings_cache"),
                      ignore_errors=True)

    print(f"\nDone in {summary['elapsed_seconds']}s. Artifacts in {output_dir}")
    print(f"Summary: {summary_path}")
    return summary


if __name__ == "__main__":
    main()
