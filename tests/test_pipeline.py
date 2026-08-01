"""The stage-2 experiment: split manifest, training, artifacts, eval, predict."""
import os
import json

import pytest
import torch

from config import ExperimentConfig
from train import make_split, run_experiment, families_from_labels
from eval import binary_metrics, family_metrics, score_rows, score_file


def _run_config(tiny_dataset, output_dir, epochs=2):
    config = ExperimentConfig()
    config.seed = 7
    config.reports_dir = tiny_dataset["reports_dir"]
    config.encoder_dir = tiny_dataset["encoder_dir"]
    config.output_dir = str(output_dir)
    config.test_size = 0.5
    config.encoder.max_len = 64
    config.model.epochs = epochs
    config.model.function_hidden = 16
    config.model.fcg_hidden = 16
    config.model.max_clusters = 4
    config.model.batch_size = 4
    return config


def test_split_is_stratified_and_disjoint(tiny_dataset):
    labels = tiny_dataset["labels"]
    train, test = make_split(labels, test_size=0.5, seed=7)

    assert not set(train) & set(test)
    assert set(train) | set(test) == set(labels)
    # every family must appear in the training half or the family head has
    # nothing to learn from
    train_families = {labels[n].get("family") for n in train}
    assert set(families_from_labels(labels)) <= train_families


def test_split_is_deterministic(tiny_dataset):
    a = make_split(tiny_dataset["labels"], 0.5, seed=7)
    b = make_split(tiny_dataset["labels"], 0.5, seed=7)
    assert a == b


def test_run_experiment_writes_every_required_artifact(tiny_dataset, tmp_path):
    """The README refuses to quote a number without these four files."""
    output_dir = tmp_path / "run"
    result = run_experiment(_run_config(tiny_dataset, output_dir),
                            labels_path=tiny_dataset["labels_path"], log=lambda *_: None)

    for name in ("config.json", "split.json", "checkpoint.pt",
                 "predictions.json", "train_history.json"):
        path = output_dir / name
        assert path.exists() and path.stat().st_size > 0, name

    predictions = json.loads((output_dir / "predictions.json").read_text())
    split = json.loads((output_dir / "split.json").read_text())
    assert len(predictions["train"]) == len(split["train"])
    assert len(predictions["test"]) == len(split["test"])
    for row in predictions["train"] + predictions["test"]:
        assert 0.0 <= row["y_score"] <= 1.0
        assert row["y_true"] in (0, 1)
    assert len(result["history"]) == 2


def test_training_loss_is_finite_and_moves(tiny_dataset, tmp_path):
    result = run_experiment(_run_config(tiny_dataset, tmp_path / "run", epochs=3),
                            labels_path=tiny_dataset["labels_path"],
                            log=lambda *_: None)
    losses = [h["loss"] for h in result["history"]]
    assert all(l == l and abs(l) != float("inf") for l in losses)  # no NaN/inf
    assert len(set(round(l, 6) for l in losses)) > 1


def test_checkpoint_round_trips_through_predict(tiny_dataset, tmp_path):
    from predict import predict, load_model

    output_dir = tmp_path / "run"
    run_experiment(_run_config(tiny_dataset, output_dir),
                   labels_path=tiny_dataset["labels_path"], log=lambda *_: None)

    checkpoint = str(output_dir / "checkpoint.pt")
    _embedder, _fenc, _clf, families = load_model(tiny_dataset["encoder_dir"],
                                                  checkpoint)
    assert families == families_from_labels(tiny_dataset["labels"])

    split = json.loads((output_dir / "split.json").read_text())
    report = os.path.join(tiny_dataset["reports_dir"], split["test"][0])
    result = predict(report, tiny_dataset["encoder_dir"], checkpoint,
                     cache_dir=str(tmp_path / "cache"))

    assert 0.0 <= result["malware_probability"] <= 1.0
    assert result["prediction"] in ("malware", "benign")
    assert result["num_functions"] > 0
    assert set(result["family_probabilities"]) == set(families)
    assert result["predicted_family"] in set(families) | {None}


def test_predict_refuses_a_checkpoint_from_a_different_encoder(tiny_dataset,
                                                               tmp_path):
    from predict import load_model

    output_dir = tmp_path / "run"
    run_experiment(_run_config(tiny_dataset, output_dir),
                   labels_path=tiny_dataset["labels_path"], log=lambda *_: None)

    checkpoint = output_dir / "checkpoint.pt"
    state = torch.load(str(checkpoint), weights_only=False)
    state["encoder_dim"] = state["encoder_dim"] + 64      # simulate a mismatch
    tampered = tmp_path / "mismatched.pt"
    torch.save(state, str(tampered))

    with pytest.raises(ValueError, match="mismatch"):
        load_model(tiny_dataset["encoder_dir"], str(tampered))


def test_sample_includes_api_nodes_in_the_call_graph(tiny_dataset):
    from embed_blocks import BlockEmbedder
    from DiffPool import FunctionEncoder
    from sample import build_sample
    from data_tokenizer import load_report

    name = sorted(tiny_dataset["labels"])[0]
    report = load_report(os.path.join(tiny_dataset["reports_dir"], name))
    embedder = BlockEmbedder(tiny_dataset["encoder_dir"], max_len=64)
    fenc = FunctionEncoder(embedder.dim, hidden_dim=16, max_clusters=4)

    s = build_sample(report, embedder, fenc, is_malware=1, cache_dir=None)
    assert s.num_functions == len(report["xcfg"])
    # api rows are appended after the functions, so the node matrix is bigger
    assert len(s.node_ids) > s.num_functions
    assert s.node_embeddings.shape[0] == len(s.node_ids)
    assert any(isinstance(n, str) and n.startswith("api:") for n in s.node_ids)
    assert s.function_embeddings.shape[0] == s.num_functions
    assert torch.isfinite(s.node_embeddings).all()


def test_metrics_handle_a_single_class_split():
    """A tiny holdout can end up all-benign; AUROC is undefined there."""
    metrics = binary_metrics([0, 0, 0], [0.1, 0.2, 0.3])
    assert metrics["auroc"] is None and metrics["auprc"] is None
    assert metrics["accuracy"] == 1.0
    assert metrics["confusion"] == [[3, 0], [0, 0]]


def test_binary_metrics_on_a_perfect_split():
    metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert metrics["auroc"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["confusion"] == [[2, 0], [0, 2]]


def test_family_metrics_ignore_benign_rows():
    rows = [
        {"y_true": 0, "y_score": 0.1, "family_true": None, "family_pred": "Worm"},
        {"y_true": 1, "y_score": 0.9, "family_true": "Worm", "family_pred": "Worm"},
        {"y_true": 1, "y_score": 0.8, "family_true": "Trojan", "family_pred": "Worm"},
    ]
    scored = score_rows(rows)
    assert scored["family"]["n"] == 2
    assert scored["binary"]["n"] == 3


def test_family_metrics_empty_is_not_an_error():
    assert family_metrics([], [])["n"] == 0


def test_eval_reads_back_exactly_what_train_wrote(tiny_dataset, tmp_path):
    output_dir = tmp_path / "run"
    result = run_experiment(_run_config(tiny_dataset, output_dir),
                            labels_path=tiny_dataset["labels_path"],
                            log=lambda *_: None)
    metrics = score_file(str(output_dir / "predictions.json"))
    assert metrics["test"]["binary"]["n"] == len(result["predictions"]["test"])
    assert 0.0 <= metrics["test"]["binary"]["accuracy"] <= 1.0


def test_config_round_trips(tmp_path):
    config = ExperimentConfig()
    config.families = ["Worm", "Trojan"]
    config.model.fcg_hidden = 77
    path = config.save(str(tmp_path / "config.json"))
    loaded = ExperimentConfig.load(path)
    assert loaded.families == ["Worm", "Trojan"]
    assert loaded.model.fcg_hidden == 77
    assert loaded.encoder.dim == config.encoder.dim


def test_config_tolerates_unknown_keys(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"seed": 3, "from_the_future": True,
                                "model": {"fcg_hidden": 12, "also_new": 1}}))
    config = ExperimentConfig.load(str(path))
    assert config.seed == 3 and config.model.fcg_hidden == 12


def test_baselines_run_on_the_same_split(tiny_dataset, tmp_path):
    from baselines import run_baselines

    output_dir = tmp_path / "run"
    run_experiment(_run_config(tiny_dataset, output_dir),
                   labels_path=tiny_dataset["labels_path"], log=lambda *_: None)
    results = run_baselines(tiny_dataset["reports_dir"],
                            str(output_dir / "split.json"),
                            str(tmp_path / "baselines.json"))
    assert set(results) == {"opcode_ngram", "graph_stats"}
    for scores in results.values():
        assert 0.0 <= scores["accuracy"] <= 1.0
    assert (tmp_path / "baselines.json").exists()


def test_data_loader_rejects_a_bad_mode(tiny_dataset):
    from Data_Loader import DataLoader
    with pytest.raises(ValueError):
        DataLoader(tiny_dataset["reports_dir"], mode="sideways")


def test_data_loader_skips_the_label_manifest(tiny_dataset):
    from Data_Loader import DataLoader
    loader = DataLoader(tiny_dataset["reports_dir"], mode="spatial")
    assert len(loader) == len(tiny_dataset["labels"])
    assert all("labels.json" not in path for path in loader.reports)


def test_temporal_mode_orders_by_capture_timestamp(tiny_dataset):
    from Data_Loader import DataLoader
    from data_tokenizer import load_report

    loader = DataLoader(tiny_dataset["reports_dir"], mode="temporal")
    stamps = [load_report(p).get("timestamp", "") for p in loader.reports]
    assert stamps == sorted(stamps)
