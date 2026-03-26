import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from scripts import evaluator


def test_compute_metrics_known_values():
    labels = np.asarray([1, 1, 0, 0], dtype=np.int64)
    predictions = np.asarray([1, 0, 1, 0], dtype=np.int64)

    metrics = evaluator.compute_metrics(labels, predictions)

    assert metrics["tp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["accuracy"] == 0.5
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5


def test_evaluate_scored_split_applies_threshold():
    labels = np.asarray([1, 1, 0, 0], dtype=np.int64)
    scores = np.asarray([0.95, 0.60, 0.70, 0.20], dtype=np.float64)

    metrics = evaluator.evaluate_scored_split(labels=labels, scores=scores, threshold=0.8)

    assert metrics["tp"] == 1
    assert metrics["tn"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 1
    assert metrics["accuracy"] == 0.75


def test_threshold_sweep_selects_best_f1_then_accuracy():
    labels = np.asarray([1, 1, 0, 0], dtype=np.int64)
    scores = np.asarray([0.90, 0.80, 0.70, 0.10], dtype=np.float64)

    best_threshold, metrics, sweep_rows = evaluator.threshold_sweep(labels=labels, scores=scores)

    assert 0.70 < best_threshold <= 0.80
    assert metrics["f1"] == 1.0
    assert len(sweep_rows) > 0


def test_score_split_count_matches_pair_count(tmp_path):
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    Image.fromarray(np.full((4, 4), 255, dtype=np.uint8), mode="L").save(image_a)
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8), mode="L").save(image_b)

    pairs_dir = tmp_path / "pairs"
    pairs_dir.mkdir()
    pair_path = pairs_dir / "val.jsonl"
    with pair_path.open("w") as file:
        file.write(json.dumps({"left_path": str(image_a), "right_path": str(image_a), "label": 1, "split": "val"}) + "\n")
        file.write(json.dumps({"left_path": str(image_a), "right_path": str(image_b), "label": 0, "split": "val"}) + "\n")

    labels, scores, rows = evaluator.score_split("val", pairs_dir, "L", (4, 4))

    assert labels.shape == (2,)
    assert scores.shape == (2,)
    assert len(rows) == 2

def test_build_confusion_matrix_dict_layout():
    metrics = {
        "tp": 3,
        "tn": 4,
        "fp": 1,
        "fn": 2,
        "accuracy": 0.0,
        "balanced_accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }

    matrix = evaluator.build_confusion_matrix_dict(metrics)

    assert matrix["rows"] == ["actual_1", "actual_0"]
    assert matrix["cols"] == ["pred_1", "pred_0"]
    assert matrix["matrix"] == [[3, 2], [1, 4]]

def test_integration_fixed_threshold_run_writes_outputs(tmp_path, monkeypatch):
    pairs_dir = tmp_path / "pairs"
    pairs_dir.mkdir(parents=True)

    bright = tmp_path / "bright.png"
    bright2 = tmp_path / "bright2.png"
    dark = tmp_path / "dark.png"
    dark2 = tmp_path / "dark2.png"

    Image.fromarray(np.full((8, 8), 255, dtype=np.uint8), mode="L").save(bright)
    Image.fromarray(np.full((8, 8), 240, dtype=np.uint8), mode="L").save(bright2)
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8), mode="L").save(dark)
    Image.fromarray(np.full((8, 8), 15, dtype=np.uint8), mode="L").save(dark2)

    val_rows = [
        {"left_path": str(bright), "right_path": str(bright2), "label": 1, "split": "val"},
        {"left_path": str(bright), "right_path": str(dark), "label": 0, "split": "val"},
    ]
    test_rows = [
        {"left_path": str(dark), "right_path": str(dark2), "label": 1, "split": "test"},
        {"left_path": str(bright2), "right_path": str(dark2), "label": 0, "split": "test"},
    ]

    with (pairs_dir / "val.jsonl").open("w") as file:
        for row in val_rows:
            file.write(json.dumps(row) + "\n")
    with (pairs_dir / "test.jsonl").open("w") as file:
        for row in test_rows:
            file.write(json.dumps(row) + "\n")

    config = {
        "run_name": "tiny_integration",
        "pairs_dir": str(pairs_dir),
        "split_for_threshold_selection": "val",
        "split_for_final_reporting": "test",
        "short_note_on_what_changed": "integration test",
        "image_mode": "L",
        "resize": [8, 8],
        "fixed_threshold": 0.9,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / ".matplotlib"))
    monkeypatch.setattr(sys, "argv", ["evaluator.py", "--config", str(config_path)])

    evaluator.main()

    summary_path = tmp_path / "outputs" / "runs" / "tiny_integration" / "tiny_integration_summary.json"
    val_scores_path = tmp_path / "outputs" / "runs" / "tiny_integration" / "tiny_integration_val_scores.jsonl"
    test_scores_path = tmp_path / "outputs" / "runs" / "tiny_integration" / "tiny_integration_test_scores.jsonl"

    assert summary_path.exists()
    assert val_scores_path.exists()
    assert test_scores_path.exists()

    summary = json.loads(summary_path.read_text())
    assert summary["run_identifier"] == "tiny_integration"
    assert set(summary["metrics"].keys()) == {"val", "test"}
    assert summary["threshold_information"]["threshold"] == 0.9
    assert "confusion_matrices" in summary
    assert set(summary["confusion_matrices"].keys()) == {"val", "test"}
