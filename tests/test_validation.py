import json
from pathlib import Path

import pytest

from src import validation


def test_validate_threshold_rejects_out_of_range():
    with pytest.raises(ValueError):
        validation.validate_threshold(1.5)


def test_read_pairs_rejects_invalid_label(tmp_path):
    image_path = tmp_path / "img.jpg"
    image_path.write_bytes(b"not_used")

    pair_path = tmp_path / "val.jsonl"
    pair_path.write_text(
        json.dumps(
            {
                "left_path": str(image_path),
                "right_path": str(image_path),
                "label": 2,
                "split": "val",
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError):
        validation.read_pairs(pair_path, expected_split="val")


def test_read_pairs_rejects_missing_required_key(tmp_path):
    image_path = tmp_path / "img.jpg"
    image_path.write_bytes(b"not_used")

    pair_path = tmp_path / "val.jsonl"
    pair_path.write_text(
        json.dumps(
            {
                "left_path": str(image_path),
                "label": 1,
                "split": "val",
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError):
        validation.read_pairs(pair_path, expected_split="val")


def test_validate_split_disjointness_detects_overlap():
    val_rows = [
        {"left_path": "a.jpg", "right_path": "b.jpg", "label": 1},
        {"left_path": "c.jpg", "right_path": "d.jpg", "label": 0},
    ]
    test_rows = [
        {"left_path": "b.jpg", "right_path": "a.jpg", "label": 1},
    ]

    with pytest.raises(ValueError):
        validation.validate_split_disjointness(val_rows, test_rows)


def test_validate_config_rejects_same_val_and_test_split(tmp_path):
    pairs_dir = tmp_path / "pairs"
    pairs_dir.mkdir()
    cfg = {
        "pairs_dir": str(pairs_dir),
        "resize": [32, 32],
        "split_for_threshold_selection": "val",
        "split_for_final_reporting": "val",
        "selection_strategy": "fixed_threshold",
        "fixed_threshold": 0.9,
    }

    with pytest.raises(ValueError):
        validation.validate_config(cfg, "dummy.json")
