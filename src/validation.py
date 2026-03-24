import json
from pathlib import Path

import numpy as np

VALID_SPLITS = {"train", "val", "test"}
VALID_SELECTION_STRATEGIES = {"fixed_threshold", "threshold_sweep"}
REQUIRED_PAIR_KEYS = {"left_path", "right_path", "label", "split"}


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    with config_path.open("r") as file:
        return json.load(file)


def validate_threshold(threshold: float) -> None:
    if not np.isfinite(threshold):
        raise ValueError(f"Threshold must be finite, got {threshold}")
    if threshold < -1.0 or threshold > 1.0:
        raise ValueError(f"Threshold must be in [-1, 1] for cosine similarity, got {threshold}")


def validate_config(cfg: dict, config_path: str | Path) -> None:
    pairs_dir = Path(cfg.get("pairs_dir", "outputs/pairs"))
    if not pairs_dir.exists():
        raise FileNotFoundError(f"pairs_dir does not exist: {pairs_dir}")

    resize = cfg.get("resize", [32, 32])
    if not isinstance(resize, list) or len(resize) != 2:
        raise ValueError(f"resize must be a list of length 2 in {config_path}")
    if any(int(value) <= 0 for value in resize):
        raise ValueError(f"resize values must be positive in {config_path}")

    selection_strategy = cfg.get("selection_strategy", "fixed_threshold")
    if selection_strategy not in VALID_SELECTION_STRATEGIES:
        raise ValueError(
            f"selection_strategy must be one of {sorted(VALID_SELECTION_STRATEGIES)}, got {selection_strategy}"
        )

    val_split = cfg.get("split_for_threshold_selection", "val")
    if val_split not in VALID_SPLITS:
        raise ValueError(f"Invalid validation split: {val_split}")

    test_split = cfg.get("split_for_final_reporting", "test")
    if test_split not in VALID_SPLITS:
        raise ValueError(f"Invalid test split: {test_split}")
    if val_split == test_split:
        raise ValueError("Validation and test split names must differ to avoid split leakage")

    if selection_strategy == "fixed_threshold":
        validate_threshold(float(cfg.get("fixed_threshold", 0.9)))


def validate_pair_record(item: dict, path: Path, line_number: int, expected_split: str | None) -> None:
    missing = REQUIRED_PAIR_KEYS - set(item.keys())
    if missing:
        raise ValueError(f"Missing keys {sorted(missing)} in {path} line {line_number}")

    if item["label"] not in (0, 1):
        raise ValueError(f"Label must be 0 or 1 in {path} line {line_number}, got {item['label']}")

    split = item["split"]
    if split not in VALID_SPLITS:
        raise ValueError(f"Invalid split {split} in {path} line {line_number}")
    if expected_split is not None and split != expected_split:
        raise ValueError(f"Expected split={expected_split}, got {split} in {path} line {line_number}")

    left_path = Path(item["left_path"])
    right_path = Path(item["right_path"])
    if not left_path.exists():
        raise FileNotFoundError(f"Missing image path in {path} line {line_number}: {left_path}")
    if not right_path.exists():
        raise FileNotFoundError(f"Missing image path in {path} line {line_number}: {right_path}")


def read_pairs(path: str | Path, expected_split: str | None = None) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pair file does not exist: {path}")
    pairs = []
    with path.open("r") as file:
        for line_number, line in enumerate(file, start=1):
            item = json.loads(line)
            validate_pair_record(item=item, path=path, line_number=line_number, expected_split=expected_split)
            pairs.append(item)
    if not pairs:
        raise ValueError(f"Pair file is empty: {path}")
    return pairs


def validate_metrics(metrics: dict) -> None:
    required = {"tp", "tn", "fp", "fn", "accuracy", "balanced_accuracy", "precision", "recall", "f1"}
    missing = required - set(metrics.keys())
    if missing:
        raise ValueError(f"Missing metric keys: {sorted(missing)}")
    for key in ["accuracy", "balanced_accuracy", "precision", "recall", "f1"]:
        value = metrics[key]
        if not np.isfinite(value):
            raise ValueError(f"Metric {key} is not finite: {value}")


def validate_split_disjointness(val_rows: list[dict], test_rows: list[dict]) -> None:
    def pair_key(row: dict) -> tuple[str, str, int]:
        ordered = tuple(sorted([row["left_path"], row["right_path"]]))
        return ordered[0], ordered[1], int(row["label"])

    overlap = {pair_key(row) for row in val_rows}.intersection({pair_key(row) for row in test_rows})
    if overlap:
        raise ValueError(f"Detected {len(overlap)} overlapping pair artifacts between validation and test")
