import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

from src.similarity_metrics import cosine_similarity_vector


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    with config_path.open("r") as file:
        return json.load(file)


def read_pairs(path: str | Path, expected_split: str | None = None) -> list[dict]:
    path = Path(path)
    pairs = []
    with path.open("r") as file:
        for line in file:
            item = json.loads(line)
            if expected_split is not None and item["split"] != expected_split:
                raise ValueError(f"Expected split={expected_split}, got {item['split']} in {path}")
            pairs.append(item)
    return pairs


def preprocess_image(image_path: str | Path, image_mode: str, resize: tuple[int, int]) -> np.ndarray:
    image_path = Path(image_path)
    with Image.open(image_path) as image:
        image = image.convert(image_mode)
        image = image.resize((resize[1], resize[0]))
        array = np.asarray(image, dtype=np.float64) / 255.0

    flat = array.reshape(-1)
    norm = np.linalg.norm(flat)
    if norm > 0:
        flat = flat / norm
    return flat


def build_image_cache(
    pairs: list[dict],
    image_mode: str,
    resize: tuple[int, int],
) -> dict[str, np.ndarray]:
    image_cache: dict[str, np.ndarray] = {}
    unique_paths = sorted(
        {item["left_path"] for item in pairs}.union({item["right_path"] for item in pairs})
    )
    for image_path in unique_paths:
        image_cache[image_path] = preprocess_image(image_path, image_mode=image_mode, resize=resize)
    return image_cache


def pairs_to_arrays(
    pairs: list[dict],
    image_cache: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left = np.stack([image_cache[item["left_path"]] for item in pairs], axis=0)
    right = np.stack([image_cache[item["right_path"]] for item in pairs], axis=0)
    labels = np.asarray([item["label"] for item in pairs], dtype=np.int64)
    return left, right, labels


def compute_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    tp = int(np.sum((predictions == 1) & (labels == 1)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))

    accuracy = (tp + tn) / max(labels.shape[0], 1)
    true_positive_rate = tp / max(tp + fn, 1)
    true_negative_rate = tn / max(tn + fp, 1)
    balanced_accuracy = 0.5 * (true_positive_rate + true_negative_rate)
    precision = tp / max(tp + fp, 1)
    recall = true_positive_rate
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def get_git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip()


def evaluate_split(
    split: str,
    pairs_dir: Path,
    image_mode: str,
    resize: tuple[int, int],
    threshold: float,
) -> tuple[dict, list[dict]]:
    pairs = read_pairs(pairs_dir / f"{split}.jsonl", expected_split=split)
    image_cache = build_image_cache(pairs, image_mode=image_mode, resize=resize)
    left_inputs, right_inputs, labels = pairs_to_arrays(pairs, image_cache)

    scores = cosine_similarity_vector(left_inputs, right_inputs)
    predictions = (scores >= threshold).astype(np.int64)
    metrics = compute_metrics(labels=labels, predictions=predictions)

    score_rows = []
    for pair, score, prediction in zip(pairs, scores.tolist(), predictions.tolist()):
        score_rows.append(
            {
                "left_path": pair["left_path"],
                "right_path": pair["right_path"],
                "label": int(pair["label"]),
                "score": float(score),
                "prediction": int(prediction),
                "split": split,
            }
        )
    return metrics, score_rows


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a simple non-learnable face embedding baseline.")
    parser.add_argument(
        "--config",
        default="configs/baseline.json",
        help="Path to evaluation config JSON.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_name = cfg.get("run_name", Path(args.config).stem)
    pairs_dir = Path(cfg.get("pairs_dir", "outputs/pairs"))
    image_mode = cfg.get("image_mode", "L")
    resize = tuple(int(value) for value in cfg.get("resize", [32, 32]))
    threshold = float(cfg.get("fixed_threshold", 0.9))
    val_split = cfg.get("split_for_threshold_selection", "val")
    test_split = cfg.get("split_for_final_reporting", "test")

    val_metrics, val_scores = evaluate_split(
        split=val_split,
        pairs_dir=pairs_dir,
        image_mode=image_mode,
        resize=resize,
        threshold=threshold,
    )
    test_metrics, test_scores = evaluate_split(
        split=test_split,
        pairs_dir=pairs_dir,
        image_mode=image_mode,
        resize=resize,
        threshold=threshold,
    )

    output_dir = Path("outputs") / "runs" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(output_dir / f"{run_name}_{val_split}_scores.jsonl", val_scores)
    save_jsonl(output_dir / f"{run_name}_{test_split}_scores.jsonl", test_scores)

    summary = {
        "run_identifier": run_name,
        "commit_hash": get_git_commit_hash(),
        "config_path": str(args.config),
        "pairs_dir": str(pairs_dir),
        "threshold_information": {
            "selection_strategy": "fixed_threshold",
            "score_metric": "cosine_similarity",
            "threshold": threshold,
        },
        "metrics": {
            val_split: val_metrics,
            test_split: test_metrics,
        },
        "short_note_on_what_changed": (
            "Simple non-learnable baseline: grayscale resize, flatten, L2 normalize, cosine similarity."
        ),
    }

    summary_path = output_dir / f"{run_name}_summary.json"
    with summary_path.open("w") as file:
        json.dump(summary, file, indent=2)

    print(f"Saved summary to {summary_path}")
    print(f"{val_split} accuracy={val_metrics['accuracy']:.4f} f1={val_metrics['f1']:.4f}")
    print(f"{test_split} accuracy={test_metrics['accuracy']:.4f} f1={test_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
