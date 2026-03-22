import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.similarity_metrics import cosine_similarity_vector, euclidean_distance_vector

EPSILON = 1e-12


def load_config(config_path: Path) -> dict:
    with config_path.open("r") as file:
        return json.load(file)


def read_pairs(split_path: Path, expected_split: str) -> list[dict]:
    pairs = []
    with split_path.open("r") as file:
        for line in file:
            record = json.loads(line)
            if record["split"] != expected_split:
                raise ValueError(
                    f"Pair split mismatch in {split_path}: expected '{expected_split}', got '{record['split']}'"
                )
            pairs.append(record)
    if not pairs:
        raise ValueError(f"No pairs found in {split_path}")
    return pairs


def simple_image_embedding(image_path: Path, simple_cfg: dict) -> np.ndarray:
    mode = simple_cfg.get("image_mode", "L")
    resize_h, resize_w = simple_cfg.get("resize", [32, 32])
    with Image.open(image_path) as image:
        image = image.convert(mode)
        image = image.resize((resize_w, resize_h), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float64) / 255.0

    vector = array.reshape(-1)

    if simple_cfg.get("normalize") == "l2":
        norm = float(np.linalg.norm(vector))
        vector = vector / (norm + EPSILON)

    expected_dim = simple_cfg.get("expected_embedding_dim")
    if expected_dim is not None and vector.shape[0] != int(expected_dim):
        raise ValueError(f"Simple embedding dim mismatch: got {vector.shape[0]}, expected {expected_dim}")

    return vector


def score_pairs(
    pairs: list[dict],
    metric: str,
    simple_cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    embedding_cache: dict[str, np.ndarray] = {}
    left_vectors = []
    right_vectors = []
    labels = []

    for pair in pairs:
        left_path = pair["left_path"]
        right_path = pair["right_path"]

        if left_path not in embedding_cache:
            embedding_cache[left_path] = simple_image_embedding(
                Path(left_path),
                simple_cfg,
            )
        if right_path not in embedding_cache:
            embedding_cache[right_path] = simple_image_embedding(
                Path(right_path),
                simple_cfg,
            )

        left_vectors.append(embedding_cache[left_path])
        right_vectors.append(embedding_cache[right_path])
        labels.append(int(pair["label"]))

    left_matrix = np.vstack(left_vectors)
    right_matrix = np.vstack(right_vectors)

    if metric == "cosine_similarity":
        scores = cosine_similarity_vector(left_matrix, right_matrix)
    elif metric == "euclidean_distance":
        scores = euclidean_distance_vector(left_matrix, right_matrix)
    else:
        raise ValueError(f"Unsupported score_metric '{metric}'")

    return scores, np.asarray(labels, dtype=np.int64)


def threshold_candidates(scores: np.ndarray) -> np.ndarray:
    unique_scores = np.unique(scores)
    if unique_scores.size == 1:
        only = unique_scores[0]
        return np.asarray([only - 1e-9, only, only + 1e-9], dtype=np.float64)

    midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0
    return np.concatenate(
        [
            np.asarray([unique_scores[0] - 1e-9], dtype=np.float64),
            unique_scores,
            midpoints,
            np.asarray([unique_scores[-1] + 1e-9], dtype=np.float64),
        ]
    )


def predict_labels(scores: np.ndarray, threshold: float, score_direction: str) -> np.ndarray:
    if score_direction == "higher_is_more_likely_same_person":
        return (scores >= threshold).astype(np.int64)
    if score_direction == "lower_is_more_likely_same_person":
        return (scores <= threshold).astype(np.int64)
    raise ValueError(f"Unsupported score_direction '{score_direction}'")


def compute_split_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float, score_direction: str) -> dict:
    predictions = predict_labels(scores, threshold, score_direction)
    acc = float(np.mean(labels == predictions))
    tp = int(np.sum((predictions == 1) & (labels == 1)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float((2.0 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "num_pairs": int(labels.size),
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def threshold_sweep(
    output_path: Path,
    scores: np.ndarray,
    labels: np.ndarray,
    score_direction: str,
) -> tuple[float, float, int]:
    candidates = np.sort(threshold_candidates(scores))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_threshold = float(candidates[0])
    best_f1 = -1.0
    best_accuracy = -1.0

    with output_path.open("w") as file:
        for threshold in candidates:
            metrics = compute_split_metrics(scores, labels, float(threshold), score_direction)
            acc = float(metrics["accuracy"])
            f1 = float(metrics["f1"])
            if (f1 > best_f1) or (f1 == best_f1 and acc > best_accuracy):
                best_f1 = f1
                best_accuracy = acc
                best_threshold = float(threshold)

            file.write(
                json.dumps(
                    {
                        "threshold": float(threshold),
                        "accuracy": acc,
                        "precision": float(metrics["precision"]),
                        "recall": float(metrics["recall"]),
                        "f1": f1,
                        "tp": int(metrics["tp"]),
                        "tn": int(metrics["tn"]),
                        "fp": int(metrics["fp"]),
                        "fn": int(metrics["fn"]),
                    }
                )
                + "\n"
            )

    return best_threshold, best_f1, int(candidates.size)


def write_jsonl_scores(output_path: Path, pairs: list[dict], scores: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as file:
        for pair, score in zip(pairs, scores):
            file.write(
                json.dumps(
                    {
                        "left_path": pair["left_path"],
                        "right_path": pair["right_path"],
                        "label": int(pair["label"]),
                        "split": pair["split"],
                        "score": float(score),
                    }
                )
                + "\n"
            )


def run(config_path: Path) -> None:
    cfg = load_config(config_path)
    baseline_id = cfg["baseline_id"]
    pairs_dir = Path(cfg["pairs_dir"])
    val_split = cfg["split_for_threshold_selection"]
    test_split = cfg["split_for_final_reporting"]
    score_metric = cfg["score_metric"]
    score_direction = cfg["score_direction"]
    embedding_backend = cfg.get("embedding_backend", "simple_flatten")
    if embedding_backend != "simple_flatten":
        raise ValueError(
            "This script is configured for simple flatten embeddings. Set embedding_backend='simple_flatten'."
        )
    simple_cfg = cfg.get("simple_embedding", {})

    val_pairs = read_pairs(pairs_dir / f"{val_split}.jsonl", expected_split=val_split)
    test_pairs = read_pairs(pairs_dir / f"{test_split}.jsonl", expected_split=test_split)

    val_scores, val_labels = score_pairs(
        val_pairs,
        score_metric,
        simple_cfg,
    )
    test_scores, test_labels = score_pairs(
        test_pairs,
        score_metric,
        simple_cfg,
    )

    output_dir = Path("outputs") / "baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_path = output_dir / f"{baseline_id}_{val_split}_threshold_sweep.jsonl"
    threshold, val_f1, sweep_count = threshold_sweep(
        sweep_path,
        val_scores,
        val_labels,
        score_direction,
    )
    val_metrics = compute_split_metrics(val_scores, val_labels, threshold, score_direction)
    test_metrics = compute_split_metrics(test_scores, test_labels, threshold, score_direction)


    write_jsonl_scores(output_dir / f"{baseline_id}_{val_split}_scores.jsonl", val_pairs, val_scores)
    write_jsonl_scores(output_dir / f"{baseline_id}_{test_split}_scores.jsonl", test_pairs, test_scores)

    summary = {
        "baseline_id": baseline_id,
        "config_path": str(config_path),
        "seed": int(cfg["seed"]),
        "pairs_dir": str(pairs_dir),
        "split_for_threshold_selection": val_split,
        "split_for_final_reporting": test_split,
        "score_metric": score_metric,
        "score_direction": score_direction,
        "embedding_backend": embedding_backend,
        "validation_threshold_sweep_path": str(sweep_path),
        "validation_threshold_candidates": int(sweep_count),
        "threshold_selection_metric": "f1",
        "selected_threshold": float(threshold),
        "threshold_selection_f1": float(val_f1),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    summary_path = output_dir / f"{baseline_id}_summary.json"
    with summary_path.open("w") as file:
        json.dump(summary, file, indent=2)

    print(f"Baseline ID: {baseline_id}")
    print(f"Threshold split: {val_split}")
    print(f"Reporting split: {test_split}")
    print(f"Score metric: {score_metric}")
    print(f"Score direction: {score_direction}")
    print(f"Embedding backend: {embedding_backend}")
    print(f"Validation threshold sweep candidates: {sweep_count}")
    print(f"Wrote threshold sweep to {sweep_path}")
    print(f"Selected threshold: {threshold:.6f}")
    print(f"Validation F1 @ threshold: {val_metrics['f1']:.4f}")
    print(f"Validation accuracy @ threshold: {val_metrics['accuracy']:.4f}")
    print(f"Test accuracy @ threshold: {test_metrics['accuracy']:.4f}")
    print(f"Wrote summary to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic baseline face verification scorer")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline_m1.json"),
        help="Path to baseline config JSON",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()