import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np

from src.learned_projection import load_projection_model, preprocess_image, project_vectors
from src.similarity_metrics import cosine_similarity_vector, euclidean_distance_vector


def get_git_commit_info() -> dict:
    try:
        full_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        short_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return {
            "git_commit": full_hash,
            "git_commit_short": short_hash,
        }
    except Exception:
        return {
            "git_commit": None,
            "git_commit_short": None,
        }


def load_config(config_path: Path) -> dict:
    with config_path.open("r") as file:
        return json.load(file)


def load_manifest(manifest_path: Path) -> dict | None:
    if not manifest_path.exists():
        return None
    with manifest_path.open("r") as file:
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


def learned_projection_embedding(image_path: Path, learned_cfg: dict, model: dict) -> np.ndarray:
    image_mode = learned_cfg.get("image_mode", model["image_mode"])
    resize = learned_cfg.get("resize", model["resize"])
    vector = preprocess_image(image_path, image_mode, resize)
    input_dim = int(model["input_dim"])
    if vector.shape[0] != input_dim:
        raise ValueError(f"Projection input dim mismatch: got {vector.shape[0]}, expected {input_dim}")

    embedding = project_vectors(
        vector[np.newaxis, :],
        model["weights"],
        model["bias"],
        learned_cfg.get("normalize_output", model["normalize_output"]),
    )[0]
    expected_dim = learned_cfg.get("expected_embedding_dim", model["embedding_dim"])
    if embedding.shape[0] != int(expected_dim):
        raise ValueError(f"Projection embedding dim mismatch: got {embedding.shape[0]}, expected {expected_dim}")
    return embedding


def score_pairs(
    pairs: list[dict],
    metric: str,
    embedding_cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    embedding_cache: dict[str, np.ndarray] = {}
    left_vectors = []
    right_vectors = []
    labels = []
    model_path = Path(embedding_cfg["model_path"])
    model = load_projection_model(model_path)

    for pair in pairs:
        left_path = pair["left_path"]
        right_path = pair["right_path"]

        if left_path not in embedding_cache:
            embedding_cache[left_path] = learned_projection_embedding(Path(left_path), embedding_cfg, model)
        if right_path not in embedding_cache:
            embedding_cache[right_path] = learned_projection_embedding(Path(right_path), embedding_cfg, model)

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

    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    tnr = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    balanced_accuracy = float((tpr + tnr) / 2.0)
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = tpr
    f1 = float((2.0 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "num_pairs": int(labels.size),
        "accuracy": acc,
        "balanced_accuracy": balanced_accuracy,
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
                        "balanced_accuracy": float(metrics["balanced_accuracy"]),
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
    run_name = cfg.get("run_name")
    run_identifier = run_name if run_name else config_path.stem
    output_stem = run_identifier
    output_dir_name = output_stem
    config_setting_name = cfg.get("config_setting_name", config_path.stem)
    pairs_dir = Path(cfg["pairs_dir"])
    val_split = cfg["split_for_threshold_selection"]
    test_split = cfg["split_for_final_reporting"]
    score_metric = cfg["score_metric"]
    score_direction = cfg["score_direction"]
    embedding_backend = "learned_projection"
    embedding_cfg = cfg.get("learned_projection", {})
    if "model_path" not in embedding_cfg:
        raise ValueError("learned_projection.model_path must be set in the evaluator config")

    val_pairs = read_pairs(pairs_dir / f"{val_split}.jsonl", expected_split=val_split)
    test_pairs = read_pairs(pairs_dir / f"{test_split}.jsonl", expected_split=test_split)

    val_scores, val_labels = score_pairs(
        val_pairs,
        score_metric,
        embedding_cfg,
    )
    test_scores, test_labels = score_pairs(
        test_pairs,
        score_metric,
        embedding_cfg,
    )

    output_dir = Path("outputs") / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path("outputs") / "manifest.json"
    manifest = load_manifest(manifest_path)

    sweep_path = output_dir / f"{output_stem}_{val_split}_threshold_sweep.jsonl"
    threshold, val_f1, sweep_count = threshold_sweep(
        sweep_path,
        val_scores,
        val_labels,
        score_direction,
    )
    val_metrics = compute_split_metrics(val_scores, val_labels, threshold, score_direction)
    test_metrics = compute_split_metrics(test_scores, test_labels, threshold, score_direction)


    write_jsonl_scores(output_dir / f"{output_stem}_{val_split}_scores.jsonl", val_pairs, val_scores)
    write_jsonl_scores(output_dir / f"{output_stem}_{test_split}_scores.jsonl", test_pairs, test_scores)

    commit_info = get_git_commit_info()
    evaluated_at_utc = datetime.now(timezone.utc).isoformat()

    data_or_pair_version = {
        "pairs_dir": str(pairs_dir),
        "manifest_path": str(manifest_path) if manifest is not None else None,
        "manifest_seed": int(manifest["seed"]) if manifest and "seed" in manifest else None,
        "split_policy": manifest.get("split_policy") if manifest else None,
        "description": cfg.get(
            "data_or_pair_version",
            "Evaluation used the pair files in pairs_dir and the current manifest when available.",
        ),
    }

    summary = {
        "run_identifier": run_identifier,
        "timestamp_utc": evaluated_at_utc,
        "tracking": {
            "output_dir": str(output_dir),
            "git_commit": commit_info["git_commit"],
            "git_commit_short": commit_info["git_commit_short"],
        },
        "config_or_run_setting_name": {
            "name": config_setting_name,
            "config_path": str(config_path),
            "embedding_backend": embedding_backend,
            "score_metric": score_metric,
            "score_direction": score_direction,
            "split_for_threshold_selection": val_split,
            "split_for_final_reporting": test_split,
            "seed": int(cfg["seed"]),
        },
        "data_or_pair_version": data_or_pair_version,
        "threshold_information": {
            "selection_metric": "f1",
            "validation_threshold_sweep_path": str(sweep_path),
            "validation_threshold_candidates": int(sweep_count),
            "selected_threshold": float(threshold),
            "selected_threshold_validation_f1": float(val_f1),
        },
        "metrics": {
            "validation": val_metrics,
            "test": test_metrics,
        },
        
    }

    summary_path = output_dir / f"{output_stem}_summary.json"
    with summary_path.open("w") as file:
        json.dump(summary, file, indent=2)

    print(f"Run identifier: {run_identifier}")
    print(f"Timestamp (UTC): {evaluated_at_utc}")
    print(f"Config setting name: {config_setting_name}")
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
    print(f"Validation balanced accuracy @ threshold: {val_metrics['balanced_accuracy']:.4f}")
    print(f"Validation precision @ threshold: {val_metrics['precision']:.4f}")
    print(f"Validation recall @ threshold: {val_metrics['recall']:.4f}")
    print(f"Test accuracy @ threshold: {test_metrics['accuracy']:.4f}")
    print(f"Test balanced accuracy @ threshold: {test_metrics['balanced_accuracy']:.4f}")
    print(f"Test precision @ threshold: {test_metrics['precision']:.4f}")
    print(f"Test recall @ threshold: {test_metrics['recall']:.4f}")
    print(f"Test F1 @ threshold: {test_metrics['f1']:.4f}")
    print(f"Wrote summary to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic baseline face verification scorer")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline.json"),
        help="Path to baseline config JSON",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
