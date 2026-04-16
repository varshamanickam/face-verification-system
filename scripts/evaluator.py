import argparse
import csv
import json
import subprocess
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.confidence_computation import add_confidence_to_rows, summarize_confidence
from scripts.embedding_generation import (build_image_cache,
                                         generate_embedding,
                                         pairs_to_arrays,
                                         preprocess_image_arcface)
from scripts.latency_measurement import measure_latency_ms
from scripts.preprocessing import preprocess_image
from scripts.similarity_scoring import score_split as score_split_stage
from scripts.threshold_decision import (compute_metrics,
                                        evaluate_scored_split,
                                        threshold_candidates,
                                        threshold_sweep)
from src.validation import (load_config, validate_config,
                            validate_split_disjointness)


def _load_inference_pairs(pairs_file: Path) -> list[dict]:
    if not pairs_file.exists():
        raise FileNotFoundError(f"pairs file does not exist: {pairs_file}")

    rows: list[dict] = []
    suffix = pairs_file.suffix.lower()
    if suffix == ".jsonl":
        with pairs_file.open("r") as file:
            for index, line in enumerate(file, start=1):
                item = json.loads(line)
                if "left_path" not in item or "right_path" not in item:
                    raise ValueError(f"Missing left_path/right_path in {pairs_file} line {index}")
                pair_id = item.get("pair_id", f"pair_{index}")
                rows.append(
                    {
                        "pair_id": str(pair_id),
                        "left_path": str(item["left_path"]),
                        "right_path": str(item["right_path"]),
                    }
                )
        return rows

    if suffix == ".csv":
        with pairs_file.open("r", newline="") as file:
            reader = csv.DictReader(file)
            for index, item in enumerate(reader, start=1):
                if "left_path" not in item or "right_path" not in item:
                    raise ValueError(f"CSV must have left_path,right_path columns: {pairs_file}")
                pair_id = item.get("pair_id") or f"pair_{index}"
                rows.append(
                    {
                        "pair_id": str(pair_id),
                        "left_path": str(item["left_path"]),
                        "right_path": str(item["right_path"]),
                    }
                )
        return rows

    raise ValueError(f"Unsupported pairs file format: {pairs_file}. Use .jsonl or .csv")


def _resolve_inference_threshold(cfg: dict, threshold_arg: float | None) -> float:
    if threshold_arg is not None:
        return float(threshold_arg)
    return float(cfg.get("fixed_threshold", 0.9))


def _run_inference_pair(
    pair_id: str,
    left_path: str,
    right_path: str,
    embedding_backend: str,
    cfg: dict,
    image_mode: str,
    resize: tuple[int, int],
    threshold: float,
) -> dict:
    left_embedding = generate_embedding(
        image_path=left_path,
        embedding_backend=embedding_backend,
        cfg=cfg,
        image_mode=image_mode,
        resize=resize,
    )
    right_embedding = generate_embedding(
        image_path=right_path,
        embedding_backend=embedding_backend,
        cfg=cfg,
        image_mode=image_mode,
        resize=resize,
    )

    score = float(np.dot(left_embedding, right_embedding))
    decision = int(score >= threshold)
    confidence_row = add_confidence_to_rows(
        [
            {
                "pair_id": pair_id,
                "left_path": left_path,
                "right_path": right_path,
                "score": score,
            }
        ],
        threshold=threshold,
    )[0]

    return {
        "pair_id": pair_id,
        "left_path": left_path,
        "right_path": right_path,
        "score": score,
        "threshold": float(threshold),
        "decision": decision,
        "confidence": float(confidence_row["confidence"]),
    }


def _print_inference_result(result: dict) -> None:
    print(f"pair_id={result['pair_id']}")
    print(f"left_path={result['left_path']}")
    print(f"right_path={result['right_path']}")
    print(f"score={result['score']:.6f}")
    print(f"threshold={result['threshold']:.6f}")
    print(f"decision={result['decision']}")
    print(f"confidence={result['confidence']:.6f}")
    print(f"latency_ms={result['latency_ms']:.3f}")
    print("---")


def run_inference_mode(args: argparse.Namespace, cfg: dict, embedding_backend: str) -> None:
    image_mode = cfg.get("image_mode", "L")
    resize = tuple(int(value) for value in cfg.get("resize", [32, 32]))
    threshold = _resolve_inference_threshold(cfg, args.threshold)

    pairs: list[dict]
    if args.pairs_file is not None:
        pairs = _load_inference_pairs(Path(args.pairs_file))
    else:
        pairs = [
            {
                "pair_id": args.pair_id or "pair_1",
                "left_path": str(args.left_image),
                "right_path": str(args.right_image),
            }
        ]

    for item in pairs:
        result, latency_ms = measure_latency_ms(
            _run_inference_pair,
            pair_id=item["pair_id"],
            left_path=item["left_path"],
            right_path=item["right_path"],
            embedding_backend=embedding_backend,
            cfg=cfg,
            image_mode=image_mode,
            resize=resize,
            threshold=threshold,
        )
        result["latency_ms"] = float(latency_ms)
        _print_inference_result(result)

def build_confusion_matrix_dict(metrics: dict) -> dict:
    return {
        "rows": ["actual_1", "actual_0"],
        "cols": ["pred_1", "pred_0"],
        "matrix": [
            [metrics["tp"], metrics["fn"]],
            [metrics["fp"], metrics["tn"]],
        ],
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


def score_split(
    split: str,
    pairs_dir: Path,
    image_mode: str,
    resize: tuple[int, int],
    embedding_backend: str = "raw_pixels",
    cfg: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    return score_split_stage(
        split=split,
        pairs_dir=pairs_dir,
        image_mode=image_mode,
        resize=resize,
        embedding_backend=embedding_backend,
        cfg=cfg,
    )


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def save_roc_style_plot(path: Path, sweep_rows: list[dict]) -> None:
    roc_points = []
    seen = set()
    for row in sweep_rows:
        fp = row["fp"]
        tn = row["tn"]
        tp = row["tp"]
        fn = row["fn"]
        fpr = fp / max(fp + tn, 1)
        tpr = tp / max(tp + fn, 1)
        key = (round(fpr, 12), round(tpr, 12))
        if key not in seen:
            seen.add(key)
            roc_points.append((fpr, tpr))

    roc_points.sort()
    path.parent.mkdir(parents=True, exist_ok=True)
    fprs = [point[0] for point in roc_points]
    tprs = [point[1] for point in roc_points]

    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=100)
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", linewidth=1.5, label="Random")
    ax.plot(fprs, tprs, color="#2463eb", linewidth=2.5, marker="o", markersize=3, label="Sweep ROC")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Validation ROC-Style Curve")
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a simple non-learnable face embedding baseline.")
    parser.add_argument(
        "--config",
        default="configs/baseline.json",
        help="Path to evaluation config JSON.",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["raw_pixels", "arcface"],
        default=None,
        help="Override embedding backend. If omitted, use config key embedding_backend.",
    )
    parser.add_argument("--left-image", default=None, help="Single-pair mode: left image path.")
    parser.add_argument("--right-image", default=None, help="Single-pair mode: right image path.")
    parser.add_argument(
        "--pairs-file",
        default=None,
        help="Batch mode file (.jsonl or .csv) with columns/keys left_path,right_path and optional pair_id.",
    )
    parser.add_argument("--pair-id", default=None, help="Optional pair identifier for single-pair mode.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold used for inference decision. Defaults to config fixed_threshold or 0.9.",
    )
    args = parser.parse_args()

    has_single_pair = args.left_image is not None or args.right_image is not None
    if has_single_pair and (args.left_image is None or args.right_image is None):
        raise ValueError("Single-pair mode requires both --left-image and --right-image")
    if args.pairs_file is not None and has_single_pair:
        raise ValueError("Use either --pairs-file or --left-image/--right-image, not both")

    cfg = load_config(args.config)
    embedding_backend = args.embedding_backend or cfg.get("embedding_backend", "raw_pixels")
    if embedding_backend not in {"raw_pixels", "arcface"}:
        raise ValueError(f"Unsupported embedding_backend: {embedding_backend}")

    if args.pairs_file is not None or has_single_pair:
        run_inference_mode(args=args, cfg=cfg, embedding_backend=embedding_backend)
        return

    validate_config(cfg, args.config)
    run_name = cfg.get("run_name", Path(args.config).stem)
    pairs_dir = Path(cfg.get("pairs_dir", "outputs/pairs"))
    image_mode = cfg.get("image_mode", "L")
    resize = tuple(int(value) for value in cfg.get("resize", [32, 32]))
    selection_strategy = cfg.get("selection_strategy", "fixed_threshold")
    short_note = cfg.get("short_note_on_what_changed", "")
    val_split = cfg.get("split_for_threshold_selection", "val")
    latency_ms = {}

    (val_labels, val_score_values, val_scores), latency_ms["similarity_scoring_val"] = measure_latency_ms(
        score_split,
        split=val_split,
        pairs_dir=pairs_dir,
        image_mode=image_mode,
        resize=resize,
        embedding_backend=embedding_backend,
        cfg=cfg,
    )

    output_dir = Path("outputs") / "runs" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if selection_strategy == "threshold_sweep":
        (
            best_threshold,
            val_metrics,
            sweep_rows,
        ), latency_ms["threshold_decision"] = measure_latency_ms(
            threshold_sweep,
            labels=val_labels,
            scores=val_score_values,
        )
        test_split = cfg.get("split_for_final_reporting", "test")
        (test_labels, test_score_values, test_scores), latency_ms["similarity_scoring_test"] = measure_latency_ms(
            score_split,
            split=test_split,
            pairs_dir=pairs_dir,
            image_mode=image_mode,
            resize=resize,
            embedding_backend=embedding_backend,
            cfg=cfg,
        )
        validate_split_disjointness(val_scores, test_scores)

        val_scores, latency_ms["confidence_computation_val"] = measure_latency_ms(
            add_confidence_to_rows,
            score_rows=val_scores,
            threshold=best_threshold,
        )
        test_scores, latency_ms["confidence_computation_test"] = measure_latency_ms(
            add_confidence_to_rows,
            score_rows=test_scores,
            threshold=best_threshold,
        )

        val_confidence = summarize_confidence(val_scores)
        test_confidence = summarize_confidence(test_scores)

        save_jsonl(output_dir / f"{run_name}_{val_split}_scores.jsonl", val_scores)
        save_jsonl(output_dir / f"{run_name}_{val_split}_threshold_sweep.jsonl", sweep_rows)
        save_jsonl(output_dir / f"{run_name}_{test_split}_scores.jsonl", test_scores)
        save_roc_style_plot(output_dir / f"{run_name}_{val_split}_roc.png", sweep_rows)
        test_metrics = evaluate_scored_split(labels=test_labels, scores=test_score_values, threshold=best_threshold)
        threshold_information = {
            "selection_strategy": "threshold_sweep",
            "selection_split": val_split,
            "selection_metric": "f1",
            "score_metric": f"cosine_similarity_{embedding_backend}",
            "best_threshold": best_threshold,
            "num_candidates": len(sweep_rows),
        }
        metrics = {
            val_split: val_metrics,
            test_split: test_metrics,
        }
        confusion_matrices = {
            val_split: build_confusion_matrix_dict(val_metrics),
            test_split: build_confusion_matrix_dict(test_metrics),
        }
    else:
        threshold = float(cfg.get("fixed_threshold", 0.9))
        test_split = cfg.get("split_for_final_reporting", "test")
        (test_labels, test_score_values, test_scores), latency_ms["similarity_scoring_test"] = measure_latency_ms(
            score_split,
            split=test_split,
            pairs_dir=pairs_dir,
            image_mode=image_mode,
            resize=resize,
            embedding_backend=embedding_backend,
            cfg=cfg,
        )
        validate_split_disjointness(val_scores, test_scores)

        val_scores, latency_ms["confidence_computation_val"] = measure_latency_ms(
            add_confidence_to_rows,
            score_rows=val_scores,
            threshold=threshold,
        )
        test_scores, latency_ms["confidence_computation_test"] = measure_latency_ms(
            add_confidence_to_rows,
            score_rows=test_scores,
            threshold=threshold,
        )

        val_confidence = summarize_confidence(val_scores)
        test_confidence = summarize_confidence(test_scores)

        save_jsonl(output_dir / f"{run_name}_{val_split}_scores.jsonl", val_scores)
        save_jsonl(output_dir / f"{run_name}_{test_split}_scores.jsonl", test_scores)
        val_metrics, latency_ms["threshold_decision_val"] = measure_latency_ms(
            evaluate_scored_split,
            labels=val_labels,
            scores=val_score_values,
            threshold=threshold,
        )
        test_metrics, latency_ms["threshold_decision_test"] = measure_latency_ms(
            evaluate_scored_split,
            labels=test_labels,
            scores=test_score_values,
            threshold=threshold,
        )
        threshold_information = {
            "selection_strategy": "fixed_threshold",
            "score_metric": f"cosine_similarity_{embedding_backend}",
            "threshold": threshold,
        }
        metrics = {
            val_split: val_metrics,
            test_split: test_metrics,
        }
        confusion_matrices = {
            val_split: build_confusion_matrix_dict(val_metrics),
            test_split: build_confusion_matrix_dict(test_metrics),
        }

    summary = {
        "run_identifier": run_name,
        "commit_hash": get_git_commit_hash(),
        "config_path": str(args.config),
        "pairs_dir": str(pairs_dir),
        "threshold_information": threshold_information,
        "metrics": metrics,
        "confidence": {
            val_split: val_confidence,
            test_split: test_confidence,
        },
        "latency_ms": latency_ms,
        "confusion_matrices": confusion_matrices,
        "short_note_on_what_changed": short_note,
    }

    summary_path = output_dir / f"{run_name}_summary.json"
    with summary_path.open("w") as file:
        json.dump(summary, file, indent=2)

    print(f"Saved summary to {summary_path}")
    print(f"{val_split} accuracy={val_metrics['accuracy']:.4f} f1={val_metrics['f1']:.4f}")
    print(f"{test_split} accuracy={test_metrics['accuracy']:.4f} f1={test_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
