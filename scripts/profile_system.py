import argparse
import json
import time
from pathlib import Path

import numpy as np

from scripts.embedding_generation import generate_embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def summarize(values):
    values = np.array(values, dtype=float)

    return {
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
    }


def load_pairs(pair_path: Path, limit: int):
    pairs = []

    with pair_path.open("r") as f:
        for line in f:
            if limit is not None and len(pairs) >= limit:
                break
            pairs.append(json.loads(line))

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Profile final ArcFace verifier stages for Milestone 4."
    )
    parser.add_argument(
        "--config",
        default="configs/arcface_best.json",
        help="Final ArcFace config file.",
    )
    parser.add_argument(
        "--pairs",
        default="outputs/pairs/test.jsonl",
        help="Pair file to profile.",
    )
    parser.add_argument(
        "--output",
        default="outputs/profiling/profile_cpu.json",
        help="Where to save profiling results.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Number of pairs to profile.",
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    pair_path = Path(args.pairs)
    output_path = Path(args.output)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if not pair_path.exists():
        raise FileNotFoundError(f"Pair file not found: {pair_path}")

    with config_path.open("r") as f:
        cfg = json.load(f)

    embedding_backend = cfg.get("embedding_backend", "arcface")
    threshold = cfg.get("threshold", cfg.get("fixed_threshold", None))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(pair_path, args.limit)

    preprocess_times = []
    embedding_times = []
    scoring_times = []
    total_times = []

    scores = []

    print(f"Profiling {len(pairs)} pairs from {pair_path}")
    print(f"Using config: {config_path}")
    print(f"Embedding backend: {embedding_backend}")

    for pair in pairs:
        left_path = pair["left_path"]
        right_path = pair["right_path"]

        total_start = time.perf_counter()

        # In this project, preprocessing is mostly represented by path validation
        # at this profiling level. Face detection/prep happens inside ArcFace embedding.
        preprocess_start = time.perf_counter()

        left_image_path = Path(left_path)
        right_image_path = Path(right_path)

        if not left_image_path.exists():
            raise FileNotFoundError(f"Left image path not found: {left_image_path}")

        if not right_image_path.exists():
            raise FileNotFoundError(f"Right image path not found: {right_image_path}")

        preprocess_end = time.perf_counter()

        # ArcFace embedding stage.
        # This includes image loading, face detection, face selection, and embedding extraction.
        embedding_start = time.perf_counter()

        left_embedding = generate_embedding(
            image_path=left_image_path,
            image_mode=cfg.get("image_mode", "RGB"),
            resize=cfg.get("resize", None),
            embedding_backend=embedding_backend,
            cfg=cfg,
        )

        right_embedding = generate_embedding(
            image_path=right_image_path,
            image_mode=cfg.get("image_mode", "RGB"),
            resize=cfg.get("resize", None),
            embedding_backend=embedding_backend,
            cfg=cfg,
        )
        embedding_end = time.perf_counter()

        # Similarity scoring stage.
        scoring_start = time.perf_counter()
        score = cosine_similarity(left_embedding, right_embedding)
        scoring_end = time.perf_counter()

        total_end = time.perf_counter()

        preprocess_times.append((preprocess_end - preprocess_start) * 1000)
        embedding_times.append((embedding_end - embedding_start) * 1000)
        scoring_times.append((scoring_end - scoring_start) * 1000)
        total_times.append((total_end - total_start) * 1000)

        scores.append(score)

    results = {
        "milestone": "4",
        "profile_type": "cpu_baseline",
        "config": str(config_path),
        "device": "cpu",
        "pair_file": str(pair_path),
        "num_pairs_profiled": len(pairs),
        "embedding_backend": embedding_backend,
        "threshold": threshold,
        "notes": (
            "Embedding timing includes image loading, face detection, face selection, "
            "and ArcFace embedding extraction because those steps are handled inside "
            "generate_embedding(). Scoring is measured separately."
        ),
        "preprocessing": summarize(preprocess_times),
        "embedding": summarize(embedding_times),
        "scoring": summarize(scoring_times),
        "end_to_end": summarize(total_times),
        "score_summary": {
            "mean_score": float(np.mean(scores)),
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
        },
    }

    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved profiling results to {output_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()