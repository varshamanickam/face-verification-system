import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean, median

import numpy as np

from scripts.confidence_computation import confidence_from_margin
from scripts.embedding_generation import generate_embedding
from src.similarity_metrics import cosine_similarity_vector


def run_one_request(left_image: str, right_image: str, cfg: dict, threshold: float, embedding_backend: str, pair_id: str) -> dict:
    start = time.perf_counter()
    try:
        left_embedding = generate_embedding(
            left_image,
            image_mode=cfg["image_mode"],
            resize=tuple(cfg["resize"]),
            embedding_backend=embedding_backend,
            cfg=cfg,
        )

        right_embedding = generate_embedding(
            right_image,
            image_mode=cfg["image_mode"],
            resize=tuple(cfg["resize"]),
            embedding_backend=embedding_backend,
            cfg=cfg,
        )

        score = float(
            cosine_similarity_vector(
            np.asarray([left_embedding], dtype=np.float64),
            np.asarray([right_embedding], dtype=np.float64),
        )[0]
)
        decision = int((np.asarray([score]) >= threshold).astype(np.int64)[0])
        confidence = float(confidence_from_margin(np.asarray([score]), threshold=threshold)[0])

        latency_ms = (time.perf_counter() - start) * 1000.0

        return {
            "pair_id": pair_id,
            "left_path": left_image,
            "right_path": right_image,
            "score": score,
            "threshold": threshold,
            "decision": decision,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "success": True,
        }
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000.0
        return {
            "pair_id": pair_id,
            "left_path": left_image,
            "right_path": right_image,
            "latency_ms": latency_ms,
            "success": False,
            "error": str(e),
        }


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def load_cfg(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


def build_requests() -> list[tuple[str, str, str]]:
    # Deterministic fixed request set for local load testing
    base_requests = [
        ("demo_same_1", "data/lfw/Aaron_Peirsol/Aaron_Peirsol_0001.jpg", "data/lfw/Aaron_Peirsol/Aaron_Peirsol_0002.jpg"),
        ("demo_diff_1", "data/lfw/Aaron_Peirsol/Aaron_Peirsol_0001.jpg", "data/lfw/George_W_Bush/George_W_Bush_0001.jpg"),
        ("demo_same_2", "data/lfw/George_W_Bush/George_W_Bush_0001.jpg", "data/lfw/George_W_Bush/George_W_Bush_0002.jpg"),
        ("demo_diff_2", "data/lfw/Winona_Ryder/Winona_Ryder_0001.jpg", "data/lfw/Tommy_Haas/Tommy_Haas_0001.jpg"),
    ]
    return base_requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config JSON, e.g. configs/arcface_best.json")
    parser.add_argument("--embedding-backend", default="arcface")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", default="outputs/load_test/load_test_summary.json")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    threshold = args.threshold if args.threshold is not None else float(cfg["fixed_threshold"])

    requests = build_requests() * args.repeats
    total_requests = len(requests)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    start_wall = time.perf_counter()
    results = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                run_one_request,
                left_image=left,
                right_image=right,
                cfg=cfg,
                threshold=threshold,
                embedding_backend=args.embedding_backend,
                pair_id=f"{pair_id}_{i}",
            )
            for i, (pair_id, left, right) in enumerate(requests)
        ]

        for future in as_completed(futures):
            results.append(future.result())

    total_wall_time_sec = time.perf_counter() - start_wall

    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]
    latencies = [r["latency_ms"] for r in successes]

    summary = {
        "config": args.config,
        "embedding_backend": args.embedding_backend,
        "workers": args.workers,
        "repeats": args.repeats,
        "total_requests": total_requests,
        "successful_requests": len(successes),
        "failed_requests": len(failures),
        "total_wall_time_sec": total_wall_time_sec,
        "throughput_req_per_sec": (len(successes) / total_wall_time_sec) if total_wall_time_sec > 0 else 0.0,
        "latency_ms": {
            "mean": mean(latencies) if latencies else 0.0,
            "median": median(latencies) if latencies else 0.0,
            "p95": percentile(latencies, 95),
            "min": min(latencies) if latencies else 0.0,
            "max": max(latencies) if latencies else 0.0,
        },
    }

    with open(args.output, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print("Load test summary")
    print(f"total_requests={summary['total_requests']}")
    print(f"successful_requests={summary['successful_requests']}")
    print(f"failed_requests={summary['failed_requests']}")
    print(f"total_wall_time_sec={summary['total_wall_time_sec']:.4f}")
    print(f"throughput_req_per_sec={summary['throughput_req_per_sec']:.4f}")
    print(f"latency_mean_ms={summary['latency_ms']['mean']:.4f}")
    print(f"latency_median_ms={summary['latency_ms']['median']:.4f}")
    print(f"latency_p95_ms={summary['latency_ms']['p95']:.4f}")
    print(f"summary_saved_to={args.output}")


if __name__ == "__main__":
    main()